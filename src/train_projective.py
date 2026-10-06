"""Anchored inverse-depth projective geometry training.

Loads the half-Gaussian reconstruction (``--load_iteration``), re-parameterises the free
``xyz`` as anchored inverse depth (``_w_raw`` / ``_uv`` through a fixed gauge,
see ``geometry.projective``) and trains geometry + colour + medium jointly:

    x_i = a_k + u_i e1_k + v_i e2_k + d_i n_k,   d_i = d0_k (1/w_i - 1)
    w_i = eps + (1-eps) sigmoid(w_raw_i) in [eps, 1)     (finite depth)

Loss = photometric on ``I_hat = t*J + B``  +  multi-view geometry loss
(two-view depth reprojection consistency + edge-aware depth smoothness). The
depth output is the *exact* optical path ``r = ||x - c||`` and is now
differentiable, so the geometry loss trains ``_w_raw`` / ``_uv``.

Scaling/rotation/opacity/normal are frozen (projection geometry is isolated);
no densification (adaptation to the reparameterisation is deferred).
"""

import os
import sys
import torch
from random import randint
from argparse import ArgumentParser
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from utils.general_utils import safe_state, get_expon_lr_func
from gaussian_renderer import render
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from medium import WaterImageModel
from geometry.losses import depth_reproj_consistency, edge_aware_depth_smooth


def compute_mean_cam_center(scene):
    centers = [cam.camera_center.cpu() for cam in scene.getTrainCameras()]
    return torch.stack(centers).float().mean(dim=0)


def load_medium_init(model_path):
    ckpt = os.path.join(model_path, "medium", "water_model.pth")
    if os.path.exists(ckpt):
        return torch.load(ckpt, map_location="cpu")
    return None


def evaluate(scene, water_model, pipe, background, iteration, args):
    torch.cuda.empty_cache()
    g = scene.gaussians

    w = g.geo.w_from_raw(g._w_raw).detach()
    print("[ITER {}] w range [{:.5f}, {:.5f}]  (eps={})  depth>0: {}".format(
        iteration, w.min().item(), w.max().item(), g.geo.eps,
        bool((w > 0).all().item())))

    for name, cameras in (
        ("test", scene.getTestCameras()),
        ("train", scene.getTrainCameras()[:min(5, len(scene.getTrainCameras()))]),
    ):
        if not cameras:
            continue
        l1 = psnr_v = ssim_v = 0.0
        for vp in cameras:
            with torch.no_grad():
                pkg = render(vp, g, pipe, background)
                I_hat, _, _ = water_model(pkg["render"], pkg["depth"])
                I_hat = torch.clamp(I_hat, 0.0, 1.0)
                gt = torch.clamp(vp.original_image.to("cuda"), 0.0, 1.0)
            l1 += l1_loss(I_hat, gt).item()
            psnr_v += psnr(I_hat, gt).mean().item()
            ssim_v += ssim(I_hat, gt).item()
        n = len(cameras)
        print("[ITER {}] {}: L1 {:.5f} PSNR {:.2f} SSIM {:.4f}".format(
            iteration, name, l1 / n, psnr_v / n, ssim_v / n))

    # Cross-view depth consistency on the first two training cameras.
    cams = scene.getTrainCameras()
    if len(cams) >= 2:
        with torch.no_grad():
            pa = render(cams[0], g, pipe, background)
            pb = render(cams[1], g, pipe, background)
        da, db = pa["depth"], pb["depth"]
        err = 0.5 * (depth_reproj_consistency(da, db, cams[0], cams[1]).item()
                     + depth_reproj_consistency(db, da, cams[1], cams[0]).item())
        print("[ITER {}] depth: mean={:.4f} max={:.3f}  reproj_consistency={:.5f}".format(
            iteration, da[da > 0].mean().item(), da.max().item(), err))
    torch.cuda.empty_cache()


def train_projective(dataset, pipe, args):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.load_iteration)

    # Freeze everything the projective geometry must not move.
    for p in (gaussians._normal, gaussians._scaling,
              gaussians._rotation, gaussians._opacity):
        p.requires_grad_(False)

    # Round-trip sanity before re-parameterising.
    xyz_before = gaussians._xyz.detach().clone()
    mean_cam_center = compute_mean_cam_center(scene).cuda()
    gaussians.enable_projective(args.K, args.eps, mean_cam_center)
    with torch.no_grad():
        xyz_after = gaussians.get_xyz.detach()
        rt_err = (xyz_after - xyz_before).abs().max().item()
    print("round-trip |xyz - geo(w_raw,uv)| max = {:.2e}".format(rt_err))

    # Medium warm start if available.
    medium_ckpt = load_medium_init(dataset.model_path)
    b_inf_init = getattr(args, "b_inf_init", None)
    if b_inf_init is None:
        b_inf_init = 0.5
    water_model = WaterImageModel(b_inf_init=b_inf_init, beta_init=args.beta_init).cuda()
    if medium_ckpt is not None:
        water_model.load_state_dict(medium_ckpt)
        print("Loaded medium init from water_model.pth")

    optimizer = torch.optim.Adam([
        {"params": [gaussians._w_raw], "lr": args.position_lr, "name": "w"},
        {"params": [gaussians._uv], "lr": args.position_lr, "name": "uv"},
        {"params": [gaussians._features_dc], "lr": args.feature_lr},
        {"params": [gaussians._features_rest], "lr": args.feature_lr / 20.0},
        {"params": list(water_model.parameters()), "lr": args.medium_lr},
    ], lr=0.0, eps=1e-15)

    pos_lr_sched = get_expon_lr_func(
        lr_init=args.position_lr, lr_final=args.position_lr * 0.01,
        lr_delay_mult=1.0, max_steps=args.iterations)

    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32, device="cuda")

    train_cams = scene.getTrainCameras()
    viewpoint_stack = None
    ema = 0.0
    progress = tqdm(range(1, args.iterations + 1), desc="Projective training")
    for iteration in range(1, args.iterations + 1):
        if not viewpoint_stack:
            viewpoint_stack = train_cams.copy()
        cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Neighbour view for the multi-view geometry loss.
        nb = train_cams[randint(0, len(train_cams) - 1)]
        while nb is cam:
            nb = train_cams[randint(0, len(train_cams) - 1)]

        pkg_a = render(cam, gaussians, pipe, background)
        pkg_b = render(nb, gaussians, pipe, background)

        I_hat, _, _ = water_model(pkg_a["render"], pkg_a["depth"])
        gt = cam.original_image.cuda()

        Ll1 = l1_loss(I_hat, gt)
        photo = (1.0 - args.lambda_dssim) * Ll1 + args.lambda_dssim * (1.0 - ssim(I_hat, gt))

        d_a, d_b = pkg_a["depth"], pkg_b["depth"]
        reproj = 0.5 * (depth_reproj_consistency(d_a, d_b, cam, nb, max_r=50.0)
                        + depth_reproj_consistency(d_b, d_a, nb, cam, max_r=50.0))
        smooth = edge_aware_depth_smooth(d_a, torch.clamp(pkg_a["render"], 0.0, 1.0))
        geo = reproj + args.lambda_smooth * smooth

        loss = photo + args.lambda_geo * geo
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        for pg in optimizer.param_groups:
            if pg.get("name") in ("w", "uv"):
                pg["lr"] = pos_lr_sched(iteration)

        ema = 0.4 * loss.item() + 0.6 * ema
        if iteration % 10 == 0:
            progress.set_postfix({
                "Loss": f"{ema:.6f}",
                "photo": f"{photo.item():.5f}",
                "geo": f"{geo.item():.5f}",
            })
            progress.update(10)
        if iteration == args.iterations:
            progress.close()

        if iteration in args.test_iterations:
            evaluate(scene, water_model, pipe, background, iteration, args)

    # Persist the projective re-parameterised geometry (PLY holds the rebuilt
    # xyz; the sidecar projective.pth holds the gauge + parameters). Save to a
    # distinct iteration so we never clobber the ``iteration_5000``
    # (medium-tuned colour) checkpoint or the reconstruction ``iteration_30000``.
    out_dir = os.path.join(dataset.model_path, "projective")
    os.makedirs(out_dir, exist_ok=True)
    scene.save(args.save_iteration)
    gaussians.save_projective(os.path.join(out_dir, "projective.pth"))
    torch.save(water_model.state_dict(), os.path.join(out_dir, "water_model.pth"))
    print("Saved projective checkpoint ->", out_dir)
    print("\nProjective training complete.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Projective-geometry training")
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--load_iteration", type=int, default=30000)
    parser.add_argument("--save_iteration", type=int, default=60000)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--position_lr", type=float, default=0.00016)
    parser.add_argument("--feature_lr", type=float, default=0.0025)
    parser.add_argument("--medium_lr", type=float, default=0.01)
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    parser.add_argument("--lambda_geo", type=float, default=0.1)
    parser.add_argument("--lambda_smooth", type=float, default=0.01)
    parser.add_argument("--beta_init", type=float, default=0.1)
    parser.add_argument("--b_inf_init", type=float, default=None)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 3000, 5000])
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    safe_state(args.quiet)
    train_projective(lp.extract(args), pp.extract(args), args)
