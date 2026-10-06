"""Joint mesh-reprojection refinement (differentiable).

Loads the projective checkpoint (``--load_iteration`` + the
``projective/projective.pth`` gauge + ``projective/water_model.pth`` medium warm
start) and the offline refined mesh, then anchors the *differentiable*
alpha-weighted depth field to that frozen mesh through two grid-sampled losses
(``mesh_reprojection`` / ``mesh_normal_loss`` in ``geometry.mesh_losses``).

Alternating optimisation (never release everything at once):

    phase 1  [0, P1)      _w_raw/_uv/_features   (water + mesh frozen)
    phase 2  [P1, P2)     water/_features        (geometry + mesh frozen)  rgb only
    phase 3  [P2, iters)  all at small LR        (+ mesh verts if requested)

The optimizer is rebuilt at each phase boundary. Every mesh term routes its
gradient through ``pkg["depth"]`` (the only differentiable depth channel) back
to ``_w_raw``/``_uv``; the surface channels stay forward-only masks/metrics.

Persists to ``<model>/mesh/s4_<tag>_*`` and a distinct ``--save_iteration`` so
Earlier-stage checkpoints are never clobbered.
"""

import os
import sys
import json
import numpy as np
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
from mesh import load_npz
from mesh.extract import project_points, _sample_at
from geometry.losses import depth_reproj_consistency, edge_aware_depth_smooth
from geometry.mesh_losses import (charbonnier, mesh_reprojection, mesh_normal_loss)


def compute_mean_cam_center(scene):
    centers = [cam.camera_center.cpu() for cam in scene.getTrainCameras()]
    return torch.stack(centers).float().mean(dim=0)


def load_medium_init(model_path):
    """Medium warm start (``projective/water_model.pth`` first, then the medium stage)."""
    for rel in ("projective/water_model.pth", "medium/water_model.pth"):
        ckpt = os.path.join(model_path, rel)
        if os.path.exists(ckpt):
            return torch.load(ckpt, map_location="cpu")
    return None


def load_mesh_tensors(path):
    """Load the refined mesh into *frozen* CUDA tensors for the reprojection losses."""
    mesh = load_npz(path)
    verts = torch.from_numpy(np.asarray(mesh["verts"], dtype=np.float32)).cuda()
    normals = torch.from_numpy(np.asarray(mesh["normals"], dtype=np.float32)).cuda()
    normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-6)
    conf = torch.from_numpy(np.asarray(mesh["confidence"], dtype=np.float32)).cuda().clamp(0.0, 1.0)
    visible = torch.from_numpy(np.asarray(mesh["visible"], dtype=np.float32)).cuda().clamp(0.0, 1.0)
    return {"verts": verts, "normals": normals,
            "confidence": conf, "visible": visible}


def _set_trainable(gaussians, water_model, mv, phase, train_mesh_verts):
    """Toggle requires_grad per phase (alternating optimisation)."""
    geo_on = phase in (1, 3)
    med_on = phase in (2, 3)
    gaussians._w_raw.requires_grad_(geo_on)
    gaussians._uv.requires_grad_(geo_on)
    gaussians._features_dc.requires_grad_(True)
    gaussians._features_rest.requires_grad_(True)
    for p in water_model.parameters():
        p.requires_grad_(med_on)
    if mv is not None:
        mv.requires_grad_(train_mesh_verts and phase == 3)


def _build_optimizer(gaussians, water_model, mv, phase, args):
    """Adam over whatever is currently trainable (rebuilt at phase boundaries)."""
    groups = []
    if gaussians._w_raw.requires_grad:
        lr = args.position_lr * (0.1 if phase == 3 else 1.0)
        groups.append({"params": [gaussians._w_raw], "lr": lr, "name": "w"})
        groups.append({"params": [gaussians._uv], "lr": lr, "name": "uv"})
    if gaussians._features_dc.requires_grad:
        flr = args.feature_lr * (0.5 if phase == 3 else 1.0)
        groups.append({"params": [gaussians._features_dc], "lr": flr})
        groups.append({"params": [gaussians._features_rest], "lr": flr / 20.0})
    wm = [p for p in water_model.parameters() if p.requires_grad]
    if wm:
        mlr = args.medium_lr * (0.5 if phase == 3 else 1.0)
        groups.append({"params": wm, "lr": mlr, "name": "medium"})
    if mv is not None and mv.requires_grad:
        groups.append({"params": [mv], "lr": args.mesh_verts_lr, "name": "mesh_verts"})
    return torch.optim.Adam(groups, lr=0.0, eps=1e-15)


def evaluate(scene, water_model, pipe, background, iteration, args, mesh):
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

    # Forward metric: does the *differentiable* depth field sit on the mesh?
    cams = scene.getTrainCameras()
    if cams:
        cam = cams[0]
        mv = mesh["verts"]
        with torch.no_grad():
            pkg = render(cam, g, pipe, background)
            d = pkg["depth"]
            u, v, z = project_points(mv, cam)
            sampled = _sample_at(d, u, v)
            r = (mv - cam.camera_center.to(d.device).float().view(1, 3)).norm(dim=-1)
            inb = (u >= 0) & (u < cam.image_width - 1) & (v >= 0) & (v < cam.image_height - 1)
            valid = inb & (z > 0) & (r < args.mesh_max_r) & (sampled > 0)
        rel = (sampled[valid] - r[valid]).abs() / (r[valid] + 1e-3)
        print("[ITER {}] mesh depth err mean={:.5f}  n_valid={}".format(
            iteration, rel.mean().item() if valid.any() else float("nan"),
            int(valid.sum().item())))
    torch.cuda.empty_cache()


def train_mesh(dataset, pipe, args):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.load_iteration)

    # Restore the projective gauge + parameters (exact continuation).
    sidecar = os.path.join(dataset.model_path, "projective", "projective.pth")
    if not os.path.exists(sidecar):
        raise FileNotFoundError("projective gauge not found: {}".format(sidecar))
    gaussians.load_projective(sidecar)
    print("Loaded projective gauge ->", sidecar)

    # Freeze everything the projective geometry must not move.
    for p in (gaussians._normal, gaussians._scaling,
              gaussians._rotation, gaussians._opacity):
        p.requires_grad_(False)

    # Medium warm start (fallback to the medium stage).
    b_inf_init = getattr(args, "b_inf_init", None)
    if b_inf_init is None:
        b_inf_init = 0.5
    water_model = WaterImageModel(b_inf_init=b_inf_init, beta_init=args.beta_init).cuda()
    medium_ckpt = load_medium_init(dataset.model_path)
    if medium_ckpt is not None:
        water_model.load_state_dict(medium_ckpt)
        print("Loaded medium warm start")

    # Frozen reference mesh.
    mesh_path = getattr(args, "mesh", None) or os.path.join(dataset.model_path, "mesh", "refined_mesh.npz")
    if not os.path.exists(mesh_path):
        raise FileNotFoundError("refined mesh not found: {}".format(mesh_path))
    mesh = load_mesh_tensors(mesh_path)
    print("Loaded refined mesh ({} verts) ->".format(mesh["verts"].shape[0]), mesh_path)

    # Mesh verts either stay a frozen tensor or become an optimisable parameter.
    if args.train_mesh_verts:
        mv = torch.nn.Parameter(mesh["verts"].clone())
    else:
        mv = mesh["verts"]

    pos_lr_sched = get_expon_lr_func(
        lr_init=args.position_lr, lr_final=args.position_lr * 0.01,
        lr_delay_mult=1.0, max_steps=args.iterations)

    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32, device="cuda")

    train_cams = scene.getTrainCameras()
    viewpoint_stack = None
    ema = 0.0
    phase = None
    optimizer = None

    progress = tqdm(range(1, args.iterations + 1), desc="Mesh refinement training")
    for iteration in range(1, args.iterations + 1):
        new_phase = 1 if iteration <= args.phase1_iters else (2 if iteration <= args.phase2_iters else 3)
        if new_phase != phase:
            phase = new_phase
            _set_trainable(gaussians, water_model, mv, phase, args.train_mesh_verts)
            optimizer = _build_optimizer(gaussians, water_model, mv, phase, args)
            print("\n[ITER {}] phase {}  (w/uv lr={}, medium lr={})".format(
                iteration, phase, args.position_lr * (0.1 if phase == 3 else 1.0),
                args.medium_lr * (0.5 if phase == 3 else 1.0)))

        if not viewpoint_stack:
            viewpoint_stack = train_cams.copy()
        cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        pkg_a = render(cam, gaussians, pipe, background)
        I_hat, _, _ = water_model(pkg_a["render"], pkg_a["depth"])
        gt = cam.original_image.cuda()

        if args.photo_charbonnier:
            photo = charbonnier(I_hat - gt).mean()
        else:
            Ll1 = l1_loss(I_hat, gt)
            photo = (1.0 - args.lambda_dssim) * Ll1 + args.lambda_dssim * (1.0 - ssim(I_hat, gt))

        loss = photo
        log = {"photo": photo.item()}

        if phase in (1, 3):
            nb = train_cams[randint(0, len(train_cams) - 1)]
            while nb is cam:
                nb = train_cams[randint(0, len(train_cams) - 1)]
            pkg_b = render(nb, gaussians, pipe, background)

            d_a, d_b = pkg_a["depth"], pkg_b["depth"]
            ms = 0.5 * (depth_reproj_consistency(d_a, d_b, cam, nb, max_r=args.mesh_max_r)
                        + depth_reproj_consistency(d_b, d_a, nb, cam, max_r=args.mesh_max_r))
            reg = edge_aware_depth_smooth(d_a, torch.clamp(pkg_a["render"], 0.0, 1.0))
            loss = loss + args.lambda_ms * ms + args.lambda_reg * reg
            log["ms"] = ms.item()
            log["reg"] = reg.item()

            L_mesh, n_mesh = mesh_reprojection(
                d_a, cam, mv, mesh["confidence"], mesh["visible"],
                surface_depth=pkg_a["surface_depth"], beta_direct=water_model.beta_direct,
                max_r=args.mesh_max_r, t_min=args.mesh_t_min,
                occl_margin=args.mesh_occl_margin)
            L_norm, n_norm = mesh_normal_loss(
                d_a, cam, mv, mesh["normals"], mesh["confidence"], mesh["visible"],
                beta_direct=water_model.beta_direct, max_r=args.mesh_max_r,
                t_min=args.mesh_t_min, depth_grad_thresh=args.norm_depth_grad_thresh)
            loss = loss + args.lambda_mesh * L_mesh + args.lambda_norm * L_norm
            log["mesh"] = L_mesh.item()
            log["norm"] = L_norm.item()
            log["n_mesh"] = n_mesh
            log["n_norm"] = n_norm

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Exponential decay on the position params across phases 1 and 3.
        for pg in optimizer.param_groups:
            if pg.get("name") in ("w", "uv"):
                pg["lr"] = pos_lr_sched(iteration) * (0.1 if phase == 3 else 1.0)

        ema = 0.4 * loss.item() + 0.6 * ema
        if iteration % 10 == 0:
            progress.set_postfix({"Loss": f"{ema:.6f}", **{k: f"{v:.5f}" if isinstance(v, float) else v for k, v in log.items()}})
            progress.update(10)
        if iteration == args.iterations:
            progress.close()

        if iteration in args.test_iterations:
            evaluate(scene, water_model, pipe, background, iteration, args, mesh)

    # Persist (distinct save_iteration + s4_<tag>_* sidecars; never clobbers earlier stages).
    scene.save(args.save_iteration)
    out_dir = os.path.join(dataset.model_path, "mesh")
    os.makedirs(out_dir, exist_ok=True)
    gaussians.save_projective(os.path.join(out_dir, "s4_{}_projective.pth".format(args.tag)))
    torch.save(water_model.state_dict(), os.path.join(out_dir, "s4_{}_water_model.pth".format(args.tag)))
    if args.train_mesh_verts:
        torch.save(mv.detach().cpu(), os.path.join(out_dir, "s4_{}_verts.pth".format(args.tag)))

    config = {
        "tag": args.tag,
        "iterations": args.iterations,
        "phase1_iters": args.phase1_iters,
        "phase2_iters": args.phase2_iters,
        "load_iteration": args.load_iteration,
        "save_iteration": args.save_iteration,
        "mesh": mesh_path,
        "lambda_dssim": args.lambda_dssim,
        "lambda_ms": args.lambda_ms,
        "lambda_mesh": args.lambda_mesh,
        "lambda_norm": args.lambda_norm,
        "lambda_reg": args.lambda_reg,
        "lambda_sil": args.lambda_sil,
        "lambda_edge": args.lambda_edge,
        "mesh_t_min": args.mesh_t_min,
        "mesh_max_r": args.mesh_max_r,
        "mesh_occl_margin": args.mesh_occl_margin,
        "norm_depth_grad_thresh": args.norm_depth_grad_thresh,
        "train_mesh_verts": args.train_mesh_verts,
        "mesh_verts_lr": args.mesh_verts_lr,
        "photo_charbonnier": args.photo_charbonnier,
        "position_lr": args.position_lr,
        "feature_lr": args.feature_lr,
        "medium_lr": args.medium_lr,
    }
    with open(os.path.join(out_dir, "s4_{}_config.json".format(args.tag)), "w") as f:
        json.dump(config, f, indent=2)

    print("Saved mesh checkpoint ->", out_dir)
    print("\nMesh refinement training complete.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Joint mesh-reprojection refinement")
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--phase1_iters", type=int, default=1500)
    parser.add_argument("--phase2_iters", type=int, default=2200)
    parser.add_argument("--load_iteration", type=int, default=60000)
    parser.add_argument("--save_iteration", type=int, default=70000)
    parser.add_argument("--mesh", type=str, default=None,
                        help="refined mesh npz (default <model>/mesh/refined_mesh.npz)")
    parser.add_argument("--position_lr", type=float, default=0.00016)
    parser.add_argument("--feature_lr", type=float, default=0.0025)
    parser.add_argument("--medium_lr", type=float, default=0.01)
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    parser.add_argument("--lambda_ms", type=float, default=0.1)
    parser.add_argument("--lambda_mesh", type=float, default=0.5)
    parser.add_argument("--lambda_norm", type=float, default=0.1)
    parser.add_argument("--lambda_reg", type=float, default=0.01)
    parser.add_argument("--lambda_sil", type=float, default=0.0,
                        help="silhouette term (eval-only, reserved; kept at 0)")
    parser.add_argument("--lambda_edge", type=float, default=0.0,
                        help="verified-edge term (eval-only, reserved; kept at 0)")
    parser.add_argument("--mesh_t_min", type=float, default=0.05)
    parser.add_argument("--mesh_max_r", type=float, default=50.0)
    parser.add_argument("--mesh_occl_margin", type=float, default=0.05)
    parser.add_argument("--norm_depth_grad_thresh", type=float, default=0.1)
    parser.add_argument("--train_mesh_verts", action="store_true")
    parser.add_argument("--mesh_verts_lr", type=float, default=1e-4)
    parser.add_argument("--photo_charbonnier", action="store_true")
    parser.add_argument("--beta_init", type=float, default=0.1)
    parser.add_argument("--b_inf_init", type=float, default=None)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 2000, 3000])
    parser.add_argument("--tag", type=str, default="full")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    safe_state(args.quiet)
    train_mesh(lp.extract(args), pp.extract(args), args)
