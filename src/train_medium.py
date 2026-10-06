"""Underwater medium forward-layer training on a frozen half-Gaussian field.

Loads an existing half-Gaussian reconstruction checkpoint (``--load_iteration``, default latest),
freezes the geometry (xyz/normal/scaling/rotation/opacity) and re-interprets
the per-Gaussian colour (features_dc / features_rest) as the target radiance
``J``, jointly with scene-shared medium parameters (beta_direct, beta_back,
B_inf). Loss is photometric on ``I_hat = t * J + B`` against the raw image.

Independent of ``train.py`` so the reconstruction path is untouched.
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
from utils.general_utils import safe_state
from gaussian_renderer import render
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from medium import WaterImageModel


def compute_b_inf_init(scene):
    """Global mean of training images as the veiling-light colour prior."""
    acc = torch.zeros(3, dtype=torch.float32)
    n = 0
    for cam in scene.getTrainCameras():
        acc += cam.original_image.mean(dim=(1, 2)).cpu()
        n += 1
    return (acc / n).clamp(0.01, 0.99)


def evaluate(scene, water_model, pipe, background, iteration):
    torch.cuda.empty_cache()
    train_cams = scene.getTrainCameras()
    for name, cameras in (
        ("test", scene.getTestCameras()),
        ("train", [train_cams[i] for i in range(min(5, len(train_cams)))]),
    ):
        if not cameras:
            continue
        l1 = psnr_v = ssim_v = 0.0
        for vp in cameras:
            with torch.no_grad():
                pkg = render(vp, scene.gaussians, pipe, background)
                I_hat, _, _ = water_model(pkg["render"], pkg["depth"])
                I_hat = torch.clamp(I_hat, 0.0, 1.0)
                gt = torch.clamp(vp.original_image.to("cuda"), 0.0, 1.0)
            l1 += l1_loss(I_hat, gt).item()
            psnr_v += psnr(I_hat, gt).mean().item()
            ssim_v += ssim(I_hat, gt).item()
        n = len(cameras)
        print("\n[ITER {}] {}: L1 {:.5f} PSNR {:.2f} SSIM {:.4f}".format(
            iteration, name, l1 / n, psnr_v / n, ssim_v / n))

    bd = water_model.beta_direct.detach().cpu().tolist()
    bb = water_model.beta_back.detach().cpu().tolist()
    Binf = water_model.B_inf.detach().cpu().tolist()
    print("[ITER {}] beta_direct={} beta_back={} B_inf={}".format(
        iteration,
        ["%.4f" % x for x in bd],
        ["%.4f" % x for x in bb],
        ["%.4f" % x for x in Binf]))
    torch.cuda.empty_cache()


def train_medium(dataset, pipe, args):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.load_iteration)

    # Freeze geometry; keep only the per-Gaussian colour (target radiance J).
    for p in (gaussians._xyz, gaussians._normal, gaussians._scaling,
              gaussians._rotation, gaussians._opacity):
        p.requires_grad_(False)

    b_inf_init = getattr(args, "b_inf_init", None)
    if b_inf_init is None:
        b_inf_init = compute_b_inf_init(scene)
    if not torch.is_tensor(b_inf_init):
        b_inf_init = torch.as_tensor(b_inf_init, dtype=torch.float32)
    print("B_inf init:", ["%.4f" % x for x in b_inf_init.tolist()])

    water_model = WaterImageModel(b_inf_init=b_inf_init, beta_init=args.beta_init).cuda()

    optimizer = torch.optim.Adam([
        {"params": [gaussians._features_dc], "lr": args.feature_lr},
        {"params": [gaussians._features_rest], "lr": args.feature_lr / 20.0},
        {"params": list(water_model.parameters()), "lr": args.medium_lr},
    ], lr=0.0, eps=1e-15)

    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32, device="cuda")

    viewpoint_stack = None
    ema = 0.0
    progress = tqdm(range(1, args.iterations + 1), desc="Medium training")
    for iteration in range(1, args.iterations + 1):
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        pkg = render(cam, gaussians, pipe, background)
        I_hat, _, _ = water_model(pkg["render"], pkg["depth"])
        gt = cam.original_image.cuda()

        Ll1 = l1_loss(I_hat, gt)
        loss = (1.0 - args.lambda_dssim) * Ll1 + args.lambda_dssim * (1.0 - ssim(I_hat, gt))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        ema = 0.4 * loss.item() + 0.6 * ema
        if iteration % 10 == 0:
            progress.set_postfix({"Loss": f"{ema:.7f}"})
            progress.update(10)
        if iteration == args.iterations:
            progress.close()

        if iteration in args.test_iterations:
            evaluate(scene, water_model, pipe, background, iteration)

    # Save the micro-tuned colour ply (J) and the medium parameters.
    out_dir = os.path.join(dataset.model_path, "medium")
    os.makedirs(out_dir, exist_ok=True)
    scene.save(args.iterations)
    torch.save(water_model.state_dict(), os.path.join(out_dir, "water_model.pth"))
    print("Saved medium params ->", os.path.join(out_dir, "water_model.pth"))
    print("\nMedium training complete.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Underwater medium-layer training (frozen geometry)")
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--load_iteration", type=int, default=-1)
    parser.add_argument("--feature_lr", type=float, default=0.0025)
    parser.add_argument("--medium_lr", type=float, default=0.01)
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    parser.add_argument("--beta_init", type=float, default=0.1)
    parser.add_argument("--b_inf_init", type=float, default=None)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 3000, 5000])
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    safe_state(args.quiet)
    train_medium(lp.extract(args), pp.extract(args), args)
