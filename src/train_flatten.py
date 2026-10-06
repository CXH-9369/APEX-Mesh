"""Surface-alignment flattening (offline shape finetune).

Loads the projective checkpoint (``--load_iteration``, default 60000), keeps
the projective positions and colours frozen, and reshapes the Gaussian *scales*
/ *rotations* / *opacities* so the density field collapses into a thin,
surface-aligned sheet.  The mesh is then extracted from the flattened field by
``extract_mtetra.py`` as before.

Loss = photometric (I_hat = t*J + B via the frozen water model)
       + lambda_flatten * flatten_loss
       + lambda_normal  * normal_alignment_loss
       + lambda_opacity * opacity_binary_loss

No densification and no position update: the stage is cheap (~3k iters) and
only ever makes Gaussians *flatter*, so it cannot break the projective geometry.
"""

import os
import sys
import torch
from random import randint
from argparse import ArgumentParser
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.loss_utils import l1_loss, ssim
from utils.general_utils import safe_state
from gaussian_renderer import render
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from medium import WaterImageModel
from geometry.flatten import flatten_loss, normal_alignment_loss, opacity_binary_loss


def train_flatten(dataset, pipe, args):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.load_iteration, shuffle=False)

    sidecar = os.path.join(dataset.model_path, "projective", "projective.pth")
    if os.path.exists(sidecar):
        gaussians.load_projective(sidecar)
        print("Loaded projective gauge ->", sidecar)

    water_model = WaterImageModel().cuda()
    ckpt = os.path.join(dataset.model_path, "projective", "water_model.pth")
    water_model.load_state_dict(torch.load(ckpt))
    water_model.eval()
    for p in water_model.parameters():
        p.requires_grad_(False)
    print("Loaded frozen water model ->", ckpt)

    # Freeze positions + colour; only Gaussian shapes are optimised.
    for p in (gaussians._w_raw, gaussians._uv,
              gaussians._features_dc, gaussians._features_rest):
        p.requires_grad_(False)
    for p in (gaussians._scaling, gaussians._rotation, gaussians._opacity):
        p.requires_grad_(True)

    optimizer = torch.optim.Adam([
        {"params": [gaussians._scaling], "lr": args.scaling_lr, "name": "scaling"},
        {"params": [gaussians._rotation], "lr": args.rotation_lr, "name": "rotation"},
        {"params": [gaussians._opacity], "lr": args.opacity_lr, "name": "opacity"},
    ], lr=0.0, eps=1e-15)

    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32, device="cuda")

    train_cams = scene.getTrainCameras()
    viewpoint_stack = None
    ema = 0.0
    progress = tqdm(range(1, args.iterations + 1), desc="Surface flattening")
    for iteration in range(1, args.iterations + 1):
        if not viewpoint_stack:
            viewpoint_stack = train_cams.copy()
        cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        pkg = render(cam, gaussians, pipe, background)
        I_hat, _, _ = water_model(pkg["render"], pkg["depth"])
        gt = cam.original_image.cuda()

        Ll1 = l1_loss(I_hat, gt)
        photo = (1.0 - args.lambda_dssim) * Ll1 + args.lambda_dssim * (1.0 - ssim(I_hat, gt))

        flat = args.lambda_flatten * flatten_loss(gaussians)
        if args.lambda_normal > 0:
            flat = flat + args.lambda_normal * normal_alignment_loss(
                gaussians, pkg["depth"], cam)
        if args.lambda_opacity > 0:
            flat = flat + args.lambda_opacity * opacity_binary_loss(gaussians)

        loss = photo + flat
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        ema = 0.4 * loss.item() + 0.6 * ema
        if iteration % 10 == 0:
            progress.set_postfix({
                "Loss": f"{ema:.6f}",
                "photo": f"{photo.item():.5f}",
                "flatten": f"{flat.item():.5f}",
            })
            progress.update(10)
        if iteration == args.iterations:
            progress.close()

    # Persist the flattened Gaussians to a distinct iteration (the projective stage stays at 60000).
    scene.save(args.save_iteration)
    print("Saved flattened Gaussians -> iteration {}".format(args.save_iteration))
    print("\nSurface flattening complete.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Surface-alignment flattening")
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--load_iteration", type=int, default=60000)
    parser.add_argument("--save_iteration", type=int, default=65000)
    parser.add_argument("--scaling_lr", type=float, default=0.005)
    parser.add_argument("--rotation_lr", type=float, default=0.001)
    parser.add_argument("--opacity_lr", type=float, default=0.05)
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    parser.add_argument("--lambda_flatten", type=float, default=1.0)
    parser.add_argument("--lambda_normal", type=float, default=0.05)
    parser.add_argument("--lambda_opacity", type=float, default=0.0)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    safe_state(args.quiet)
    train_flatten(lp.extract(args), pp.extract(args), args)
