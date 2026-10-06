"""Export projective geometry results.

Loads a projective checkpoint (PLY + ``projective/projective.pth`` sidecar +
``projective/water_model.pth``) and writes per-view PNGs under
``<model>/projective/ours_<iter>/<split>/<channel>/`` for raw/I_hat/J/t/B/depth
(exact optical path)/residual. The per-Gaussian ``w`` / ``uv`` are scalar
statistics (not per-pixel images), so they are reported to stdout as ranges.
"""

import os
import sys
import torch
from argparse import ArgumentParser
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from medium import WaterImageModel
from utils.general_utils import safe_state


def save_depth_map(depth, path):
    d = depth.detach().cpu().numpy().squeeze()
    valid = d[d > 0]
    vmin = float(valid.min()) if valid.size else 0.0
    vmax = float(d.max())
    norm = (d - vmin) / max(vmax - vmin, 1e-6)
    plt.imsave(path, plt.get_cmap("viridis")(norm)[..., :3])


def save_residual(res, path):
    x = torch.clamp(res.detach().cpu() * 2.0 + 0.5, 0.0, 1.0)
    torchvision.utils.save_image(x, path)


def render_projective(dataset, pipe, iteration, skip_train, skip_test):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        sidecar = os.path.join(dataset.model_path, "projective", "projective.pth")
        if os.path.exists(sidecar):
            gaussians.load_projective(sidecar)
            print("Loaded projective gauge ->", sidecar)
        else:
            print("WARNING: no projective sidecar found; falling back to raw xyz")

        water_model = WaterImageModel().cuda()
        ckpt = os.path.join(dataset.model_path, "projective", "water_model.pth")
        water_model.load_state_dict(torch.load(ckpt))
        water_model.eval()

        if gaussians.projective:
            w = gaussians.geo.w_from_raw(gaussians._w_raw)
            print("w  range [{:.5f}, {:.5f}]  (eps={})".format(
                w.min().item(), w.max().item(), gaussians.geo.eps))
            print("uv range [{:.5f}, {:.5f}]".format(
                gaussians._uv.min().item(), gaussians._uv.max().item()))

        background = torch.tensor(
            [1, 1, 1] if dataset.white_background else [0, 0, 0],
            dtype=torch.float32, device="cuda")

        out_root = os.path.join(dataset.model_path, "projective", "ours_{}".format(scene.loaded_iter))
        sets = []
        if not skip_test:
            sets.append(("test", scene.getTestCameras()))
        if not skip_train:
            sets.append(("train", scene.getTrainCameras()))

        channels = ("raw", "I_hat", "J", "t", "B", "depth", "residual")
        for name, views in sets:
            dirs = {k: os.path.join(out_root, name, k) for k in channels}
            for d in dirs.values():
                os.makedirs(d, exist_ok=True)

            for idx, view in enumerate(tqdm(views, desc="Rendering {}".format(name))):
                pkg = render(view, gaussians, pipe, background)
                J = pkg["render"]
                depth = pkg["depth"]
                I_hat, t, B = water_model(J, depth)
                raw = view.original_image[0:3]
                pref = "{:05d}.png".format(idx)

                torchvision.utils.save_image(raw, os.path.join(dirs["raw"], pref))
                torchvision.utils.save_image(torch.clamp(I_hat, 0, 1), os.path.join(dirs["I_hat"], pref))
                torchvision.utils.save_image(torch.clamp(J, 0, 1), os.path.join(dirs["J"], pref))
                torchvision.utils.save_image(torch.clamp(t, 0, 1), os.path.join(dirs["t"], pref))
                torchvision.utils.save_image(torch.clamp(B, 0, 1), os.path.join(dirs["B"], pref))
                save_depth_map(depth, os.path.join(dirs["depth"], pref))
                save_residual(I_hat - raw.to(I_hat.device), os.path.join(dirs["residual"], pref))

        print("Saved projective exports ->", out_root)


if __name__ == "__main__":
    parser = ArgumentParser(description="Projective export (J/t/B/depth/w/u/v)")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=60000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)

    render_projective(model.extract(args), pipeline.extract(args),
                      args.iteration, args.skip_train, args.skip_test)
