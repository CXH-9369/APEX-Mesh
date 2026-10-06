"""Smoke test: depth output + medium layer + consecutive backward."""
import os, sys, argparse
import torch
from random import randint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arguments import ModelParams, PipelineParams
from scene import Scene
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim
from medium import WaterImageModel


def main():
    safe_state(False)
    parser = argparse.ArgumentParser()
    mp_p = ModelParams(parser)
    pp_p = PipelineParams(parser)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args([
        "--model_path", "../outputs/baseline/Panama",
        "--source_path", "/home/cxh/APEX/apex-mesh/datasets/seathru_undist/Panama",
        "--sh_degree", "3",
    ])
    mp = mp_p.extract(args)
    pp = pp_p.extract(args)

    g = GaussianModel(3)
    scene = Scene(mp, g, load_iteration=-1)

    # freeze geometry, keep colour (mirrors train_medium.py)
    for p in (g._xyz, g._normal, g._scaling, g._rotation, g._opacity):
        p.requires_grad_(False)

    water = WaterImageModel(b_inf_init=0.5, beta_init=0.1).cuda()
    opt = torch.optim.Adam([
        {"params": [g._features_dc], "lr": 0.0025},
        {"params": [g._features_rest], "lr": 0.000125},
        {"params": list(water.parameters()), "lr": 0.01},
    ], lr=0.0, eps=1e-15)

    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    cams = scene.getTrainCameras()

    # --- single render depth sanity check ---
    cam0 = cams[0]
    pkg = render(cam0, g, pp, bg)
    depth = pkg["depth"]
    J = pkg["render"]
    print(f"depth: shape={tuple(depth.shape)} dtype={depth.dtype} "
          f"min={depth.min().item():.3f} max={depth.max().item():.3f} "
          f"requires_grad={depth.requires_grad}")
    print(f"J:    shape={tuple(J.shape)} requires_grad={J.requires_grad}")

    I_hat, t, B = water(J, depth)
    print(f"I_hat shape={tuple(I_hat.shape)}  t mean={t.mean().item():.4f}  "
          f"B mean={B.mean().item():.4f}")

    # --- consecutive backward with medium in the graph ---
    vs = None
    for it in range(1, args.iters + 1):
        if not vs:
            vs = cams.copy()
        cam = vs.pop(randint(0, len(vs) - 1))
        pkg = render(cam, g, pp, bg)
        I_hat, _, _ = water(pkg["render"], pkg["depth"])
        gt = cam.original_image.cuda()
        loss = (1 - 0.2) * l1_loss(I_hat, gt) + 0.2 * (1 - ssim(I_hat, gt))
        loss.backward()

        fg = g._features_dc.grad
        assert fg is not None, "features_dc grad is None!"
        assert g._xyz.grad is None, "xyz should be frozen!"
        wg = water.raw_bd.grad
        assert wg is not None and wg.abs().sum() > 0, "medium grad is None/zero!"

        opt.step()
        opt.zero_grad(set_to_none=True)
        print(f"iter {it}: loss={loss.item():.5f} "
              f"|feat_grad|={fg.norm().item():.4f} "
              f"|bd_grad|={wg.abs().sum().item():.6f} "
              f"beta_d={water.beta_direct.detach().cpu().tolist()[0]:.4f}")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
