"""Projective-geometry tests.

Pure-geometry tests (no scene load): round-trip reconstruction, finite depth,
gradcheck. Scene tests: exact optical path (single-Gaussian decisive check vs
camera-z) and differentiable depth (consecutive backwards, no "second time").
"""

import os
import sys
import math
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from geometry.projective import AnchoredProjectiveGeometry, build_blocks, enable
from arguments import ModelParams, PipelineParams
from scene import Scene
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state


# ---------------------------------------------------------------------------
# Pure geometry (no scene)
# ---------------------------------------------------------------------------
def test_roundtrip():
    torch.manual_seed(0)
    N, K = 512, 8
    xyz = torch.randn(N, 3)
    mean_cam = torch.randn(3)
    w_raw, uv, geo = enable(xyz, K, mean_cam, eps=1e-3)

    xyz_hat = geo(w_raw, uv)
    err = (xyz_hat - xyz).abs().max().item()
    print("[roundtrip] max |geo(w_raw,uv) - xyz| = {:.2e}".format(err))
    assert err < 1e-4, "round-trip reconstruction is not exact"

    w = geo.w_from_raw(w_raw)
    assert (w >= 1e-3).all() and (w < 1.0).all(), "w out of [eps, 1)"
    d = geo.d0[geo.block_id] * (1.0 / w.squeeze(-1) - 1.0)
    assert (d > 0).all(), "finite depth d must be > 0"
    print("[finite depth] w in [{:.5f}, {:.5f}]  d>0 OK".format(w.min().item(), w.max().item()))
    return True


def test_gradcheck():
    torch.manual_seed(0)
    N, K = 16, 3
    xyz = torch.randn(N, 3).double()
    mean_cam = torch.randn(3).double()
    anchor, d0, basis, block_id = build_blocks(xyz.float(), K, mean_cam.float(), eps=1e-3)
    geo = AnchoredProjectiveGeometry(anchor, d0, basis, block_id, eps=1e-3).double()
    w_raw, uv = geo.inverse(xyz)
    w_raw = w_raw.double().requires_grad_()
    uv = uv.double().requires_grad_()

    ok = torch.autograd.gradcheck(geo.forward, (w_raw, uv), eps=1e-6, atol=1e-4, rtol=1e-3)
    print("[gradcheck]", "OK" if ok else "FAILED")
    assert ok, "gradcheck on geo.forward failed"
    return True


# ---------------------------------------------------------------------------
# Scene-based (loads Panama baseline)
# ---------------------------------------------------------------------------
def test_exact_optical_path(cam):
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    H, W = cam.image_height, cam.image_width
    view = cam.world_view_transform
    proj = cam.full_proj_transform
    campos = cam.camera_center
    fovx, fovy = cam.FoVx, cam.FoVy

    # Off-axis camera-space direction (projects ~85% toward the right edge).
    tx = 0.7 * math.tan(fovx * 0.5)
    ty = 0.2 * math.tan(fovy * 0.5)
    dir_cam = torch.tensor([tx, ty, 1.0], device="cuda", dtype=torch.float32)
    dir_cam = dir_cam / dir_cam.norm()
    R = view[:3, :3]
    dir_world = dir_cam @ R.T
    dir_world = dir_world / dir_world.norm()

    d0 = 2.0
    P = (campos + d0 * dir_world).unsqueeze(0)          # (1, 3)

    r = torch.tensor(d0, device="cuda")
    # world_view_transform is stored glm-transposed, so the world->camera
    # transform on the Python side is view.T @ p_hom.
    z_cam = (view.T @ torch.cat([P[0], torch.ones(1, device="cuda")]))[2]

    settings = GaussianRasterizationSettings(
        image_height=H, image_width=W,
        tanfovx=math.tan(fovx * 0.5), tanfovy=math.tan(fovy * 0.5),
        bg=torch.zeros(3, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=view, projmatrix=proj,
        sh_degree=0, campos=campos,
        prefiltered=False, debug=False)
    rasterizer = GaussianRasterizer(settings)

    cov = torch.zeros(1, 6, device="cuda")
    s2 = 0.02 ** 2
    cov[0, 0] = cov[0, 3] = cov[0, 5] = s2

    color, _, depth, _, _, _, _ = rasterizer(
        means3D=P,
        means2D=torch.zeros_like(P),
        normal=dir_world.unsqueeze(0),
        shs=None,
        colors_precomp=torch.tensor([[1.0, 0.0, 0.0]], device="cuda"),
        opacities=torch.full((1, 2), 0.99, device="cuda"),
        scales=None,
        rotations=None,
        cov3D_precomp=cov,
        cov3D_precomp_small=None)

    depth_img = depth.squeeze(0)
    peak = torch.argmax(color[0])
    dv = depth_img.reshape(-1)[peak].item()

    print("[optical path] r={:.4f} z_cam={:.4f} depth@peak={:.4f}".format(
        r.item(), z_cam.item(), dv))
    assert abs(dv - r.item()) < 0.05 * r.item(), \
        "rendered depth does not match exact optical path ||x - c||"
    assert abs(dv - z_cam.item()) > 0.02 * r.item(), \
        "rendered depth equals camera-z; optical path is NOT in effect"
    return True


def test_differentiable_depth(scene, pipe, bg, K):
    g = scene.gaussians

    centers = torch.stack([c.camera_center.cpu() for c in scene.getTrainCameras()]).float()
    mean_cam = centers.mean(dim=0).cuda()
    g.enable_projective(K, 1e-3, mean_cam)

    cams = scene.getTrainCameras()
    pkg = render(cams[0], g, pipe, bg)
    depth = pkg["depth"]
    assert depth.requires_grad, "depth must be differentiable now"
    depth.sum().backward()

    wg = g._w_raw.grad
    ug = g._uv.grad
    assert wg is not None and wg.abs().sum() > 0, "w_raw grad is None/zero"
    assert ug is not None and ug.abs().sum() > 0, "uv grad is None/zero"
    print("[differentiable depth] |w_raw.grad|={:.4e} |uv.grad|={:.4e}".format(
        wg.abs().sum().item(), ug.abs().sum().item()))

    # Consecutive backward on a fresh render must not raise "second time".
    pkg2 = render(cams[1], g, pipe, bg)
    pkg2["depth"].sum().backward()
    print("[differentiable depth] consecutive backward OK")
    return True


def main():
    safe_state(False)
    parser = argparse.ArgumentParser()
    mp_p = ModelParams(parser)
    pp_p = PipelineParams(parser)
    parser.add_argument("--load_iteration", type=int, default=30000)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--skip_geometry", action="store_true")
    parser.add_argument("--skip_scene", action="store_true")
    args = parser.parse_args([
        "--model_path", "../outputs/baseline/Panama",
        "--source_path", "/home/cxh/APEX/apex-mesh/datasets/seathru_undist/Panama",
        "--sh_degree", "3",
    ])
    mp = mp_p.extract(args)
    pp = pp_p.extract(args)

    if not args.skip_geometry:
        test_roundtrip()
        test_gradcheck()

    if not args.skip_scene:
        g = GaussianModel(3)
        scene = Scene(mp, g, load_iteration=args.load_iteration)
        bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

        test_exact_optical_path(scene.getTrainCameras()[0])
        test_differentiable_depth(scene, pp, bg, args.K)

    print("\nALL PROJECTIVE TESTS PASSED")


if __name__ == "__main__":
    main()
