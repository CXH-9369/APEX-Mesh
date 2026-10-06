"""Differentiable mesh-reprojection tests (mesh-reprojection feedback through depth).

Synthetic tests (no scene): normal-from-depth plane recovery + gradient,
occlusion masking, low-transmission masking. Scene tests (loads Panama):
mesh_reprojection gradient reaches _w_raw/_uv (the *binding* constraint that
feedback flows through the differentiable depth channel), the four surface
channels stay non-differentiable, and consecutive backwards don't raise
"second time".
"""

import os
import sys
import math
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from scene import Scene
from gaussian_renderer import render, GaussianModel
from medium import WaterImageModel
from arguments import ModelParams, PipelineParams
from utils.general_utils import safe_state
from geometry.mesh_losses import normal_from_depth, mesh_reprojection


class _FakeCam:
    def __init__(self, H=64, W=64):
        self.image_height = H
        self.image_width = W
        self.FoVx = math.radians(60.0)
        self.FoVy = math.radians(60.0)
        self.world_view_transform = torch.eye(4, dtype=torch.float32, device="cuda")
        self.camera_center = torch.zeros(3, dtype=torch.float32, device="cuda")


# ---------------------------------------------------------------------------
# Synthetic tests (no scene)
# ---------------------------------------------------------------------------
def test_normal_from_depth_plane():
    cam = _FakeCam()
    d = torch.full((64, 64), 10.0, device="cuda", requires_grad=True)
    n = normal_from_depth(d, cam)                          # (3,H,W)
    c = n[:, 32, 32]
    print("[normal_from_depth] center normal = ({:.3f},{:.3f},{:.3f})".format(
        c[0].item(), c[1].item(), c[2].item()))
    assert abs(c[0]) < 1e-3 and abs(c[1]) < 1e-3, "plane normal must be axial"
    assert abs(c[2] + 1.0) < 1e-3, "plane normal must point toward the camera (-z)"
    n.sum().backward()
    assert d.grad is not None and d.grad.abs().sum() > 0, "gradient must flow back to depth"
    return True


def test_occlusion_mask():
    cam = _FakeCam()
    depth = torch.full((64, 64), 10.0, device="cuda")
    verts = torch.tensor([[0.0, 0.0, 10.0]], device="cuda")
    conf = torch.ones(1, device="cuda")
    vis = torch.ones(1, device="cuda")

    sd_ok = torch.full((64, 64), 10.0, device="cuda")       # surface at the vertex
    _, n1 = mesh_reprojection(depth, cam, verts, conf, vis, surface_depth=sd_ok,
                              max_r=50.0, t_min=0.05)
    sd_occ = torch.full((64, 64), 5.0, device="cuda")       # closer surface occludes
    _, n2 = mesh_reprojection(depth, cam, verts, conf, vis, surface_depth=sd_occ,
                              max_r=50.0, t_min=0.05)
    print("[occlusion] n_valid no-occ={} occ={}".format(n1, n2))
    assert n1 == 1, "unoccluded vertex must contribute"
    assert n2 == 0, "occluded vertex must be masked out"
    return True


def test_low_transmission_mask():
    cam = _FakeCam()
    depth = torch.full((64, 64), 10.0, device="cuda")
    verts = torch.tensor([[0.0, 0.0, 10.0]], device="cuda")
    conf = torch.ones(1, device="cuda")
    vis = torch.ones(1, device="cuda")

    _, n_clear = mesh_reprojection(depth, cam, verts, conf, vis,
                                   beta_direct=torch.tensor([0.001], device="cuda"),
                                   max_r=50.0, t_min=0.05)
    _, n_turbid = mesh_reprojection(depth, cam, verts, conf, vis,
                                    beta_direct=torch.tensor([1.0], device="cuda"),
                                    max_r=50.0, t_min=0.05)
    print("[transmission] n_valid clear={} turbid={}".format(n_clear, n_turbid))
    assert n_clear == 1, "clear water must keep the vertex"
    assert n_turbid == 0, "low transmission must mask the vertex"
    return True


# ---------------------------------------------------------------------------
# Scene tests (Panama)
# ---------------------------------------------------------------------------
def _load_mesh_sub(path, step=20):
    from mesh import load_npz
    m = load_npz(path)
    s = slice(None, None, step)
    return {
        "verts": torch.from_numpy(m["verts"][s].astype(np.float32)).cuda(),
        "confidence": torch.from_numpy(m["confidence"][s].astype(np.float32)).cuda().clamp(0.0, 1.0),
        "visible": torch.from_numpy(m["visible"][s].astype(np.float32)).cuda().clamp(0.0, 1.0),
    }


def test_mesh_reprojection_grad(scene, pipe, bg, mesh, water):
    g = scene.gaussians
    cam = scene.getTrainCameras()[0]
    pkg = render(cam, g, pipe, bg)
    loss, n = mesh_reprojection(pkg["depth"], cam, mesh["verts"],
                                mesh["confidence"], mesh["visible"],
                                surface_depth=pkg["surface_depth"],
                                beta_direct=water.beta_direct,
                                max_r=50.0, t_min=0.05)
    print("[mesh_reproj] n_valid={} loss={:.5f}".format(n, loss.item()))
    assert n > 0, "no valid mesh verts in the first view"
    loss.backward()

    wg = g._w_raw.grad
    ug = g._uv.grad
    assert wg is not None and wg.abs().sum() > 0, "gradient must reach _w_raw via depth"
    assert ug is not None and ug.abs().sum() > 0, "gradient must reach _uv via depth"
    print("[mesh_reproj grad] |_w_raw.grad|={:.4e} |_uv.grad|={:.4e}".format(
        wg.abs().sum().item(), ug.abs().sum().item()))
    return True


def test_non_diff_channels(scene, pipe, bg):
    pkg = render(scene.getTrainCameras()[0], scene.gaussians, pipe, bg)
    for k in ("surface_depth", "accum_opacity", "normal", "edge"):
        assert pkg[k].requires_grad is False, "{} must be non-differentiable".format(k)
    assert pkg["depth"].requires_grad, "depth must be differentiable"
    assert pkg["render"].requires_grad, "render must be differentiable"
    print("[non-diff] four surface channels detached; depth/render differentiable")
    return True


def test_consecutive_backward(scene, pipe, bg, mesh, water):
    g = scene.gaussians
    cam = scene.getTrainCameras()[0]

    pkg1 = render(cam, g, pipe, bg)
    l1, _ = mesh_reprojection(pkg1["depth"], cam, mesh["verts"], mesh["confidence"],
                              mesh["visible"], surface_depth=pkg1["surface_depth"],
                              beta_direct=water.beta_direct, max_r=50.0, t_min=0.05)
    l1.backward()

    pkg2 = render(cam, g, pipe, bg)
    l2, _ = mesh_reprojection(pkg2["depth"], cam, mesh["verts"], mesh["confidence"],
                              mesh["visible"], surface_depth=pkg2["surface_depth"],
                              beta_direct=water.beta_direct, max_r=50.0, t_min=0.05)
    l2.backward()   # must not raise "Trying to backward through the graph a second time"
    print("[consecutive] two mesh_reprojection backwards OK")
    return True


def main():
    safe_state(False)
    parser = argparse.ArgumentParser()
    mp_p = ModelParams(parser)
    pp_p = PipelineParams(parser)
    parser.add_argument("--load_iteration", type=int, default=60000)
    parser.add_argument("--skip_scene", action="store_true")
    args = parser.parse_args([
        "--model_path", "../outputs/baseline/Panama",
        "--source_path", "/home/cxh/APEX/apex-mesh/datasets/seathru_undist/Panama",
        "--sh_degree", "3",
    ])
    mp = mp_p.extract(args)
    pp = pp_p.extract(args)

    test_normal_from_depth_plane()
    test_occlusion_mask()
    test_low_transmission_mask()

    if not args.skip_scene:
        g = GaussianModel(3)
        scene = Scene(mp, g, load_iteration=args.load_iteration)
        sidecar = os.path.join(mp.model_path, "projective", "projective.pth")
        g.load_projective(sidecar)
        for p in (g._normal, g._scaling, g._rotation, g._opacity):
            p.requires_grad_(False)

        water = WaterImageModel().cuda()
        water.load_state_dict(torch.load(os.path.join(mp.model_path, "projective", "water_model.pth")))
        water.eval()
        bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        mesh = _load_mesh_sub(os.path.join(mp.model_path, "mesh", "refined_mesh.npz"))

        test_mesh_reprojection_grad(scene, pp, bg, mesh, water)
        test_non_diff_channels(scene, pp, bg)
        test_consecutive_backward(scene, pp, bg, mesh, water)

    print("\nALL TRAIN TESTS PASSED")


if __name__ == "__main__":
    main()
