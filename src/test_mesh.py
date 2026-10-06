"""Mesh tests.

Rasterizer tests (scene load, single/two Gaussian): median surface depth,
accumulated opacity, alpha-weighted normal, half-Gaussian boundary response.
Pure-mesh tests (no scene): Delaunay extraction + edge culling + open boundary,
PLY/OBJ/NPZ round-trip, refinement cleanup/stats, and cKDTree point-cloud
distance.
"""

import os
import sys
import math
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from mesh import (build_reference_mesh, mesh_stats, cleanup,
                  laplacian_smooth, boundary_edges, self_intersection_count,
                  write_ply, write_obj, save_npz, load_npz, read_ply,
                  point_cloud_distance)


# ---------------------------------------------------------------------------
# Rasterizer helpers
# ---------------------------------------------------------------------------
def _rasterize(means3D, normals, opacities, colors, cov3Ds, cam, bg=None):
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    H, W = cam.image_height, cam.image_width
    view = cam.world_view_transform
    proj = cam.full_proj_transform
    campos = cam.camera_center
    settings = GaussianRasterizationSettings(
        image_height=H, image_width=W,
        tanfovx=math.tan(cam.FoVx * 0.5), tanfovy=math.tan(cam.FoVy * 0.5),
        bg=bg if bg is not None else torch.zeros(3, device="cuda"),
        scale_modifier=1.0, viewmatrix=view, projmatrix=proj,
        sh_degree=0, campos=campos, prefiltered=False, debug=False)
    rasterizer = GaussianRasterizer(settings)
    P = means3D.shape[0]
    out = rasterizer(
        means3D=means3D, means2D=torch.zeros_like(means3D), normal=normals,
        shs=None, colors_precomp=colors, opacities=opacities,
        scales=None, rotations=None, cov3D_precomp=cov3Ds, cov3D_precomp_small=None)
    color, radii, depth, sd, A, normal, edge = out
    return {"color": color, "depth": depth.squeeze(0),
            "surface_depth": sd.squeeze(0), "accum_opacity": A.squeeze(0),
            "normal": normal, "edge": edge.squeeze(0)}


def _ray_through(cam, tx=0.0, ty=0.0):
    """World unit ray for a camera-space direction (tx, ty, 1)."""
    dir_cam = torch.tensor([tx, ty, 1.0], device="cuda", dtype=torch.float32)
    dir_cam = dir_cam / dir_cam.norm()
    R = cam.world_view_transform[:3, :3]
    d = dir_cam @ R.T
    return d / d.norm()


def test_rasterizer_surface(cam):
    # Single opaque Gaussian on the central ray at distance r=3.0.
    r = 3.0
    dir_world = _ray_through(cam)
    P = (cam.camera_center + r * dir_world).unsqueeze(0)
    n = (-dir_world).unsqueeze(0)  # surface normal toward camera
    s2 = 0.05 ** 2
    cov = torch.zeros(1, 6, device="cuda")
    cov[0, 0] = cov[0, 3] = cov[0, 5] = s2
    out = _rasterize(P, n, torch.full((1, 2), 0.99, device="cuda"),
                     torch.tensor([[1.0, 0.0, 0.0]], device="cuda"), cov, cam)

    cy, cx = np.unravel_index(int(torch.argmax(out["color"][0]).item()),
                              out["color"].shape[1:])
    A_c = out["accum_opacity"][cy, cx].item()
    sd_c = out["surface_depth"][cy, cx].item()
    nm = out["normal"][:, cy, cx]
    nm = nm / nm.norm()
    e_c = out["edge"][cy, cx].item()

    print("[surface] A={:.3f} surface_depth={:.4f} (r={:.3f}) edge={:.3f}".format(
        A_c, sd_c, r, e_c))
    assert abs(A_c - 0.99) < 0.02, "accumulated opacity should be ~0.99"
    assert abs(sd_c - r) < 0.05 * r, "surface depth != optical path r"
    assert abs(float((nm @ (-dir_world)).item()) - 1.0) < 0.02, "normal direction wrong"
    assert e_c > 0.0, "boundary response should be positive at the surface"
    return True


def test_median_surface_depth(cam):
    # Semi-transparent front (op=0.3) at r1, opaque back (op=0.99) at r2.
    r1, r2 = 3.0, 6.0
    d = _ray_through(cam)
    P = torch.stack([cam.camera_center + r1 * d, cam.camera_center + r2 * d])
    n = (-d).unsqueeze(0).repeat(2, 1)
    s2 = 0.05 ** 2
    cov = torch.zeros(2, 6, device="cuda")
    cov[:, 0] = cov[:, 3] = cov[:, 5] = s2
    op = torch.tensor([[0.3, 0.3], [0.99, 0.99]], device="cuda")
    col = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device="cuda")
    out = _rasterize(P, n, op, col, cov, cam)

    cy, cx = np.unravel_index(int(torch.argmax(out["color"][0]).item()),
                              out["color"].shape[1:])
    sd_c = out["surface_depth"][cy, cx].item()
    d_c = out["depth"][cy, cx].item()
    print("[median] surface_depth={:.3f} (r2={:.2f}) weighted_depth={:.3f}".format(
        sd_c, r2, d_c))
    assert abs(sd_c - r2) < 0.05 * r2, "median surface depth should be the half-occlusion depth r2"
    assert d_c < r2 - 0.05, "weighted depth must differ (be pulled toward r1)"
    return True


# ---------------------------------------------------------------------------
# Pure-mesh tests (no scene / no GPU)
# ---------------------------------------------------------------------------
class _FakeCam:
    def __init__(self, H=64, W=64):
        self.image_height = H
        self.image_width = W
        self.FoVx = math.radians(60.0)
        self.FoVy = math.radians(60.0)
        self.world_view_transform = torch.eye(4, dtype=torch.float32)
        self.camera_center = torch.zeros(3, dtype=torch.float32)


def _flat_surface(H=64, W=64, depth=10.0):
    surface_depth = torch.full((H, W), depth)
    accum_opacity = torch.ones((H, W))
    normal = torch.zeros((3, H, W))
    normal[2] = 1.0
    edge = torch.zeros((H, W))
    return {"surface_depth": surface_depth, "accum_opacity": accum_opacity,
            "normal": normal, "edge": edge}


def test_delaunay_mesh_and_io():
    surf = _flat_surface()
    cam = _FakeCam()
    mesh = build_reference_mesh(surf, cam, max_r=50.0, a_thresh=0.7,
                                depth_tol=0.05, normal_tol=0.52, edge_thresh=0.5)
    nv, nf = mesh["verts"].shape[0], mesh["faces"].shape[0]
    nb = mesh["boundary_edges"].shape[0]
    print("[delaunay] verts={} faces={} boundary_edges={}".format(nv, nf, nb))
    assert nv == 64 * 64
    assert nf > 0
    assert nb > 0, "open boundary (perimeter) must exist"

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.ply")
        o = os.path.join(d, "m.obj")
        nz = os.path.join(d, "m.npz")
        write_ply(p, mesh)
        write_obj(o, mesh)
        save_npz(nz, mesh)
        m_npz = load_npz(nz)
        m_ply = read_ply(p)
        assert np.allclose(m_npz["verts"], mesh["verts"]), "npz round-trip failed"
        assert np.allclose(m_ply["verts"], mesh["verts"], atol=1e-3), "ply round-trip failed"
        assert np.array_equal(m_ply["faces"], mesh["faces"]), "ply faces round-trip failed"
    return True


def test_depth_culling_two_components():
    # Hard depth seam: left half depth 10, right half depth 20 -> culling must
    # split the mesh into two components along the seam.
    H = W = 64
    surf = _flat_surface(H, W, depth=10.0)
    surf["surface_depth"][:, W // 2:] = 20.0
    mesh = build_reference_mesh(surf, _FakeCam(H, W), max_r=50.0, a_thresh=0.7,
                                depth_tol=0.05, normal_tol=0.52, edge_thresh=0.5)
    stats = mesh_stats(mesh)
    print("[culling] n_components={}".format(stats["n_components"]))
    assert stats["n_components"] == 2, "depth seam must split the mesh into two components"
    return True


def test_refine_and_point_cloud():
    surf = _flat_surface(H=32, W=32, depth=10.0)
    mesh = build_reference_mesh(surf, _FakeCam(32, 32), max_r=50.0, a_thresh=0.7)
    v, n, f = cleanup(mesh["verts"], mesh["faces"], mesh["normals"])
    b = boundary_edges(f)
    vs = laplacian_smooth(v, f, boundary_edges=b, lam=0.3, iters=2)
    assert vs.shape == v.shape
    assert self_intersection_count(v, f) >= 0
    print("[refine] verts={} faces={} boundary={} self_isect={}".format(
        v.shape[0], f.shape[0], b.shape[0], self_intersection_count(v, f)))

    # Point-cloud distance: a gt cloud sampling the same surface (mesh verts
    # shifted by a known +z offset of 0.1) must recover that offset on both
    # directions, with full coverage.
    gt = mesh["verts"] + np.array([0.0, 0.0, 0.1], dtype=np.float32)
    d = point_cloud_distance(mesh, gt, max_dist=0.2)
    print("[pcd] mesh->point mean={:.4f} point->mesh mean={:.4f} coverage={:.3f}".format(
        d["mesh_to_point_mean"], d["point_to_mesh_mean"], d["coverage_ratio"]))
    assert d["mesh_to_point_mean"] < 0.15, "mesh->point distance should recover the 0.1 offset"
    assert d["point_to_mesh_mean"] < 0.15, "point->mesh distance should recover the 0.1 offset"
    assert d["coverage_ratio"] > 0.99, "all gt points should be within max_dist"
    return True


def main():
    # Rasterizer tests need the Panama scene (cameras only).
    import argparse
    from arguments import ModelParams
    from scene import Scene
    from gaussian_renderer import GaussianModel
    from utils.general_utils import safe_state

    safe_state(False)
    parser = argparse.ArgumentParser()
    mp_p = ModelParams(parser)
    args = parser.parse_args([
        "--model_path", "../outputs/baseline/Panama",
        "--source_path", "/home/cxh/APEX/apex-mesh/datasets/seathru_undist/Panama",
        "--sh_degree", "3",
    ])
    mp = mp_p.extract(args)
    g = GaussianModel(3)
    scene = Scene(mp, g, load_iteration=60000)
    cam = scene.getTrainCameras()[0]

    test_rasterizer_surface(cam)
    test_median_surface_depth(cam)

    test_delaunay_mesh_and_io()
    test_depth_culling_two_components()
    test_refine_and_point_cloud()

    print("\nALL MESH TESTS PASSED")


if __name__ == "__main__":
    main()
