"""Offline mesh tests (fusion / edge-weighted Laplacian / cleanup).

All tests are synthetic (no scene load): flat multi-view fusion, outlier
robustness (median vs mean), edge-weighted vs uniform Laplacian crease
preservation, small-component removal + remapping, and silhouette support.
"""

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from mesh import (build_reference_mesh, fuse_multiview, silhouette_support,
                  edge_weighted_laplacian, laplacian_smooth, remove_small_components,
                  boundary_edges)
from mesh.extract import project_points


# ---------------------------------------------------------------------------
# Synthetic camera + plane surfaces
# ---------------------------------------------------------------------------
class _TransCam:
    """Camera at ``center`` looking along world +z (identity rotation)."""
    def __init__(self, center, H=64, W=64):
        self.image_height = H
        self.image_width = W
        self.FoVx = math.radians(60.0)
        self.FoVy = math.radians(60.0)
        self.camera_center = torch.tensor(center, dtype=torch.float32, device="cuda")
        v = torch.eye(4, dtype=torch.float32, device="cuda")
        v[0, 3] = -center[0]
        v[1, 3] = -center[1]
        v[2, 3] = -center[2]
        self.world_view_transform = v


def _plane_surface(cam, plane_z=10.0):
    """Analytic surface channels for a flat world ``z = plane_z`` plane."""
    from geometry.losses import ray_dirs
    dirs = ray_dirs(cam, device="cuda")                     # (H,W,3)
    dz = dirs[..., 2]
    t = (plane_z - cam.camera_center[2]) / dz.clamp_min(1e-3)
    sd = torch.where(dz > 0, t, torch.zeros_like(t))
    H, W = sd.shape
    A = (dz > 0).float()
    normal = torch.zeros(3, H, W, device="cuda")
    normal[2] = 1.0
    edge = torch.zeros(H, W, device="cuda")
    return {"surface_depth": sd, "accum_opacity": A, "normal": normal, "edge": edge}


# ---------------------------------------------------------------------------
# Test 1+2: multi-view fusion (flat agreement + outlier robustness)
# ---------------------------------------------------------------------------
def test_fuse_multiview():
    H = W = 64
    ref_cam = _TransCam((0.0, 0.0, 0.0), H, W)
    good_cams = [_TransCam((0.3, 0.0, 0.0), H, W),
                 _TransCam((0.6, 0.0, 0.0), H, W),
                 _TransCam((0.9, 0.0, 0.0), H, W)]

    # (1) flat: all views agree on plane z=10 -> verts stay on the plane and
    # every vertex is supported by (nearly) every neighbour.
    cams = [ref_cam] + good_cams
    surfaces = [_plane_surface(c, 10.0) for c in cams]
    mesh = build_reference_mesh(surfaces[0], ref_cam, max_r=50.0, a_thresh=0.7,
                                depth_tol=0.05, normal_tol=0.52, edge_thresh=0.5, stride=2)
    verts, normals, support, c_view = fuse_multiview(
        mesh, ref_cam, cams, surfaces, tol=0.05, max_r=50.0, a_thresh=0.7, robust="median")

    z_err = (verts[:, 2] - 10.0).abs().max().item()
    print("[fuse flat] verts={} support mean={:.2f} max={} |z-10|max={:.4f}".format(
        verts.shape[0], support.float().mean().item(), int(support.max().item()), z_err))
    assert z_err < 0.05, "flat fusion must keep verts on the z=10 plane"
    assert int(support.max().item()) == len(good_cams), "some vertex should see all neighbours"
    assert support.float().mean().item() > 0.8 * len(good_cams), \
        "most verts should be supported by most neighbours"

    # (2) outlier: one neighbour is a *slightly* biased plane (z=10.4, still
    # within tol) -> median rejects the bias, mean is pulled toward it.
    biased = [_plane_surface(c, 10.0) for c in cams]
    biased[2] = _plane_surface(cams[2], 10.4)   # replace the 3rd neighbour
    v_med, _, _, _ = fuse_multiview(mesh, ref_cam, cams, biased, tol=0.05,
                                    max_r=50.0, a_thresh=0.7, robust="median")
    v_mean, _, _, _ = fuse_multiview(mesh, ref_cam, cams, biased, tol=0.05,
                                     max_r=50.0, a_thresh=0.7, robust="mean")
    z_med = v_med[:, 2].mean().item()
    z_mean = v_mean[:, 2].mean().item()
    print("[fuse outlier] median mean-z={:.4f}  mean mean-z={:.4f} (true=10.0)".format(z_med, z_mean))
    assert abs(z_med - 10.0) < 0.01, "median should reject the biased neighbour"
    assert z_mean > z_med + 0.05, "mean should be degraded toward the biased neighbour"
    return True


# ---------------------------------------------------------------------------
# Test 3: edge-weighted vs uniform Laplacian crease preservation
# ---------------------------------------------------------------------------
def _grid_mesh(H, W, z_fn):
    verts = []
    for i in range(H):
        for j in range(W):
            x = (j / (W - 1) - 0.5) * 2.0
            y = (i / (H - 1) - 0.5) * 2.0
            verts.append([x, y, z_fn(x, y)])
    verts = np.asarray(verts, dtype=np.float64)
    faces = []
    for i in range(H - 1):
        for j in range(W - 1):
            a = i * W + j
            b = i * W + j + 1
            c = (i + 1) * W + j
            d = (i + 1) * W + j + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    return verts, np.asarray(faces, dtype=np.int32)


def test_edge_weighted_laplacian():
    H, W = 3, 21
    verts, faces = _grid_mesh(H, W, lambda x, y: abs(x))
    b = boundary_edges(faces)

    # Boundary response: high on the crease column (x == 0), low elsewhere.
    edge = np.zeros(verts.shape[0], dtype=np.float64)
    crease = (np.abs(verts[:, 0]) < 1e-6)
    edge[crease] = 1.0

    vs_u = laplacian_smooth(verts, faces, boundary_edges=b, lam=0.3, iters=3)
    vs_w = edge_weighted_laplacian(verts, faces, boundary_edges=b, edge=edge,
                                   gamma=5.0, lam=0.3, iters=3)

    # Interior crease vertex is row 1, col 10.
    idx = 1 * W + 10
    z_u = vs_u[idx, 2]
    z_w = vs_w[idx, 2]
    print("[crease] uniform z={:.4f}  edge-weighted z={:.4f} (orig=0.0)".format(z_u, z_w))
    assert z_w < z_u, "edge-weighted Laplacian must preserve the crease better"
    assert z_w < 0.01, "edge-weighted smoothing should not round the crease"
    assert z_u > 0.01, "uniform smoothing should round the crease"
    return True


# ---------------------------------------------------------------------------
# Test 4: small-component removal + remapping (and silhouette retention)
# ---------------------------------------------------------------------------
def test_remove_small_components():
    big_v, big_f = _grid_mesh(10, 10, lambda x, y: 0.0)      # 100 verts, 162 faces
    # Three tiny isolated triangles (1 face each, disconnected).
    tiny_v = np.asarray([[100.0, 0, 0], [100.1, 0, 0], [100.0, 0.1, 0],
                         [101.0, 0, 0], [101.1, 0, 0], [101.0, 0.1, 0],
                         [102.0, 0, 0], [102.1, 0, 0], [102.0, 0.1, 0]], dtype=np.float64)
    tiny_f = np.asarray([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int32) + big_v.shape[0]
    verts = np.concatenate([big_v, tiny_v], axis=0)
    faces = np.concatenate([big_f, tiny_f], axis=0)

    # (a) no silhouette -> tiny components are dropped and everything remaps.
    mesh = {"verts": verts.astype(np.float32), "faces": faces}
    out = remove_small_components(mesh, min_faces=50)
    print("[small-comp] verts {}->{}  faces {}->{}".format(
        verts.shape[0], out["verts"].shape[0], faces.shape[0], out["faces"].shape[0]))
    assert out["faces"].shape[0] == big_f.shape[0], "tiny components must be dropped"
    assert out["verts"].shape[0] == big_v.shape[0], "dangling verts must be removed"
    assert out["faces"].max() < out["verts"].shape[0], "face indices must be remapped"
    assert out["boundary_edges"].shape[0] > 0, "boundary edges must be recomputed"

    # (b) a tiny component with high silhouette support is retained.
    sil = np.zeros(verts.shape[0], dtype=np.float32)
    sil[big_v.shape[0]:] = 1.0    # tiny comps -> strong silhouette
    mesh2 = {"verts": verts.astype(np.float32), "faces": faces, "silhouette": sil}
    out2 = remove_small_components(mesh2, min_faces=50, silhouette=sil, silhouette_thresh=0.2)
    print("[small-comp sil] faces kept {} (big={} + 3 tiny)".format(out2["faces"].shape[0], big_f.shape[0]))
    assert out2["faces"].shape[0] == big_f.shape[0] + 3, "high-silhouette tiny comps must be kept"
    return True


# ---------------------------------------------------------------------------
# Test 5: silhouette_support separates true edge from flat patch
# ---------------------------------------------------------------------------
def test_silhouette_support():
    cam0 = _TransCam((0.0, 0.0, 0.0), 32, 32)
    cam1 = _TransCam((0.5, 0.0, 0.0), 32, 32)
    cams = [cam0, cam1]

    bv = torch.tensor([[0.0, 0.0, 10.0], [2.0, 2.0, 10.0]], device="cuda")
    edge_maps, accum_maps = [], []
    for cam in cams:
        edge = torch.zeros(cam.image_height, cam.image_width, device="cuda")
        A = torch.ones(cam.image_height, cam.image_width, device="cuda")
        u, v, z = project_points(bv[:1], cam)          # the true-edge vertex
        ui, vi = int(round(u[0].item())), int(round(v[0].item()))
        edge[vi, ui] = 1.0                             # edge evidence at that pixel only
        edge_maps.append(edge)
        accum_maps.append(A)

    sil = silhouette_support(bv, cams, edge_maps, accum_maps, edge_thresh=0.5)
    print("[silhouette] true-edge={:.3f}  flat={:.3f}".format(sil[0].item(), sil[1].item()))
    assert sil[0].item() > 0.9, "true-edge vertex should get high silhouette support"
    assert sil[1].item() < 0.1, "flat vertex should get ~0 silhouette support"
    return True


def main():
    from utils.general_utils import safe_state
    safe_state(False)

    test_fuse_multiview()
    test_edge_weighted_laplacian()
    test_remove_small_components()
    test_silhouette_support()

    print("\nALL MESH TESTS PASSED")


if __name__ == "__main__":
    main()
