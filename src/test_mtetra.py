"""Unit tests: marching-tetrahedra extraction + Gaussian density field.

Run from ``src/``:

    python test_mtetra.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mesh.mtetra import (marching_tetrahedra, gaussian_density,
                         splat_density_volume, compute_normals,
                         estimate_threshold, sample_trilinear)
from mesh.refine import boundary_edges


def _id_quat(n):
    q = np.zeros((n, 4), dtype=np.float32)
    q[:, 0] = 1.0
    return torch.tensor(q, device="cuda")


def test_sphere():
    """MT on an analytic sphere: verts on the shell, watertight, outward normals."""
    origin = np.array([0.0, 0.0, 0.0])
    spacing = np.array([0.04, 0.04, 0.04])
    n = 34
    c = np.array([0.66, 0.66, 0.66])
    R = 0.42
    x = origin[0] + spacing[0] * np.arange(n)
    y = origin[1] + spacing[1] * np.arange(n)
    z = origin[2] + spacing[2] * np.arange(n)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    # signed distance: positive inside the ball -> iso-surface d=0 is the sphere
    vol = R - np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2)

    verts, faces = marching_tetrahedra(vol, spacing, origin, tau=0.0)
    assert verts.shape[0] > 0 and faces.shape[0] > 0, "empty extraction"

    # every vertex sits on the sphere (within a couple of cells of linear-interp error)
    r = np.linalg.norm(verts - c, axis=1)
    err = np.abs(r - R)
    assert err.max() < 2.5 * spacing[0], "vertices off the sphere: max err {:.4f}".format(err.max())

    # watertight: no boundary edges (sphere fully inside the grid)
    be = boundary_edges(faces)
    assert be.size == 0, "sphere mesh is not watertight ({} boundary edges)".format(be.shape[0])

    # genus-0 Euler characteristic: F = 2V - 4 for a closed triangulated surface
    v, f = verts.shape[0], faces.shape[0]
    assert abs(f - (2 * v - 4)) <= 4, "Euler check failed: V={} F={}".format(v, f)

    # outward normals from the density gradient
    normals, _ = compute_normals(verts, faces, vol, origin, spacing)
    dots = (normals * (verts - c)).sum(axis=1)
    assert (dots > 0).mean() > 0.98, "normals not outward"

    print("  [sphere] V={} F={} max_err={:.4f} boundary={} OK".format(
        v, f, err.max(), be.shape[0]))


def test_gaussian_density_analytic():
    """Single isotropic Gaussian matches the closed-form kernel."""
    mu = torch.zeros((1, 3), device="cuda")
    scale = torch.tensor([[0.3, 0.3, 0.3]], device="cuda")
    rot = _id_quat(1)
    alpha = torch.ones((1,), device="cuda")

    pts = torch.tensor([[0.0, 0.0, 0.0],
                        [0.3, 0.0, 0.0],
                        [0.0, 0.6, 0.0]], device="cuda")
    d = gaussian_density(pts, mu, rot, scale, alpha).cpu().numpy()
    r = np.linalg.norm(pts.cpu().numpy(), axis=1) / 0.3
    expected = np.exp(-0.5 * r ** 2)
    assert np.allclose(d, expected, atol=1e-4), "density mismatch: {} vs {}".format(d, expected)

    # anisotropic: scales (0.3, 0.1, 0.1) -> d at (0,0.2,0) = exp(-0.5*(0.2/0.1)^2)
    scale2 = torch.tensor([[0.3, 0.1, 0.1]], device="cuda")
    pts2 = torch.tensor([[0.0, 0.2, 0.0]], device="cuda")
    d2 = gaussian_density(pts2, mu, rot, scale2, alpha).cpu().numpy()
    assert np.allclose(d2, [np.exp(-0.5 * 4.0)], atol=1e-4), "anisotropic mismatch: {}".format(d2)

    print("  [density] isotropic + anisotropic closed-form OK")


def test_splat_consistency():
    """Splatting onto a grid matches the analytic kernel at grid vertices."""
    mu = torch.tensor([[0.5, 0.5, 0.5]], device="cuda")
    scale = torch.tensor([[0.1, 0.1, 0.1]], device="cuda")
    rot = _id_quat(1)
    alpha = torch.ones((1,), device="cuda")
    origin = np.array([0.0, 0.0, 0.0])
    spacing = np.array([0.05, 0.05, 0.05])
    shape = (21, 21, 21)

    vol = splat_density_volume(mu, rot, scale, alpha, origin, spacing, shape,
                               alpha_min=0.0)
    vol = vol.cpu().numpy()

    # center grid vertex is exactly at mu -> d = 1.0
    assert abs(vol[10, 10, 10] - 1.0) < 1e-3, "center value {}".format(vol[10, 10, 10])
    # one spacing away (0.05 = 0.5 sigma) -> exp(-0.5 * 0.25) = 0.8825
    expected = np.exp(-0.5 * 0.5 ** 2)
    assert abs(vol[11, 10, 10] - expected) < 1e-3, "1-cell value {} vs {}".format(
        vol[11, 10, 10], expected)

    # estimate_threshold samples at the Gaussian centers
    tau = estimate_threshold(vol, mu.cpu().numpy(), origin, spacing, percentile=50.0)
    assert abs(tau - 1.0) < 1e-3, "threshold estimate {}".format(tau)

    print("  [splat] grid splat + threshold estimate OK (tau={:.3f})".format(tau))


if __name__ == "__main__":
    print("=== test_mtetra ===")
    test_sphere()
    test_gaussian_density_analytic()
    test_splat_consistency()
    print("ALL PASS")
