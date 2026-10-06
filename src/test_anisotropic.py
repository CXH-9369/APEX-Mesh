"""Unit tests: anisotropic (APSS-style) MLS kernel vs. plain isotropic.

Run from ``src/``:

    python test_anisotropic.py

The flattened Gaussians are thin surface-aligned discs, so extracting them with
an *isotropic* kernel re-blurs the surface along its normal and rounds sharp
edges off.  A kernel that is wide tangentially (``sigma``) but thin along each
disc's normal (``sigma_n ~ 0.5 * spacing``) recovers sharp creases and a much
more accurate surface.  These tests check that on a cube (sharp corners) and a
sphere (smooth, must not degrade) with no holes.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mesh.implicit import implicit_surface_field
from mesh.mtetra import marching_tetrahedra
from mesh.refine import boundary_edges

OUT = "/home/cxh/APEX/apex-mesh/outputs/test_anisotropic"


def _grid(means, spacing, sigma):
    bmin = means.cpu().numpy().min(axis=0)
    bmax = means.cpu().numpy().max(axis=0)
    extent = bmax - bmin
    pad = 3.0 * sigma
    origin = bmin - pad
    shape = tuple(int(s) for s in (np.ceil((extent + 2.0 * pad) / spacing) + 2).astype(np.int64))
    return origin, shape


def _extract(means, normals, alphas, spacing, sigma, sigma_n, w_thresh=0.2):
    spacing = np.full(3, spacing, dtype=np.float64)
    origin, shape = _grid(means, spacing, sigma)
    num, den = implicit_surface_field(
        means, normals, alphas, origin, spacing, shape, sigma,
        alpha_min=0.0, sigma_n=sigma_n)
    den.clamp_min_(1e-6)
    f = num / den
    f[den < w_thresh] = 1.0
    vol = (-f).cpu().numpy()
    return marching_tetrahedra(vol, spacing, origin, tau=0.0)


def cube_gaussians(H=0.5, gs=0.05, device="cuda"):
    means, normals = [], []
    coords = np.arange(-H + gs / 2.0, H, gs)
    for axis in range(3):
        for sign in (-1.0, 1.0):
            n = np.zeros(3, dtype=np.float32)
            n[axis] = sign
            others = [a for a in range(3) if a != axis]
            for a in coords:
                for b in coords:
                    p = np.zeros(3, dtype=np.float32)
                    p[axis] = sign * H
                    p[others[0]] = a
                    p[others[1]] = b
                    means.append(p)
                    normals.append(n)
    return (torch.tensor(np.array(means), device=device),
            torch.tensor(np.array(normals), device=device))


def sphere_gaussians(R=0.45, n=3000, device="cuda"):
    idx = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * idx / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * idx
    p = np.stack([np.sin(phi) * np.cos(theta),
                  np.sin(phi) * np.sin(theta),
                  np.cos(phi)], axis=1) * R
    return torch.tensor(p, device=device), torch.tensor(p / R, device=device)


def _cube_rounding(verts, H=0.5):
    d = np.abs(np.asarray(verts)).max(axis=1) - H
    return float(np.abs(d).max()), float(np.abs(d).mean())


def _sphere_err(verts, R=0.45):
    r = np.linalg.norm(np.asarray(verts), axis=1)
    err = np.abs(r - R)
    return float(err.max()), float(err.mean())


def _write_obj(path, verts, faces):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for v in verts:
            f.write("v %.6f %.6f %.6f\n" % (v[0], v[1], v[2]))
        for tri in faces:
            f.write("f %d %d %d\n" % (tri[0] + 1, tri[1] + 1, tri[2] + 1))


def test_cube_sharpening():
    H, gs, spacing, sigma = 0.5, 0.05, 0.02, 0.06
    means, normals = cube_gaussians(H=H, gs=gs)
    alphas = torch.ones(means.shape[0], device=means.device)

    vi, fi = _extract(means, normals, alphas, spacing, sigma, sigma_n=None)   # isotropic
    vb, fb = _extract(means, normals, alphas, spacing, sigma, sigma_n=0.5 * spacing)

    ri = _cube_rounding(vi, H)
    rb = _cube_rounding(vb, H)
    bei, beb = boundary_edges(fi).size, boundary_edges(fb).size

    print("  [cube] isotropic V=%d max_round=%.4f mean=%.4f  boundary=%d" %
          (vi.shape[0], ri[0], ri[1], bei))
    print("  [cube] aniso(.5s) V=%d max_round=%.4f mean=%.4f  boundary=%d" %
          (vb.shape[0], rb[0], rb[1], beb))

    _write_obj(os.path.join(OUT, "cube_iso.obj"), vi, fi)
    _write_obj(os.path.join(OUT, "cube_aniso.obj"), vb, fb)

    assert bei == 0 and beb == 0, "cube meshes must be watertight"
    assert rb[0] <= 0.7 * ri[0], "anisotropic should sharpen the corner by >=30%%"
    print("  [cube] sharpening: max_round %.4f -> %.4f" % (ri[0], rb[0]))


def test_sphere_accuracy():
    R, spacing, sigma = 0.45, 0.03, 0.06
    means, normals = sphere_gaussians(R=R)
    alphas = torch.ones(means.shape[0], device=means.device)

    vi, fi = _extract(means, normals, alphas, spacing, sigma, sigma_n=None)
    vb, fb = _extract(means, normals, alphas, spacing, sigma, sigma_n=0.5 * spacing)

    ei = _sphere_err(vi, R)
    eb = _sphere_err(vb, R)
    bei, beb = boundary_edges(fi).size, boundary_edges(fb).size

    print("  [sphere] isotropic max_err=%.4f mean=%.4f  boundary=%d" % (ei[0], ei[1], bei))
    print("  [sphere] aniso(.5s) max_err=%.4f mean=%.4f  boundary=%d" % (eb[0], eb[1], beb))

    _write_obj(os.path.join(OUT, "sphere_iso.obj"), vi, fi)
    _write_obj(os.path.join(OUT, "sphere_aniso.obj"), vb, fb)

    assert bei == 0 and beb == 0, "sphere meshes must be watertight"
    assert eb[0] <= 0.7 * ei[0], "anisotropic should sharpen the sphere, not degrade it"
    print("  [sphere] sharpening: max_err %.4f -> %.4f" % (ei[0], eb[0]))


def test_sigma_n_too_small_holes():
    """A sigma_n far below the Gaussian spacing fragments the surface (holes)."""
    H, gs, spacing, sigma = 0.5, 0.05, 0.02, 0.06
    means, normals = cube_gaussians(H=H, gs=gs)
    alphas = torch.ones(means.shape[0], device=means.device)

    v_tiny, _ = _extract(means, normals, alphas, spacing, sigma, sigma_n=0.15 * spacing)
    v_ok, _ = _extract(means, normals, alphas, spacing, sigma, sigma_n=0.5 * spacing)
    print("  [holes] sigma_n=0.15s V=%d  vs sigma_n=0.5s V=%d" % (v_tiny.shape[0], v_ok.shape[0]))
    assert v_tiny.shape[0] < 0.5 * v_ok.shape[0], "too-small sigma_n should lose surface"


if __name__ == "__main__":
    print("=== test_anisotropic ===")
    test_cube_sharpening()
    test_sphere_accuracy()
    test_sigma_n_too_small_holes()
    print("ALL PASS")
