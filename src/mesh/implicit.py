"""Implicit moving-least-squares surface from flattened Gaussians (offline).

The standard pipeline reconstructs a surface from *flattened* Gaussians with Poisson.  Without
a Poisson implementation we use the equivalent first half of that idea: each
Gaussian ``i`` votes for a plane through its centre ``mu_i`` with normal ``n_i``
(its thinnest axis), and the surface is the zero set of the weighted average

    f(x) = sum_i w_i(x) * n_i . (x - mu_i)  /  sum_i w_i(x)
    w_i(x) = alpha_i * exp(-||x - mu_i||^2 / (2 sigma^2))

``f`` is a signed distance to the surface (positive in front / free space,
negative behind), and — crucially — it is built from the Gaussian *centres*,
which are consistent 3D points, not from the per-view median depth (which the
flat discs render inconsistently across views, the source of the 17% reproj
misalignment).  The zero set is extracted with marching tetrahedra.

Normals must be consistently oriented first (``orient_normals``): the Gaussian
quaternion's thinnest axis has an arbitrary sign, so it is flipped to point
toward the nearest camera (outward for an open reef scene).
"""

import math

import torch

from utils.general_utils import build_rotation


def orient_normals(means, normals, cam_centers):
    """Flip each normal to point toward its nearest camera (outward).

    Args:
        means: (N,3) tensor.
        normals: (N,3) tensor, unit vectors of arbitrary sign.
        cam_centers: (C,3) tensor of camera centres.
    Returns:
        (N,3) oriented normals.
    """
    means = means.float()
    normals = normals.float()
    cc = cam_centers.float().to(means.device)
    d2 = ((means[:, None, :] - cc[None, :, :]) ** 2).sum(-1)      # (N, C)
    nearest = cc[d2.argmin(dim=-1)]                               # (N, 3)
    dirs = nearest - means                                       # (N, 3)
    flip = (normals * dirs).sum(dim=-1) < 0
    normals = normals.clone()
    normals[flip] = -normals[flip]
    return normals


def gaussian_normals(gaussians):
    """Per-Gaussian normal = the rotation axis of the *smallest* scale."""
    scales = gaussians.get_scaling                       # (N, 3)
    idx = scales.argmin(dim=1)                           # (N,)
    R = build_rotation(gaussians.get_rotation)           # (N, 3, 3)
    n = torch.arange(R.shape[0], device=scales.device)
    return R[n, :, idx]                                  # (N, 3)


def implicit_surface_field(means, normals, alphas, origin, spacing, shape,
                           sigma, alpha_min=0.05, sigma_n=None, chunk=2048):
    """Splat the MLS implicit function into a voxel grid.

    Returns ``(num, den)`` grids (each (nx,ny,nz)) where ``f = num / den``; the
    surface is ``f == 0``.  ``den`` is the accumulated weight and can be used to
    mask unobserved regions.

    ``sigma`` is the tangential kernel radius.  When ``sigma_n`` is given the
    kernel is *anisotropic*: wide (``sigma``) in the tangent plane of each
    flattened Gaussian and thin (``sigma_n``) along its normal ``n_i``,

        w_i(x) = alpha_i exp(-||d_perp||^2 / 2sigma^2 - (d . n_i)^2 / 2sigma_n^2)

    This is the APSS-style kernel the flattened discs actually call for: the
    isotropic form re-blurs the surface along its normal by ``sigma`` and rounds
    sharp edges off, while a small ``sigma_n ~ 0.5 * spacing`` keeps each disc's
    vote in a thin slab and recovers sharp creases (validated on synthetic cube
    / sphere: max surface error drops several-fold, no holes).
    """
    means = means.float()
    normals = normals.float()
    alphas = alphas.float().reshape(-1)
    keep = alphas >= alpha_min
    means, normals, alphas = means[keep], normals[keep], alphas[keep]
    N = means.shape[0]
    nx, ny, nz = (int(shape[0]), int(shape[1]), int(shape[2]))

    origin_t = torch.as_tensor(origin, device=means.device, dtype=torch.float32)
    spacing_t = torch.as_tensor(spacing, device=means.device, dtype=torch.float32)
    sx, sy, sz = (float(spacing_t[0]), float(spacing_t[1]), float(spacing_t[2]))

    num = torch.zeros((nx, ny, nz), device=means.device, dtype=torch.float32)
    den = torch.zeros((nx, ny, nz), device=means.device, dtype=torch.float32)
    num_flat, den_flat = num.view(-1), den.view(-1)
    n_total = nx * ny * nz

    R = int(math.ceil(3.0 * sigma / min(sx, sy, sz))) + 1
    oxr = torch.arange(-R, R + 1, device=means.device)
    OX, OY, OZ = torch.meshgrid(oxr, oxr, oxr, indexing="ij")
    off = torch.stack([OX.reshape(-1), OY.reshape(-1), OZ.reshape(-1)], dim=-1)

    for st in range(0, N, chunk):
        m = means[st:st + chunk]
        n = normals[st:st + chunk]
        a = alphas[st:st + chunk]
        base = torch.round((m - origin_t) / spacing_t).long()      # (C, 3)
        cell = base[:, None, :] + off[None, :, :]                   # (C, K, 3)
        inb = ((cell[..., 0] >= 0) & (cell[..., 0] < nx) &
               (cell[..., 1] >= 0) & (cell[..., 1] < ny) &
               (cell[..., 2] >= 0) & (cell[..., 2] < nz))
        pos = cell.float() * spacing_t + origin_t
        d = pos - m[:, None, :]                                    # (C, K, 3)
        ndot = (n[:, None, :] * d).sum(-1)                         # (C, K) d . n_i
        r2 = (d ** 2).sum(-1)
        if sigma_n is None:
            w = a[:, None] * torch.exp(-r2 / (2.0 * sigma * sigma))
        else:
            dperp2 = r2 - ndot ** 2                                # ||d - (d.n)n||^2
            w = a[:, None] * torch.exp(-0.5 * (dperp2 / (sigma * sigma) +
                                               ndot ** 2 / (sigma_n * sigma_n)))
        w = torch.where(inb, w, torch.zeros_like(w))
        idx = (cell[..., 0] * (ny * nz) + cell[..., 1] * nz + cell[..., 2])
        idx = idx.clamp(0, n_total - 1)
        num_flat.index_add_(0, idx.reshape(-1), (w * ndot).reshape(-1))
        den_flat.index_add_(0, idx.reshape(-1), w.reshape(-1))

    return num, den
