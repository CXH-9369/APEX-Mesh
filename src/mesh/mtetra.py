"""Marching-tetrahedra iso-surface extraction over a 3D Gaussian density field.

The density field follows the Gaussian Opacity Fields idea: every 3D Gaussian
contributes a weighted anisotropic normal kernel

    d(x) = sum_i  alpha_i * exp(-0.5 * ||R_i^T (x - mu_i) / s_i||^2)

where ``alpha_i`` is the Gaussian's (half-Gaussian-mean) opacity, ``mu_i`` its
position, ``R_i`` its rotation and ``s_i`` its *exponentiated* scale.  The
observable surface is the iso-surface ``d(x) = tau``, extracted with a
watertight 6-tetrahedron decomposition of the sampling grid.

Everything here is offline: it only *reads* a trained checkpoint, it never
changes training.  See the paper plan ``peppy-prancing-dusk.md``.
"""

import math

import numpy as np
import torch

from utils.general_utils import build_rotation


# ---------------------------------------------------------------------------
# Marching-tetrahedra tables
# ---------------------------------------------------------------------------
# The 8 cube corners (bit k = corner k) and the 6 tetrahedra that partition a
# cube using the body diagonal 0-7.  The face diagonals are consistent across
# neighbouring cells (bottom 0-3, top 4-7, front 0-6, back 1-7, left 0-5,
# right 2-7), which guarantees a watertight decomposition.
CUBE_CORNERS = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
], dtype=np.int64)

CUBE_TETS = np.array([
    [0, 1, 3, 7],
    [0, 2, 3, 7],
    [0, 2, 6, 7],
    [0, 4, 6, 7],
    [0, 1, 5, 7],
    [0, 4, 5, 7],
], dtype=np.int64)

# The 6 edges of a tetrahedron, as pairs of *tet-local* corner indices.
TET_EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=np.int64)

# 16-case triangle table: ``bitmask`` (bit k = tet corner k is INSIDE, d > tau)
# -> list of triangles, each triangle a triple of tet-edge indices.  This is
# topology only; the winding is fixed afterwards from the density gradient
# (outward normals), so the entries are un-oriented.
TET_CASES = [
    [],                        # 0
    [[0, 2, 1]],               # 1  v0
    [[0, 3, 4]],               # 2  v1
    [[1, 3, 4], [1, 4, 2]],    # 3  v0,v1
    [[1, 3, 5]],               # 4  v2
    [[0, 3, 5], [0, 5, 2]],    # 5  v0,v2
    [[0, 1, 5], [0, 5, 4]],    # 6  v1,v2
    [[2, 4, 5]],               # 7  v0,v1,v2 (v3 out)
    [[2, 4, 5]],               # 8  v3
    [[0, 1, 5], [0, 5, 4]],    # 9  v0,v3
    [[0, 2, 5], [0, 5, 3]],    # 10 v1,v3
    [[1, 3, 5]],               # 11 v0,v1,v3 (v2 out)
    [[1, 2, 4], [1, 4, 3]],    # 12 v2,v3
    [[0, 3, 4]],               # 13 v0,v2,v3 (v1 out)
    [[0, 1, 2]],               # 14 v1,v2,v3 (v0 out)
    [],                        # 15
]


# ---------------------------------------------------------------------------
# Density field evaluation
# ---------------------------------------------------------------------------
def gaussian_density(pts, means, rotations, scales, opacities, chunk=262144):
    """Evaluate ``d(x)`` at arbitrary points (naive, chunked — for tests / small Q).

    ``opacities`` are already sigmoided (effective per-Gaussian alpha);
    ``scales`` are already exponentiated; ``rotations`` are unit quaternions.
    """
    R = build_rotation(rotations.float())
    out = torch.zeros(pts.shape[0], device=pts.device, dtype=torch.float32)
    for i in range(0, means.shape[0], chunk):
        m = means[i:i + chunk]
        s = scales[i:i + chunk]
        r = R[i:i + chunk]
        a = opacities[i:i + chunk].reshape(-1)
        d = pts[:, None, :] - m[None, :, :]            # (M, C, 3)
        q = torch.einsum('mci,cij->mcj', d, r)          # R^T (x - mu)
        dist2 = ((q / s[None, :, :]) ** 2).sum(-1)      # (M, C)
        out += (a[None, :] * torch.exp(-0.5 * dist2)).sum(-1)
    return out


def splat_density_volume(means, rotations, scales, opacities, origin, spacing,
                         shape, alpha_min=0.05, s_cut=None, chunk=2048):
    """Splat Gaussians into a density *grid* (values at grid vertices).

    Only Gaussians with ``alpha >= alpha_min`` and (optionally) maximum scale
    ``<= s_cut`` are kept — this drops the low-opacity background fog and the
    huge-scale Gaussians that would otherwise blur the field.  Each kept
    Gaussian adds its kernel to the grid vertices within a 3-sigma box.

    Grid convention: vertex ``(i, j, k)`` sits at ``origin + (i*sx, j*sy, k*sz)``.
    """
    alpha = opacities.reshape(-1)
    keep = alpha >= alpha_min
    if s_cut is not None:
        keep = keep & (scales.max(dim=1).values <= s_cut)
    means = means[keep].float()
    scales = scales[keep].float()
    alpha = alpha[keep]
    rotations = rotations[keep]
    N = means.shape[0]
    if N == 0:
        return torch.zeros(tuple(shape), device=means.device, dtype=torch.float32)

    R = build_rotation(rotations.float())            # (N, 3, 3)
    max_s = scales.max(dim=1).values                 # (N,)
    order = torch.argsort(max_s, descending=True)
    means, scales, alpha, R, max_s = (
        means[order], scales[order], alpha[order], R[order], max_s[order])

    origin_t = torch.as_tensor(origin, device=means.device, dtype=torch.float32)
    spacing_t = torch.as_tensor(spacing, device=means.device, dtype=torch.float32)
    sx, sy, sz = (float(spacing_t[0]), float(spacing_t[1]), float(spacing_t[2]))
    nx, ny, nz = (int(shape[0]), int(shape[1]), int(shape[2]))

    density = torch.zeros((nx, ny, nz), device=means.device, dtype=torch.float32)
    flat = density.view(-1)
    n_total = nx * ny * nz

    for st in range(0, N, chunk):
        m = means[st:st + chunk]
        s = scales[st:st + chunk]
        r = R[st:st + chunk]
        a = alpha[st:st + chunk]
        r_max = 3.0 * float(max_s[st].item())        # sorted desc -> chunk max
        Rx = int(math.ceil(r_max / sx)) + 1
        Ry = int(math.ceil(r_max / sy)) + 1
        Rz = int(math.ceil(r_max / sz)) + 1
        oxr = torch.arange(-Rx, Rx + 1, device=m.device)
        oyr = torch.arange(-Ry, Ry + 1, device=m.device)
        ozr = torch.arange(-Rz, Rz + 1, device=m.device)
        OX, OY, OZ = torch.meshgrid(oxr, oyr, ozr, indexing='ij')
        off = torch.stack([OX.reshape(-1), OY.reshape(-1), OZ.reshape(-1)], dim=-1)

        base = torch.round((m - origin_t) / spacing_t).long()          # (C, 3)
        cell = base[:, None, :] + off[None, :, :]                       # (C, K, 3)
        inb = ((cell[..., 0] >= 0) & (cell[..., 0] < nx) &
               (cell[..., 1] >= 0) & (cell[..., 1] < ny) &
               (cell[..., 2] >= 0) & (cell[..., 2] < nz))
        pos = cell.float() * spacing_t + origin_t
        d = pos - m[:, None, :]
        q = torch.einsum('cki,cij->ckj', d, r)
        dist2 = ((q / s[:, None, :]) ** 2).sum(-1)
        contrib = a[:, None] * torch.exp(-0.5 * dist2)
        contrib = torch.where(inb, contrib, torch.zeros_like(contrib))
        idx = (cell[..., 0] * (ny * nz) + cell[..., 1] * nz + cell[..., 2])
        idx = idx.clamp(0, n_total - 1)
        flat.index_add_(0, idx.reshape(-1), contrib.reshape(-1))

    return density


# ---------------------------------------------------------------------------
# Marching tetrahedra
# ---------------------------------------------------------------------------
def _weld(verts, tol, chunk=20_000_000):
    """Merge coincident vertices and remap face indices.

    MT shared-edge vertices are *bit-identical* (same two grid samples and
    interpolation weight), so a single exact ``np.unique`` pass dedups them
    correctly with ~3x less peak RAM than the old two-level int64 chunked dedup
    (which materialised every vertex as an int64 key and re-sorted the per-chunk
    uniques again globally, OOM'ing the machine at res >= 1920).
    """
    verts = np.asarray(verts, dtype=np.float32)
    if verts.shape[0] == 0:
        return verts, np.zeros(0, dtype=np.int64)
    u, inverse = np.unique(verts, axis=0, return_inverse=True)
    return u.astype(np.float32, copy=False), inverse


def marching_tetrahedra(volume, spacing, origin, tau):
    """Extract the iso-surface ``volume == tau`` from a (nx, ny, nz) grid.

    Returns ``(verts (V,3) float32, faces (F,3) int32)`` in world coordinates
    (vertices welded, so faces share vertices and normals/boundary make sense).
    """
    volume = np.asarray(volume, dtype=np.float32)
    nx, ny, nz = volume.shape
    sx, sy, sz = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
    ox, oy, oz = (float(origin[0]), float(origin[1]), float(origin[2]))

    tris = []  # list of (nsel, 3, 3) float32 world-space triangles
    for k in range(nz - 1):
        I, J = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing='ij')
        I = I.ravel().astype(np.int64)
        J = J.ravel().astype(np.int64)
        for t in range(6):
            tets = CUBE_TETS[t]
            vals = np.empty((4, I.size), dtype=np.float32)
            for cc in range(4):
                dx, dy, dz = CUBE_CORNERS[tets[cc]]
                vals[cc] = volume[I + dx, J + dy, k + dz]
            inside = vals > tau
            bit = (inside[0].astype(np.int64) * 1 + inside[1].astype(np.int64) * 2 +
                   inside[2].astype(np.int64) * 4 + inside[3].astype(np.int64) * 8)
            for case, case_tris in enumerate(TET_CASES):
                if not case_tris:
                    continue
                sel = np.where(bit == case)[0]
                if sel.size == 0:
                    continue
                pos = np.empty((4, sel.size, 3), dtype=np.float32)
                for cc in range(4):
                    dx, dy, dz = CUBE_CORNERS[tets[cc]]
                    pos[cc, :, 0] = (I[sel] + dx) * sx + ox
                    pos[cc, :, 1] = (J[sel] + dy) * sy + oy
                    pos[cc, :, 2] = (k + dz) * sz + oz
                vsel = vals[:, sel]
                for tri in case_tris:
                    pts = []
                    for e in tri:
                        a, b = TET_EDGES[e]
                        va, vb = vsel[a], vsel[b]
                        denom = (vb - va).astype(np.float32)
                        safe = np.where(np.abs(denom) < 1e-12, np.float32(1.0), denom)
                        w = np.where(np.abs(denom) < 1e-12, np.float32(0.5),
                                     (np.float32(tau) - va) / safe)
                        pa, pb = pos[a], pos[b]
                        pts.append(pa + w[:, None] * (pb - pa))
                    tris.append(np.stack(pts, axis=1))

    if not tris:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int32))

    verts = np.concatenate(tris, axis=0).reshape(-1, 3)              # (T*3, 3) float32
    del tris
    # faces are just arange(3*T).reshape(-1,3), so rebuild them from the weld
    # inverse directly — avoids materialising a second (T,3) int64 buffer
    # alongside the weld's inverse (the two were the MT-phase peak).
    tol = max(1e-6, min(sx, sy, sz) * 1e-3)
    verts, remap = _weld(verts, tol)
    faces = remap.reshape(-1, 3)                                     # (T, 3)
    del remap
    # Drop degenerate faces produced by exact vertex hits / welding.
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & \
           (faces[:, 0] != faces[:, 2])
    return verts.astype(np.float32), faces[keep].astype(np.int32)


# ---------------------------------------------------------------------------
# Threshold estimation + normals
# ---------------------------------------------------------------------------
def sample_trilinear(volume, pts, origin, spacing):
    """Trilinear-sample a scalar grid at world points ``pts (M,3)`` -> ``(M,)``."""
    volume = np.asarray(volume, dtype=np.float32)
    nx, ny, nz = volume.shape
    origin = np.asarray(origin, dtype=np.float32)
    spacing = np.asarray(spacing, dtype=np.float32)
    pts = np.asarray(pts, dtype=np.float32)
    f = (pts - origin) / spacing
    i = np.floor(f).astype(np.int64)
    fx = f - i
    i0 = np.clip(i, 0, [nx - 1, ny - 1, nz - 1])
    i1 = np.clip(i + 1, 0, [nx - 1, ny - 1, nz - 1])
    v000 = volume[i0[:, 0], i0[:, 1], i0[:, 2]]
    v100 = volume[i1[:, 0], i0[:, 1], i0[:, 2]]
    v010 = volume[i0[:, 0], i1[:, 1], i0[:, 2]]
    v110 = volume[i1[:, 0], i1[:, 1], i0[:, 2]]
    v001 = volume[i0[:, 0], i0[:, 1], i1[:, 2]]
    v101 = volume[i1[:, 0], i0[:, 1], i1[:, 2]]
    v011 = volume[i0[:, 0], i1[:, 1], i1[:, 2]]
    v111 = volume[i1[:, 0], i1[:, 1], i1[:, 2]]
    x, y, z = fx[:, 0], fx[:, 1], fx[:, 2]
    v00 = v000 * (1 - x) + v100 * x
    v10 = v010 * (1 - x) + v110 * x
    v01 = v001 * (1 - x) + v101 * x
    v11 = v011 * (1 - x) + v111 * x
    v0 = v00 * (1 - y) + v10 * y
    v1 = v01 * (1 - y) + v11 * y
    return v0 * (1 - z) + v1 * z


def estimate_threshold(volume, means, origin, spacing, percentile=50.0):
    """Auto-estimate tau as a percentile of the density at the Gaussian centers."""
    means = np.asarray(means, dtype=np.float64).reshape(-1, 3)
    if means.shape[0] == 0:
        return 0.0
    d = sample_trilinear(volume, means, origin, spacing)
    d = d[d > 0]
    if d.size == 0:
        return 0.0
    return float(np.percentile(d, percentile))


def compute_normals(verts, faces, volume, origin, spacing):
    """Outward vertex normals from the density gradient + faces oriented to match.

    ``-grad d`` points toward empty space (decreasing density), i.e. outward.
    Returns ``(normals (V,3) float32, faces (F,3) int64)``.
    """
    # Central differences sampled *at the vertices* (no full-size np.gradient
    # grid): np.gradient materialises one float32 copy of the whole volume per
    # axis (~7.8 GB at res 1920), which OOM'd the machine.  Sampling the field
    # at verts +/- spacing is mathematically identical (central difference
    # commutes with trilinear interpolation) and needs only a few small buffers.
    verts = np.asarray(verts, dtype=np.float32)
    origin = np.asarray(origin, dtype=np.float32)
    spacing = np.asarray(spacing, dtype=np.float32)
    grad = np.empty((verts.shape[0], 3), dtype=np.float32)
    for a in range(3):
        h = spacing[a]
        e = np.zeros(3, dtype=np.float32)
        e[a] = h
        plus = sample_trilinear(volume, verts + e, origin, spacing)
        minus = sample_trilinear(volume, verts - e, origin, spacing)
        grad[:, a] = (plus - minus) / (2.0 * h)
        del plus, minus
    norm = np.linalg.norm(grad, axis=-1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    outward = -(grad / norm)

    if faces.size == 0:
        return outward.astype(np.float32), faces

    v = verts[faces]
    fn = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    fn_norm = np.linalg.norm(fn, axis=-1, keepdims=True)
    fn_norm[fn_norm < 1e-12] = 1.0
    fn = fn / fn_norm
    mean_out = outward[faces].mean(axis=1)
    flip = (fn * mean_out).sum(axis=-1) < 0
    faces = faces.copy()
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return outward.astype(np.float32), faces
