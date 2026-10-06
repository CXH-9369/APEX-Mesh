"""Offline surface extraction.

From the new rasterizer outputs (median surface depth, accumulated opacity,
alpha-weighted normal, half-Gaussian boundary response) build an *open*
reference-view depth mesh: reliable surface samples -> 2D Delaunay on the
valid pixel grid -> cull edges across depth / normal / boundary-response
discontinuities (open boundary, no hole-filling, no back-face closure).

Multi-view support votes each vertex by reprojecting into neighbour views and
comparing the sampled surface depth against the exact optical path
``||x - c_k||``; the per-vertex confidence combines medium transmittance
``t``, view support and a robust photometric residual.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import Delaunay

from geometry.losses import unproject


# ---------------------------------------------------------------------------
# Rendering helper
# ---------------------------------------------------------------------------
def render_surface(view, gaussians, pipe, bg, water_model):
    """Render one view and return the surface-relevant channels (GPU tensors)."""
    from gaussian_renderer import render
    pkg = render(view, gaussians, pipe, bg)
    J = pkg["render"]
    depth = pkg["depth"]
    I_hat, t, B = water_model(J, depth)
    return {
        "render": J,
        "depth": depth,
        "surface_depth": pkg["surface_depth"],
        "accum_opacity": pkg["accum_opacity"],
        "normal": pkg["normal"],
        "edge": pkg["edge"],
        "I_hat": I_hat,
        "t": t,
        "B": B,
    }


def surface_mask(surface_depth, accum_opacity, max_r, a_thresh):
    """Reliable surface-sample mask (median depth exists + well-occluded)."""
    return (surface_depth > 0) & (accum_opacity >= a_thresh) & (surface_depth < max_r)


# ---------------------------------------------------------------------------
# Reference-view depth mesh
# ---------------------------------------------------------------------------
def _edge_ok_vectorized(a, b, depth, normals, edge, depth_tol, normal_tol, edge_thresh):
    da, db = depth[a], depth[b]
    dd = np.abs(da - db) / (np.minimum(da, db) + 1e-6)
    cos = np.clip((normals[a] * normals[b]).sum(-1), -1.0, 1.0)
    ang = np.arccos(cos)
    emax = np.maximum(edge[a], edge[b])
    return (dd <= depth_tol) & (ang <= normal_tol) & (emax <= edge_thresh)


def build_reference_mesh(surface, cam, max_r=50.0, a_thresh=0.7,
                         depth_tol=0.05, normal_tol=0.52, edge_thresh=0.5,
                         stride=1):
    """Build an open reference-view mesh from one view's surface channels.

    ``stride`` subsamples the valid pixel grid (stride=2 -> 1/4 the vertices)
    *after* unprojection, so per-pixel rays stay correct while the Delaunay
    triangulation is kept tractable.

    Returns a ``mesh.Mesh`` dict (see ``mesh.io``) with ``verts``/``normals``
    in world space and culled ``faces``; ``confidence``/``support_count`` are
    filled in later by ``multiview_support``.
    """
    sd = surface["surface_depth"]
    A = surface["accum_opacity"]
    normal = surface["normal"]
    edge = surface["edge"]

    # Expected surface normal = alpha-weighted normal / accumulated opacity.
    nrm = normal / A.clamp_min(1e-6).unsqueeze(0)             # (3,H,W)
    nrm = nrm / nrm.norm(dim=0, keepdim=True).clamp_min(1e-6)

    # Unproject at full resolution so per-pixel ray directions stay exact,
    # then (optionally) subsample the resulting point set.
    pts = unproject(sd, cam)                                  # (H,W,3)

    if stride > 1:
        sd = sd[::stride, ::stride]
        A = A[::stride, ::stride]
        edge = edge[::stride, ::stride]
        pts = pts[::stride, ::stride]
        nrm = nrm[:, ::stride, ::stride]

    H, W = sd.shape
    valid = surface_mask(sd, A, max_r, a_thresh)
    flat = valid.reshape(-1)
    verts = pts.reshape(-1, 3)[flat]                          # (M,3)
    norms = nrm.reshape(3, -1).T[flat]                        # (M,3)

    sd_v = sd.reshape(-1)[flat]                               # (M,)
    edge_v = edge.reshape(-1)[flat]                           # (M,)

    # Integer pixel coordinates for 2D Delaunay triangulation.
    vv, uu = torch.meshgrid(torch.arange(H, device=sd.device),
                            torch.arange(W, device=sd.device), indexing="ij")
    uv = torch.stack([uu, vv], dim=-1).reshape(-1, 2)[flat]   # (M,2)

    verts_np = verts.detach().cpu().numpy().astype(np.float64)
    norms_np = norms.detach().cpu().numpy().astype(np.float64)
    sd_np = sd_v.detach().cpu().numpy()
    edge_np = edge_v.detach().cpu().numpy()
    uv_np = uv.detach().cpu().numpy()

    if uv_np.shape[0] < 3:
        return {"verts": verts_np.astype(np.float32),
                "normals": norms_np.astype(np.float32),
                "faces": np.zeros((0, 3), dtype=np.int32),
                "confidence": np.zeros((uv_np.shape[0],), dtype=np.float32),
                "support_count": np.zeros((uv_np.shape[0],), dtype=np.int32),
                "visible": np.ones((uv_np.shape[0],), dtype=np.uint8),
                "boundary_edges": np.zeros((0, 2), dtype=np.int32),
                "uv": uv_np.astype(np.int32),
                "edge": edge_np.astype(np.float32),
                "ref_depth": sd_np.astype(np.float32),
                "tri": np.zeros((0, 3), dtype=np.int32)}

    tri = Delaunay(uv_np)
    S = tri.simplices.astype(np.int64)                        # (F,3)
    e0 = _edge_ok_vectorized(S[:, 0], S[:, 1], sd_np, norms_np, edge_np,
                             depth_tol, normal_tol, edge_thresh)
    e1 = _edge_ok_vectorized(S[:, 1], S[:, 2], sd_np, norms_np, edge_np,
                             depth_tol, normal_tol, edge_thresh)
    e2 = _edge_ok_vectorized(S[:, 2], S[:, 0], sd_np, norms_np, edge_np,
                             depth_tol, normal_tol, edge_thresh)
    faces = S[e0 & e1 & e2].astype(np.int32)

    # Boundary edges = edges referenced by exactly one kept face.
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]) if faces.size else np.zeros((0, 2), dtype=np.int32)
    edges = np.sort(edges, axis=1)
    boundary = np.zeros((0, 2), dtype=np.int32)
    if edges.size:
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        boundary = unique[counts == 1].astype(np.int32)

    M = verts_np.shape[0]
    return {"verts": verts_np.astype(np.float32),
            "normals": norms_np.astype(np.float32),
            "faces": faces,
            "confidence": np.ones((M,), dtype=np.float32),
            "support_count": np.zeros((M,), dtype=np.int32),
            "visible": np.ones((M,), dtype=np.uint8),
            "boundary_edges": boundary,
            "uv": uv_np.astype(np.int32),
            "edge": edge_np.astype(np.float32),
            "ref_depth": sd_np.astype(np.float32),
            "tri": tri.simplices.astype(np.int32)}


# ---------------------------------------------------------------------------
# Multi-view support voting
# ---------------------------------------------------------------------------
def project_points(pts, cam):
    """Project world points ``(M,3)`` into ``cam`` -> ``(u, v, z)`` pixel coords."""
    ones = torch.ones(pts.shape[0], 1, device=pts.device, dtype=pts.dtype)
    pts_h = torch.cat([pts, ones], dim=-1)                          # (M,4)
    cam_pts = pts_h @ cam.world_view_transform.to(pts.device).float().T  # (M,4)
    z = cam_pts[..., 2].clamp_min(1e-6)
    W, H = cam.image_width, cam.image_height
    fx = W / (2.0 * math.tan(cam.FoVx * 0.5))
    fy = H / (2.0 * math.tan(cam.FoVy * 0.5))
    u = cam_pts[..., 0] / z * fx + W / 2.0
    v = cam_pts[..., 1] / z * fy + H / 2.0
    return u, v, z


def _sample_at(sd, u, v):
    H, W = sd.shape
    gu = u / (W - 1) * 2.0 - 1.0
    gv = v / (H - 1) * 2.0 - 1.0
    grid = torch.stack([gu, gv], dim=-1).unsqueeze(0).unsqueeze(0)  # (1,1,M,2)
    return F.grid_sample(sd.view(1, 1, H, W), grid, mode="bilinear",
                         padding_mode="zeros", align_corners=True).view(-1)


def multiview_support(ref_verts, cam_ref, cams, surface_depths, tol=0.05, max_r=50.0,
                      chunk=8_000_000):
    """Per-vertex support count + view-agreement score across neighbour views.

    Vertices are processed in ``chunk``-sized slabs so the per-camera
    intermediates (projected coords, warped depth, residuals) never materialise
    for all M vertices at once — the anisotropic kernel can emit 70M+ pre-clean
    vertices, and an all-at-once ``project_points`` would allocate several GB of
    (M,4) temporaries per view.

    Args:
        ref_verts: (M,3) torch (cuda) reference-view vertices.
        cam_ref: the reference camera (skipped).
        cams: list of neighbour cameras.
        surface_depths: list of (H,W) median surface-depth maps, aligned to cams.
    Returns:
        support_count: (M,) int tensor (# agreeing views).
        c_view: (M,) float tensor in [0,1] (mean agreement).
    """
    M = ref_verts.shape[0]
    device = ref_verts.device
    support = torch.zeros(M, dtype=torch.int32, device=device)
    c_view = torch.zeros(M, dtype=torch.float32, device=device)
    n = 0
    for cam, sd in zip(cams, surface_depths):
        if cam is cam_ref:
            continue
        n += 1
        cc = cam.camera_center.to(device).float().view(1, 3)
        for s in range(0, M, chunk):
            e = min(s + chunk, M)
            vv = ref_verts[s:e]
            u, v, z = project_points(vv, cam)
            warped = _sample_at(sd, u, v)
            expected = (vv - cc).norm(dim=-1)
            rel = (warped - expected).abs() / (expected + 1e-3)
            inb = (u >= 0) & (u < cam.image_width - 1) & (v >= 0) & (v < cam.image_height - 1)
            agree = (z > 0) & (expected < max_r) & inb & (rel < tol)
            support[s:e] += agree.to(torch.int32)
            c_view[s:e] += (1.0 - rel.clamp(0.0, 1.0)) * agree.to(torch.float32)
    if n > 0:
        c_view = c_view / n
    return support, c_view


def photometric_confidence(I_hat, raw):
    """Robust photometric confidence c_photo = exp(-|residual| / median)."""
    resid = (I_hat - raw).abs().mean(dim=0)                       # (H,W)
    med = resid[resid > 0].median() if (resid > 0).any() else resid.mean()
    return torch.exp(-resid / (med + 1e-6))


def confidence(t, c_view, c_photo):
    """C_med = clip(t,0,1) * c_view * c_photo."""
    return (t.clamp(0.0, 1.0) * c_view * c_photo).clamp(0.0, 1.0)
