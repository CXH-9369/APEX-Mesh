"""Differentiable mesh-reprojection losses.

Every term is routed through the *alpha-weighted* ``depth`` channel, which is
the one rasterizer output that is differentiable back to the gaussian
parameters (``_w_raw`` / ``_uv`` through the projective gauge). The surface
channels (``surface_depth`` / ``accum_opacity`` / ``normal`` / ``edge``) are
``mark_non_differentiable`` and are only used here as detached masks / metrics.

No new CUDA. Reuses ``geometry.losses.unproject`` and ``mesh.extract``'s
``project_points`` / ``_sample_at`` (all pure-torch, differentiable).

Note: this module is imported directly (``from geometry.mesh_losses import ...``)
rather than via ``geometry.__init__`` to avoid a mesh<->geometry import cycle.
"""

import torch
import torch.nn.functional as F

from geometry.losses import unproject, normal_from_depth
from mesh.extract import project_points, _sample_at


def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)


def _sample_channels(t, u, v):
    """Sample a ``(C, H, W)`` field at ``(u, v)`` -> ``(M, C)``."""
    C, H, W = t.shape
    gu = u / (W - 1) * 2.0 - 1.0
    gv = v / (H - 1) * 2.0 - 1.0
    grid = torch.stack([gu, gv], dim=-1).unsqueeze(0).unsqueeze(0)   # (1,1,M,2)
    out = F.grid_sample(t.unsqueeze(0), grid, mode="bilinear",
                        padding_mode="zeros", align_corners=True)    # (1,C,1,M)
    return out.squeeze(0).squeeze(1).permute(1, 0)                   # (M,C)


def mesh_reprojection(depth, cam, mesh_verts, mesh_conf, mesh_visible,
                      surface_depth=None, beta_direct=None, max_r=50.0,
                      t_min=0.05, occl_margin=0.05, eps=1e-3):
    """L_mesh_reprojection: rendered depth vs the frozen reference mesh.

    Projects each mesh vertex into ``cam``, samples the *differentiable* rendered
    depth there and compares against the exact optical path ``||v - c||`` (NOT
    the camera-space z, which is a different quantity). Robust Charbonnier on the
    relative error, weighted by per-vertex confidence/visibility.

    Masks: in-view, in-front, within ``max_r``, rendered depth > 0, transmission
    > ``t_min`` (no strong supervision through near-opaque water), and not
    occluded (the detached median ``surface_depth`` at the pixel is not clearly
    in front of the vertex).

    Returns:
        (loss scalar, n_valid int).
    """
    device = depth.device
    u, v, z = project_points(mesh_verts, cam)
    d = _sample_at(depth, u, v)                                     # (M,) DIFF
    r = (mesh_verts - cam.camera_center.to(device).float().view(1, 3)).norm(dim=-1)
    rel = (d - r) / (r + eps)

    inb = (u >= 0) & (u < cam.image_width - 1) & (v >= 0) & (v < cam.image_height - 1)
    valid = inb & (z > 0) & (r < max_r) & (d > 0)

    if beta_direct is not None:
        t = torch.exp(-beta_direct.detach().reshape(-1).mean() * r)  # (M,)
        valid = valid & (t > t_min)

    if surface_depth is not None:
        sd_w = _sample_at(surface_depth, u, v)                       # non-diff
        occluded = (sd_w > 0) & (sd_w < r * (1.0 - occl_margin))
        valid = valid & ~occluded

    if not valid.any():
        return torch.tensor(0.0, device=device, dtype=depth.dtype), 0

    w = (mesh_conf * mesh_visible).clamp_min(0.0)
    loss = (w[valid] * charbonnier(rel[valid], eps)).mean()
    return loss, int(valid.sum().item())


def mesh_normal_loss(depth, cam, mesh_verts, mesh_normals, mesh_conf, mesh_visible,
                     beta_direct=None, max_r=50.0, t_min=0.05,
                     depth_grad_thresh=0.1, eps=1e-3):
    """L_normal: normal-from-depth vs mesh vertex normal, masked to interior.

    Compares the (depth-derived) surface normal sampled at each reprojected mesh
    vertex against the mesh vertex normal (both oriented toward the camera), via
    a robust angular loss ``charbonnier(1 - cos)``. Only interior vertices (low
    depth-gradient) contribute, since normal-from-depth is unreliable at
    discontinuities.

    Returns:
        (loss scalar, n_valid int).
    """
    device = depth.device
    u, v, z = project_points(mesh_verts, cam)
    r = (mesh_verts - cam.camera_center.to(device).float().view(1, 3)).norm(dim=-1)

    n_depth = normal_from_depth(depth, cam)                         # (3,H,W)
    n_hat = _sample_channels(n_depth, u, v)                         # (M,3)
    n_hat = n_hat / (n_hat.norm(dim=-1, keepdim=True) + 1e-6)

    n_mesh = mesh_normals
    c = cam.camera_center.to(device).float()
    view = c.unsqueeze(0) - mesh_verts
    flip = (n_mesh * view).sum(dim=-1, keepdim=True) < 0
    n_mesh = torch.where(flip, -n_mesh, n_mesh)
    cos = (n_mesh * n_hat).sum(dim=-1).clamp(-1.0, 1.0)             # (M,)

    # Interior mask: exclude depth discontinuities where the normal is bogus.
    gx = depth[:, 1:] - depth[:, :-1]                               # (H, W-1)
    gy = depth[1:, :] - depth[:-1, :]                               # (H-1, W)
    gm = torch.zeros_like(depth)
    gm[:, 1:] += gx.abs()
    gm[:, :-1] += gx.abs()
    gm[1:, :] += gy.abs()
    gm[:-1, :] += gy.abs()
    interior = _sample_at(gm, u, v) < depth_grad_thresh

    inb = (u >= 0) & (u < cam.image_width - 1) & (v >= 0) & (v < cam.image_height - 1)
    valid = inb & (z > 0) & (r < max_r) & (n_hat.norm(dim=-1) > 0.5) & interior
    if beta_direct is not None:
        t = torch.exp(-beta_direct.detach().reshape(-1).mean() * r)
        valid = valid & (t > t_min)

    if not valid.any():
        return torch.tensor(0.0, device=device, dtype=depth.dtype), 0

    w = (mesh_conf * mesh_visible).clamp_min(0.0)
    loss = (w[valid] * charbonnier(1.0 - cos[valid], eps)).mean()
    return loss, int(valid.sum().item())
