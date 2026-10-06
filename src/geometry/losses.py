"""Differentiable multi-view geometry losses.

All functions operate on the *alpha-weighted* per-pixel depth produced by the
rasterizer. This is a soft composite (see the surface-depth for the hard
variant), so the losses are phrased as a photometric-geometric *consistency*
rather than a hard surface reprojection.

The depth stored by the rasterizer is the exact optical path ``r = ||x - c||``
along each pixel ray, so unprojecting with the pixel ray direction
``x = c + d * ray_dir`` is the natural inverse.
"""

import math

import torch
import torch.nn.functional as F


def ray_dirs(cam, device=None):
    """Unit pixel ray directions in *world* space.

    Args:
        cam: Camera with FoVx/FoVy, image_width/image_height and
             world_view_transform (world->camera 4x4).
    Returns:
        dirs: (H, W, 3) unit world-space ray directions.
    """
    H, W = cam.image_height, cam.image_width
    fx = W / (2.0 * math.tan(cam.FoVx * 0.5))
    fy = H / (2.0 * math.tan(cam.FoVy * 0.5))
    cx, cy = W / 2.0, H / 2.0

    if device is None:
        device = cam.world_view_transform.device
    u, v = torch.meshgrid(
        torch.arange(W, device=device, dtype=torch.float32),
        torch.arange(H, device=device, dtype=torch.float32),
        indexing="xy",
    )
    x = (u - cx) / fx
    y = (v - cy) / fy
    dirs_cam = torch.stack([x, y, torch.ones_like(x)], dim=-1)      # (H, W, 3)
    dirs_cam = dirs_cam / dirs_cam.norm(dim=-1, keepdim=True)

    # v_world = R_w2c^T @ v_cam  <=>  dirs_world = dirs_cam @ R_w2c^T
    R_w2c = cam.world_view_transform[:3, :3]
    dirs_world = dirs_cam @ R_w2c.T
    return dirs_world


def unproject(depth, cam):
    """Lift alpha-weighted depth back to world space.

    Args:
        depth: (H, W) optical-path depth.
        cam: source camera.
    Returns:
        pts: (H, W, 3) world-space points ``c + d * ray``.
    """
    dirs = ray_dirs(cam, device=depth.device)                       # (H, W, 3)
    c = cam.camera_center.to(depth.device).float().view(1, 1, 3)
    return c + depth.unsqueeze(-1) * dirs


def normal_from_depth(depth, cam):
    """Differentiable surface normal from alpha-weighted depth (world space).

    Central differences of the unprojected point cloud, cross product, oriented
    toward the camera. Border ring is zero-padded (callers mask it out).

    Args:
        depth: (H, W) optical-path depth.
        cam: source camera.
    Returns:
        (3, H, W) unit normals.
    """
    pts = unproject(depth, cam)                                     # (H,W,3)
    du = pts[:, 2:, :] - pts[:, :-2, :]                             # (H, W-2, 3)
    dv = pts[2:, :, :] - pts[:-2, :, :]                             # (H-2, W, 3)
    du = du[1:-1, :, :]                                             # (H-2, W-2, 3)
    dv = dv[:, 1:-1, :]                                             # (H-2, W-2, 3)
    n = torch.cross(du, dv, dim=-1)
    n = n / (n.norm(dim=-1, keepdim=True) + 1e-6)

    c = cam.camera_center.to(depth.device).float().view(1, 1, 3)
    center_pts = pts[1:-1, 1:-1, :]                                 # (H-2, W-2, 3)
    view = c - center_pts                                           # surface -> camera
    flip = (n * view).sum(dim=-1, keepdim=True) < 0
    n = torch.where(flip, -n, n)

    H, W = depth.shape
    out = torch.zeros(3, H, W, device=depth.device, dtype=depth.dtype)
    out[:, 1:-1, 1:-1] = n.permute(2, 0, 1)
    return out


def _project(pts, cam):
    """Project world points into ``cam``'s pixel coordinates.

    Returns:
        u, v: (H, W) pixel coordinates (float).
        z: (H, W) camera-space depth (positive in front of camera).
    """
    H, W = pts.shape[:2]
    ones = torch.ones(H, W, 1, device=pts.device)
    pts_h = torch.cat([pts, ones], dim=-1)                          # (H, W, 4)
    cam_pts = pts_h @ cam.world_view_transform.to(pts.device).float().T  # (H, W, 4)
    z = cam_pts[..., 2].clamp_min(1e-6)
    fx = W / (2.0 * math.tan(cam.FoVx * 0.5))
    fy = H / (2.0 * math.tan(cam.FoVy * 0.5))
    u = cam_pts[..., 0] / z * fx + W / 2.0
    v = cam_pts[..., 1] / z * fy + H / 2.0
    return u, v, z


def depth_reproj_consistency(d_A, d_B, camA, camB, max_r=50.0):
    """Two-view depth reprojection consistency.

    Unproject A's depth to 3D, project into B, sample B's depth there and
    compare against the exact optical path from B's centre to the point.

    Args:
        d_A: (H, W) depth rendered from camA (exact optical path).
        d_B: (H, W) depth rendered from camB.
        camA, camB: the two cameras.
        max_r: reject pixels whose unprojected point is farther than this.

    Returns:
        scalar mean absolute consistency error over valid pixels.
    """
    H, W = d_A.shape
    pts = unproject(d_A, camA)                                      # (H, W, 3)

    u, v, z = _project(pts, camB)

    # Sample B's depth at the projected location.
    grid_u = u / (W - 1) * 2.0 - 1.0
    grid_v = v / (H - 1) * 2.0 - 1.0
    grid = torch.stack([grid_u, grid_v], dim=-1).unsqueeze(0)       # (1, H, W, 2)
    d_B_flat = d_B.view(1, 1, H, W)
    warped = F.grid_sample(
        d_B_flat, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    ).view(H, W)

    expected = (pts - camB.camera_center.to(pts.device).float().view(1, 1, 3)).norm(dim=-1)

    valid = (
        (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        & (z > 0) & (d_A > 0) & (expected < max_r)
    )
    if valid.any():
        # Relative error: the scene is not unit-normalised (optical path is
        # O(10) units), so an absolute mean would be scale-dependent and would
        # need a per-scene loss weight. Relative keeps the loss ~O(0.1).
        rel = (warped - expected).abs() / (expected + 1e-3)
        return rel[valid].mean()
    return torch.tensor(0.0, device=d_A.device, dtype=d_A.dtype)


def edge_aware_depth_smooth(depth, color, lam=10.0):
    """Edge-aware depth smoothness regulariser (low weight).

    Penalises depth gradients weighted down where the color gradient is large,
    so depth stays smooth on flat textureless regions but can break at edges.

    Args:
        depth: (H, W).
        color: (3, H, W) in [0, 1].
        lam: edge falloff strength.

    Returns:
        scalar regulariser.
    """
    d = depth.view(1, 1, *depth.shape[-2:])
    c = color.view(1, 3, *color.shape[-2:])

    dx = d[..., 1:] - d[..., :-1]
    dy = d[..., 1:, :] - d[..., :-1, :]

    gx = torch.exp(-lam * (c[..., 1:] - c[..., :-1]).abs().mean(dim=1, keepdim=True))
    gy = torch.exp(-lam * (c[..., 1:, :] - c[..., :-1, :]).abs().mean(dim=1, keepdim=True))

    return (dx.abs() * gx).mean() + (dy.abs() * gy).mean()
