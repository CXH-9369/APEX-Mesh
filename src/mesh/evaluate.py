"""Geometry evaluation.

Without MOUD lidar or instance annotations on this machine, the runnable
held-out-view geometric checks are:

* ``reproject_consistency_heldout`` — reproject the mesh into a held-out view
  and compare its exact optical path against that view's rendered median
  surface depth (relative error).
* ``boundary_edge_agreement`` — do reprojected mesh boundary vertices land on
  above-average half-Gaussian boundary response (E_v) in the held-out view?

``point_cloud_distance`` implements the MOUD lidar metric spec (mesh->point
mean/median/p90, point->mesh coverage) via scipy cKDTree; it is exercised by
unit tests and stays latent until MOUD data is provided.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from mesh.extract import project_points, _sample_at
from geometry.losses import (unproject, _project, depth_reproj_consistency,
                             normal_from_depth)


def reproject_consistency_heldout(mesh_verts, cam, surface_depth, max_r=50.0):
    """Mean relative reprojection error of mesh verts against a held-out view.

    Args:
        mesh_verts: (M,3) torch (cuda) world vertices.
        cam: held-out camera.
        surface_depth: (H,W) torch median surface depth rendered from cam.
    Returns:
        dict with mean relative error and number of in-view vertices.
    """
    u, v, z = project_points(mesh_verts, cam)
    warped = _sample_at(surface_depth, u, v)
    expected = (mesh_verts - cam.camera_center.to(mesh_verts.device).float().view(1, 3)).norm(dim=-1)
    inb = (u >= 0) & (u < cam.image_width - 1) & (v >= 0) & (v < cam.image_height - 1)
    valid = (z > 0) & (expected < max_r) & inb & (warped > 0)
    if not valid.any():
        return {"mean_rel_err": float("nan"), "n_verts": 0}
    rel = (warped[valid] - expected[valid]).abs() / (expected[valid] + 1e-3)
    return {"mean_rel_err": float(rel.mean().item()), "n_verts": int(valid.sum().item())}


def boundary_edge_agreement(boundary_verts, cam, edge_map):
    """Mean E_v at reprojected boundary verts vs the view-wide mean E_v.

    Returns a ratio; >1 means mesh boundaries align with edge evidence.
    """
    u, v, z = project_points(boundary_verts, cam)
    inb = (u >= 0) & (u < cam.image_width - 1) & (v >= 0) & (v < cam.image_height - 1) & (z > 0)
    if not inb.any():
        return {"ratio": float("nan"), "n_verts": 0}
    sampled = _sample_at(edge_map, u[inb], v[inb])
    view_mean = edge_map[edge_map > 0].mean() if (edge_map > 0).any() else edge_map.mean()
    ratio = float((sampled.mean() / (view_mean + 1e-6)).item())
    return {"ratio": ratio, "n_verts": int(inb.sum().item())}


@torch.no_grad()
def depth_boundary_metric(boundary_verts, cams, edge_maps):
    """Batch boundary-edge agreement over multiple views (eval-only log metric).

    Thin batch wrapper around :func:`boundary_edge_agreement` for training logs:
    reports the mean ratio across the given views (a single scalar), ignoring
    views where no boundary vertex is in-frame. Forward-only (no_grad), matching
    the plan's "evaluation-only" role.

    Args:
        boundary_verts: (M,3) world vertices on the mesh open boundary.
        cams: sequence of camera objects.
        edge_maps: sequence of (H,W) half-Gaussian edge-response maps, one per cam.
    Returns:
        dict with ``mean_ratio`` (nan if no view contributed) and ``n_views``.
    """
    ratios = []
    for cam, emap in zip(cams, edge_maps):
        b = boundary_edge_agreement(boundary_verts, cam, emap)
        if b["n_verts"]:
            ratios.append(b["ratio"])
    return {"mean_ratio": float(np.mean(ratios)) if ratios else float("nan"),
            "n_views": int(len(ratios))}


def point_cloud_distance(mesh, gt_points, max_dist):
    """MOUD lidar metric (mesh <-> held-out point cloud), via cKDTree.

    Args:
        mesh: mesh.Mesh dict (uses ``verts``).
        gt_points: (G,3) held-out lidar points.
        max_dist: coverage threshold (physical units).
    Returns:
        dict with mesh->point and point->mesh distances (mean/median/p90) and
        the coverage ratio of gt points within ``max_dist`` of the mesh.
    """
    mesh_verts = np.asarray(mesh["verts"], dtype=np.float64)
    gt = np.asarray(gt_points, dtype=np.float64)
    tree_gt = cKDTree(gt)
    tree_m = cKDTree(mesh_verts)
    d_m2p, _ = tree_gt.query(mesh_verts)
    d_p2m, _ = tree_m.query(gt)

    def q(x):
        return float(np.quantile(x, 0.9))

    return {
        "mesh_to_point_mean": float(d_m2p.mean()),
        "mesh_to_point_median": float(np.median(d_m2p)),
        "mesh_to_point_p90": q(d_m2p),
        "point_to_mesh_mean": float(d_p2m.mean()),
        "point_to_mesh_median": float(np.median(d_p2m)),
        "point_to_mesh_p90": q(d_p2m),
        "coverage_ratio": float((d_p2m < max_dist).mean()),
    }


# ---------------------------------------------------------------------------
# Forward-view stability (evaluation-only)
# ---------------------------------------------------------------------------
@torch.no_grad()
def forward_view_stability(cams, depths, max_r=50.0):
    """Adjacent-view pairwise depth reprojection consistency (forward pass).

    For each consecutive pair of views, unproject view i's alpha-weighted depth
    and compare it against view i+1's depth at the reprojected pixels
    (bidirectional, same metric as ``depth_reproj_consistency``). A low value
    means the forward depth field is stable across nearby held-out views.

    Args:
        cams: list of cameras (aligned with ``depths``).
        depths: list of (H,W) alpha-weighted depth tensors.
        max_r: reject unprojected points farther than this.

    Returns:
        dict with ``mean`` (mean consistency error over pairs) and ``n_pairs``.
    """
    errs = []
    for i in range(len(cams) - 1):
        a, b = cams[i], cams[i + 1]
        da, db = depths[i], depths[i + 1]
        e = 0.5 * (depth_reproj_consistency(da, db, a, b, max_r=max_r).item()
                   + depth_reproj_consistency(db, da, b, a, max_r=max_r).item())
        errs.append(e)
    return {"mean": float(np.mean(errs)) if errs else float("nan"),
            "n_pairs": int(len(errs))}


@torch.no_grad()
def normal_consistency(cams, depths, max_r=50.0):
    """Adjacent-view normal-from-depth angular consistency (forward pass).

    For each consecutive pair of views, warp view i's depth-derived world
    normals into view i+1 and measure the mean cosine agreement against view
    i+1's own depth-derived normals. Both are oriented toward their own camera;
    on a consistent surface they agree in world space.

    Args:
        cams: list of cameras (aligned with ``depths``).
        depths: list of (H,W) alpha-weighted depth tensors.
        max_r: reject unprojected points farther than this.

    Returns:
        dict with ``mean_cos`` and ``n_pairs``.
    """
    cos_means = []
    for i in range(len(cams) - 1):
        a, b = cams[i], cams[i + 1]
        da, db = depths[i], depths[i + 1]
        na = normal_from_depth(da, a)                       # (3,H,W)
        nb = normal_from_depth(db, b)                       # (3,H,W)
        pts = unproject(da, a)                              # (H,W,3)
        u, v, z = _project(pts, b)
        H, W = da.shape
        gu = u / (W - 1) * 2.0 - 1.0
        gv = v / (H - 1) * 2.0 - 1.0
        grid = torch.stack([gu, gv], dim=-1).unsqueeze(0)   # (1,H,W,2)
        nb_w = F.grid_sample(nb.unsqueeze(0), grid, mode="bilinear",
                             padding_mode="zeros", align_corners=True)  # (1,3,H,W)
        nb_w = nb_w / (nb_w.norm(dim=1, keepdim=True) + 1e-6)
        cos = (na * nb_w).sum(dim=0)                        # (H,W)
        r = (pts - b.camera_center.to(da.device).float().view(1, 1, 3)).norm(dim=-1)
        inb = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        valid = (inb & (z > 0) & (r < max_r) & (da > 0)
                 & (na.norm(dim=0) > 0.5) & (nb_w.norm(dim=0) > 0.5))
        if valid.any():
            cos_means.append(float(cos[valid].mean().item()))
    return {"mean_cos": float(np.mean(cos_means)) if cos_means else float("nan"),
            "n_pairs": int(len(cos_means))}
