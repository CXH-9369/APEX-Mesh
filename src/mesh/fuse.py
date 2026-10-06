"""Offline multi-view surface fusion + silhouette support.

The reference mesh is built from a *single* reference view's median
surface depth, which is a per-view soft statistic (this is the root cause of
the 46.8% held-out reprojection error). ``fuse_multiview`` replaces each
reference vertex with a robust average of surface samples lifted from every
agreeing neighbour view, reducing the single-view bias before any refinement.

``silhouette_support`` scores each boundary vertex by the fraction of neighbour
views where its reprojection lands on half-Gaussian boundary response (E_v),
and feeds ``refine.remove_small_components`` to avoid dropping real surface
patches that merely happen to be small.
"""

import torch
import torch.nn.functional as F

from geometry.losses import ray_dirs
from mesh.extract import project_points, _sample_at


def _sample_field3(field, u, v):
    """Sample a ``(H, W, 3)`` field at ``(u, v)`` -> ``(M, 3)`` via grid_sample."""
    H, W = field.shape[0], field.shape[1]
    gu = u / (W - 1) * 2.0 - 1.0
    gv = v / (H - 1) * 2.0 - 1.0
    grid = torch.stack([gu, gv], dim=-1).unsqueeze(0).unsqueeze(0)   # (1,1,M,2)
    d = field.permute(2, 0, 1).unsqueeze(0)                          # (1,3,H,W)
    out = F.grid_sample(d, grid, mode="bilinear", padding_mode="zeros",
                        align_corners=True)                          # (1,3,1,M)
    return out.squeeze(0).squeeze(1).permute(1, 0)                   # (M,3)


def fuse_multiview(mesh_ref, ref_cam, cams, surfaces, tol=0.05, max_r=50.0,
                   a_thresh=0.7, robust="median"):
    """Multi-view surface fusion of a reference-view mesh.

    Args:
        mesh_ref: dict from ``build_reference_mesh`` (needs ``verts``/``normals``).
        ref_cam: the reference camera (skipped).
        cams: list of neighbour cameras.
        surfaces: list of surface-channel dicts aligned with ``cams`` (each with
                  ``surface_depth``, ``accum_opacity``, ``normal``).
        tol: relative depth agreement threshold for a view to support a vertex.
        max_r: reject vertices farther than this.
        a_thresh: accumulated-opacity threshold for a neighbour sample to count.
        robust: "median" (coordinate-wise median, outlier-robust) or "mean"
                (weighted average).

    Returns:
        verts: (M,3) float32 fused world positions (cuda).
        normals: (M,3) float32 fused world normals (cuda, unit).
        support_count: (M,) int32 # agreeing neighbour views.
        c_view: (M,) float32 mean (1 - rel) over agreeing views, in [0,1].
    """
    verts_ref = torch.from_numpy(mesh_ref["verts"].astype("float32")).cuda()
    norms_ref = torch.from_numpy(mesh_ref["normals"].astype("float32")).cuda()
    M = verts_ref.shape[0]
    device = verts_ref.device

    # Per-vertex candidate lists: (3D pos, weight) and (normal, weight).
    candidates = [[] for _ in range(M)]
    norm_candidates = [[] for _ in range(M)]
    support = torch.zeros(M, dtype=torch.int32, device=device)
    c_view = torch.zeros(M, dtype=torch.float32, device=device)

    n = 0
    for cam, surf in zip(cams, surfaces):
        if cam is ref_cam:
            continue
        n += 1
        sd = surf["surface_depth"]
        A = surf["accum_opacity"]
        normal = surf["normal"]                       # (3,H,W)
        u, v, z = project_points(verts_ref, cam)
        warped = _sample_at(sd, u, v)
        expected = (verts_ref - cam.camera_center.to(device).float().view(1, 3)).norm(dim=-1)
        rel = (warped - expected).abs() / (expected + 1e-3)
        A_warped = _sample_at(A, u, v)
        inb = (u >= 0) & (u < cam.image_width - 1) & (v >= 0) & (v < cam.image_height - 1)
        agree = (z > 0) & (expected < max_r) & inb & (warped > 0) & (A_warped >= a_thresh) & (rel < tol)
        support += agree.to(torch.int32)
        c_view += (1.0 - rel.clamp(0.0, 1.0)) * agree.to(torch.float32)

        aidx = agree.nonzero(as_tuple=False).view(-1)
        if aidx.numel() == 0:
            continue
        # Lift each agreeing neighbour surface sample back to 3D along its ray.
        dirs = _sample_field3(ray_dirs(cam, device=device), u[aidx], v[aidx])  # (K,3)
        cc = cam.camera_center.to(device).float().view(1, 3)
        p_k = cc + warped[aidx].unsqueeze(1) * dirs                          # (K,3)
        w_k = 1.0 - rel[aidx].clamp(0.0, 1.0)                                # (K,)
        nrm = normal.permute(1, 2, 0)                                        # (H,W,3)
        n_k = _sample_field3(nrm, u[aidx], v[aidx])                          # (K,3)
        n_k = n_k / (n_k.norm(dim=-1, keepdim=True) + 1e-6)
        for j, idx in enumerate(aidx.tolist()):
            candidates[idx].append((p_k[j], float(w_k[j])))
            norm_candidates[idx].append((n_k[j], float(w_k[j])))

    verts_out = verts_ref.clone()
    norms_out = norms_ref.clone()
    active = [i for i in range(M) if candidates[i]]
    for i in active:
        poss = [verts_ref[i]] + [p for p, _ in candidates[i]]
        wts = [1.0] + [w for _, w in candidates[i]]
        if robust == "median":
            fused = torch.stack(poss).median(dim=0).values
        else:
            w = torch.tensor(wts, device=device, dtype=torch.float32)
            w = w / w.sum()
            fused = (torch.stack(poss) * w.unsqueeze(1)).sum(dim=0)
        verts_out[i] = fused

        nposs = [norms_ref[i]] + [nn for nn, _ in norm_candidates[i]]
        if robust == "median":
            fn = torch.stack(nposs).median(dim=0).values
        else:
            w = torch.tensor(wts, device=device, dtype=torch.float32)
            w = w / w.sum()
            fn = (torch.stack(nposs) * w.unsqueeze(1)).sum(dim=0)
        norms_out[i] = fn / (fn.norm() + 1e-6)

    if n > 0:
        c_view = c_view / n
    return verts_out.float(), norms_out.float(), support, c_view


def silhouette_support(boundary_verts, cams, edge_maps, accum_maps, edge_thresh=0.5):
    """Per-boundary-vertex silhouette agreement across neighbour views.

    Args:
        boundary_verts: (M,3) torch (cuda) boundary vertices.
        cams / edge_maps / accum_maps: aligned neighbour views (``edge`` and
                                       ``accum_opacity`` channels).
        edge_thresh: E_v threshold for a pixel to count as silhouette evidence.

    Returns:
        (M,) float32 in [0,1]: fraction of in-view neighbours where the vertex
        lands on E_v >= edge_thresh (over well-occluded pixels only).
    """
    M = boundary_verts.shape[0]
    device = boundary_verts.device
    hit = torch.zeros(M, dtype=torch.float32, device=device)
    denom = torch.zeros(M, dtype=torch.float32, device=device)
    for cam, edge_map, A in zip(cams, edge_maps, accum_maps):
        u, v, z = project_points(boundary_verts, cam)
        inb = (u >= 0) & (u < cam.image_width - 1) & (v >= 0) & (v < cam.image_height - 1) & (z > 0)
        e_s = _sample_at(edge_map, u, v)
        a_s = _sample_at(A, u, v)
        valid = inb & (a_s >= 0.5)
        hit += ((e_s >= edge_thresh) & valid).float()
        denom += valid.float()
    return hit / denom.clamp_min(1.0)
