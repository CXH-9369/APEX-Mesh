"""Multi-view truncated signed-distance fusion (offline, 2DGS-style extraction).

Instead of iso-surfacing the raw Gaussian *density* field (which is a heuristic
that drifts from the rendered surface — the source of the overlay misalignment),
fuse the per-view median ``surface_depth`` maps into a TSDF.  Because the mesh
is the zero level set of the fused rendered depth, it is the *rendered surface*
by construction, so reprojection against the images is consistent.

Each voxel is projected into every view; the signed distance is the optical-path
difference ``surface_depth(pixel) - ||x - camera_centre||`` (positive in front
of the surface, negative behind), truncated to ``trunc`` and averaged with a
binary in-view/near-surface weight.  The surface is the zero crossing.
"""

import torch

from mesh.extract import project_points, _sample_at


def fuse_tsdf(cams, surface_depths, origin, spacing, shape, trunc,
              max_r=50.0, chunk=2000000, accum_maps=None, a_thresh=0.5):
    """Fuse ``surface_depths`` (one (H,W) per cam) into a TSDF voxel grid.

    Args:
        cams: camera objects aligned with ``surface_depths``.
        surface_depths: list of (H,W) median surface depth maps.
        origin, spacing, shape: voxel grid (axis 0=x, 1=y, 2=z).
        trunc: truncation distance (world units); set to ~3 * spacing.
        max_r: reject depth samples farther than this.
        accum_maps: optional list of (H,W) accumulated-opacity maps; pixels below
            ``a_thresh`` are treated as unreliable (floaters / holes) and skipped.
        a_thresh: accumulated-opacity threshold (only used with ``accum_maps``).

    Returns:
        tsdf: (nx, ny, nz) float32, negative behind / positive in front.
        weight: (nx, ny, nz) float32 accumulated sample count.
    """
    device = surface_depths[0].device
    origin = torch.as_tensor(origin, dtype=torch.float32, device=device)
    spacing = torch.as_tensor(spacing, dtype=torch.float32, device=device)
    nx, ny, nz = int(shape[0]), int(shape[1]), int(shape[2])
    n = nx * ny * nz
    use_accum = accum_maps is not None

    tsdf = torch.zeros(n, device=device)
    wsum = torch.zeros(n, device=device)

    for st in range(0, n, chunk):
        en = min(st + chunk, n)
        idx = torch.arange(st, en, device=device)
        iz = idx % nz
        iy = torch.div(idx, nz, rounding_mode="trunc") % ny
        ix = torch.div(idx, ny * nz, rounding_mode="trunc")
        centers = torch.stack([ix, iy, iz], dim=1).float() * spacing + origin  # (C,3)

        acc_s = torch.zeros(en - st, device=device)
        acc_w = torch.zeros(en - st, device=device)
        for ci, (cam, sd) in enumerate(zip(cams, surface_depths)):
            u, v, _ = project_points(centers, cam)
            d = _sample_at(sd, u, v)                                  # (C,)
            r = (centers - cam.camera_center.to(device).float().view(1, 3)).norm(dim=-1)
            H, W = sd.shape
            inb = ((u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1) &
                   (d > 0) & (d < max_r))
            if use_accum:
                a = _sample_at(accum_maps[ci], u, v)
                inb = inb & (a >= a_thresh)
            sdf = d - r                                              # + in front, - behind
            w = inb.float() * (sdf.abs() < trunc).float()
            acc_s = acc_s + w * sdf.clamp(-trunc, trunc)
            acc_w = acc_w + w
        tsdf[st:en] = acc_s
        wsum[st:en] = acc_w

    tsdf = tsdf / wsum.clamp_min(1.0)
    return tsdf.view(nx, ny, nz), wsum.view(nx, ny, nz)
