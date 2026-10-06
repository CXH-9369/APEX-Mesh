"""Surface-alignment regularisers (offline flattening stage).

These losses reshape *already-trained* Gaussians into thin, surface-aligned
discs so that the density field

    d(x) = sum_i  alpha_i * exp(-0.5 * ||R_i^T (x - mu_i) / s_i||^2)

collapses into a clean thin sheet whose iso-surface hugs the observed surface.
Only the Gaussian shapes (scale / rotation / opacity) are optimised by the
caller; positions and colour stay frozen, so this never disturbs the
projective geometry.

Two terms define the surface alignment:

* ``flatten_loss`` — minimise each Gaussian's *smallest* scale (the axis
  perpendicular to the surface).  The photometric loss keeps the in-plane
  scales alive, so the result is a flat disc, not a point.
* ``normal_alignment_loss`` — pull the Gaussian's thinnest axis (its normal)
  into agreement with the surface normal obtained from the alpha-weighted
  rendered depth.  The smallest-scale eigenvector is the normal because the
  covariance is ``Sigma = R diag(s^2) R^T`` (columns of ``R`` are the axes).
"""

import torch
import torch.nn.functional as F

from utils.general_utils import build_rotation
from geometry.losses import normal_from_depth
from mesh.extract import project_points


def flatten_loss(gaussians):
    """Mean smallest (perpendicular) scale — flatten every Gaussian into a disc."""
    scales = gaussians.get_scaling                       # (N, 3) exponentiated
    return scales.min(dim=1).values.mean()


def gaussian_normals(gaussians):
    """Per-Gaussian normal = the rotation axis of the *smallest* scale."""
    scales = gaussians.get_scaling                       # (N, 3)
    idx = scales.argmin(dim=1)                           # (N,)
    R = build_rotation(gaussians.get_rotation)           # (N, 3, 3)
    n = torch.arange(R.shape[0], device=scales.device)
    return R[n, :, idx]                                  # (N, 3)


def opacity_binary_loss(gaussians):
    """Nudge the half-Gaussian mean opacity toward 0 or 1 (crisper field)."""
    alpha = gaussians.get_opacity.mean(dim=1)            # (N,)
    return (alpha * (1.0 - alpha)).mean()


def _sample_vector(map3, u, v):
    """Bilinear-sample a (3, H, W) map at pixel coords ``(u, v)`` -> (3, N)."""
    C, H, W = map3.shape
    gu = u / (W - 1) * 2.0 - 1.0
    gv = v / (H - 1) * 2.0 - 1.0
    grid = torch.stack([gu, gv], dim=-1).unsqueeze(0).unsqueeze(0)   # (1, 1, N, 2)
    out = F.grid_sample(map3.unsqueeze(0), grid, mode="bilinear",
                        padding_mode="zeros", align_corners=True)     # (1, C, 1, N)
    return out.view(C, -1)                               # (3, N)


def normal_alignment_loss(gaussians, depth, cam, max_r=50.0):
    """Mean ``1 - |n_g . n_s|`` over in-view Gaussians.

    ``n_s`` is the world-space outward normal of the alpha-weighted rendered
    depth (computed on the fly from the *live* scales, so it tracks the
    flattening), and ``n_g`` is each Gaussian's thinnest axis.
    """
    n_s = normal_from_depth(depth, cam)                  # (3, H, W) world space
    means = gaussians.get_xyz                            # (N, 3)
    n_g = gaussian_normals(gaussians)                    # (N, 3)

    u, v, z = project_points(means, cam)                 # (N,)
    expected = (means - cam.camera_center.to(means.device).float().view(1, 3)).norm(dim=-1)
    inb = ((u >= 0) & (u < cam.image_width - 1) &
           (v >= 0) & (v < cam.image_height - 1) &
           (z > 0) & (expected < max_r))
    if not inb.any():
        return torch.tensor(0.0, device=means.device, dtype=means.dtype)

    n_samp = _sample_vector(n_s, u, v)                   # (3, N)
    cos = (n_g * n_samp.t()).sum(dim=-1)                 # (N,)
    return (1.0 - cos.abs())[inb].mean()
