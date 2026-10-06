"""Anchored inverse-depth projective geometry.

Re-parameterises the free Gaussian position ``xyz`` as

    x_i = a_k + u_i * e1_k + v_i * e2_k + d_i * n_k
    d_i = d0_k * (1 / w_i - 1)
    w_i = eps + (1 - eps) * sigmoid(w_raw_i)        in [eps, 1)

where the *gauge* (anchor ``a_k``, depth scale ``d0_k``, orthonormal frame
``[e1_k, e2_k, n_k]`` and the per-Gaussian block assignment) is **fixed** and
never optimised. Only ``w_raw`` (inverse depth) and ``uv`` (tangential offset)
carry gradients, so the optimiser moves Gaussians along a bounded, well-posed
projective manifold instead of a free 3D coordinate.

The frame is aligned with the viewing direction: ``n_k`` points from the block
toward the mean camera centre. The anchor is placed *behind* the block (on the
far side along ``n_k``) so that every ``d_i = n_k . (x_i - a_k) > 0``, which is
what makes ``inverse`` an exact inverse of ``forward`` and keeps ``w`` strictly
inside ``(eps, 1)``.
"""

import torch
import torch.nn as nn


def _kmeans(xyz, K, iters=20):
    """GPU Lloyd k-means on the given points (no sklearn dependency)."""
    xyz = xyz.detach().float()
    N = xyz.shape[0]
    K = min(K, N)
    idx = torch.randperm(N, device=xyz.device)[:K]
    centroids = xyz[idx].clone()
    assign = torch.zeros(N, dtype=torch.long, device=xyz.device)
    for _ in range(iters):
        dist = torch.cdist(xyz, centroids)          # (N, K)
        assign = dist.argmin(dim=1)
        for k in range(K):
            m = assign == k
            if m.any():
                centroids[k] = xyz[m].mean(dim=0)
            else:
                centroids[k] = xyz[torch.randint(0, N, (1,), device=xyz.device)]
    return assign, centroids


def _orthonormal_frame(n):
    """Gram-Schmidt orthogonal complement of the (already unit) vectors ``n``.

    Args:
        n: (K, 3) unit vectors.
    Returns:
        e1, e2: (K, 3) unit vectors spanning the plane orthogonal to ``n``.
    """
    ref = torch.where(
        n[:, 0:1].abs() < 0.9,
        torch.tensor([1.0, 0.0, 0.0], device=n.device).expand_as(n),
        torch.tensor([0.0, 1.0, 0.0], device=n.device).expand_as(n),
    )
    e1 = ref - (ref * n).sum(dim=1, keepdim=True) * n
    e1 = e1 / (e1.norm(dim=1, keepdim=True) + 1e-8)
    e2 = torch.cross(n, e1, dim=1)
    e2 = e2 / (e2.norm(dim=1, keepdim=True) + 1e-8)
    return e1, e2


def build_blocks(xyz, K, mean_cam_center, eps=1e-3):
    """Cluster ``xyz`` into K local blocks and build the fixed projective gauge.

    Args:
        xyz: (N, 3) positions (world space).
        K: number of blocks (clamped to N).
        mean_cam_center: (3,) mean camera centre (world space).
        eps: inverse-depth lower bound.

    Returns:
        anchor: (K, 3) block anchors (placed on the far side of each block).
        d0: (K,) per-block depth scale.
        basis: (K, 3, 3) orthonormal frame, columns = [e1, e2, n].
        block_id: (N,) long tensor assigning each Gaussian to a block.
    """
    xyz = xyz.detach().float()
    N = xyz.shape[0]
    K = min(K, N)
    assign, centroids = _kmeans(xyz, K)

    mc = mean_cam_center.detach().float().reshape(1, 3)

    # n_k points from the block toward the camera (viewing direction).
    n_raw = mc - centroids
    n = n_raw / (n_raw.norm(dim=1, keepdim=True) + 1e-8)
    e1, e2 = _orthonormal_frame(n)

    block_id = assign
    c_pt = centroids[block_id]                      # (N, 3)
    n_pt = n[block_id]                              # (N, 3)
    s = ((xyz - c_pt) * n_pt).sum(dim=1)            # signed depth rel. centroid

    s_min = torch.zeros(K, device=xyz.device)
    s_max = torch.zeros(K, device=xyz.device)
    for k in range(K):
        m = block_id == k
        if m.any():
            s_min[k] = s[m].min()
            s_max[k] = s[m].max()

    spread = (s_max - s_min).clamp_min(1e-4)
    margin = 0.05 * spread                          # 5% of block depth
    d0 = spread + margin

    # Push the anchor to the far side so every d_i = n.(x - a) >= margin > 0.
    anchor = centroids + (s_min - margin).unsqueeze(-1) * n

    basis = torch.stack([e1, e2, n], dim=-1)        # (K, 3, 3), columns e1/e2/n
    return anchor, d0, basis, block_id


class AnchoredProjectiveGeometry(nn.Module):
    """Fixed projective gauge mapping ``(w_raw, uv) -> xyz`` (and its inverse)."""

    def __init__(self, anchor, d0, basis, block_id, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.register_buffer("anchor", anchor.float())
        self.register_buffer("d0", d0.float())
        self.register_buffer("basis", basis.float())
        self.register_buffer("block_id", block_id.long())

    def w_from_raw(self, w_raw):
        return self.eps + (1.0 - self.eps) * torch.sigmoid(w_raw)

    def forward(self, w_raw, uv):
        # w_raw: (N, 1), uv: (N, 2) -> xyz: (N, 3)
        w = self.w_from_raw(w_raw).squeeze(-1)       # (N,)
        d = self.d0[self.block_id] * (1.0 / w - 1.0)
        e1 = self.basis[self.block_id][:, :, 0]
        e2 = self.basis[self.block_id][:, :, 1]
        n = self.basis[self.block_id][:, :, 2]
        a = self.anchor[self.block_id]
        x = a + uv[:, 0:1] * e1 + uv[:, 1:2] * e2 + d.unsqueeze(-1) * n
        return x

    def inverse(self, xyz):
        # xyz: (N, 3) -> (w_raw: (N,1), uv: (N,2))
        a = self.anchor[self.block_id]
        e1 = self.basis[self.block_id][:, :, 0]
        e2 = self.basis[self.block_id][:, :, 1]
        n = self.basis[self.block_id][:, :, 2]
        r = xyz - a
        d = (r * n).sum(dim=1)
        u = (r * e1).sum(dim=1)
        v = (r * e2).sum(dim=1)
        d0 = self.d0[self.block_id]
        w = d0 / (d0 + d)
        w = torch.clamp(w, self.eps, 1.0 - self.eps)
        # Invert w = eps + (1-eps)*sigmoid(w_raw): w_raw = log((w-eps)/(1-w)).
        w_raw = torch.log((w - self.eps) / (1.0 - w))
        uv = torch.stack([u, v], dim=-1)
        return w_raw.unsqueeze(-1), uv


def enable(xyz, K, mean_cam_center, eps=1e-3):
    """Build the projective gauge for ``xyz`` and return initial parameters.

    Returns:
        w_raw: (N, 1) initial inverse-depth parameters (exact round-trip).
        uv: (N, 2) initial tangential offsets (exact round-trip).
        geo: AnchoredProjectiveGeometry holding the fixed gauge.
    """
    anchor, d0, basis, block_id = build_blocks(xyz, K, mean_cam_center, eps=eps)
    geo = AnchoredProjectiveGeometry(anchor, d0, basis, block_id, eps=eps)
    w_raw, uv = geo.inverse(xyz.detach().float())
    return w_raw, uv, geo
