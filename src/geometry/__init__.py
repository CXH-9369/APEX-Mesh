"""Projective geometry: anchored inverse-depth reparameterisation and
differentiable multi-view geometry losses."""

from .projective import (
    AnchoredProjectiveGeometry,
    build_blocks,
    enable,
)
from .losses import (
    depth_reproj_consistency,
    edge_aware_depth_smooth,
    ray_dirs,
    unproject,
)

__all__ = [
    "AnchoredProjectiveGeometry",
    "build_blocks",
    "enable",
    "depth_reproj_consistency",
    "edge_aware_depth_smooth",
    "ray_dirs",
    "unproject",
]
