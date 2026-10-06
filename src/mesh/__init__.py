"""Offline observable-surface mesh (extract / fuse / refine / evaluate / io)."""

from mesh.io import write_ply, write_obj, save_npz, load_npz, read_ply
from mesh.extract import (
    render_surface, surface_mask, build_reference_mesh, multiview_support,
    project_points, photometric_confidence, confidence,
)
from mesh.fuse import fuse_multiview, silhouette_support
from mesh.mtetra import (gaussian_density, splat_density_volume, marching_tetrahedra,
                         estimate_threshold, sample_trilinear, compute_normals)
from mesh.refine import (
    mesh_stats, cleanup, laplacian_smooth, edge_weighted_laplacian,
    remove_small_components, self_intersection_count, boundary_edges,
)
from mesh.evaluate import (
    reproject_consistency_heldout, boundary_edge_agreement, point_cloud_distance,
    forward_view_stability, normal_consistency, depth_boundary_metric,
)

__all__ = [
    "write_ply", "write_obj", "save_npz", "load_npz", "read_ply",
    "render_surface", "surface_mask", "build_reference_mesh", "multiview_support",
    "project_points", "photometric_confidence", "confidence",
    "fuse_multiview", "silhouette_support",
    "gaussian_density", "splat_density_volume", "marching_tetrahedra",
    "estimate_threshold", "sample_trilinear", "compute_normals",
    "mesh_stats", "cleanup", "laplacian_smooth", "edge_weighted_laplacian",
    "remove_small_components", "self_intersection_count", "boundary_edges",
    "reproject_consistency_heldout", "boundary_edge_agreement", "point_cloud_distance",
    "forward_view_stability", "normal_consistency", "depth_boundary_metric",
]
