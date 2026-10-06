"""Refine an extracted observable-surface mesh.

Loads a mesh NPZ, removes degenerate/flipped faces, applies edge-aware
(open-boundary-preserving) Laplacian smoothing, and writes the refined mesh.
Reports mesh quality statistics before/after.
"""

import os
import sys
import json
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from arguments import ModelParams, get_combined_args
from utils.general_utils import safe_state
from mesh import (load_npz, save_npz, write_ply, write_obj,
                  mesh_stats, cleanup, laplacian_smooth, edge_weighted_laplacian,
                  remove_small_components, boundary_edges)


def refine(dataset, args):
    src = os.path.join(dataset.model_path, "mesh", "reference_mesh.npz")
    mesh = load_npz(src)
    before = mesh_stats(mesh)
    print("before:", json.dumps(before))

    verts, normals, faces = cleanup(
        mesh["verts"], mesh["faces"], mesh["normals"],
        orient_thresh=args.orient_thresh)

    # phase 3: drop small spurious components (retain thin real patches by
    # silhouette support when available).
    work = dict(mesh)
    work["verts"], work["normals"], work["faces"] = verts, normals, faces
    work = remove_small_components(
        work, min_faces=args.min_component_faces,
        silhouette=work.get("silhouette"), silhouette_thresh=args.silhouette_thresh)
    verts, normals, faces = work["verts"], work["normals"], work["faces"]

    b = boundary_edges(faces)
    if args.uniform:
        verts = laplacian_smooth(verts, faces, boundary_edges=b,
                                 lam=args.lam, iters=args.iters)
    else:
        verts = edge_weighted_laplacian(
            verts, faces, boundary_edges=b, edge=work.get("edge"),
            gamma=args.gamma, lam=args.lam, iters=args.iters,
            vert_confidence=work.get("confidence"))
    # Re-normalise normals after smoothing (vertex-normal estimate from faces).
    normals = _vertex_normals(verts, faces)
    b = boundary_edges(faces)

    out = dict(work)
    out["verts"] = verts
    out["normals"] = normals
    out["faces"] = faces
    out["boundary_edges"] = b
    # confidence/support/visible/silhouette unchanged (already remapped by
    # remove_small_components).

    after = mesh_stats(out)
    print("after :", json.dumps(after))

    root = os.path.join(dataset.model_path, "mesh")
    write_ply(os.path.join(root, "refined_mesh.ply"), out)
    write_obj(os.path.join(root, "refined_mesh.obj"), out)
    save_npz(os.path.join(root, "refined_mesh.npz"), out)
    with open(os.path.join(root, "refine_stats.json"), "w") as f:
        json.dump({"before": before, "after": after}, f, indent=2)
    print("Saved refined mesh ->", root)


def _vertex_normals(verts, faces):
    normals = np.zeros_like(verts)
    v = verts[faces]
    fn = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    for k in range(3):
        np.add.at(normals, faces[:, k], fn)
    n = np.linalg.norm(normals, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return (normals / n).astype(np.float32)


if __name__ == "__main__":
    parser = ArgumentParser(description="Mesh refinement")
    lp = ModelParams(parser, sentinel=True)
    parser.add_argument("--lam", type=float, default=0.3)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=5.0,
                        help="edge falloff for the edge-weighted Laplacian")
    parser.add_argument("--orient_thresh", type=float, default=0.0,
                        help="orientation cosine cutoff for flipped-face pruning; "
                             "lenient (e.g. -0.5) for self-overlapping meshes")
    parser.add_argument("--min_component_faces", type=int, default=50,
                        help="drop connected components smaller than this (0 = keep all)")
    parser.add_argument("--silhouette_thresh", type=float, default=0.2,
                        help="keep a small component if its mean silhouette support >= this")
    parser.add_argument("--uniform", action="store_true",
                        help="use the uniform Laplacian instead of the edge-weighted default")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)
    refine(lp.extract(args), args)
