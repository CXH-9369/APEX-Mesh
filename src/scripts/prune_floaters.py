#!/usr/bin/env python
"""Remove floating fragments from a coloured mesh (non-destructive).

Connected components are found over the face-adjacency graph and every
component below a size threshold is dropped (by default only the single
*largest* component survives).  This strips the sparse shards / thin sheets
that the implicit-MLS extractor leaves floating above and beside the reef,
which in the interactive viewer show up as the "lines" around the scene.

Output is written to NEW files -- the original ``colored_mesh.npz`` /
``colored_mesh.ply`` are left untouched:

    colored_mesh_pruned.npz / .ply   pruned mesh
    viewer.bin                       regenerated from the pruned mesh

Usage:
  python scripts/prune_floaters.py AT1 [AT2 ...] [--keep_top 1] [--min_verts 1000]
"""
import os
import sys
import struct

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

REPO = "/home/cxh/APEX/apex-mesh"
MODEL_ROOT = os.path.join(REPO, "outputs", "moud")
MAGIC = b"APXM"


def _components(faces, nv):
    """Labels (nv,) for connected components of the triangle mesh."""
    f = faces.astype(np.int64)
    i, j, k = f[:, 0], f[:, 1], f[:, 2]
    e = np.concatenate([
        np.stack([np.minimum(i, j), np.maximum(i, j)], 1),
        np.stack([np.minimum(j, k), np.maximum(j, k)], 1),
        np.stack([np.minimum(i, k), np.maximum(i, k)], 1),
    ], axis=0).astype(np.int32)
    g = coo_matrix((np.ones(len(e), dtype=np.int8), (e[:, 0], e[:, 1])),
                   shape=(nv, nv)).tocsr()
    g.data[:] = 1
    ncomp, labels = connected_components(g, directed=False)
    return labels, ncomp


def _write_ply(path, verts, normals, faces, colors):
    verts = verts.astype(np.float32)
    normals = normals.astype(np.float32)
    colors = colors.astype(np.uint8)
    nv, nf = verts.shape[0], faces.shape[0]
    header = ("ply\nformat binary_little_endian 1.0\n"
              "comment pruned largest-component coloured mesh\n"
              "element vertex %d\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property float nx\nproperty float ny\nproperty float nz\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "element face %d\n"
              "property list uchar int vertex_indices\nend_header\n" % (nv, nf))
    v = np.zeros(nv, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                            ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                            ("r", "u1"), ("g", "u1"), ("b", "u1")])
    v["x"], v["y"], v["z"] = verts[:, 0], verts[:, 1], verts[:, 2]
    v["nx"], v["ny"], v["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    v["r"], v["g"], v["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
    fa = np.zeros(nf, dtype=[("n", "u1"), ("idx", "<i4", (3,))])
    fa["n"] = 3
    fa["idx"] = faces.astype(np.int32)
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(v.tobytes())
        fh.write(fa.tobytes())


def _write_viewer_bin(path, verts, normals, colors, faces):
    pos = verts.astype("<f4").reshape(-1, 3).tobytes()
    nor = normals.astype("<f4").reshape(-1, 3).tobytes()
    col = colors.astype("u1").reshape(-1, 3).tobytes()
    col_pad = b"\x00" * ((-len(col)) % 4)
    fac = faces.astype("<u4").reshape(-1, 3).tobytes()
    nv, nf = verts.shape[0], faces.shape[0]
    header = struct.pack("<4sII", MAGIC, nv, nf)
    with open(path, "wb") as f:
        f.write(header); f.write(pos); f.write(nor); f.write(col)
        f.write(col_pad); f.write(fac)


def prune(mesh_dir, keep_top=1, min_verts=0):
    src = os.path.join(mesh_dir, "colored_mesh.npz")
    if not os.path.exists(src):
        src = os.path.join(mesh_dir, "refined_mesh.npz")
    m = np.load(src)
    verts = m["verts"]
    faces = m["faces"]
    nv = verts.shape[0]

    labels, ncomp = _components(faces, nv)
    cnt = np.bincount(labels)
    order = np.argsort(-cnt)

    if min_verts > 0:
        keep_comp = set(order[cnt[order] >= min_verts].tolist())
        keep_comp.update(order[:max(1, keep_top)].tolist())
    else:
        keep_comp = set(order[:max(1, keep_top)].tolist())

    kept = np.zeros(nv, dtype=bool)
    for c in keep_comp:
        kept |= (labels == c)

    remap = np.full(nv, -1, dtype=np.int64)
    keep_idx = np.nonzero(kept)[0]
    remap[keep_idx] = np.arange(keep_idx.shape[0], dtype=np.int64)

    fkept = kept[faces].all(axis=1)
    faces_new = remap[faces[fkept]].astype(np.int64)

    out = {}
    for key in m.files:
        a = m[key]
        if key == "faces":
            out[key] = faces_new
        elif key == "boundary_edges":
            ok = kept[a].all(axis=1)
            out[key] = remap[a[ok]].astype(np.int64)
        elif a.shape[0] == nv:
            out[key] = a[keep_idx]
        else:
            out[key] = a
    out["verts"] = verts[keep_idx]
    out["faces"] = faces_new

    print("components: %d -> keeping %d (%s), verts %d -> %d, faces %d -> %d"
          % (ncomp, len(keep_comp), sorted(keep_comp)[:8],
             nv, keep_idx.shape[0], faces.shape[0], faces_new.shape[0]))

    color = out.get("color")
    if color is None:
        color = np.full((keep_idx.shape[0], 3), 200, dtype=np.uint8)
    normals = out.get("normals")

    np.savez_compressed(os.path.join(mesh_dir, "colored_mesh_pruned.npz"), **out)
    _write_ply(os.path.join(mesh_dir, "colored_mesh_pruned.ply"),
               out["verts"], normals, faces_new, color)
    _write_viewer_bin(os.path.join(mesh_dir, "viewer.bin"),
                      out["verts"], normals, color, faces_new)
    print("wrote colored_mesh_pruned.npz/.ply + viewer.bin (%.1f MB)"
          % (os.path.getsize(os.path.join(mesh_dir, "viewer.bin")) / 1e6))


def main():
    args = sys.argv[1:]
    keep_top, min_verts = 1, 0
    scenes = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--keep_top":
            keep_top = int(args[i + 1]); i += 2
        elif a == "--min_verts":
            min_verts = int(args[i + 1]); i += 2
        else:
            scenes.append(a); i += 1
    if not scenes:
        scenes = ["AT1"]
    for s in scenes:
        mesh_dir = os.path.join(MODEL_ROOT, s, "mesh")
        print("=== prune_floaters %s (keep_top=%d min_verts=%d)" % (s, keep_top, min_verts))
        prune(mesh_dir, keep_top, min_verts)
    print("done")


if __name__ == "__main__":
    main()
