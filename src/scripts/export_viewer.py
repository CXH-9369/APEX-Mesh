#!/usr/bin/env python
"""Copy the interactive mesh viewer + vendored three.js into <model>/mesh/, and
write a fast-loading ``viewer.bin`` so the browser does not parse the 380 MB
ASCII-header PLY on the main thread (that stalls for tens of seconds and shows a
black canvas).

``viewer.bin`` layout (all little-endian)::

    magic     "APXM"          4 bytes
    nv        uint32          4
    nf        uint32          4
    positions float32[nv*3]   x, y, z
    normals   float32[nv*3]
    colors    uint8  [nv*3]   r, g, b  (padded to 4-byte alignment)
    faces     uint32 [nf*3]

It is generated straight from ``colored_mesh.npz`` (already baked by
``bake_vertex_color.py``), so it does not re-run the GPU bake.

To use::

    cd outputs/moud/<scene>/mesh
    python -m http.server 8000
    # open http://localhost:8000/viewer.html

or just open viewer.html and drag ``colored_mesh.ply`` onto the page.

Usage:
  python scripts/export_viewer.py AT1 [AT2 ...]
"""
import os
import sys
import shutil
import struct

import numpy as np

REPO = "/home/cxh/APEX/apex-mesh"
MODEL_ROOT = os.path.join(REPO, "outputs", "moud")
HERE = os.path.dirname(os.path.abspath(__file__))
MAGIC = b"APXM"


def write_viewer_bin(mesh_dir):
    """Write viewer.bin from the baked coloured mesh (no GPU work)."""
    colored = os.path.join(mesh_dir, "colored_mesh.npz")
    path = colored if os.path.exists(colored) else os.path.join(mesh_dir, "refined_mesh.npz")
    if not os.path.exists(path):
        print("  ! no colored_mesh.npz / refined_mesh.npz in %s — skipping viewer.bin" % mesh_dir)
        return
    m = np.load(path)
    verts = np.asarray(m["verts"], dtype="<f4").reshape(-1, 3)
    faces = np.asarray(m["faces"], dtype="<u4").reshape(-1, 3)
    nv, nf = verts.shape[0], faces.shape[0]
    if "color" in m.files:
        color = np.asarray(m["color"], dtype="u1").reshape(-1, 3)
    else:
        color = np.full((nv, 3), 200, dtype="u1")
    if "normals" in m.files:
        normals = np.asarray(m["normals"], dtype="<f4").reshape(-1, 3)
    else:
        normals = np.zeros((nv, 3), dtype="<f4")

    pos = verts.tobytes()
    nor = normals.tobytes()
    col = color.tobytes()
    col_pad = b"\x00" * ((-len(col)) % 4)
    fac = faces.tobytes()
    header = struct.pack("<4sII", MAGIC, nv, nf)

    out = os.path.join(mesh_dir, "viewer.bin")
    with open(out, "wb") as f:
        f.write(header)
        f.write(pos)
        f.write(nor)
        f.write(col)
        f.write(col_pad)
        f.write(fac)
    print("  viewer.bin: %d verts, %d faces, %.1f MB -> %s"
          % (nv, nf, os.path.getsize(out) / 1e6, out))


def main():
    scenes = sys.argv[1:] if len(sys.argv) > 1 else ["AT1"]
    for s in scenes:
        mesh_dir = os.path.join(MODEL_ROOT, s, "mesh")
        os.makedirs(mesh_dir, exist_ok=True)
        shutil.copy(os.path.join(HERE, "viewer.html"), os.path.join(mesh_dir, "viewer.html"))
        shutil.copy(os.path.join(HERE, "three.min.js"), os.path.join(mesh_dir, "three.min.js"))
        write_viewer_bin(mesh_dir)
        print("viewer + three.js + viewer.bin ->", mesh_dir)


if __name__ == "__main__":
    main()
