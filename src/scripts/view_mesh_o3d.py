#!/usr/bin/env python
"""Interactive native window to inspect the extracted mesh (open3d).

Pops up a real GUI window (independent of the browser) so you can orbit / zoom
the coloured or white mesh and check fineness + orientation.  A coordinate
frame is drawn at the mesh min-corner (RGB axes = X/Y/Z) to confirm direction.

Controls (open3d defaults):
    left-drag              orbit / rotate
    ctrl + left-drag       pan
    wheel / right-drag     zoom
    H                      print the full shortcut list (lighting, normals, ...)

Usage:
  python scripts/view_mesh_o3d.py AT1 [--white] [--sample N] [--no-axes]
"""
import os
import sys

import numpy as np
import open3d as o3d

REPO = "/home/cxh/APEX/apex-mesh"
MODEL_ROOT = os.path.join(REPO, "outputs", "moud")


def load_mesh(mesh_dir):
    for name in ("colored_mesh_pruned.npz", "colored_mesh.npz", "refined_mesh.npz"):
        path = os.path.join(mesh_dir, name)
        if os.path.exists(path):
            break
    m = np.load(path)
    verts = m["verts"].astype(np.float64)
    faces = m["faces"].astype(np.int32)
    color = (m["color"].astype(np.float64) if "color" in m.files
             else np.full((verts.shape[0], 3), 200.0))
    normals = (m["normals"].astype(np.float64)
               if "normals" in m.files else None)
    return verts, faces, color, normals


def main():
    args = sys.argv[1:]
    white = "--white" in args
    axes = "--no-axes" not in args
    sample = 1
    scenes = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--sample":
            sample = int(args[i + 1]); i += 2
        elif a in ("--white", "--no-axes") or a.startswith("--"):
            i += 1
        else:
            scenes.append(a); i += 1
    scenes = scenes or ["AT1"]

    for s in scenes:
        mesh_dir = os.path.join(MODEL_ROOT, s, "mesh")
        verts, faces, color, normals = load_mesh(mesh_dir)

        if sample > 1:
            faces = faces[::sample]
            print("  subsampled faces 1/%d -> %d" % (sample, faces.shape[0]))

        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(verts)
        m.triangles = o3d.utility.Vector3iVector(faces)
        if white:
            m.vertex_colors = o3d.utility.Vector3dVector(
                np.full((verts.shape[0], 3), 0.85))
        else:
            m.vertex_colors = o3d.utility.Vector3dVector(color / 255.0)
        if normals is not None:
            m.vertex_normals = o3d.utility.Vector3dVector(normals)
        else:
            m.compute_vertex_normals()

        print("%s: verts=%d faces=%d  %s" %
              (s, verts.shape[0], faces.shape[0],
               "white" if white else "colour"))

        geoms = [m]
        if axes:
            lo = verts.min(axis=0)
            size = float(np.ptp(verts, axis=0).max()) * 0.15
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size, lo)
            geoms.append(frame)

        o3d.visualization.draw_geometries(
            geoms,
            window_name="APEX-Mesh %s (%d verts, %d faces)"
                        % (s, verts.shape[0], faces.shape[0]),
            width=1400, height=1000, mesh_show_back_face=True)


if __name__ == "__main__":
    main()
