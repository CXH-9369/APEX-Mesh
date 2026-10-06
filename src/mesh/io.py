"""Minimal PLY/OBJ/NPZ mesh I/O (no open3d/trimesh dependency).

The mesh is carried internally as a plain dict (``mesh.Mesh``):

    {
      "verts":        (N, 3) float32 world positions,
      "normals":      (N, 3) float32 unit surface normals,
      "faces":        (F, 3) int32 vertex indices,
      "confidence":   (N,)   float32 in [0, 1] (C_med),
      "support_count":(N,)   int32 number of supporting neighbour views,
      "visible":      (N,)   uint8 0/1 visible-domain mask,
      "boundary_edges":(E,2) int32 edges shared by a single face (open boundary),
    }

``write_ply`` / ``write_obj`` emit the geometric deliverable; ``save_npz`` /
``load_npz`` persist the full structure (including confidence / support /
boundary) for refine/evaluate round-trips.
"""

import os

import numpy as np


def _as_array(a, n, dtype):
    if a is None:
        return np.zeros((n,), dtype=dtype)
    a = np.asarray(a)
    if a.ndim == 2:
        a = a.reshape(-1)
    return a.astype(dtype, copy=False)


def write_ply(path, mesh):
    verts = np.asarray(mesh["verts"], dtype=np.float32)
    normals = np.asarray(mesh["normals"], dtype=np.float32)
    faces = np.asarray(mesh["faces"], dtype=np.int32)
    n = verts.shape[0]
    confidence = _as_array(mesh.get("confidence"), n, np.float32)
    support = _as_array(mesh.get("support_count"), n, np.int32)
    visible = _as_array(mesh.get("visible"), n, np.uint8)

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment APEX-Mesh observable-surface mesh\n")
        f.write("element vertex {}\n".format(n))
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write("property float confidence\n")
        f.write("property int support_count\n")
        f.write("property uchar visible\n")
        f.write("element face {}\n".format(faces.shape[0]))
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for i in range(n):
            f.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {} {}\n".format(
                verts[i, 0], verts[i, 1], verts[i, 2],
                normals[i, 0], normals[i, 1], normals[i, 2],
                confidence[i], int(support[i]), int(visible[i])))
        for tri in faces:
            f.write("3 {} {} {}\n".format(int(tri[0]), int(tri[1]), int(tri[2])))


def write_obj(path, mesh):
    verts = np.asarray(mesh["verts"], dtype=np.float32)
    faces = np.asarray(mesh["faces"], dtype=np.int32)
    with open(path, "w") as f:
        f.write("# APEX-Mesh observable-surface mesh\n")
        for v in verts:
            f.write("v {:.6f} {:.6f} {:.6f}\n".format(v[0], v[1], v[2]))
        for tri in faces:
            f.write("f {} {} {}\n".format(int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1))


def save_npz(path, mesh):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arrays = {k: np.asarray(v) for k, v in mesh.items()}
    np.savez_compressed(path, **arrays)


def load_npz(path):
    data = np.load(path)
    return {k: data[k] for k in data.files}


def read_ply(path):
    """Minimal ASCII PLY reader (our own writer's format)."""
    with open(path, "r") as f:
        lines = f.read().splitlines()
    assert lines[0].strip() == "ply", "not a ply file"
    n_vert = n_face = 0
    header_end = 0
    i = 1
    while i < len(lines):
        parts = lines[i].split()
        if parts[0] == "element" and parts[1] == "vertex":
            n_vert = int(parts[2])
        elif parts[0] == "element" and parts[1] == "face":
            n_face = int(parts[2])
        elif parts[0] == "end_header":
            header_end = i
            break
        i += 1
    verts = np.zeros((n_vert, 3), dtype=np.float32)
    normals = np.zeros((n_vert, 3), dtype=np.float32)
    confidence = np.zeros((n_vert,), dtype=np.float32)
    support = np.zeros((n_vert,), dtype=np.int32)
    visible = np.zeros((n_vert,), dtype=np.uint8)
    for j in range(n_vert):
        p = lines[header_end + 1 + j].split()
        verts[j] = [float(p[0]), float(p[1]), float(p[2])]
        normals[j] = [float(p[3]), float(p[4]), float(p[5])]
        confidence[j] = float(p[6])
        support[j] = int(p[7])
        visible[j] = int(p[8])
    faces = np.zeros((n_face, 3), dtype=np.int32)
    for j in range(n_face):
        p = lines[header_end + 1 + n_vert + j].split()
        faces[j] = [int(p[1]), int(p[2]), int(p[3])]
    return {"verts": verts, "normals": normals, "faces": faces,
            "confidence": confidence, "support_count": support, "visible": visible}
