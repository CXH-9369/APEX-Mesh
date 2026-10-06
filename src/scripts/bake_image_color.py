#!/usr/bin/env python
"""Bake per-vertex colour by sampling the *original source photos*.

The stock ``bake_vertex_color.py`` colours each vertex from its nearest
Gaussian's SH0 base colour, so colour is quantised to the ~500k Gaussian
Voronoi cells (coarse, blocky).  This script instead reprojects every vertex
into every camera, tests visibility with a software z-buffer of the mesh
itself, and bilinearly samples the 4096x3008 source JPEG at the projected
pixel.  Visible views are blended with a Lambertian weight (normal . view-dir),
giving photo-resolution colour on the mesh surface.

Reads ``colored_mesh_pruned.npz`` (falls back to ``colored_mesh.npz``) and
writes the result back to the same file with the ``color`` channel replaced,
then re-exports ``viewer.bin`` so the interactive viewer reloads it.

Usage:
  python scripts/bake_image_color.py AT1 [AT2 ...] [--tol 0.03]
"""
import os
import sys
import json
import struct

import numpy as np
from PIL import Image

REPO = "/home/cxh/APEX/apex-mesh"
SRC_ROOT = "/home/cxh/APEX/MOUD"
MODEL_ROOT = os.path.join(REPO, "outputs", "moud")
MAGIC = b"APXM"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".PNG")


def _find_image(img_dir, base_name):
    for e in IMG_EXTS:
        p = os.path.join(img_dir, base_name + e)
        if os.path.exists(p):
            return p
    return None


def _bilinear(img, u, v):
    """Sample img (H,W,3) float at (u,v) float pixel coords, bilinear."""
    H, W = img.shape[:2]
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    du = (u - u0).astype(np.float32)
    dv = (v - v0).astype(np.float32)
    u0 = np.clip(u0, 0, W - 1)
    v0 = np.clip(v0, 0, H - 1)
    u1 = np.clip(u0 + 1, 0, W - 1)
    v1 = np.clip(v0 + 1, 0, H - 1)
    flat = img.reshape(H * W, 3)
    c00 = flat[v0 * W + u0]
    c01 = flat[v0 * W + u1]
    c10 = flat[v1 * W + u0]
    c11 = flat[v1 * W + u1]
    w00 = ((1 - du) * (1 - dv))[:, None]
    w01 = (du * (1 - dv))[:, None]
    w10 = ((1 - du) * dv)[:, None]
    w11 = (du * dv)[:, None]
    return c00 * w00 + c01 * w01 + c10 * w10 + c11 * w11


def _write_viewer_bin(path, verts, normals, colors, faces):
    pos = verts.astype("<f4").reshape(-1, 3).tobytes()
    nor = normals.astype("<f4").reshape(-1, 3).tobytes()
    col = colors.astype("u1").reshape(-1, 3).tobytes()
    col_pad = b"\x00" * ((-len(col)) % 4)
    fac = faces.astype("<u4").reshape(-1, 3).tobytes()
    header = struct.pack("<4sII", MAGIC, verts.shape[0], faces.shape[0])
    with open(path, "wb") as f:
        f.write(header); f.write(pos); f.write(nor); f.write(col)
        f.write(col_pad); f.write(fac)


def bake(mesh_dir, img_dir, cams, tol=0.03):
    src = os.path.join(mesh_dir, "colored_mesh_pruned.npz")
    if not os.path.exists(src):
        src = os.path.join(mesh_dir, "colored_mesh.npz")
    m = np.load(src)
    verts = m["verts"].astype(np.float32)
    faces = m["faces"].astype(np.int64)
    normals = m["normals"].astype(np.float32)
    old_color = (m["color"] if "color" in m.files
                 else np.full((verts.shape[0], 3), 200, np.uint8))

    M = verts.shape[0]
    # face centroids + normals for a denser z-buffer splat
    tri = verts[faces]
    fc = tri.mean(axis=1)
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12)
    del tri

    P = np.concatenate([verts, fc], axis=0).astype(np.float32)   # (M+F,3)
    NF = fc.shape[0]

    color_sum = np.zeros((M, 3), np.float64)
    weight_sum = np.zeros(M, np.float64)
    seen = np.zeros(M, np.int32)

    for ci, cam in enumerate(cams):
        R = np.asarray(cam["rotation"], dtype=np.float32)        # R_c2w
        pos = np.asarray(cam["position"], dtype=np.float32)
        W, H = int(cam["width"]), int(cam["height"])
        fx, fy = float(cam["fx"]), float(cam["fy"])
        cx, cy = W / 2.0, H / 2.0

        # --- project every point (verts + face centroids) -------------------
        Xc = (P - pos) @ R                                       # (M+F,3)
        z = Xc[:, 2]
        u = fx * Xc[:, 0] / np.maximum(z, 1e-6) + cx
        v = fy * Xc[:, 1] / np.maximum(z, 1e-6) + cy

        ui = np.floor(u).astype(np.int64)
        vi = np.floor(v).astype(np.int64)
        ok = (z > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

        # software z-buffer (min depth per pixel)
        depth = np.full(W * H, np.inf, np.float32)
        np.minimum.at(depth, vi[ok] * W + ui[ok], z[ok])

        # --- visibility of the mesh vertices --------------------------------
        zV, uV, vV = z[:M], u[:M], v[:M]
        uVi = np.floor(uV).astype(np.int64)
        vVi = np.floor(vV).astype(np.int64)
        okV = (zV > 0) & (uVi >= 0) & (uVi < W) & (vVi >= 0) & (vVi < H)
        dV = depth[np.where(okV, vVi * W + uVi, 0)]
        front = okV & (dV >= zV - tol)                           # front surface

        img_path = _find_image(img_dir, cam["img_name"])
        if img_path is None:
            continue
        img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32)
        rgb = _bilinear(img, uV[front], vV[front])               # (k,3) 0..255

        # Frontality weight: |normal . view_dir| (sign-agnostic so the sheet's
        # inconsistently-oriented normals do not zero out the weight and force
        # half the surface onto the Gaussian-colour fallback).
        view = pos - verts[front]
        view = view / (np.linalg.norm(view, axis=1, keepdims=True) + 1e-8)
        w = np.abs((normals[front] * view).sum(axis=1))

        idx = np.nonzero(front)[0]
        color_sum[idx] += rgb * w[:, None]
        weight_sum[idx] += w
        seen[idx] += 1
        if (ci + 1) % 10 == 0:
            print("    cam %2d/%d  verts seen so far >=1 view: %.1f%%"
                  % (ci + 1, len(cams), 100.0 * (seen > 0).mean()))

    # blend; fall back to the old (Gaussian-SH) colour where unseen
    color = np.where(weight_sum[:, None] > 0,
                     (color_sum / np.maximum(weight_sum, 1e-6)[:, None]),
                     old_color.astype(np.float32))
    color = np.clip(color, 0, 255).astype(np.uint8)

    cov = float((weight_sum > 0).mean())
    print("  coverage: %.1f%% verts with >=1 view, mean %.2f views/vertex"
          % (100.0 * cov, seen.mean()))

    # write back the colour channel + re-export viewer.bin
    out = {k: m[k] for k in m.files}
    m.close()
    out["color"] = color
    np.savez_compressed(src, **out)
    _write_viewer_bin(os.path.join(mesh_dir, "viewer.bin"),
                      verts, normals, color, faces)
    print("  wrote %s (colour) + viewer.bin" % os.path.basename(src))


def main():
    args = sys.argv[1:]
    tol = 0.03
    scenes = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--tol":
            tol = float(args[i + 1]); i += 2
        else:
            scenes.append(a); i += 1
    if not scenes:
        scenes = ["AT1"]
    for s in scenes:
        mesh_dir = os.path.join(MODEL_ROOT, s, "mesh")
        img_dir = os.path.join(SRC_ROOT, s, "images")
        cams = json.load(open(os.path.join(MODEL_ROOT, s, "cameras.json")))
        print("=== bake_image_color %s (%d cams, tol=%.3f)" % (s, len(cams), tol))
        bake(mesh_dir, img_dir, cams, tol)
    print("done")


if __name__ == "__main__":
    main()
