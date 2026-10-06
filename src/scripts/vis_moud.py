#!/usr/bin/env python
"""MOUD mesh + render-overlay visualizations (paper figures).

Produces, per scene, under ``<model>/mesh/vis/``:
  01_reference_view_depth.png    2D reference-view projection, colored by optical depth
  02_reference_view_confidence.png  same, colored by per-face confidence
  03_mesh3d_confidence.png       3D refined mesh, colored by confidence
  04_mesh3d_depth.png            3D refined mesh, colored by optical depth
  05_overlay_grid_4views.png     refined mesh reprojected over raw underwater images
  06_refview_faces.png / 07_refview_boundary.png  single reference view, faces / boundary only

Usage:
  python vis_moud.py AT1 [AT2 AT3 AT4 AT5]

Paths are derived from the conventional layout (no argparse needed):
  model dir = <repo>/outputs/moud/<scene>, source images = /home/cxh/APEX/MOUD/<scene>/images
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

REPO = "/home/cxh/APEX/apex-mesh"
SRC_ROOT = "/home/cxh/APEX/MOUD"
MODEL_ROOT = os.path.join(REPO, "outputs", "moud")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".PNG")


def _mesh_path(model_dir):
    return os.path.join(model_dir, "mesh", "refined_mesh.npz")


def _find_image(img_dir, base_name):
    for e in IMG_EXTS:
        p = os.path.join(img_dir, base_name + e)
        if os.path.exists(p):
            return p
    return None


def mesh_vis(model_dir, out_dir):
    m = np.load(_mesh_path(model_dir))
    verts = m["verts"].astype(np.float64)
    faces = m["faces"]
    uv = m["uv"].astype(np.float64)
    ref_depth = m["ref_depth"].astype(np.float64)
    confidence = m["confidence"].astype(np.float64)
    support = m["support_count"].astype(np.float64)
    boundary = m["boundary_edges"]

    face_depth = ref_depth[faces].mean(axis=1)
    face_conf = confidence[faces].mean(axis=1)
    face_support = support[faces].mean(axis=1)

    # --- Fig 1: reference-view 2D projection colored by optical depth ---
    fig, ax = plt.subplots(figsize=(12, 8))
    tri = uv[faces]
    coll = PolyCollection(tri, array=face_depth, cmap="turbo", edgecolor="none", linewidths=0)
    ax.add_collection(coll)
    if boundary.size:
        ax.add_collection(LineCollection(uv[boundary], colors="black",
                                         linewidths=0.4, alpha=0.5))
    ax.set_xlim(uv[:, 0].min(), uv[:, 0].max())
    ax.set_ylim(uv[:, 1].min(), uv[:, 1].max())
    ax.invert_yaxis(); ax.set_aspect("equal")
    ax.set_title("Refined mesh - reference-view projection (optical depth)")
    fig.colorbar(coll, ax=ax, label="ref_depth")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_reference_view_depth.png"), dpi=130)
    plt.close(fig)

    # --- Fig 2: reference-view 2D projection colored by confidence ---
    fig, ax = plt.subplots(figsize=(12, 8))
    coll = PolyCollection(tri, array=face_conf, cmap="magma", edgecolor="none", linewidths=0)
    ax.add_collection(coll)
    ax.set_xlim(uv[:, 0].min(), uv[:, 0].max())
    ax.set_ylim(uv[:, 1].min(), uv[:, 1].max())
    ax.invert_yaxis(); ax.set_aspect("equal")
    ax.set_title("Refined mesh - confidence (mean per face)")
    fig.colorbar(coll, ax=ax, label="confidence")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_reference_view_confidence.png"), dpi=130)
    plt.close(fig)

    # --- Fig 3/4: 3D mesh (subsample faces for interactive-time rendering) ---
    step = max(1, int(len(faces) // 4000))
    fsub = faces[::step]
    tri3 = verts[fsub]

    def save_3d(values, title, fname, cmap):
        fig = plt.figure(figsize=(11, 9))
        ax = fig.add_subplot(111, projection="3d")
        pc = Poly3DCollection(tri3, cmap=cmap, edgecolor="none", linewidths=0)
        pc.set_array(values[::step])
        ax.add_collection3d(pc)
        c = verts.mean(axis=0)
        r = np.abs(verts - c).max()
        ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=25, azim=-60)
        ax.set_title(title)
        fig.colorbar(pc, ax=ax, shrink=0.6)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=120)
        plt.close(fig)

    save_3d(face_conf, "Refined mesh 3D - confidence", "03_mesh3d_confidence.png", "magma")
    save_3d(face_depth, "Refined mesh 3D - optical depth", "04_mesh3d_depth.png", "turbo")

    print("  [mesh] verts={} faces={} boundary_edges={}".format(
        verts.shape[0], faces.shape[0], boundary.shape[0]))


def render_overlay(model_dir, src_dir, out_dir):
    m = np.load(_mesh_path(model_dir))
    verts = m["verts"].astype(np.float64)
    faces = m["faces"]
    boundary = m["boundary_edges"]

    cams = json.load(open(os.path.join(model_dir, "cameras.json")))
    img_dir = os.path.join(src_dir, "images")

    def project(cam):
        R = np.array(cam["rotation"], dtype=np.float64)      # world -> camera
        pos = np.array(cam["position"], dtype=np.float64)
        fx, fy = cam["fx"], cam["fy"]
        W, H = cam["width"], cam["height"]
        Xc = (verts - pos) @ R.T
        z = Xc[:, 2]
        u = fx * Xc[:, 0] / z + W / 2.0
        v = fy * Xc[:, 1] / z + H / 2.0
        return u, v, z

    def draw_overlay(ax, cam, alpha=0.45, show_faces=True, show_boundary=True, cmap="turbo"):
        img_path = _find_image(img_dir, cam["img_name"])
        if img_path is None:
            ax.text(0.5, 0.5, "missing image", transform=ax.transAxes)
            return
        ax.imshow(np.asarray(Image.open(img_path)))
        u, v, z = project(cam)
        ok = (z > 0) & (u >= 0) & (u < cam["width"]) & (v >= 0) & (v < cam["height"])
        fok = ok[faces].all(axis=1)
        if show_faces and fok.any():
            tri = np.stack([u[faces[fok]], v[faces[fok]]], axis=-1)
            fdepth = z[faces[fok]].mean(axis=1)
            coll = PolyCollection(tri, array=fdepth, cmap=cmap, edgecolor="none", linewidths=0)
            coll.set_alpha(alpha)
            ax.add_collection(coll)
        if show_boundary and boundary.size:
            bok = ok[boundary].all(axis=1)
            be = np.stack([u[boundary[bok]], v[boundary[bok]]], axis=-1)
            ax.add_collection(LineCollection(be, colors="black", linewidths=0.5, alpha=0.7))
        ax.set_xlim(0, cam["width"]); ax.set_ylim(cam["height"], 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("{} (cam id={})".format(cam["img_name"], cam["id"]), fontsize=8)

    n = len(cams)
    ids = [int(round(k)) for k in np.linspace(0, n - 1, 4)]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for ax, cid in zip(axes.ravel(), ids):
        draw_overlay(ax, cams[cid])
    fig.suptitle("Refined mesh reprojected over raw underwater images (camera-space depth)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_overlay_grid_4views.png"), dpi=110)
    plt.close(fig)

    # Single high-res reference view (pick the middle id), faces vs boundary only.
    ref = ids[len(ids) // 2]
    for mode, fname in [("faces", "06_refview_faces.png"), ("boundary", "07_refview_boundary.png")]:
        fig, ax = plt.subplots(figsize=(12, 8))
        draw_overlay(ax, cams[ref], show_faces=(mode == "faces"), show_boundary=(mode == "boundary"))
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=130)
        plt.close(fig)

    print("  [overlay] {} cameras, used ids {}".format(n, ids))


def main():
    scenes = sys.argv[1:] if len(sys.argv) > 1 else ["AT1"]
    for scene in scenes:
        model_dir = os.path.join(MODEL_ROOT, scene)
        src_dir = os.path.join(SRC_ROOT, scene)
        out_dir = os.path.join(model_dir, "mesh", "vis")
        os.makedirs(out_dir, exist_ok=True)
        print("=== {} -> {}".format(scene, out_dir))
        if not os.path.exists(_mesh_path(model_dir)):
            print("  SKIP: refined_mesh.npz not found; run the pipeline first.")
            continue
        mesh_vis(model_dir, out_dir)
        render_overlay(model_dir, src_dir, out_dir)
        print("  saved ->", out_dir)


if __name__ == "__main__":
    main()
