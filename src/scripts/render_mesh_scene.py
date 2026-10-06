#!/usr/bin/env python
"""Whole-scene shaded mesh render (paper figure).

The stock ``vis_moud.py`` 03/04 panels subsample to ~4000 faces, which reads as
a point cloud, not a mesh.  This script rasterises the *full* refined mesh with
a software z-buffer (perspective projection + Lambertian shading + 1px dilation)
so the whole AT1 scene is visible as a solid, shaded surface.  It also emits a
wireframe panel so the triangulation is explicit.

Output: ``<model>/mesh/vis/08_mesh_scene_shaded.png`` and
``09_mesh_scene_wireframe.png``.

Usage:
  python scripts/render_mesh_scene.py AT1 [AT2 ...]
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy import ndimage

REPO = "/home/cxh/APEX/apex-mesh"
MODEL_ROOT = os.path.join(REPO, "outputs", "moud")


def load_mesh(model_dir):
    """Load refined mesh; prefer the pruned coloured mesh when present, else
    colored_mesh.npz (has per-vertex 'color'), else refined_mesh.npz."""
    root = os.path.join(model_dir, "mesh")
    for name in ("colored_mesh_pruned.npz", "colored_mesh.npz", "refined_mesh.npz"):
        path = os.path.join(root, name)
        if os.path.exists(path):
            break
    m = np.load(path)
    vcolors = m["color"].astype(np.float32) if "color" in m.files else None
    return (m["verts"].astype(np.float64),
            m["faces"].astype(np.int64),
            m["normals"].astype(np.float64),
            vcolors)


def look_at(eye, target, up):
    """Return world->camera rotation (rows = camera axes) for a RH camera
    looking from ``eye`` toward ``target`` (z_cam positive in front)."""
    z = target - eye
    z = z / np.linalg.norm(z)              # camera forward (points at target)
    x = np.cross(z, up)
    x = x / np.linalg.norm(x)
    y = np.cross(x, z)
    y = y / np.linalg.norm(y)
    return np.stack([x, y, z], axis=0)     # R: Xc = (P - eye) @ R.T


def render(verts, faces, vnormals, eye, target, up, W, H, fov=42.0,
           vcolors=None, zmap="turbo"):
    """Software z-buffer: splat verts + face centroids, Lambertian shade,
    dilate 1px to close gaps.  Colours by baked vertex colour when ``vcolors``
    is given, else by depth (``zmap``).  Returns RGB uint8 image (H, W, 3)."""
    R = look_at(eye, target, up)
    Xc = (verts - eye) @ R.T               # camera space, z forward
    z = Xc[:, 2]
    fx = fy = 0.5 * H / np.tan(np.deg2rad(fov) / 2.0)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    u = fx * Xc[:, 0] / z + cx
    v = fy * Xc[:, 1] / z + cy

    # per-face centroid (denser splat) + per-face normal for flat shading
    fc = verts[faces].mean(axis=1)
    Xc_c = (fc - eye) @ R.T
    zc = Xc_c[:, 2]
    uc = fx * Xc_c[:, 0] / zc + cx
    vc = fy * Xc_c[:, 1] / zc + cy
    fn = np.cross(verts[faces][:, 1] - verts[faces][:, 0],
                  verts[faces][:, 2] - verts[faces][:, 0])
    fn = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12)

    pu = np.concatenate([u, uc])
    pv = np.concatenate([v, vc])
    pz = np.concatenate([z, zc])
    # vertex normals for verts, face normals for centroids
    pn = np.concatenate([vnormals, fn])

    light = np.array([-0.35, 0.45, 0.83])
    light = light / np.linalg.norm(light)
    ndot = (pn * light).sum(axis=1)
    shade = (0.35 + 0.65 * np.abs(ndot))[:, None]   # (P,1)

    # per-point RGB: baked colour if available, else depth colormap
    if vcolors is not None:
        fc_colors = vcolors[faces].mean(axis=1)      # (F,3) face centroid colour
        prgb = np.concatenate([vcolors, fc_colors]).astype(np.float32) / 255.0
    else:
        cmap = plt.get_cmap(zmap)
        zz_norm = (pz - pz.min()) / (pz.max() - pz.min() + 1e-12)
        prgb = cmap(zz_norm)[..., :3].astype(np.float32)

    ui = np.floor(pu).astype(np.int64)
    vi = np.floor(pv).astype(np.int64)
    ok = (pz > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, pz = ui[ok], vi[ok], pz[ok]
    prgb, shade = prgb[ok], shade[ok]
    idx = vi * W + ui

    depth = np.full(W * H, np.inf)
    np.minimum.at(depth, idx, pz)
    near = depth[idx] <= pz + 1e-7          # this point is (one of) the nearest

    rgb_buf = np.ones((W * H, 3), dtype=np.float32)
    sh_buf = np.zeros(W * H, dtype=np.float32)
    rgb_buf[idx[near]] = prgb[near]          # nearest splat wins (ties -> later)
    sh_buf[idx[near]] = shade[near, 0]

    rgb = rgb_buf.reshape(H, W, 3)
    sh = sh_buf.reshape(H, W)
    mask = np.isfinite(depth.reshape(H, W))

    # 1px dilation (fill the occasional single-pixel gap between splats)
    mask = ndimage.binary_dilation(mask, iterations=1)
    rgb = np.stack([ndimage.grey_dilation(rgb[..., c], size=(3, 3))
                    for c in range(3)], axis=-1)
    sh = ndimage.grey_dilation(sh, size=(3, 3))

    out = rgb * sh[..., None]
    out[~mask] = 1.0
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def scene_render(model_dir, out_dir):
    verts, faces, vnormals, vcolors = load_mesh(model_dir)
    c = verts.mean(axis=0)
    r = float(np.abs(verts - c).max())
    dist = 2.3 * r
    up = np.array([0.0, 0.0, -1.0])        # z is downward depth -> up is -z

    views = []
    for azim in (25, 115, 205, 295):
        phi = np.deg2rad(38)
        th = np.deg2rad(azim)
        d = np.array([np.cos(phi) * np.cos(th), np.cos(phi) * np.sin(th), -np.sin(phi)])
        eye = c + dist * d
        views.append(("azim=%d" % azim, eye))

    W, H = 1500, 1100
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, (name, eye) in zip(axes.ravel(), views):
        img = render(verts, faces, vnormals, eye, c, up, W, H, vcolors=vcolors)
        ax.imshow(img)
        ax.set_title("shaded mesh  " + name, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("AT1 whole-scene refined mesh (Lambertian shaded, baked colour)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "08_mesh_scene_shaded.png"), dpi=120)
    plt.close(fig)
    print("  saved 08_mesh_scene_shaded.png")

    # --- wireframe (triangulation explicit) from one view ---
    eye = views[0][1]
    R = look_at(eye, c, up)
    Xc = (verts - eye) @ R.T
    z = Xc[:, 2]
    fx = fy = 0.5 * H / np.tan(np.deg2rad(42) / 2.0)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    u = fx * Xc[:, 0] / z + cx
    v = fy * Xc[:, 1] / z + cy
    ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    # unique edges, subsampled for a legible wireframe
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.sort(e, axis=1)
    e = np.unique(e, axis=0)
    step = max(1, int(len(e) // 250000))
    e = e[::step]
    ok_e = ok[e].all(axis=1)

    fig, ax = plt.subplots(figsize=(15, 11))
    seg = np.stack([u[e[ok_e]], v[e[ok_e]]], axis=-1)
    ax.add_collection(LineCollection(seg, colors="black", linewidths=0.2, alpha=0.6))
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("AT1 refined mesh - wireframe (triangulation, %d edges shown)"
                 % len(seg))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "09_mesh_scene_wireframe.png"), dpi=120)
    plt.close(fig)
    print("  saved 09_mesh_scene_wireframe.png  (verts=%d faces=%d)" %
          (verts.shape[0], faces.shape[0]))


def topdown_render(model_dir, out_dir, W=1800, H=1400):
    """Straight-down (nadir) view of the whole reef sheet, single high-res panel."""
    verts, faces, vnormals, vcolors = load_mesh(model_dir)
    c = verts.mean(axis=0)
    ext = verts.max(axis=0) - verts.min(axis=0)
    dist = 1.6 * float(ext.max())
    eye = np.array([c[0], c[1], verts[:, 2].min() - dist])   # above the surface
    up = np.array([0.0, 1.0, 0.0])

    img = render(verts, faces, vnormals, eye, c, up, W, H, fov=40.0, vcolors=vcolors)
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(img)
    ax.set_title("AT1 refined mesh - top-down (nadir) shaded view", fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    p = os.path.join(out_dir, "08b_mesh_scene_topdown.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)

    # wireframe companion
    R = look_at(eye, c, up)
    Xc = (verts - eye) @ R.T
    z = Xc[:, 2]
    fx = fy = 0.5 * H / np.tan(np.deg2rad(40) / 2.0)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    u = fx * Xc[:, 0] / z + cx
    v = fy * Xc[:, 1] / z + cy
    ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.sort(e, axis=1); e = np.unique(e, axis=0)
    step = max(1, int(len(e) // 300000)); e = e[::step]
    ok_e = ok[e].all(axis=1)
    fig, ax = plt.subplots(figsize=(16, 12))
    seg = np.stack([u[e[ok_e]], v[e[ok_e]]], axis=-1)
    ax.add_collection(LineCollection(seg, colors="black", linewidths=0.2, alpha=0.6))
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("AT1 refined mesh - top-down wireframe (%d edges)" % len(seg))
    fig.tight_layout()
    p2 = os.path.join(out_dir, "09b_mesh_scene_topdown_wireframe.png")
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    print("  saved 08b_mesh_scene_topdown.png / 09b_mesh_scene_topdown_wireframe.png")


def main():
    args = sys.argv[1:]
    topdown = "--topdown" in args
    scenes = [a for a in args if a != "--topdown"] or ["AT1"]
    for s in scenes:
        model_dir = os.path.join(MODEL_ROOT, s)
        out_dir = os.path.join(model_dir, "mesh", "vis")
        os.makedirs(out_dir, exist_ok=True)
        print("=== %s -> %s" % (s, out_dir))
        if topdown:
            topdown_render(model_dir, out_dir)
        else:
            scene_render(model_dir, out_dir)


if __name__ == "__main__":
    main()
