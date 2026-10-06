#!/usr/bin/env python
"""Per-input-view mesh inspection: original image | white mesh | coloured mesh.

For every training camera in ``cameras.json``, render the refined mesh from that
exact viewpoint (camera intrinsics + extrinsics) and save a three-panel figure:

    left    original underwater image
    middle  Lambertian-shaded *white* mesh (geometry only)
    right   Lambertian-shaded *baked-colour* mesh

The white panel is a neutral matte render so you can inspect the geometry
without texture colour; the coloured panel shows the baked vertex colour.  The
renderer is the same software z-buffer + Lambertian shading used by
``render_mesh_scene.py``, but driven by each camera's real pose/focal length.

Output:
    <model>/mesh/vis/perview/000_<img_name>.png   one per camera (4096->1024)
    <model>/mesh/vis/perview_grid.png             contact sheet of the coloured meshes

Usage:
  python scripts/vis_perview_mesh.py AT1 [AT2 ...] [--scale 0.25]
"""
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image

REPO = "/home/cxh/APEX/apex-mesh"
SRC_ROOT = "/home/cxh/APEX/MOUD"
MODEL_ROOT = os.path.join(REPO, "outputs", "moud")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".PNG")
LIGHT = np.array([-0.35, 0.45, 0.83], dtype=np.float64)
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def load_mesh(model_dir):
    root = os.path.join(model_dir, "mesh")
    for name in ("colored_mesh_pruned.npz", "colored_mesh.npz", "refined_mesh.npz"):
        path = os.path.join(root, name)
        if os.path.exists(path):
            break
    m = np.load(path)
    verts = m["verts"].astype(np.float64)
    faces = m["faces"].astype(np.int64)
    normals = m["normals"].astype(np.float64)
    vcolors = m["color"].astype(np.float32) if "color" in m.files else None
    return verts, faces, normals, vcolors


def _find_image(img_dir, base_name):
    for e in IMG_EXTS:
        p = os.path.join(img_dir, base_name + e)
        if os.path.exists(p):
            return p
    return None


def render_camera(verts, vnormals, vcolors, fc_world, fn, fc_color,
                  R, pos, fx, fy, W, H, colored):
    """Software z-buffer render of the mesh from one camera.

    ``fc_world`` (F,3) / ``fn`` (F,3) are precomputed face centroids + normals;
    ``fc_color`` (F,3) the per-face baked colour.  When ``colored`` is False the
    mesh is shaded as a neutral matte ("white model").  Returns RGB uint8 (H,W,3)
    on a white background.
    """
    # cameras.json ``rotation`` is camera->world (R_c2w = R_w2c^T), so a world
    # point maps to camera space as  Xc[i] = R_w2c @ (x[i] - pos)  which in numpy
    # is  (x - pos) @ rotation  (NO transpose — the transpose mirrors the scene,
    # which is exactly the "viewing the bottom from below" flip).
    Xc = (verts - pos) @ R
    z = Xc[:, 2]
    u = fx * Xc[:, 0] / z + W / 2.0
    v = fy * Xc[:, 1] / z + H / 2.0

    Xc_c = (fc_world - pos) @ R
    zc = Xc_c[:, 2]
    uc = fx * Xc_c[:, 0] / zc + W / 2.0
    vc = fy * Xc_c[:, 1] / zc + H / 2.0

    pu = np.concatenate([u, uc])
    pv = np.concatenate([v, vc])
    pz = np.concatenate([z, zc])
    pn = np.concatenate([vnormals, fn])

    ndot = (pn * LIGHT).sum(axis=1)
    shade = (0.35 + 0.65 * np.abs(ndot))[:, None]

    if colored:
        prgb = np.concatenate([vcolors, fc_color]).astype(np.float32) / 255.0
    else:
        prgb = np.full((pu.shape[0], 3), 0.8, dtype=np.float32)

    ui = np.floor(pu).astype(np.int64)
    vi = np.floor(pv).astype(np.int64)
    ok = (pz > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, pz = ui[ok], vi[ok], pz[ok]
    prgb, shade = prgb[ok], shade[ok]
    idx = vi * W + ui

    depth = np.full(W * H, np.inf)
    np.minimum.at(depth, idx, pz)
    near = depth[idx] <= pz + 1e-7

    rgb_buf = np.ones((W * H, 3), dtype=np.float32)
    sh_buf = np.zeros(W * H, dtype=np.float32)
    rgb_buf[idx[near]] = prgb[near]
    sh_buf[idx[near]] = shade[near, 0]

    rgb = rgb_buf.reshape(H, W, 3)
    sh = sh_buf.reshape(H, W)
    mask = np.isfinite(depth.reshape(H, W))

    mask = ndimage.binary_dilation(mask, iterations=1)
    rgb = np.stack([ndimage.grey_dilation(rgb[..., c], size=(3, 3))
                    for c in range(3)], axis=-1)
    sh = ndimage.grey_dilation(sh, size=(3, 3))

    out = rgb * sh[..., None]
    out[~mask] = 1.0
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def perview(model_dir, src_dir, out_dir, scale):
    verts, faces, vnormals, vcolors = load_mesh(model_dir)

    # Precompute view-independent face data once.
    tri = verts[faces]                                   # (F,3,3)
    fc_world = tri.mean(axis=1)                          # (F,3)
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12)
    del tri
    fc_color = (vcolors[faces].mean(axis=1).astype(np.float32)
                if vcolors is not None else np.full((faces.shape[0], 3), 200, np.float32))

    cams = json.load(open(os.path.join(model_dir, "cameras.json")))
    img_dir = os.path.join(src_dir, "images")
    os.makedirs(out_dir, exist_ok=True)

    n = len(cams)
    for i, cam in enumerate(cams):
        R = np.array(cam["rotation"], dtype=np.float64)
        pos = np.array(cam["position"], dtype=np.float64)
        W, H = int(cam["width"] * scale), int(cam["height"] * scale)
        fx, fy = cam["fx"] * scale, cam["fy"] * scale

        white = render_camera(verts, vnormals, vcolors, fc_world, fn, fc_color,
                              R, pos, fx, fy, W, H, colored=False)
        color = render_camera(verts, vnormals, vcolors, fc_world, fn, fc_color,
                              R, pos, fx, fy, W, H, colored=True)

        # original image, downscaled to match
        img_path = _find_image(img_dir, cam["img_name"])
        orig = None
        if img_path is not None:
            im = Image.open(img_path).convert("RGB")
            im = im.resize((W, H), Image.BILINEAR)
            orig = np.asarray(im)

        fig, axes = plt.subplots(1, 3, figsize=(3 * W / 150, H / 150 + 0.5))
        labels = ["original", "white mesh", "coloured mesh"]
        panels = [orig, white, color]
        for ax, lab, panel in zip(axes, labels, panels):
            if panel is None:
                ax.text(0.5, 0.5, "missing image", transform=ax.transAxes, ha="center")
            else:
                ax.imshow(panel)
            ax.set_title(lab, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("{}  (cam {})".format(cam["img_name"], cam["id"]), fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = os.path.join(out_dir, "{:03d}_{}.png".format(i, cam["img_name"]))
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("  [{:02d}/{:02d}] {}".format(i + 1, n, os.path.basename(out)))

    # contact sheet of the coloured meshes (one small thumbnail per view)
    _contact_sheet(cams, verts, vnormals, vcolors, fc_world, fn, fc_color, out_dir, scale)


def _contact_sheet(cams, verts, vnormals, vcolors, fc_world, fn, fc_color, out_dir, scale):
    cols = 7
    tw = 240
    rows = int(np.ceil(len(cams) / cols))
    th = int(tw * 3008 / 4096) + 18
    sheet = np.full((rows * th, cols * tw, 3), 255, dtype=np.uint8)
    for i, cam in enumerate(cams):
        R = np.array(cam["rotation"], dtype=np.float64)
        pos = np.array(cam["position"], dtype=np.float64)
        W, H = tw, int(tw * 3008 / 4096)
        fx, fy = cam["fx"] * (tw / 4096.0), cam["fy"] * (tw / 4096.0)
        img = render_camera(verts, vnormals, vcolors, fc_world, fn, fc_color,
                            R, pos, fx, fy, W, H, colored=True)
        r, c = divmod(i, cols)
        sheet[r * th:r * th + H, c * tw:(c + 1) * tw] = img
        sheet[r * th + H:r * th + H + 16, c * tw:(c + 1) * tw] = 235
    plt.imsave(os.path.join(out_dir, "..", "perview_grid.png"), sheet)
    print("  contact sheet ->", os.path.join(out_dir, "..", "perview_grid.png"))


def main():
    args = sys.argv[1:]
    scale = 0.25
    if "--scale" in args:
        k = args.index("--scale")
        scale = float(args[k + 1]); del args[k:k + 2]
    scenes = args or ["AT1"]
    for scene in scenes:
        model_dir = os.path.join(MODEL_ROOT, scene)
        src_dir = os.path.join(SRC_ROOT, scene)
        out_dir = os.path.join(model_dir, "mesh", "vis", "perview")
        os.makedirs(out_dir, exist_ok=True)
        print("=== %s -> %s (scale %.2f)" % (scene, out_dir, scale))
        perview(model_dir, src_dir, out_dir, scale)


if __name__ == "__main__":
    main()
