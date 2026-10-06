#!/usr/bin/env python
"""Bake per-vertex RGB colour onto a mesh from the nearest Gaussian's SH0 colour.

The mesh is the implicit-MLS zero-set of the flattened Gaussians, so every mesh
vertex sits near a Gaussian centre; its "own colour" is the nearest Gaussian's
degree-0 SH colour (the view-independent diffuse base), matching the CUDA
rasteriser's formula ``colour = SH_C0 * f_dc + 0.5`` (clamped >= 0).

Writes, next to ``refined_mesh.*``:
  colored_mesh.npz   -- all refined-mesh channels + ``color`` (V,3) uint8
  colored_mesh.ply   -- binary little-endian PLY with x,y,z,nx,ny,nz,red,green,blue

Usage:
  python bake_vertex_color.py --source_path <src> --model_path <m> --load_iteration 65000
"""
import os
import sys
import numpy as np
import torch
from argparse import ArgumentParser
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scene import Scene, GaussianModel
from arguments import ModelParams, get_combined_args

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
         -1.0925484305920792, 0.5462742152960396]
SH_C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
         0.3731763325901154, -0.4570457994644658, 1.445305721320277,
         -0.5900435899266435]


def _sh_color(means, features, campos):
    """Evaluate the full degree-3 SH colour (matches ``computeColorFromSH``)
    at each Gaussian in the direction toward ``campos`` -> (N,3) in ~[0,1]."""
    d = means - campos
    d = d / torch.norm(d, dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    sh = features                                       # (N, 16, 3)
    result = SH_C0 * sh[:, 0]
    result = (result - SH_C1 * y[:, None] * sh[:, 1]
              + SH_C1 * z[:, None] * sh[:, 2]
              - SH_C1 * x[:, None] * sh[:, 3])
    result = (result + SH_C2[0] * xy[:, None] * sh[:, 4]
              + SH_C2[1] * yz[:, None] * sh[:, 5]
              + SH_C2[2] * (2 * zz - xx - yy)[:, None] * sh[:, 6]
              + SH_C2[3] * xz[:, None] * sh[:, 7]
              + SH_C2[4] * (xx - yy)[:, None] * sh[:, 8])
    result = (result + SH_C3[0] * (y * (3 * xx - yy))[:, None] * sh[:, 9]
              + SH_C3[1] * (xy * z)[:, None] * sh[:, 10]
              + SH_C3[2] * (y * (4 * zz - xx - yy))[:, None] * sh[:, 11]
              + SH_C3[3] * (z * (2 * zz - 3 * xx - 3 * yy))[:, None] * sh[:, 12]
              + SH_C3[4] * (x * (4 * zz - xx - yy))[:, None] * sh[:, 13]
              + SH_C3[5] * (z * (xx - yy))[:, None] * sh[:, 14]
              + SH_C3[6] * (x * (xx - 3 * yy))[:, None] * sh[:, 15])
    return result + 0.5


def _write_colored_ply(path, verts, normals, faces, colors):
    verts = verts.astype(np.float32)
    normals = normals.astype(np.float32)
    colors = colors.astype(np.uint8)
    nv, nf = verts.shape[0], faces.shape[0]
    header = ("ply\n"
              "format binary_little_endian 1.0\n"
              "comment baked vertex colour from nearest Gaussian SH0\n"
              "element vertex %d\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property float nx\nproperty float ny\nproperty float nz\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "element face %d\n"
              "property list uchar int vertex_indices\n"
              "end_header\n" % (nv, nf))
    v = np.zeros(nv, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                            ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                            ("r", "u1"), ("g", "u1"), ("b", "u1")])
    v["x"], v["y"], v["z"] = verts[:, 0], verts[:, 1], verts[:, 2]
    v["nx"], v["ny"], v["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    v["r"], v["g"], v["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
    f = np.zeros(nf, dtype=[("n", "u1"), ("idx", "<i4", (3,))])
    f["n"] = 3
    f["idx"] = faces.astype(np.int32)
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(v.tobytes())
        fh.write(f.tobytes())
    return path


def main():
    parser = ArgumentParser(description="bake per-vertex mesh colour")
    lp = ModelParams(parser, sentinel=True)
    parser.add_argument("--load_iteration", type=int, default=65000)
    parser.add_argument("--alpha_min", type=float, default=0.05,
                        help="only colour from Gaussians above this alpha")
    parser.add_argument("--max_dist", type=float, default=None,
                        help="vertices farther than this from any Gaussian are greyed")
    parser.add_argument("--saturation", type=float, default=1.0,
                        help="colour saturation boost (1.0 = none)")
    parser.add_argument("--contrast", type=float, default=1.0,
                        help="colour contrast boost (1.0 = none)")
    args = get_combined_args(parser)

    dataset = lp.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.load_iteration, shuffle=False)
    sidecar = os.path.join(dataset.model_path, "projective", "projective.pth")
    if os.path.exists(sidecar):
        gaussians.load_projective(sidecar)

    means_t = gaussians.get_xyz.detach()                                   # (N,3) cuda
    alphas = gaussians.get_opacity.detach().mean(dim=1).cpu().numpy()      # (N,)
    features = gaussians.get_features.detach()                             # (N,16,3)
    cams = scene.getTrainCameras()
    cam_centers = torch.stack([c.camera_center.cpu() for c in cams]).float()
    campos = cam_centers.mean(dim=0).to(means_t.device)

    rgb = _sh_color(means_t, features, campos).clamp(0.0, 1.0)
    sat = getattr(args, "saturation", 1.0)
    con = getattr(args, "contrast", 1.0)
    if sat != 1.0:
        gray = rgb.mean(dim=-1, keepdim=True)
        rgb = (gray + (rgb - gray) * sat).clamp(0.0, 1.0)
    if con != 1.0:
        rgb = ((rgb - 0.5) * con + 0.5).clamp(0.0, 1.0)
    rgb = rgb.cpu().numpy().astype(np.float32)
    means = means_t.detach().cpu().numpy().astype(np.float64)
    keep = alphas >= args.alpha_min
    print("Gaussians: %d (alpha>=%.2f: %d)" % (means.shape[0], args.alpha_min, int(keep.sum())))
    tree = cKDTree(means[keep])

    root = os.path.join(dataset.model_path, "mesh")
    src = os.path.join(root, "refined_mesh.npz")
    m = np.load(src)
    verts = m["verts"].astype(np.float64)
    dist, idx = tree.query(verts, k=1)
    idx = np.where(keep)[0][idx]                      # map back to full gaussian index
    colors = (rgb[idx] * 255.0 + 0.5).astype(np.uint8)

    max_dist = getattr(args, "max_dist", None)
    if max_dist is not None:
        colors[dist > max_dist] = 128                 # grey unknown regions
    print("coloured %d verts; median dist to nearest Gaussian %.4f m, "
          "max %.4f m" % (verts.shape[0], float(np.median(dist)), float(dist.max())))

    out = {k: m[k] for k in m.files}
    out["color"] = colors
    np.savez_compressed(os.path.join(root, "colored_mesh.npz"), **out)
    _write_colored_ply(os.path.join(root, "colored_mesh.ply"),
                       verts.astype(np.float32), m["normals"].astype(np.float32),
                       m["faces"], colors)
    print("saved colored_mesh.npz / colored_mesh.ply ->", root)


if __name__ == "__main__":
    main()
