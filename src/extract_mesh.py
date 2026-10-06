"""Extract an open observable-surface mesh from the projective checkpoint.

Loads the projective model (``--load_iteration`` + ``projective/projective.pth``
+ ``projective/water_model.pth``), renders the new surface channels (median
surface depth, accumulated opacity, alpha-weighted normal, half-Gaussian
boundary response) for every training view, builds a *reference-view* depth
mesh (2D Delaunay + edge culling, open boundary), votes cross-view support and
assigns per-vertex confidence, then prunes low-support vertices.

Outputs (under ``<model_path>/mesh/``): reference_mesh.{ply,obj,npz},
mesh_stats.json and per-view surface PNGs.
"""

import os
import sys
import json
import math

import numpy as np
import torch
from argparse import ArgumentParser
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from medium import WaterImageModel
from utils.general_utils import safe_state
from mesh import (render_surface, surface_mask, build_reference_mesh,
                  multiview_support, fuse_multiview, silhouette_support,
                  photometric_confidence, confidence,
                  write_ply, write_obj, save_npz, mesh_stats, boundary_edges)
from mesh.extract import _edge_ok_vectorized


def _save_map(t, path, cmap="viridis"):
    x = t.detach().cpu().numpy()
    valid = x[x > 0]
    vmin = float(valid.min()) if valid.size else 0.0
    vmax = float(x.max())
    norm = (x - vmin) / max(vmax - vmin, 1e-6)
    plt.imsave(path, plt.get_cmap(cmap)(norm)[..., :3])


def _save_normal(n, path):
    x = n.detach().cpu().numpy()  # (3,H,W)
    img = (np.clip(x, -1, 1) * 0.5 + 0.5).transpose(1, 2, 0)
    plt.imsave(path, img)


def _nearest_cam(scene, mean_center, idx):
    cams = scene.getTrainCameras()
    if idx >= 0:
        return cams[idx]
    centers = torch.stack([c.camera_center.cpu() for c in cams]).float()
    d = (centers - mean_center.cpu()).norm(dim=1)
    return cams[int(d.argmin().item())]


def prune_by_support(mesh, min_support):
    keep = mesh["support_count"] >= min_support
    n_old = keep.shape[0]
    remap = np.full(n_old, -1, dtype=np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    new = {k: v for k, v in mesh.items()}
    # Remap every per-vertex (N, ...) array; faces get remapped below, and
    # ``tri`` (the pre-cull Delaunay simplices) is stale after pruning.
    for k in list(new.keys()):
        if k in ("faces", "boundary_edges", "tri"):
            continue
        arr = np.asarray(new[k])
        if arr.ndim >= 1 and arr.shape[0] == n_old:
            new[k] = arr[keep]
    faces = mesh["faces"]
    keep_f = faces.size and np.all(remap[faces] >= 0, axis=1)
    new["faces"] = remap[faces[keep_f]] if faces.size else np.zeros((0, 3), dtype=np.int32)
    new["boundary_edges"] = boundary_edges(new["faces"])
    new.pop("tri", None)
    return new


def extract(dataset, pipe, args):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.load_iteration, shuffle=False)

    sidecar = os.path.join(dataset.model_path, "projective", "projective.pth")
    if os.path.exists(sidecar):
        gaussians.load_projective(sidecar)
        print("Loaded projective gauge ->", sidecar)
    water_model = WaterImageModel().cuda()
    ckpt = os.path.join(dataset.model_path, "projective", "water_model.pth")
    water_model.load_state_dict(torch.load(ckpt))
    water_model.eval()

    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    cams = scene.getTrainCameras()

    # Render surface channels for every training view (once).
    surfaces = []
    for cam in tqdm(cams, desc="Rendering surface channels"):
        surfaces.append(render_surface(cam, gaussians, pipe, bg, water_model))

    # Reference camera (nearest the mean camera centre, or explicit index).
    mean_center = torch.stack([c.camera_center.cpu() for c in cams]).float().mean(dim=0)
    ref_cam = _nearest_cam(scene, mean_center, args.reference_idx)
    ref_idx = cams.index(ref_cam)
    surf = surfaces[ref_idx]
    print("Reference view: train[{:d}] (uid={})".format(ref_idx, ref_cam.uid))

    # Build the reference-view open mesh.
    mesh = build_reference_mesh(
        surf, ref_cam, max_r=args.max_r, a_thresh=args.a_thresh,
        depth_tol=args.depth_tol, normal_tol=args.normal_tol, edge_thresh=args.edge_thresh,
        stride=args.stride)
    print("Reference mesh: {} verts, {} faces (pre-prune)".format(
        mesh["verts"].shape[0], mesh["faces"].shape[0]))

    # Multi-view support + confidence on the reference vertices.
    if args.fuse == "none":
        ref_verts = torch.from_numpy(mesh["verts"]).cuda()
        other_sds = [s["surface_depth"] for s in surfaces]
        support, c_view = multiview_support(
            ref_verts, ref_cam, cams, other_sds, tol=args.support_tol, max_r=args.max_r)
    else:
        # phase 3: fuse the reference median-depth surface with every agreeing
        # neighbour view, then re-cull edges on the fused normals.
        verts_f, norms_f, support, c_view = fuse_multiview(
            mesh, ref_cam, cams, surfaces, tol=args.fusion_tol,
            max_r=args.max_r, a_thresh=args.fusion_a_thresh, robust=args.fuse)
        mesh["verts"] = verts_f.detach().cpu().numpy()
        mesh["normals"] = norms_f.detach().cpu().numpy()
        S = mesh["tri"]
        if S.size:
            sd_v = mesh["ref_depth"].astype(np.float64)
            nrm = mesh["normals"].astype(np.float64)
            e_v = mesh["edge"].astype(np.float64)
            e0 = _edge_ok_vectorized(S[:, 0], S[:, 1], sd_v, nrm, e_v,
                                     args.depth_tol, args.normal_tol, args.edge_thresh)
            e1 = _edge_ok_vectorized(S[:, 1], S[:, 2], sd_v, nrm, e_v,
                                     args.depth_tol, args.normal_tol, args.edge_thresh)
            e2 = _edge_ok_vectorized(S[:, 2], S[:, 0], sd_v, nrm, e_v,
                                     args.depth_tol, args.normal_tol, args.edge_thresh)
            mesh["faces"] = S[e0 & e1 & e2].astype(np.int32)
            mesh["boundary_edges"] = boundary_edges(mesh["faces"])

    # Confidence is evaluated on the SAME subsampled grid as build_reference_mesh.
    st = args.stride
    valid = surface_mask(surf["surface_depth"][::st, ::st],
                         surf["accum_opacity"][::st, ::st], args.max_r, args.a_thresh)
    flat = valid.reshape(-1)
    t_scalar = surf["t"].mean(dim=0)[::st, ::st].reshape(-1)[flat]          # (M,)
    c_photo = photometric_confidence(
        surf["I_hat"][:, ::st, ::st],
        ref_cam.original_image.to("cuda")[:, ::st, ::st]).reshape(-1)[flat]
    mesh["confidence"] = confidence(t_scalar, c_view, c_photo).detach().cpu().numpy()
    mesh["support_count"] = support.cpu().numpy()
    mesh["visible"] = (mesh["support_count"] >= args.min_support).astype(np.uint8)

    # Per-vertex silhouette support (boundary verts), for component retention.
    bv = mesh["boundary_edges"]
    sil_full = np.ones((mesh["verts"].shape[0],), dtype=np.float32)
    if bv.size:
        bv_idx = np.unique(bv.ravel())
        bv_verts = torch.from_numpy(mesh["verts"][bv_idx]).cuda()
        sil = silhouette_support(
            bv_verts, cams, [s["edge"] for s in surfaces],
            [s["accum_opacity"] for s in surfaces])
        sil_full[bv_idx] = sil.detach().cpu().numpy()
    mesh["silhouette"] = sil_full

    print("support_count range [{}, {}]  mean={:.2f}".format(
        int(mesh["support_count"].min()), int(mesh["support_count"].max()),
        float(mesh["support_count"].mean())))

    mesh = prune_by_support(mesh, args.min_support)
    stats = mesh_stats(mesh)
    print("Pruned mesh: {} verts, {} faces".format(mesh["verts"].shape[0], mesh["faces"].shape[0]))
    print("stats:", json.dumps(stats, indent=2))

    # Persist.
    out = os.path.join(dataset.model_path, "mesh")
    os.makedirs(out, exist_ok=True)
    write_ply(os.path.join(out, "reference_mesh.ply"), mesh)
    write_obj(os.path.join(out, "reference_mesh.obj"), mesh)
    save_npz(os.path.join(out, "reference_mesh.npz"), mesh)
    with open(os.path.join(out, "mesh_stats.json"), "w") as f:
        json.dump({"reference_idx": ref_idx, "stats": stats}, f, indent=2)

    # Per-view surface visualisations.
    view_root = os.path.join(out, "views")
    for i, s in enumerate(surfaces):
        d = os.path.join(view_root, "{:03d}".format(i))
        os.makedirs(d, exist_ok=True)
        _save_map(s["surface_depth"], os.path.join(d, "surface_depth.png"))
        _save_map(s["accum_opacity"], os.path.join(d, "accum_opacity.png"), cmap="gray")
        _save_normal(s["normal"], os.path.join(d, "normal.png"))
        _save_map(s["edge"], os.path.join(d, "edge.png"), cmap="gray")
    print("Saved mesh + stats + views ->", out)


if __name__ == "__main__":
    parser = ArgumentParser(description="Offline mesh extraction")
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    parser.add_argument("--load_iteration", type=int, default=60000)
    parser.add_argument("--reference_idx", type=int, default=-1)
    parser.add_argument("--a_thresh", type=float, default=0.7)
    parser.add_argument("--max_r", type=float, default=50.0)
    parser.add_argument("--depth_tol", type=float, default=0.05)
    parser.add_argument("--normal_tol", type=float, default=0.52)
    parser.add_argument("--edge_thresh", type=float, default=0.5)
    parser.add_argument("--stride", type=int, default=2,
                        help="subsample the valid pixel grid (stride=2 -> 1/4 verts)")
    parser.add_argument("--support_tol", type=float, default=0.05)
    parser.add_argument("--min_support", type=int, default=2)
    parser.add_argument("--fuse", type=str, default="median",
                        choices=["none", "median", "mean"],
                        help="multi-view surface fusion mode (none = default behaviour)")
    parser.add_argument("--fusion_tol", type=float, default=0.05)
    parser.add_argument("--fusion_a_thresh", type=float, default=0.7)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)
    extract(lp.extract(args), pp.extract(args), args)
