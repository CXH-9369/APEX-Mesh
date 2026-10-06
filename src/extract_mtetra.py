"""Marching-tetrahedra extraction: extract a mesh from the trained Gaussian *density* field.

Offline, mirrors the assembly of ``extract_mesh.py`` (loads the projective
checkpoint, renders the per-view surface channels, assigns multi-view support /
confidence / silhouette) but replaces the reference-view 2D-Delaunay depth mesh
with a marching-tetrahedra iso-surface of

    d(x) = sum_i  alpha_i * exp(-0.5 * ||R_i^T (x - mu_i) / s_i||^2)

(Gaussian Opacity Fields style).  Outputs ``reference_mesh.{ply,obj,npz}`` with
the full 11-key schema, so refine / train_mesh / evaluate / vis are unchanged.

Usage:
  python extract_mtetra.py --source_path <src> --model_path <m> --load_iteration 60000 \
      [--res 384] [--alpha_min 0.05] [--s_cut 0.3] [--tau 0.0] [--percentile 50]
"""

import os
import sys
import json
import math

import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from medium import WaterImageModel
from utils.general_utils import safe_state
from mesh import (render_surface, multiview_support, silhouette_support,
                  photometric_confidence, confidence, boundary_edges,
                  remove_small_components, mesh_stats, write_ply, write_obj,
                  save_npz)
from mesh.extract import project_points, _sample_at
from mesh.mtetra import (splat_density_volume, marching_tetrahedra,
                         estimate_threshold, compute_normals)


def _nearest_cam(scene, mean_center, idx):
    cams = scene.getTrainCameras()
    if idx >= 0:
        return cams[idx]
    centers = torch.stack([c.camera_center.cpu() for c in cams]).float()
    d = (centers - mean_center.cpu()).norm(dim=1)
    return cams[int(d.argmin().item())]


def _prune_by_support(mesh, min_support):
    keep = mesh["support_count"] >= min_support
    n_old = keep.shape[0]
    remap = np.full(n_old, -1, dtype=np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    new = dict(mesh)
    for k in list(new.keys()):
        if k in ("faces", "boundary_edges"):
            continue
        arr = np.asarray(new[k])
        if arr.ndim >= 1 and arr.shape[0] == n_old:
            new[k] = arr[keep]
    faces = mesh["faces"]
    keep_f = faces.size and np.all(remap[faces] >= 0, axis=1)
    new["faces"] = remap[faces[keep_f]] if faces.size else np.zeros((0, 3), dtype=np.int32)
    new["boundary_edges"] = boundary_edges(new["faces"])
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

    # --- Gaussian density-field primitives ---------------------------------
    means = gaussians.get_xyz.detach()                       # (N, 3)
    scales = gaussians.get_scaling.detach()                  # (N, 3) exp
    rotations = gaussians.get_rotation.detach()              # (N, 4) quat
    opacities = gaussians.get_opacity.detach()               # (N, 2) sigmoid
    alpha = opacities.mean(dim=1)                            # (N,) half-Gaussian peak
    print("Gaussians: {} (alpha>=0: {})".format(means.shape[0], int((alpha > 0).sum())))

    # --- grid --------------------------------------------------------------
    bbox_min = means.min(dim=0).values.cpu().numpy()
    bbox_max = means.max(dim=0).values.cpu().numpy()
    extent = bbox_max - bbox_min
    pad = max(3.0 * args.s_cut, args.pad_frac * float(extent.max()))
    origin = bbox_min - pad
    grid_extent = extent + 2.0 * pad
    spacing_arg = getattr(args, "spacing", None)
    if spacing_arg is not None:
        spacing = np.full(3, spacing_arg, dtype=np.float64)
    else:
        spacing = np.full(3, grid_extent.max() / args.res, dtype=np.float64)
    shape = tuple(int(s) for s in (np.ceil(grid_extent / spacing) + 2).astype(np.int64))
    print("bbox {}  extent {}  pad {:.3f}".format(bbox_min.round(2), extent.round(2), pad))
    print("grid: origin {} spacing {:.4f} shape {}".format(
        origin.round(2), spacing[0], shape))

    # --- density field + threshold + iso-surface ---------------------------
    volume = splat_density_volume(means, rotations, scales, alpha, origin, spacing,
                                  shape, alpha_min=args.alpha_min, s_cut=args.s_cut)
    vol_cpu = volume.cpu().numpy()
    nz = float((vol_cpu > 0).mean() * 100)
    print("density field: max={:.3f}  non-zero {:.1f}% of {} cells".format(
        float(volume.max()), nz, vol_cpu.size))

    tau_arg = getattr(args, "tau", None)
    if tau_arg is None:
        tau = estimate_threshold(vol_cpu, means.cpu().numpy(), origin, spacing,
                                 percentile=args.percentile)
        print("estimated tau = {:.4f} (percentile {})".format(tau, args.percentile))
    else:
        tau = tau_arg
        print("tau = {:.4f} (explicit)".format(tau))

    verts, faces = marching_tetrahedra(vol_cpu, spacing, origin, tau)
    print("MT: {} verts, {} faces (pre-clean)".format(verts.shape[0], faces.shape[0]))

    if faces.size == 0:
        raise RuntimeError("marching tetrahedra produced an empty surface; "
                           "try a lower --tau or --percentile")

    normals, faces = compute_normals(verts, faces, vol_cpu, origin, spacing)

    # --- surface channels (for support / confidence / silhouette / uv) -----
    surfaces = []
    for cam in cams:
        surfaces.append(render_surface(cam, gaussians, pipe, bg, water_model))
    mean_center = torch.stack([c.camera_center.cpu() for c in cams]).float().mean(dim=0)
    ref_cam = _nearest_cam(scene, mean_center, args.reference_idx)
    ref_idx = cams.index(ref_cam)
    surf = surfaces[ref_idx]
    print("Reference view: train[{:d}]".format(ref_idx))

    verts_t = torch.from_numpy(verts).cuda()
    u, v, z = project_points(verts_t, ref_cam)
    uv = torch.stack([u, v], dim=1).cpu().numpy()          # (V, 2)
    ref_depth = z.cpu().numpy()                            # (V,)
    edge = _sample_at(surf["edge"], u, v).cpu().numpy()    # (V,)

    # multi-view support + confidence
    support, c_view = multiview_support(
        verts_t, ref_cam, cams, [s["surface_depth"] for s in surfaces],
        tol=args.support_tol, max_r=args.max_r)
    t_map = surf["t"].mean(dim=0)
    t_scalar = _sample_at(t_map, u, v)
    c_photo_map = photometric_confidence(surf["I_hat"], ref_cam.original_image.to("cuda"))
    c_photo = _sample_at(c_photo_map, u, v)
    conf = confidence(t_scalar, c_view, c_photo).detach().cpu().numpy()

    # silhouette support on boundary verts
    b = boundary_edges(faces)
    sil_full = np.ones((verts.shape[0],), dtype=np.float32)
    if b.size:
        bv_idx = np.unique(b.ravel())
        bv_verts = verts_t[bv_idx]
        sil = silhouette_support(
            bv_verts, cams, [s["edge"] for s in surfaces],
            [s["accum_opacity"] for s in surfaces])
        sil_full[bv_idx] = sil.detach().cpu().numpy()

    mesh = {
        "verts": verts.astype(np.float32),
        "normals": normals,
        "faces": faces.astype(np.int32),
        "confidence": conf.astype(np.float32),
        "support_count": support.cpu().numpy().astype(np.int32),
        "visible": (support.cpu().numpy() >= args.min_support).astype(np.uint8),
        "boundary_edges": b.astype(np.int32),
        "uv": uv.astype(np.float32),
        "ref_depth": ref_depth.astype(np.float32),
        "edge": edge.astype(np.float32),
        "silhouette": sil_full,
    }

    # support pruning first, then small-component cleanup: pruning can split a
    # component into isolated triangles, so those fragments must be dropped after.
    # NB: the MT surface is a closed shell, so small components are floaters to
    # drop by size — the silhouette exception (designed for the thin open
    # Delaunay surfaces) would keep them all here, so it is disabled.
    mesh = _prune_by_support(mesh, args.min_support)
    mesh = remove_small_components(
        mesh, min_faces=args.min_component_faces, silhouette=None)

    stats = mesh_stats(mesh)
    print("Pruned mesh: {} verts, {} faces".format(mesh["verts"].shape[0], mesh["faces"].shape[0]))
    print("stats:", json.dumps(stats, indent=2))

    out = os.path.join(dataset.model_path, "mesh")
    os.makedirs(out, exist_ok=True)
    write_ply(os.path.join(out, "reference_mesh.ply"), mesh)
    write_obj(os.path.join(out, "reference_mesh.obj"), mesh)
    save_npz(os.path.join(out, "reference_mesh.npz"), mesh)
    with open(os.path.join(out, "mesh_stats.json"), "w") as f:
        json.dump({"reference_idx": ref_idx, "tau": float(tau), "stats": stats}, f, indent=2)
    print("Saved MT mesh + stats ->", out)


if __name__ == "__main__":
    parser = ArgumentParser(description="Marching-tetrahedra mesh extraction")
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    parser.add_argument("--load_iteration", type=int, default=60000)
    parser.add_argument("--reference_idx", type=int, default=-1)
    # density field
    parser.add_argument("--alpha_min", type=float, default=0.05,
                        help="drop Gaussians with alpha below this (fog)")
    parser.add_argument("--s_cut", type=float, default=0.3,
                        help="drop Gaussians with max scale above this (background)")
    # grid
    parser.add_argument("--res", type=int, default=384,
                        help="target cells on the longest axis (isotropic spacing)")
    parser.add_argument("--spacing", type=float, default=None,
                        help="explicit isotropic spacing (overrides --res)")
    parser.add_argument("--pad_frac", type=float, default=0.02,
                        help="bbox padding as a fraction of the longest extent")
    # threshold
    parser.add_argument("--tau", type=float, default=None,
                        help="iso-value (default: auto percentile estimate)")
    parser.add_argument("--percentile", type=float, default=50.0,
                        help="percentile of density-at-centers used for auto tau")
    # support / cleanup
    parser.add_argument("--support_tol", type=float, default=0.05)
    parser.add_argument("--max_r", type=float, default=50.0)
    parser.add_argument("--min_support", type=int, default=1)
    parser.add_argument("--min_component_faces", type=int, default=100,
                        help="drop connected components smaller than this")
    parser.add_argument("--silhouette_thresh", type=float, default=0.2)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)
    extract(lp.extract(args), pp.extract(args), args)
