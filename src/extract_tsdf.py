"""TSDF extraction: extract a mesh by multi-view TSDF fusion of the rendered depth.

Offline alternative to ``extract_mtetra.py``.  Instead of iso-surfacing the raw
Gaussian *density* field (whose iso-surface drifts from the rendered surface —
the source of the overlay misalignment), this fuses the per-view median
``surface_depth`` maps into a truncated signed-distance field and extracts its
zero crossing with marching tetrahedra.

Because the mesh is the zero level set of the *fused rendered depth*, it is the
rendered surface by construction, so the overlay is aligned and the held-out
reprojection error collapses.  This is the same idea 2DGS uses for mesh
extraction, applied to our flattened Gaussians whose depth is now
crisp.

Outputs ``reference_mesh.{ply,obj,npz}`` with the full 11-key schema, so
refine / train_mesh / evaluate / vis are unchanged.

Usage:
  python extract_tsdf.py --source_path <src> --model_path <m> --load_iteration 65000 \
      [--res 512] [--trunc 0.16]
"""

import os
import sys
import json

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
from mesh.mtetra import marching_tetrahedra, compute_normals
from mesh.tsdf import fuse_tsdf


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

    # --- per-view surface depth (the input to the TSDF) ---------------------
    surfaces = []
    for cam in cams:
        surfaces.append(render_surface(cam, gaussians, pipe, bg, water_model))

    mean_center = torch.stack([c.camera_center.cpu() for c in cams]).float().mean(dim=0)
    ref_cam = _nearest_cam(scene, mean_center, args.reference_idx)
    ref_idx = cams.index(ref_cam)
    print("Reference view: train[{:d}]".format(ref_idx))

    # --- grid (bbox from the Gaussian surface positions) --------------------
    means = gaussians.get_xyz.detach()
    bbox_min = means.min(dim=0).values.cpu().numpy()
    bbox_max = means.max(dim=0).values.cpu().numpy()
    extent = bbox_max - bbox_min
    spacing_arg = getattr(args, "spacing", None)
    if spacing_arg is not None:
        spacing = np.full(3, spacing_arg, dtype=np.float64)
    else:
        spacing = np.full(3, float(extent.max()) / args.res, dtype=np.float64)
    trunc = getattr(args, "trunc", None)
    if trunc is None:
        trunc = 3.0 * float(spacing[0])
    pad = max(3.0 * trunc, args.pad_frac * float(extent.max()))
    origin = bbox_min - pad
    grid_extent = extent + 2.0 * pad
    shape = tuple(int(s) for s in (np.ceil(grid_extent / spacing) + 2).astype(np.int64))
    print("bbox {}  extent {}  pad {:.3f}".format(bbox_min.round(2), extent.round(2), pad))
    print("grid: origin {} spacing {:.4f} shape {}  trunc {:.3f}".format(
        origin.round(2), spacing[0], shape, trunc))

    # --- TSDF fusion + zero-crossing iso-surface ----------------------------
    tsdf, wsum = fuse_tsdf(cams, [s["surface_depth"] for s in surfaces],
                           origin, spacing, shape, trunc=trunc, max_r=args.max_r,
                           accum_maps=[s["accum_opacity"] for s in surfaces],
                           a_thresh=args.a_thresh)
    w = wsum.cpu().numpy()
    covered = float((w > 0).mean() * 100)
    kept = float((w >= args.min_weight).mean() * 100)
    print("TSDF fused: mean weight {:.1f} views/voxel, {:.1f}% observed, {:.1f}% >= min_weight".format(
        float(w[w > 0].mean()) if (w > 0).any() else 0.0, covered, kept))
    if (w > 0).any():
        hist = np.bincount(w[w > 0].astype(np.int64).clip(0, 32), minlength=33)
        print("weight histogram (obs cells):",
              " ".join("{:d}:{:.1f}%".format(i, 100.0 * h / (w > 0).sum())
                       for i, h in enumerate(hist) if h > 0))

    # Under-observed cells are single-view noise / floaters — treat them as free
    # space so they cannot form spurious surfaces.  Only well-agreed regions
    # (>= min_weight views) can cross zero.
    tsdf = torch.where(wsum >= args.min_weight, tsdf,
                       torch.full_like(tsdf, trunc))

    # inside = behind the surface (tsdf < 0) -> extract -tsdf == 0.
    vol = (-tsdf).cpu().numpy()
    verts, faces = marching_tetrahedra(vol, spacing, origin, tau=0.0)
    print("MT: {} verts, {} faces (pre-clean)".format(verts.shape[0], faces.shape[0]))

    if faces.size == 0:
        raise RuntimeError("TSDF marching tetrahedra produced an empty surface; "
                           "check --res / --trunc (surface may be clipped by bbox)")

    # compute_normals(-tsdf) -> outward = +grad(tsdf) = free space direction.
    normals, faces = compute_normals(verts, faces, vol, origin, spacing)

    # --- surface channels (support / confidence / silhouette / uv) ----------
    verts_t = torch.from_numpy(verts).cuda()
    u, v, z = project_points(verts_t, ref_cam)
    uv = torch.stack([u, v], dim=1).cpu().numpy()          # (V, 2)
    ref_depth = z.cpu().numpy()                            # (V,)
    surf = surfaces[ref_idx]
    edge = _sample_at(surf["edge"], u, v).cpu().numpy()    # (V,)

    support, c_view = multiview_support(
        verts_t, ref_cam, cams, [s["surface_depth"] for s in surfaces],
        tol=args.support_tol, max_r=args.max_r)
    t_map = surf["t"].mean(dim=0)
    t_scalar = _sample_at(t_map, u, v)
    c_photo_map = photometric_confidence(surf["I_hat"], ref_cam.original_image.to("cuda"))
    c_photo = _sample_at(c_photo_map, u, v)
    conf = confidence(t_scalar, c_view, c_photo).detach().cpu().numpy()

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
        json.dump({"reference_idx": ref_idx, "trunc": float(trunc), "stats": stats}, f, indent=2)
    print("Saved TSDF mesh + stats ->", out)


if __name__ == "__main__":
    parser = ArgumentParser(description="TSDF-fusion mesh extraction")
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    parser.add_argument("--load_iteration", type=int, default=60000)
    parser.add_argument("--reference_idx", type=int, default=-1)
    # grid
    parser.add_argument("--res", type=int, default=512,
                        help="target cells on the longest axis (isotropic spacing)")
    parser.add_argument("--spacing", type=float, default=None,
                        help="explicit isotropic spacing (overrides --res)")
    parser.add_argument("--pad_frac", type=float, default=0.02,
                        help="bbox padding as a fraction of the longest extent")
    # TSDF
    parser.add_argument("--trunc", type=float, default=None,
                        help="truncation distance (default: 3 * spacing)")
    parser.add_argument("--min_weight", type=float, default=3.0,
                        help="drop cells seen by fewer than this many views "
                             "(single-view noise / floaters)")
    parser.add_argument("--a_thresh", type=float, default=0.5,
                        help="skip depth pixels with accumulated opacity below this")
    # support / cleanup
    parser.add_argument("--support_tol", type=float, default=0.05)
    parser.add_argument("--max_r", type=float, default=50.0)
    parser.add_argument("--min_support", type=int, default=1)
    parser.add_argument("--min_component_faces", type=int, default=100,
                        help="drop connected components smaller than this")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)
    extract(lp.extract(args), pp.extract(args), args)
