"""Implicit extraction: extract a mesh from flattened Gaussians via an implicit MLS surface.

Surface extraction without Poisson.  After the flattening stage the
Gaussians are thin, surface-aligned discs; their centres ``mu_i`` lie on the
surface and their thinnest axes ``n_i`` are (up to sign) the surface normals.
We orient ``n_i`` toward the nearest camera and build

    f(x) = sum_i w_i(x) n_i . (x - mu_i) / sum_i w_i(x),   w_i = alpha_i e^{-r^2/2sigma^2}

whose zero set is the surface, extracted with marching tetrahedra.

Because ``f`` is built from the consistent Gaussian centres (not the
view-dependent median depth), the mesh is a coherent 3D surface and the overlay
is aligned.  Outputs ``reference_mesh.{ply,obj,npz}`` with the full 11-key
schema so refine / train_mesh / evaluate / vis are unchanged.

Usage:
  python extract_implicit.py --source_path <src> --model_path <m> --load_iteration 65000 \
      [--res 512] [--sigma 0.05] [--alpha_min 0.05] [--w_thresh 0.5]
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
from mesh.implicit import implicit_surface_field, orient_normals, gaussian_normals


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

    # --- flattened-Gaussian surface primitives ------------------------------
    means = gaussians.get_xyz.detach()                       # (N, 3) on-surface
    alphas = gaussians.get_opacity.detach().mean(dim=1)      # (N,)
    normals = gaussian_normals(gaussians).detach()           # (N, 3) thinnest axis
    cam_centers = torch.stack([c.camera_center.cpu() for c in cams]).float()
    normals = orient_normals(means, normals, cam_centers)
    print("Gaussians: {} (alpha>={}: {})".format(
        means.shape[0], args.alpha_min, int((alphas >= args.alpha_min).sum())))

    # --- grid (bbox from the *kept* Gaussian surface positions) ------------
    # Using only alpha-filtered Gaussians avoids the far-away fog/background
    # outliers stretching the bbox, so the grid hugs the reef and the same
    # --res yields a much finer effective spacing (the field is otherwise
    # ~98% empty cells).
    keep_bb = alphas >= args.alpha_min
    bbox_min = means[keep_bb].min(dim=0).values.cpu().numpy()
    bbox_max = means[keep_bb].max(dim=0).values.cpu().numpy()
    extent = bbox_max - bbox_min
    spacing_arg = getattr(args, "spacing", None)
    if spacing_arg is not None:
        spacing = np.full(3, spacing_arg, dtype=np.float64)
    else:
        spacing = np.full(3, float(extent.max()) / args.res, dtype=np.float64)
    sigma = getattr(args, "sigma", None)
    if sigma is None:
        sigma = 1.5 * float(spacing[0])
    pad = max(3.0 * sigma, args.pad_frac * float(extent.max()))
    origin = bbox_min - pad
    grid_extent = extent + 2.0 * pad
    shape = tuple(int(s) for s in (np.ceil(grid_extent / spacing) + 2).astype(np.int64))
    print("bbox {}  extent {}  pad {:.3f}".format(bbox_min.round(2), extent.round(2), pad))
    sigma_n = getattr(args, "sigma_n", None)
    if sigma_n is None:
        sigma_n = 0.6 * float(spacing[0])          # anisotropic default (~sharp)
    print("grid: origin {} spacing {:.4f} shape {}  sigma {:.3f}  sigma_n {:.4f}".format(
        origin.round(2), spacing[0], shape, sigma, sigma_n))

    # --- implicit field + zero-set iso-surface ------------------------------
    num, den = implicit_surface_field(
        means, normals, alphas, origin, spacing, shape, sigma,
        alpha_min=args.alpha_min, sigma_n=sigma_n)
    # Report support before consuming the weight grid (bool sum avoids a
    # full-size float32 intermediate).
    frac = float((den >= args.w_thresh).sum().item()) / den.numel()
    print("implicit field: {:.1f}% cells with weight >= {:.1f}".format(
        100.0 * frac, args.w_thresh))

    # f = num / den, then mask weakly-supported cells to "outside" (f=1).
    # In-place clamp + aggressive frees keep the peak ~28 GB at res 2048, where
    # each float32 grid is ~9.5 GB (num+den alone are ~19 GB).
    den.clamp_min_(1e-6)
    f = num / den
    del num
    mask_out = den < args.w_thresh
    del den
    f[mask_out] = 1.0
    del mask_out

    # inside = behind the surface (f < 0) -> extract -f == 0.
    vol = (-f).cpu().numpy()
    del f
    torch.cuda.empty_cache()
    verts, faces = marching_tetrahedra(vol, spacing, origin, tau=0.0)
    print("MT: {} verts, {} faces (pre-clean)".format(verts.shape[0], faces.shape[0]))

    if faces.size == 0:
        raise RuntimeError("implicit surface produced an empty mesh; "
                           "check --sigma / --alpha_min / --w_thresh")

    # compute_normals(-f) -> outward = +grad(f) = free-space direction.
    normals, faces = compute_normals(verts, faces, vol, origin, spacing)
    del vol

    # --- surface channels (support / confidence / silhouette / uv) ----------
    # no_grad: extraction is pure inference; keeping autograd graphs for 46
    # renders would pin GBs of host-side node state for no benefit.
    surfaces = []
    with torch.no_grad():
        for cam in cams:
            surfaces.append(render_surface(cam, gaussians, pipe, bg, water_model))
    mean_center = torch.stack([c.camera_center.cpu() for c in cams]).float().mean(dim=0)
    ref_cam = _nearest_cam(scene, mean_center, args.reference_idx)
    ref_idx = cams.index(ref_cam)
    surf = surfaces[ref_idx]
    print("Reference view: train[{:d}]".format(ref_idx))

    verts_t = torch.from_numpy(verts).cuda()

    # --- multi-view support over ALL pre-clean verts (chunked on the GPU) ---
    support, c_view = multiview_support(
        verts_t, ref_cam, cams, [s["surface_depth"] for s in surfaces],
        tol=args.support_tol, max_r=args.max_r)

    # --- prune by support FIRST (on the GPU) so the per-vertex attribute
    #     buffers (uv/ref_depth/edge/conf/sil) and the boundary computation
    #     only touch the surviving vertices.  The anisotropic kernel recovers
    #     far sharper creases and roughly doubles the pre-clean vertex count
    #     (32.7M -> 72.6M on AT1); holding every float buffer at 72.6M — and
    #     running boundary_edges over the 146M pre-clean faces — OOM'd the
    #     31 GB host. ---
    keep = support >= args.min_support                       # (M,) GPU bool
    keep_np = keep.cpu().numpy()
    remap = np.full(verts.shape[0], -1, dtype=np.int64)
    remap[keep_np] = np.arange(int(keep_np.sum()))
    faces = faces.astype(np.int32)                           # final dtype now (drop int64)
    # column-wise membership test avoids a full (F,3) int64 remap[faces] temp.
    keep_f = ((remap[faces[:, 0]] >= 0) & (remap[faces[:, 1]] >= 0) &
              (remap[faces[:, 2]] >= 0))
    faces = remap[faces[keep_f]].astype(np.int32)
    verts = verts[keep_np]
    normals = normals[keep_np]
    verts_t = verts_t[keep]
    support = support[keep]
    c_view = c_view[keep]
    del keep, keep_np, remap, keep_f

    # --- per-vertex attributes only for the surviving vertices ---
    u, v, z = project_points(verts_t, ref_cam)
    uv = torch.stack([u, v], dim=1).cpu().numpy()
    ref_depth = z.cpu().numpy()
    edge = _sample_at(surf["edge"], u, v).cpu().numpy()
    t_map = surf["t"].mean(dim=0)
    t_scalar = _sample_at(t_map, u, v)
    c_photo_map = photometric_confidence(surf["I_hat"], ref_cam.original_image.to("cuda"))
    c_photo = _sample_at(c_photo_map, u, v)
    conf = confidence(t_scalar, c_view, c_photo).detach().cpu().numpy()

    # --- boundary + silhouette on the pruned mesh ---
    b = boundary_edges(faces)
    sil_full = np.ones((verts.shape[0],), dtype=np.float32)
    if b.size:
        bv_idx = np.unique(b.ravel())
        sil = silhouette_support(
            verts_t[bv_idx], cams, [s["edge"] for s in surfaces],
            [s["accum_opacity"] for s in surfaces])
        sil_full[bv_idx] = sil.detach().cpu().numpy()

    mesh = {
        "verts": verts,
        "normals": normals,
        "faces": faces,
        "confidence": conf.astype(np.float32),
        "support_count": support.cpu().numpy().astype(np.int32),
        "visible": (support.cpu().numpy() >= args.min_support).astype(np.uint8),
        "boundary_edges": b.astype(np.int32),
        "uv": uv.astype(np.float32),
        "ref_depth": ref_depth.astype(np.float32),
        "edge": edge.astype(np.float32),
        "silhouette": sil_full,
    }

    # support pruning already done above; drop remaining floaters by component
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
        json.dump({"reference_idx": ref_idx, "sigma": float(sigma),
                   "sigma_n": float(sigma_n), "stats": stats}, f, indent=2)
    print("Saved implicit mesh + stats ->", out)


if __name__ == "__main__":
    parser = ArgumentParser(description="Implicit-surface mesh extraction")
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    parser.add_argument("--load_iteration", type=int, default=60000)
    parser.add_argument("--reference_idx", type=int, default=-1)
    # field
    parser.add_argument("--alpha_min", type=float, default=0.05,
                        help="drop Gaussians with alpha below this (fog)")
    parser.add_argument("--sigma", type=float, default=None,
                        help="MLS kernel radius (default: 1.5 * spacing)")
    parser.add_argument("--w_thresh", type=float, default=0.5,
                        help="drop field cells with accumulated weight below this")
    parser.add_argument("--sigma_n", type=float, default=None,
                        help="anisotropic normal-direction kernel radius (default: 0.6 * "
                             "spacing; pass a large value for the isotropic fallback)")
    # grid
    parser.add_argument("--res", type=int, default=512,
                        help="target cells on the longest axis (isotropic spacing)")
    parser.add_argument("--spacing", type=float, default=None,
                        help="explicit isotropic spacing (overrides --res)")
    parser.add_argument("--pad_frac", type=float, default=0.02,
                        help="bbox padding as a fraction of the longest extent")
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
