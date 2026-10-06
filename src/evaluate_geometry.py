"""Held-out-view geometry evaluation.

Loads the refined mesh and evaluates it against *held-out* (test) views:
(1) reprojection depth consistency (mesh -> rendered median surface depth),
(2) boundary edge agreement (mesh boundary vs half-Gaussian edge response E_v),
(3) mesh quality statistics (open boundary / non-manifold / self-intersection).

If ``--gt_points`` (a Nx3 .npy/.txt) is supplied, also reports the MOUD-style
point-cloud distance (mesh <-> held-out lidar) via scipy cKDTree.
"""

import os
import sys
import json

import numpy as np
import torch
from argparse import ArgumentParser
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from medium import WaterImageModel
from utils.general_utils import safe_state
from mesh import (load_npz, render_surface, mesh_stats,
                  reproject_consistency_heldout, boundary_edge_agreement,
                  point_cloud_distance, forward_view_stability, normal_consistency)


def _load_gt_points(path):
    if path is None:
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    return np.loadtxt(path)


def evaluate(dataset, pipe, args):
    mesh = load_npz(args.mesh)
    stats = mesh_stats(mesh)
    print("mesh stats:", json.dumps(stats))

    # In --stability mode, evaluate the mesh-reprojection checkpoint (s4_<tag>_* sidecars);
    # otherwise the extraction baseline (projective/ sidecars). Output filenames differ
    # so the extraction geometry_eval.json is preserved as the baseline row.
    if args.stability:
        sidecar = os.path.join(dataset.model_path, "mesh", "s4_{}_projective.pth".format(args.tag))
        ckpt = os.path.join(dataset.model_path, "mesh", "s4_{}_water_model.pth".format(args.tag))
        load_iteration = args.save_iteration
        out_name = "s4_{}_geometry_eval.json".format(args.tag)
    else:
        sidecar = os.path.join(dataset.model_path, "projective", "projective.pth")
        ckpt = os.path.join(dataset.model_path, "projective", "water_model.pth")
        load_iteration = args.load_iteration
        out_name = "geometry_eval.json"

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=False)
    if os.path.exists(sidecar):
        gaussians.load_projective(sidecar)
    water_model = WaterImageModel().cuda()
    water_model.load_state_dict(torch.load(ckpt))
    water_model.eval()

    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    test_cams = scene.getTestCameras()
    if not test_cams:
        print("No held-out (test) cameras; falling back to train cameras.")
        test_cams = scene.getTrainCameras()

    verts = torch.from_numpy(mesh["verts"]).cuda()
    bv_idx = np.unique(np.asarray(mesh["boundary_edges"]).ravel())
    bv = torch.from_numpy(mesh["verts"][bv_idx]).cuda()

    surfs = []
    reproj_errs, boundary_ratios = [], []
    for cam in tqdm(test_cams, desc="Held-out evaluation"):
        surf = render_surface(cam, gaussians, pipe, bg, water_model)
        surfs.append(surf)
        r = reproject_consistency_heldout(verts, cam, surf["surface_depth"], max_r=args.max_r)
        b = boundary_edge_agreement(bv, cam, surf["edge"])
        if r["n_verts"]:
            reproj_errs.append(r["mean_rel_err"])
        if b["n_verts"]:
            boundary_ratios.append(b["ratio"])

    report = {
        "mesh_stats": stats,
        "heldout_reproj_mean_rel_err": float(np.nanmean(reproj_errs)) if reproj_errs else float("nan"),
        "heldout_reproj_n_views": int(len(reproj_errs)),
        "heldout_boundary_edge_ratio_mean": float(np.nanmean(boundary_ratios)) if boundary_ratios else float("nan"),
    }

    if args.stability:
        depths = [s["depth"] for s in surfs]
        report["forward_view_stability"] = forward_view_stability(test_cams, depths, max_r=args.max_r)
        report["normal_consistency"] = normal_consistency(test_cams, depths, max_r=args.max_r)

    gt = _load_gt_points(args.gt_points)
    if gt is not None:
        report["point_cloud_distance"] = point_cloud_distance(mesh, gt, args.max_dist)
    else:
        print("No --gt_points provided; skipping MOUD point-cloud distance (latent).")

    print(json.dumps(report, indent=2))
    out = os.path.join(dataset.model_path, "mesh", out_name)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print("Saved ->", out)


if __name__ == "__main__":
    parser = ArgumentParser(description="Held-out geometry evaluation")
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    parser.add_argument("--load_iteration", type=int, default=60000)
    parser.add_argument("--mesh", type=str, default=None)
    parser.add_argument("--max_r", type=float, default=50.0)
    parser.add_argument("--gt_points", type=str, default=None)
    parser.add_argument("--max_dist", type=float, default=0.05)
    parser.add_argument("--stability", action="store_true",
                        help="evaluate the mesh-reprojection checkpoint (s4_<tag>_*) incl. forward-view stability")
    parser.add_argument("--tag", type=str, default="full")
    parser.add_argument("--save_iteration", type=int, default=70000)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    # get_combined_args drops cmdline args whose value is None (e.g. --mesh /
    # --gt_points), so recover them with defaults explicitly.
    if getattr(args, "mesh", None) is None:
        args.mesh = os.path.join(lp.extract(args).model_path, "mesh", "refined_mesh.npz")
    if not hasattr(args, "gt_points"):
        args.gt_points = None
    if not hasattr(args, "max_dist"):
        args.max_dist = 0.05
    if not hasattr(args, "stability"):
        args.stability = False
    if not hasattr(args, "tag"):
        args.tag = "full"
    if not hasattr(args, "save_iteration"):
        args.save_iteration = 70000
    safe_state(args.quiet)
    evaluate(lp.extract(args), pp.extract(args), args)
