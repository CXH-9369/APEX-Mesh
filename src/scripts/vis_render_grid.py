#!/usr/bin/env python
"""Assemble paper rendering figures from render_projective.py exports.

Reads ``<model>/projective/ours_<iter>/test/{raw,I_hat,depth}/`` and stacks
selected views as a 3-row grid (raw GT | reconstruction I_hat | optical depth),
plus a residual strip. Saves under ``<model>/projective/ours_<iter>/grid/``.

Usage:
  python vis_render_grid.py AT1 [AT2 ...] [--iter 60000] [--n 6]
"""
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO = "/home/cxh/APEX/apex-mesh"
SRC_ROOT = "/home/cxh/APEX/MOUD"
MODEL_ROOT = os.path.join(REPO, "outputs", "moud")


def _pngs(d):
    return sorted(glob.glob(os.path.join(d, "*.png")))


def grid_one(scene, iteration, n):
    root = os.path.join(MODEL_ROOT, scene, "projective", "ours_{}".format(iteration), "test")
    for ch in ("raw", "I_hat", "depth", "residual"):
        if not os.path.isdir(os.path.join(root, ch)):
            print("  [{}] missing {} (run render_projective.py first)".format(scene, ch))
            return
    raws = _pngs(os.path.join(root, "raw"))
    sel = raws[:n]

    def load(p):
        return np.asarray(Image.open(p).convert("RGB"))

    rows = ["raw", "I_hat", "depth"]
    fig, axes = plt.subplots(len(rows), len(sel), figsize=(3.2 * len(sel), 3.0 * len(rows)))
    if len(sel) == 1:
        axes = axes[:, None]
    for j, ch in enumerate(rows):
        for i, rp in enumerate(sel):
            base = os.path.splitext(os.path.basename(rp))[0]
            axes[j, i].imshow(load(os.path.join(root, ch, base + ".png")))
            axes[j, i].set_xticks([]); axes[j, i].set_yticks([])
            if i == 0:
                axes[j, i].set_ylabel({"raw": "GT", "I_hat": "Ours (I_hat)", "depth": "Depth"}[ch], fontsize=9)
    fig.suptitle("{} — rendered test views (iter {})".format(scene, iteration), fontsize=11)
    fig.tight_layout()
    outdir = os.path.join(root, "grid")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "grid_raw_that_depth.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("  [{}] saved -> {}".format(scene, out))


def main():
    argv = sys.argv[1:]
    iteration = 60000
    if "--iter" in argv:
        iteration = int(argv[argv.index("--iter") + 1])
    n = 6
    if "--n" in argv:
        n = int(argv[argv.index("--n") + 1])
    scenes = [a for a in argv if not a.startswith("--") and not a.isdigit()]
    if not scenes:
        scenes = ["AT1"]
    for s in scenes:
        print("=== {} ===".format(s))
        grid_one(s, iteration, n)


if __name__ == "__main__":
    main()
