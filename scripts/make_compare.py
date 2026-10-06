#!/usr/bin/env python
"""Generate side-by-side "GT | Render" comparisons (from the renders/ produced by render.py).

Usage:
  python scripts/make_compare.py \
    --scene datasets/seathru_undist/Panama \
    --model outputs/baseline/Panama --iter 30000
"""
import os, sys, argparse
from PIL import Image, ImageDraw, ImageFont

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)
from scene.colmap_loader import read_extrinsics_binary  # noqa: E402


def llff_split(n):
    test_idx = [i for i in range(n) if i % 8 == 0]
    train_idx = [i for i in range(n) if i % 8 != 0]
    return test_idx, train_idx


def composite(render_path, gt_path, out_path):
    r = Image.open(render_path).convert("RGB")
    g = Image.open(gt_path).convert("RGB").resize(r.size, Image.BILINEAR)
    w, h = r.size
    gap, label_h = 6, 28
    canvas = Image.new("RGB", (w * 2 + gap, h + label_h), (255, 255, 255))
    canvas.paste(g, (0, label_h))
    canvas.paste(r, (w + gap, label_h))
    d = ImageDraw.Draw(canvas)
    try:
        f = ImageFont.load_default()
    except Exception:
        f = None
    d.text((6, 6), "GT", fill=(200, 30, 30), font=f)
    d.text((w + gap + 6, 6), "Render", fill=(30, 90, 200), font=f)
    canvas.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--iter", default="30000")
    ap.add_argument("--n_train", type=int, default=4)
    args = ap.parse_args()

    imgs = read_extrinsics_binary(os.path.join(args.scene, "sparse", "0", "images.bin"))
    ordered = sorted(imgs.values(), key=lambda x: x.id)
    names = [im.name for im in ordered]
    test_idx, train_idx = llff_split(len(names))

    out_dir = os.path.join(args.model, "compare")
    os.makedirs(out_dir, exist_ok=True)

    for k, ti in enumerate(test_idx):
        composite(
            os.path.join(args.model, "test", f"ours_{args.iter}", "renders", f"{k:05d}.png"),
            os.path.join(args.scene, "images", names[ti]),
            os.path.join(out_dir, f"test_{k:02d}_{os.path.splitext(names[ti])[0]}.png"),
        )
    for k, ti in enumerate(train_idx[: args.n_train]):
        composite(
            os.path.join(args.model, "train", f"ours_{args.iter}", "renders", f"{k:05d}.png"),
            os.path.join(args.scene, "images", names[ti]),
            os.path.join(out_dir, f"train_{k:02d}_{os.path.splitext(names[ti])[0]}.png"),
        )
    print("saved ->", out_dir)
    for f in sorted(os.listdir(out_dir)):
        print(" ", f)


if __name__ == "__main__":
    main()
