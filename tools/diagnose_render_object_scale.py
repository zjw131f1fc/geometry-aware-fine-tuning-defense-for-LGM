#!/usr/bin/env python3
"""
Quick diagnostic for "object appears smaller/larger in-frame" across render roots.

Heuristic:
- OmniObject3D renders: use normals PNG, treat RGB==0 as background.
- GSO (our render_omni_format.py outputs): use alpha>0 as foreground.

Outputs bbox-area ratio statistics (foreground bounding-box area / image area).
This is robust to resolution changes and is a good proxy for apparent object scale.
"""

from __future__ import annotations

import argparse
import random
import statistics
from pathlib import Path

import numpy as np
from PIL import Image


def _bbox_ratio_from_mask(mask: np.ndarray) -> float | None:
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got {mask.shape}")
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    h, w = mask.shape
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    bbox_area = (x1 - x0) * (y1 - y0)
    return float(bbox_area / (w * h))


def omni_bbox_ratio(obj_dir: Path, view: int = 0) -> float | None:
    p = obj_dir / "render" / "normals" / f"r_{view}_normal.png"
    if not p.exists():
        return None
    im = np.array(Image.open(p))
    if im.ndim != 3 or im.shape[2] < 3:
        return None
    rgb = im[:, :, :3]
    mask = np.any(rgb != 0, axis=2)
    return _bbox_ratio_from_mask(mask)


def gso_bbox_ratio(obj_dir: Path, view: int = 0) -> float | None:
    p = obj_dir / "render" / "images" / f"r_{view}.png"
    if not p.exists():
        return None
    im = np.array(Image.open(p))
    if im.ndim != 3 or im.shape[2] < 3:
        return None
    if im.shape[2] >= 4:
        mask = im[:, :, 3] > 0
    else:
        mask = np.any(im[:, :, :3] != 0, axis=2)
    return _bbox_ratio_from_mask(mask)


def _summarize(values: list[float]) -> str:
    if not values:
        return "n=0"
    return (
        f"n={len(values)} "
        f"min={min(values):.6f} "
        f"median={statistics.median(values):.6f} "
        f"max={max(values):.6f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--omni_root",
        type=Path,
        default=Path("datas/omniobject3d___OmniObject3D-New/raw/blender_renders"),
        help="OmniObject3D blender_renders root",
    )
    ap.add_argument(
        "--gso_root",
        type=Path,
        default=Path("datas/GSO/render_same_pose_all_100v_512"),
        help="GSO render root (folders with */render/images)",
    )
    ap.add_argument("--view", type=int, default=0, help="view index to inspect (default: 0)")
    ap.add_argument("--sample", type=int, default=50, help="max objects to sample per root")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show_smallest", type=int, default=5, help="show N smallest examples per root")
    args = ap.parse_args()

    random.seed(args.seed)

    omni_dirs = [p for p in args.omni_root.iterdir() if p.is_dir()]
    gso_dirs = [p for p in args.gso_root.iterdir() if p.is_dir() and p.name != "_logs"]

    omni_pick = random.sample(omni_dirs, min(args.sample, len(omni_dirs)))
    gso_pick = random.sample(gso_dirs, min(args.sample, len(gso_dirs)))

    omni_vals: list[tuple[str, float]] = []
    for d in omni_pick:
        v = omni_bbox_ratio(d, view=args.view)
        if v is not None:
            omni_vals.append((d.name, v))

    gso_vals: list[tuple[str, float]] = []
    for d in gso_pick:
        v = gso_bbox_ratio(d, view=args.view)
        if v is not None:
            gso_vals.append((d.name, v))

    omni_only = [v for _, v in omni_vals]
    gso_only = [v for _, v in gso_vals]

    print("[ScaleDiag] bbox_ratio stats (bbox area / image area)")
    print(f"[ScaleDiag] Omni: {_summarize(omni_only)}  root={args.omni_root}")
    print(f"[ScaleDiag]  GSO: {_summarize(gso_only)}  root={args.gso_root}")

    if omni_only and gso_only:
        scale_linear = (statistics.median(omni_only) / statistics.median(gso_only)) ** 0.5
        print(f"[ScaleDiag] Suggested linear scale multiplier (GSO→Omni): ~{scale_linear:.3f}x")

    if args.show_smallest > 0:
        print("\n[ScaleDiag] Smallest examples:")
        for name, v in sorted(omni_vals, key=lambda x: x[1])[: args.show_smallest]:
            print(f"  Omni  {name:30s}  bbox_ratio={v:.6f}")
        for name, v in sorted(gso_vals, key=lambda x: x[1])[: args.show_smallest]:
            print(f"  GSO   {name:30s}  bbox_ratio={v:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

