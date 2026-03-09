#!/usr/bin/env python3
"""
Plot a summary figure for a `main_experiment_*/` folder (multiple runs).

This script is CPU-only (no GPU needed) and reads `metrics.json` produced by
`script/run_pipeline.py`.

Example:
  python script/plot_main_experiment_summary.py \
    output/experiments_output/main_experiment_20260304_032535
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib

# Headless + avoid writing caches to system disk when possible.
matplotlib.use("Agg")
_tmp_base = os.environ.get("TMPDIR")
if not _tmp_base and os.path.isdir("/root/autodl-tmp"):
    _tmp_base = "/root/autodl-tmp/tmp"
if _tmp_base:
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(_tmp_base, "mpl"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class RunRow:
    run_name: str
    category: str
    defense_method: str
    baseline_target_psnr: float
    baseline_target_lpips: float
    postdef_target_psnr: float
    postdef_target_lpips: float
    baseline_source_psnr: float
    baseline_source_lpips: float
    postdef_source_psnr: float
    postdef_source_lpips: float


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_category(run_dir: str, metrics: Dict[str, Any]) -> str:
    # Prefer the dumped pipeline_config.json (most reliable).
    cfg_path = os.path.join(run_dir, "pipeline_config.json")
    if os.path.exists(cfg_path):
        cfg = _read_json(cfg_path)
        cats = (((cfg.get("data") or {}).get("target") or {}).get("categories") or [])
        if isinstance(cats, list) and cats:
            return str(cats[0])

    # Fall back to dir name pattern: "<cat>_<method...>"
    base = os.path.basename(os.path.abspath(run_dir))
    if "_" in base:
        return base.split("_", 1)[0]
    return base


def _infer_method(run_dir: str) -> str:
    cfg_path = os.path.join(run_dir, "pipeline_config.json")
    if os.path.exists(cfg_path):
        cfg = _read_json(cfg_path)
        m = ((cfg.get("defense") or {}).get("method") or "").strip()
        if m:
            return str(m)
    base = os.path.basename(os.path.abspath(run_dir))
    if "_" in base:
        return base.split("_", 1)[1]
    return "unknown"


def _get_metric(d: Dict[str, Any], key: str, default: float = float("nan")) -> float:
    v = d.get(key, default)
    try:
        return float(v)
    except Exception:
        return float("nan")


def _extract_row(run_dir: str) -> Optional[RunRow]:
    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        return None
    metrics = _read_json(metrics_path)

    bt = metrics.get("baseline_target") or {}
    pt = metrics.get("postdefense_target") or {}
    bs = metrics.get("baseline_source") or {}
    ps = metrics.get("postdefense_source") or {}

    row = RunRow(
        run_name=os.path.basename(os.path.abspath(run_dir)),
        category=_infer_category(run_dir, metrics),
        defense_method=_infer_method(run_dir),
        baseline_target_psnr=_get_metric(bt, "psnr"),
        baseline_target_lpips=_get_metric(bt, "lpips"),
        postdef_target_psnr=_get_metric(pt, "psnr"),
        postdef_target_lpips=_get_metric(pt, "lpips"),
        baseline_source_psnr=_get_metric(bs, "psnr"),
        baseline_source_lpips=_get_metric(bs, "lpips"),
        postdef_source_psnr=_get_metric(ps, "psnr"),
        postdef_source_lpips=_get_metric(ps, "lpips"),
    )
    return row


def _scan_main_dir(main_dir: str) -> List[RunRow]:
    rows: List[RunRow] = []
    for name in sorted(os.listdir(main_dir)):
        run_dir = os.path.join(main_dir, name)
        if not os.path.isdir(run_dir):
            continue
        row = _extract_row(run_dir)
        if row is not None:
            rows.append(row)
    return rows


def _group_indices(values: List[str]) -> Dict[str, int]:
    uniq = []
    seen = set()
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        uniq.append(v)
    return {v: i for i, v in enumerate(uniq)}


def plot_summary(
    rows: List[RunRow],
    *,
    out_path: str,
    mode: str = "delta",
    title: str = "",
) -> None:
    if not rows:
        raise ValueError("No runs found (missing metrics.json).")

    categories = sorted({r.category for r in rows})
    methods = sorted({r.defense_method for r in rows})
    cat_to_i = {c: i for i, c in enumerate(categories)}
    method_to_i = {m: i for i, m in enumerate(methods)}

    # Build arrays [num_methods, num_categories] with NaN default.
    M, C = len(methods), len(categories)
    def _nan():
        return np.full((M, C), np.nan, dtype=np.float64)

    base_t_psnr, base_t_lpips = _nan(), _nan()
    post_t_psnr, post_t_lpips = _nan(), _nan()
    base_s_psnr, base_s_lpips = _nan(), _nan()
    post_s_psnr, post_s_lpips = _nan(), _nan()

    for r in rows:
        mi = method_to_i[r.defense_method]
        ci = cat_to_i[r.category]
        base_t_psnr[mi, ci] = r.baseline_target_psnr
        base_t_lpips[mi, ci] = r.baseline_target_lpips
        post_t_psnr[mi, ci] = r.postdef_target_psnr
        post_t_lpips[mi, ci] = r.postdef_target_lpips
        base_s_psnr[mi, ci] = r.baseline_source_psnr
        base_s_lpips[mi, ci] = r.baseline_source_lpips
        post_s_psnr[mi, ci] = r.postdef_source_psnr
        post_s_lpips[mi, ci] = r.postdef_source_lpips

    if mode == "postdef":
        # Absolute post-defense numbers.
        y_tp = post_t_psnr
        y_tl = post_t_lpips
        y_sp = post_s_psnr
        y_sl = post_s_lpips
        ylabels = ("Target PSNR↑ (attack quality)", "Target LPIPS↓ (attack quality)",
                   "Source PSNR↑ (retention)", "Source LPIPS↓ (retention)")
        suptitle = title or "Main Experiment Summary (Post-defense metrics)"
    elif mode == "delta":
        # Deltas vs baseline (post - baseline).
        y_tp = post_t_psnr - base_t_psnr
        y_tl = post_t_lpips - base_t_lpips
        y_sp = post_s_psnr - base_s_psnr
        y_sl = post_s_lpips - base_s_lpips
        ylabels = ("Δ Target PSNR (post - baseline)  ↓ better defense",
                   "Δ Target LPIPS (post - baseline) ↑ better defense",
                   "Δ Source PSNR (post - baseline)  ~0 best",
                   "Δ Source LPIPS (post - baseline) ~0 best")
        suptitle = title or "Main Experiment Summary (Delta vs Baseline)"
    else:
        raise ValueError(f"Unknown mode: {mode} (use 'delta' or 'postdef')")

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.reshape(-1)
    fig.suptitle(suptitle, fontsize=14)

    xs = np.arange(C, dtype=np.float64)
    group_width = 0.8
    bar_width = group_width / max(M, 1)
    offsets = (np.arange(M) - (M - 1) / 2.0) * bar_width

    series = [
        ("target_psnr", y_tp, ylabels[0]),
        ("target_lpips", y_tl, ylabels[1]),
        ("source_psnr", y_sp, ylabels[2]),
        ("source_lpips", y_sl, ylabels[3]),
    ]

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", None) or ["C0", "C1", "C2", "C3"]

    for ax, (_, Y, ylabel) in zip(axes, series):
        for mi, method in enumerate(methods):
            ax.bar(
                xs + offsets[mi],
                Y[mi],
                width=bar_width * 0.95,
                label=method,
                color=colors[mi % len(colors)],
                alpha=0.85,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(categories, rotation=0)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
        ax.axhline(y=0.0, color="black", linewidth=0.8, alpha=0.6)
        ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Plot summary metrics for a main_experiment folder")
    p.add_argument("main_dir", type=str, help="Path to output/experiments_output/main_experiment_*/")
    p.add_argument(
        "--mode",
        type=str,
        default="delta",
        choices=["delta", "postdef"],
        help="Plot mode: delta (post-baseline) or postdef (absolute post-defense).",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path. Default: <main_dir>/main_experiment_summary_<mode>.png",
    )
    p.add_argument("--title", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    main_dir = os.path.abspath(os.path.expanduser(args.main_dir))
    if not os.path.isdir(main_dir):
        raise FileNotFoundError(f"main_dir not found: {main_dir}")

    rows = _scan_main_dir(main_dir)
    out_path = args.output or os.path.join(main_dir, f"main_experiment_summary_{args.mode}.png")
    plot_summary(rows, out_path=out_path, mode=args.mode, title=args.title)
    print(f"[MainSummary] saved: {out_path}")
    print(f"[MainSummary] runs: {len(rows)}")


if __name__ == "__main__":
    main()

