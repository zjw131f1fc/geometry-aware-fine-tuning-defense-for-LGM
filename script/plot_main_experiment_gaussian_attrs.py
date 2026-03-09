#!/usr/bin/env python3
"""
Plot Gaussian diagnostic attributes for a `main_experiment_*/` folder.

This script is CPU-only and uses the `gaussian_diag` blocks stored inside
`metrics.json` produced by `script/run_pipeline.py`.

What it plots:
  - Baseline Attack gaussian_diag (on target)
  - Post-Defense Attack gaussian_diag (on target)
  - Delta (post - baseline) for numeric attributes

Example:
  TMPDIR=/root/autodl-tmp/tmp python script/plot_main_experiment_gaussian_attrs.py \
    output/experiments_output/main_experiment_20260304_032535
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

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
class RunDiag:
    run_name: str
    category: str
    defense_method: str
    baseline_diag: Dict[str, Any]
    postdef_diag: Dict[str, Any]


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_category(run_dir: str) -> str:
    cfg_path = os.path.join(run_dir, "pipeline_config.json")
    if os.path.exists(cfg_path):
        cfg = _read_json(cfg_path)
        cats = (((cfg.get("data") or {}).get("target") or {}).get("categories") or [])
        if isinstance(cats, list) and cats:
            return str(cats[0])
    base = os.path.basename(os.path.abspath(run_dir))
    return base.split("_", 1)[0] if "_" in base else base


def _infer_method(run_dir: str) -> str:
    cfg_path = os.path.join(run_dir, "pipeline_config.json")
    if os.path.exists(cfg_path):
        cfg = _read_json(cfg_path)
        method = ((cfg.get("defense") or {}).get("method") or "").strip()
        if method:
            return str(method)
    base = os.path.basename(os.path.abspath(run_dir))
    return base.split("_", 1)[1] if "_" in base else "unknown"


def _extract_gaussian_diag(target_block: Any) -> Dict[str, Any]:
    """
    Handle both standard and semantic-deflection metric formats.
    """
    if not isinstance(target_block, dict):
        return {}
    if isinstance(target_block.get("gaussian_diag"), dict):
        return target_block["gaussian_diag"]
    # Semantic deflection: nested dicts.
    for key in ("input", "supervision"):
        nested = target_block.get(key)
        if isinstance(nested, dict) and isinstance(nested.get("gaussian_diag"), dict):
            return nested["gaussian_diag"]
    return {}


def _scan_main_dir(main_dir: str) -> List[RunDiag]:
    rows: List[RunDiag] = []
    for name in sorted(os.listdir(main_dir)):
        run_dir = os.path.join(main_dir, name)
        if not os.path.isdir(run_dir):
            continue
        metrics_path = os.path.join(run_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            continue
        metrics = _read_json(metrics_path)

        baseline_diag = _extract_gaussian_diag(metrics.get("baseline_target"))
        postdef_diag = _extract_gaussian_diag(metrics.get("postdefense_target"))
        if not baseline_diag or not postdef_diag:
            # Keep it strict: main plots expect both stages.
            continue

        rows.append(
            RunDiag(
                run_name=name,
                category=_infer_category(run_dir),
                defense_method=_infer_method(run_dir),
                baseline_diag=baseline_diag,
                postdef_diag=postdef_diag,
            )
        )
    return rows


def _get_float(d: Dict[str, Any], key: str) -> float:
    v = d.get(key)
    try:
        return float(v)
    except Exception:
        return float("nan")


@dataclass(frozen=True)
class AttrSpec:
    key: str
    title: str
    transform: Callable[[np.ndarray], np.ndarray]
    ylims: Optional[Tuple[float, float]] = None


def _log10_safe(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log10(np.maximum(x, eps))


def _identity(x: np.ndarray) -> np.ndarray:
    return x


DEFAULT_ATTRS: List[AttrSpec] = [
    AttrSpec("opacity_mean", "opacity_mean", _identity, (0.0, 1.0)),
    AttrSpec("opacity_lt_01", "opacity_lt_01", _identity, (0.0, 1.0)),
    AttrSpec("opacity_lt_001", "opacity_lt_001", _identity, (0.0, 1.0)),
    AttrSpec("render_white_ratio", "render_white_ratio", _identity, (0.0, 1.0)),
    AttrSpec("rgb_white_ratio", "rgb_white_ratio", _identity, (0.0, 1.0)),
    AttrSpec("rgb_mean", "rgb_mean", _identity, (0.0, 1.0)),
    AttrSpec("pos_spread", "pos_spread", _identity, None),
    AttrSpec("pos_out_of_range", "pos_out_of_range", _identity, (0.0, 1.0)),
    AttrSpec("pos_far_away", "pos_far_away", _identity, (0.0, 1.0)),
    AttrSpec("scale_mean", "log10(scale_mean)", _log10_safe, None),
    AttrSpec("scale_tiny", "scale_tiny", _identity, (0.0, 1.0)),
    AttrSpec("scale_aniso_ratio", "log10(scale_aniso_ratio)", _log10_safe, None),
    AttrSpec("trap_position", "trap_position", _identity, None),
    AttrSpec("trap_scale", "trap_scale", _identity, None),
    AttrSpec("trap_opacity", "trap_opacity", _identity, None),
    AttrSpec("trap_rotation", "trap_rotation", _identity, None),
    AttrSpec("gaussian_dist_to_baseline", "gaussian_dist_to_baseline", _identity, None),
]


def _ordered_unique(values: List[str]) -> List[str]:
    seen = set()
    out = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _collect_matrix(
    rows: List[RunDiag],
    *,
    phase: str,
    key: str,
    categories: List[str],
    methods: List[str],
) -> np.ndarray:
    """
    Returns:
        Y[method_i, category_i] with NaNs for missing.
    """
    Y = np.full((len(methods), len(categories)), np.nan, dtype=np.float64)
    cat_to_i = {c: i for i, c in enumerate(categories)}
    method_to_i = {m: i for i, m in enumerate(methods)}

    for r in rows:
        ci = cat_to_i.get(r.category)
        mi = method_to_i.get(r.defense_method)
        if ci is None or mi is None:
            continue
        if phase == "baseline":
            Y[mi, ci] = _get_float(r.baseline_diag, key)
        elif phase == "postdef":
            Y[mi, ci] = _get_float(r.postdef_diag, key)
        elif phase == "delta":
            b = _get_float(r.baseline_diag, key)
            p = _get_float(r.postdef_diag, key)
            Y[mi, ci] = p - b
        else:
            raise ValueError(f"Unknown phase: {phase}")
    return Y


def plot_attr_grid(
    rows: List[RunDiag],
    *,
    phase: str,
    out_path: str,
    title: str,
    attrs: List[AttrSpec],
) -> None:
    if not rows:
        raise ValueError("No runs found with baseline_target/postdefense_target gaussian_diag.")

    categories = sorted({r.category for r in rows})
    methods = _ordered_unique([r.defense_method for r in rows])

    n_attrs = len(attrs)
    ncols = 4
    nrows = int(np.ceil(n_attrs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.6 * nrows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(-1)
    fig.suptitle(title, fontsize=14)

    xs = np.arange(len(categories), dtype=np.float64)
    group_width = 0.82
    bar_width = group_width / max(len(methods), 1)
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * bar_width
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", None) or ["C0", "C1", "C2", "C3"]

    legend_handles = []

    for ai, spec in enumerate(attrs):
        ax = axes[ai]
        Y = _collect_matrix(rows, phase=phase, key=spec.key, categories=categories, methods=methods)
        Yt = spec.transform(Y)

        for mi, method in enumerate(methods):
            bars = ax.bar(
                xs + offsets[mi],
                Yt[mi],
                width=bar_width * 0.95,
                label=method,
                color=colors[mi % len(colors)],
                alpha=0.85,
            )
            if ai == 0:
                legend_handles.append(bars)

        ax.set_title(spec.title)
        ax.set_xticks(xs)
        ax.set_xticklabels(categories, rotation=0)
        ax.grid(True, axis="y", alpha=0.25)
        if spec.ylims is not None:
            ax.set_ylim(spec.ylims)
        ax.axhline(y=0.0, color="black", linewidth=0.8, alpha=0.5)

    # Hide remaining empty axes.
    for j in range(n_attrs, len(axes)):
        axes[j].axis("off")

    # One shared legend (avoid repeating in every subplot).
    if methods and legend_handles:
        fig.legend(
            legend_handles,
            methods,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            fontsize=10,
            frameon=True,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Plot gaussian_diag attributes for main_experiment folder")
    p.add_argument("main_dir", type=str, help="Path to output/experiments_output/main_experiment_*/")
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Default: main_dir",
    )
    p.add_argument(
        "--phases",
        type=str,
        default="baseline,postdef,delta",
        help="Comma-separated phases to plot: baseline,postdef,delta",
    )
    return p.parse_args()


def main():
    args = parse_args()
    main_dir = os.path.abspath(os.path.expanduser(args.main_dir))
    if not os.path.isdir(main_dir):
        raise FileNotFoundError(f"main_dir not found: {main_dir}")

    out_dir = os.path.abspath(os.path.expanduser(args.out_dir or main_dir))
    os.makedirs(out_dir, exist_ok=True)

    rows = _scan_main_dir(main_dir)
    if not rows:
        raise RuntimeError(
            "No valid runs found. Expect run subfolders containing metrics.json with:\n"
            "  baseline_target.gaussian_diag and postdefense_target.gaussian_diag\n"
            f"main_dir={main_dir}"
        )

    phases = [p.strip() for p in (args.phases or "").split(",") if p.strip()]
    for phase in phases:
        if phase not in ("baseline", "postdef", "delta"):
            raise ValueError(f"Unknown phase in --phases: {phase}")
        out_path = os.path.join(out_dir, f"main_experiment_gaussian_attrs_{phase}.png")
        title = f"Main Experiment Gaussian Attributes ({phase})"
        plot_attr_grid(rows, phase=phase, out_path=out_path, title=title, attrs=DEFAULT_ATTRS)
        print(f"[GaussianAttrs] saved: {out_path}")
    print(f"[GaussianAttrs] runs: {len(rows)}")


if __name__ == "__main__":
    main()
