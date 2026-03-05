"""
Gaussian distribution plotting utilities (paper-style).

This module is intentionally lightweight:
- Uses matplotlib (Agg) + numpy + torch.
- SciPy is optional (only for KDE curves).
- Writes outputs to user-provided paths (project data disk).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import gaussian_kde  # type: ignore
except Exception:  # pragma: no cover
    gaussian_kde = None


@dataclass(frozen=True)
class GaussianExport:
    stage: str
    sample_keys: List[Dict[str, Any]]
    gaussians: List[torch.Tensor]
    meta: Dict[str, Any]


def _flatten_gaussians(gaussians_list: List[torch.Tensor]) -> torch.Tensor:
    if not gaussians_list:
        return torch.empty((0, 14), dtype=torch.float32)
    flat = []
    for g in gaussians_list:
        if not torch.is_tensor(g):
            continue
        if g.ndim == 3:
            # [B, N, 14] -> flatten batch
            g = g.reshape(-1, g.shape[-1])
        elif g.ndim == 2:
            pass
        else:
            g = g.reshape(-1, g.shape[-1])
        flat.append(g.detach().float().cpu())
    if not flat:
        return torch.empty((0, 14), dtype=torch.float32)
    return torch.cat(flat, dim=0)


def _safe_logit(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x) - np.log1p(-x)


def _compute_stage_arrays(flat: torch.Tensor) -> Dict[str, np.ndarray]:
    """
    Compute per-Gaussian scalar arrays used for plotting.

    flat: [M, 14] on CPU float32
    """
    if flat.numel() == 0:
        return {
            "opacity": np.zeros((0,), dtype=np.float32),
            "opacity_logit": np.zeros((0,), dtype=np.float32),
            "scale_log10": np.zeros((0,), dtype=np.float32),
            "scale_aniso_log10": np.zeros((0,), dtype=np.float32),
            "pos_norm": np.zeros((0,), dtype=np.float32),
            "rgb_max": np.zeros((0,), dtype=np.float32),
        }

    pos = flat[:, 0:3]
    opacity = flat[:, 3].clamp(1e-6, 1.0 - 1e-6)
    scale = flat[:, 4:7].clamp_min(1e-12)
    rgb = flat[:, 11:14].clamp(0.0, 1.0)

    opacity_np = opacity.numpy()
    scale_mean_np = scale.mean(dim=-1).numpy()
    scale_aniso_np = (scale.max(dim=-1).values / (scale.min(dim=-1).values + 1e-12)).numpy()
    pos_norm_np = pos.norm(dim=-1).numpy()
    rgb_max_np = rgb.max(dim=-1).values.numpy()

    return {
        "opacity": opacity_np,
        "opacity_logit": _safe_logit(opacity_np),
        "scale_log10": np.log10(scale_mean_np),
        "scale_aniso_log10": np.log10(scale_aniso_np),
        "pos_norm": pos_norm_np,
        "rgb_max": rgb_max_np,
    }


def _quantile_range(x: np.ndarray, q_lo: float = 0.005, q_hi: float = 0.995) -> Tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0
    lo = float(np.quantile(x, q_lo))
    hi = float(np.quantile(x, q_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.min(x))
        hi = float(np.max(x))
    if lo == hi:
        hi = lo + 1e-6
    return lo, hi


def summarize_gaussians(gaussians_list: List[torch.Tensor]) -> Dict[str, Any]:
    """
    Compute summary numbers similar to Evaluator.diagnose_gaussians, but using raw gaussian tensors.
    """
    per_sample = []
    for g in gaussians_list:
        if not torch.is_tensor(g) or g.numel() == 0:
            continue
        g = g.detach().float().cpu()
        if g.ndim == 3:
            # [B, N, 14] -> flatten batch dimension into samples
            for bi in range(g.shape[0]):
                per_sample.append(g[bi])
            continue
        per_sample.append(g)

    if not per_sample:
        return {"num_samples": 0, "num_gaussians_total": 0}

    sample_stats = []
    total_gaussians = 0
    for g in per_sample:
        total_gaussians += int(g.shape[-2])
        pos = g[:, 0:3]
        opacity = g[:, 3]
        scale = g[:, 4:7]
        rgb = g[:, 11:14]

        pos_abs_max = pos.abs().max(dim=-1).values
        scale_mean = scale.mean(dim=-1)
        rgb_white = (rgb > 0.95).all(dim=-1)

        sample_stats.append(
            {
                "num_gaussians": int(g.shape[-2]),
                "opacity_mean": float(opacity.mean().item()),
                "opacity_lt_01": float((opacity < 0.1).float().mean().item()),
                "opacity_lt_001": float((opacity < 0.01).float().mean().item()),
                "scale_mean": float(scale.mean().item()),
                "scale_tiny": float((scale_mean < 1e-4).float().mean().item()),
                "scale_aniso_ratio": float(
                    (scale.max(dim=-1).values / (scale.min(dim=-1).values + 1e-8)).mean().item()
                ),
                "pos_spread": float(pos.std(dim=0).mean().item()),
                "pos_out_of_range": float((pos_abs_max > 1.0).float().mean().item()),
                "pos_far_away": float((pos_abs_max > 2.0).float().mean().item()),
                "rgb_mean": float(rgb.mean().item()),
                "rgb_white_ratio": float(rgb_white.float().mean().item()),
            }
        )

    def _mean(key: str) -> float:
        vals = [s.get(key) for s in sample_stats if isinstance(s.get(key), (int, float))]
        return float(np.mean(vals)) if vals else 0.0

    def _std(key: str) -> float:
        vals = [s.get(key) for s in sample_stats if isinstance(s.get(key), (int, float))]
        return float(np.std(vals)) if vals else 0.0

    return {
        "num_samples": int(len(sample_stats)),
        "num_gaussians_total": int(total_gaussians),
        "mean": {k: _mean(k) for k in sample_stats[0].keys() if k != "num_gaussians"},
        "std": {k: _std(k) for k in sample_stats[0].keys() if k != "num_gaussians"},
    }


def load_gaussian_export(path: str) -> GaussianExport:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"gaussian export must be a dict, got {type(obj)}: {path}")
    stage = str(obj.get("stage", "")) or os.path.basename(path)
    sample_keys = obj.get("sample_keys", []) or []
    gaussians = obj.get("gaussians", []) or []
    meta = {k: v for k, v in obj.items() if k not in ("gaussians",)}
    return GaussianExport(stage=stage, sample_keys=sample_keys, gaussians=gaussians, meta=meta)


def plot_gaussian_distributions(
    exports: List[GaussianExport],
    *,
    save_path: str,
    summary_path: Optional[str] = None,
    title: str = "Gaussian Distributions (per-Gaussian)",
    bins: int = 80,
    kde: bool = True,
    kde_bw_method: Optional[str] = "scott",
    kde_grid_size: int = 512,
    kde_max_points: int = 50000,
) -> Dict[str, Any]:
    """
    Plot overlayed distributions for multiple stages.

    Notes:
      - Histograms are always plotted (density=True).
      - KDE is optional and uses SciPy if available. For large point sets,
        it will be randomly subsampled to `kde_max_points` (per stage per metric)
        for speed and numerical stability.
    """
    exports = [e for e in exports if e is not None and e.gaussians]
    if not exports:
        raise ValueError("No valid gaussian exports provided.")

    arrays_by_stage: Dict[str, Dict[str, np.ndarray]] = {}
    summaries: Dict[str, Any] = {}
    for e in exports:
        flat = _flatten_gaussians(e.gaussians)
        arrays_by_stage[e.stage] = _compute_stage_arrays(flat)
        summaries[e.stage] = summarize_gaussians(e.gaussians)

    metrics = [
        ("opacity", "Opacity", (0.0, 1.0)),
        ("opacity_logit", "logit(Opacity)", None),
        ("scale_log10", "log10(Scale mean)", None),
        ("scale_aniso_log10", "log10(Scale aniso ratio)", None),
        ("pos_norm", "Position L2 norm", None),
        ("rgb_max", "RGB max", (0.0, 1.0)),
    ]

    n_metrics = len(metrics)
    ncols = 3
    nrows = int(np.ceil(n_metrics / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows))
    axes = np.array(axes).reshape(-1)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])

    stage_names = [e.stage for e in exports]
    fig.suptitle(f"{title}\nStages: {', '.join(stage_names)}", fontsize=14)

    kde_enabled = bool(kde) and (gaussian_kde is not None)
    if bool(kde) and (gaussian_kde is None):
        print("[GaussianPlot] SciPy not found; KDE disabled (hist only).")

    rng = np.random.default_rng(0)

    for ax, (mkey, mtitle, fixed_range) in zip(axes, metrics):
        # Determine a common range across stages (robust quantiles).
        if fixed_range is not None:
            lo, hi = fixed_range
        else:
            combined = np.concatenate([arrays_by_stage[s][mkey] for s in stage_names], axis=0)
            lo, hi = _quantile_range(combined)

        edges = np.linspace(lo, hi, int(bins) + 1)
        kde_x = np.linspace(lo, hi, int(kde_grid_size)) if kde_enabled else None

        for si, stage in enumerate(stage_names):
            x = arrays_by_stage[stage][mkey]
            x = x[np.isfinite(x)]
            if x.size == 0:
                continue
            ax.hist(
                x,
                bins=edges,
                density=True,
                histtype="step",
                linewidth=1.6,
                alpha=0.55,
                color=colors[si % len(colors)],
                label=stage,
            )
            if kde_enabled and kde_x is not None:
                try:
                    x_kde = x.astype(np.float64, copy=False)
                    if x_kde.size > int(kde_max_points):
                        idx = rng.choice(x_kde.size, size=int(kde_max_points), replace=False)
                        x_kde = x_kde[idx]
                    if x_kde.size >= 2 and float(np.std(x_kde)) > 1e-12:
                        kde_fn = gaussian_kde(x_kde, bw_method=kde_bw_method)
                        y = kde_fn(kde_x)
                        ax.plot(
                            kde_x,
                            y,
                            color=colors[si % len(colors)],
                            linewidth=2.2,
                            alpha=0.9,
                        )
                except Exception:
                    # KDE can fail on near-constant distributions; keep hist as fallback.
                    pass

        ax.set_title(mtitle)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)

    # Hide extra axes if any.
    for ax in axes[len(metrics):]:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    out_summary: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "save_path": save_path,
        "plot_config": {
            "bins": int(bins),
            "kde": bool(kde_enabled),
            "kde_bw_method": kde_bw_method,
            "kde_grid_size": int(kde_grid_size),
            "kde_max_points": int(kde_max_points),
        },
        "stages": stage_names,
        "stage_summaries": summaries,
    }
    if summary_path:
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(out_summary, f, indent=2, ensure_ascii=False)
    return out_summary
