#!/usr/bin/env python3
"""
Plot paper-style Gaussian parameter distributions from exported Gaussian samples.

Two usage styles:

1) Give a pipeline run directory (recommended, simplest):
   python script/plot_gaussian_distributions.py output/.../robust_default

   The script will auto-discover:
     <run_dir>/gaussian_samples/baseline_attack.pt
     <run_dir>/gaussian_samples/defense.pt
     <run_dir>/gaussian_samples/postdefense_attack.pt

2) Explicitly specify export files:
   python script/plot_gaussian_distributions.py --inputs a.pt b.pt c.pt

These `.pt` files are produced by `script/run_pipeline.py` when running with:
  --export_gaussian_samples
"""

import os
import argparse
from typing import List, Optional

# Prefer data-disk tmp for matplotlib cache.
_tmp_base = os.environ.get("TMPDIR")
if not _tmp_base:
    _tmp_base = "/root/autodl-tmp/tmp" if os.path.isdir("/root/autodl-tmp") else None
if _tmp_base:
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(_tmp_base, "mpl"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from tools.gaussian_plotting import load_gaussian_export, plot_gaussian_distributions


def parse_args():
    p = argparse.ArgumentParser(description="Plot Gaussian distributions from exported samples")
    p.add_argument(
        "run_dir",
        type=str,
        nargs="?",
        default=None,
        help="Pipeline output dir containing gaussian_samples/ (or a parent dir that contains one sub-run).",
    )
    p.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        default=None,
        help="One or more gaussian export .pt files (baseline/defense/postdef, etc.). Overrides run_dir auto-discovery.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (.png). Default: <run_dir>/gaussian_distributions.png",
    )
    p.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Optional JSON summary path. Default: <run_dir>/gaussian_distributions_summary.json",
    )
    p.add_argument("--title", type=str, default="Gaussian Distributions (per-Gaussian)")
    p.add_argument("--bins", type=int, default=80)
    p.add_argument("--kde", dest="kde", action="store_true", help="Overlay KDE curves (requires SciPy).")
    p.add_argument("--no_kde", dest="kde", action="store_false", help="Disable KDE overlay (hist only).")
    p.set_defaults(kde=True)
    p.add_argument(
        "--kde_bw_method",
        type=str,
        default="scott",
        help="KDE bandwidth method passed to scipy.stats.gaussian_kde (e.g. scott/silverman or a float as string).",
    )
    p.add_argument("--kde_grid_size", type=int, default=512, help="Number of x points for KDE evaluation.")
    p.add_argument(
        "--kde_max_points",
        type=int,
        default=50000,
        help="Max points used for KDE (per stage per metric); larger may be slower / unstable.",
    )
    return p.parse_args()


def _resolve_inputs_from_run_dir(run_dir: str) -> (str, List[str]):
    """
    Returns:
        (workspace_dir, export_paths)
    """
    run_dir = os.path.abspath(os.path.expanduser(run_dir))

    def _is_export_dir(d: str) -> bool:
        if not os.path.isdir(d):
            return False
        # Require at least one export file to exist.
        for name in ("baseline_attack.pt", "defense.pt", "postdefense_attack.pt"):
            if os.path.exists(os.path.join(d, name)):
                return True
        return False

    # Case A: user passed <run_dir>/gaussian_samples directly.
    if os.path.basename(run_dir) == "gaussian_samples" and _is_export_dir(run_dir):
        export_dir = run_dir
        workspace_dir = os.path.dirname(export_dir)
    else:
        # Case B: user passed the pipeline workspace dir.
        direct = os.path.join(run_dir, "gaussian_samples")
        if _is_export_dir(direct):
            export_dir = direct
            workspace_dir = run_dir
        else:
            # Case C: user passed a parent dir (e.g., experiments/<exp>/) that contains exactly one sub-run.
            candidates: List[str] = []
            try:
                for name in sorted(os.listdir(run_dir)):
                    d = os.path.join(run_dir, name, "gaussian_samples")
                    if _is_export_dir(d):
                        candidates.append(d)
            except FileNotFoundError:
                pass

            if len(candidates) == 1:
                export_dir = candidates[0]
                workspace_dir = os.path.dirname(export_dir)
            elif len(candidates) > 1:
                msg = (
                    f"[GaussianPlot] Found multiple runs under: {run_dir}\n"
                    "Please point to a specific run dir, e.g. one of:\n"
                    + "\n".join([f"  - {os.path.dirname(c)}" for c in candidates])
                )
                raise RuntimeError(msg)
            else:
                raise FileNotFoundError(
                    f"[GaussianPlot] No gaussian_samples exports found under: {run_dir}\n"
                    "This run likely did not enable --export_gaussian_samples.\n"
                    "Re-run the pipeline with --export_gaussian_samples (and usually --no_baseline_cache)."
                )

    stage_order = [
        ("baseline_attack", os.path.join(export_dir, "baseline_attack.pt")),
        ("defense", os.path.join(export_dir, "defense.pt")),
        ("postdefense_attack", os.path.join(export_dir, "postdefense_attack.pt")),
    ]
    exports: List[str] = []
    for _, p in stage_order:
        if os.path.exists(p):
            exports.append(p)

    if not exports:
        raise FileNotFoundError(f"[GaussianPlot] No export .pt files found in: {export_dir}")

    return workspace_dir, exports


def main():
    args = parse_args()
    inputs: Optional[List[str]] = args.inputs
    base_dir: Optional[str] = None

    if inputs:
        base_dir = os.path.dirname(os.path.abspath(inputs[0]))
    elif args.run_dir:
        base_dir, inputs = _resolve_inputs_from_run_dir(args.run_dir)
    else:
        raise RuntimeError("Must provide either a run_dir positional argument or --inputs.")

    exports = [load_gaussian_export(p) for p in inputs]

    out_png = args.output or os.path.join(base_dir, "gaussian_distributions.png")
    out_json = args.summary or os.path.join(base_dir, "gaussian_distributions_summary.json")

    plot_gaussian_distributions(
        exports,
        save_path=out_png,
        summary_path=out_json,
        title=args.title,
        bins=args.bins,
        kde=bool(args.kde),
        kde_bw_method=(None if args.kde_bw_method in ("", "none", "None") else args.kde_bw_method),
        kde_grid_size=int(args.kde_grid_size),
        kde_max_points=int(args.kde_max_points),
    )
    print(f"[GaussianPlot] saved: {out_png}")
    print(f"[GaussianPlot] summary: {out_json}")


if __name__ == "__main__":
    main()
