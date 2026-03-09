#!/usr/bin/env python3
"""
Re-run pipeline runs to export *per-Gaussian* distributions (hist + KDE) for:
  - Baseline Attack
  - Post-Defense Attack (defensed attack)

Why this script exists:
  - Existing `main_experiment_*/` folders usually contain only summary metrics/renders.
  - True per-Gaussian hist/KDE requires exporting raw Gaussian tensors (not metrics.json means).
  - `script/run_pipeline.py` now supports `--export_gaussian_samples` to do that.

This script scans a `main_experiment_*/` folder, reads each run's `pipeline_args.json`,
and re-runs `script/run_pipeline.py` with:
  - a NEW output_dir (so we don't overwrite existing results)
  - `--export_gaussian_samples` enabled

Example:
  python script/run_main_experiment_gaussian_distributions.py \
    output/experiments_output/main_experiment_20260304_032535 \
    --gpu 0 --samples 32 --bins 80

Only run a subset:
  python script/run_main_experiment_gaussian_distributions.py \
    output/experiments_output/main_experiment_20260304_032535 \
    --gpu 0 --include "plant_.*" --samples 32

Notes on disk safety:
  - This script forces TMP/cache dirs onto `/root/autodl-tmp/` when available.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_data_disk_env(base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    base = "/root/autodl-tmp"
    if not os.path.isdir(base):
        return env

    cache_root = os.path.join(base, ".cache")
    tmp_root = os.path.join(base, "tmp")
    os.makedirs(cache_root, exist_ok=True)
    os.makedirs(tmp_root, exist_ok=True)

    # Hard override TMPDIR to prevent tempfile usage on system disk.
    env["TMPDIR"] = tmp_root

    env.setdefault("XDG_CACHE_HOME", cache_root)
    env.setdefault("TORCH_HOME", os.path.join(cache_root, "torch"))
    env.setdefault("HF_HOME", os.path.join(cache_root, "huggingface"))
    env.setdefault("WANDB_DIR", os.path.join(cache_root, "wandb"))
    env.setdefault("MPLCONFIGDIR", os.path.join(tmp_root, "mpl"))

    for k in ("XDG_CACHE_HOME", "TORCH_HOME", "HF_HOME", "WANDB_DIR", "MPLCONFIGDIR"):
        try:
            os.makedirs(env[k], exist_ok=True)
        except Exception:
            pass
    return env


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_pipeline_run_dir(d: str) -> bool:
    return os.path.exists(os.path.join(d, "pipeline_args.json"))


def _scan_run_dirs(
    main_dir: str,
    *,
    runs: Optional[Sequence[str]] = None,
    include: Optional[str] = None,
    exclude: Optional[str] = None,
) -> List[Tuple[str, str]]:
    main_dir = os.path.abspath(os.path.expanduser(main_dir))
    if _is_pipeline_run_dir(main_dir):
        return [(os.path.basename(main_dir), main_dir)]

    if not os.path.isdir(main_dir):
        raise FileNotFoundError(f"Not found: {main_dir}")

    run_set = set(runs) if runs else None
    include_re = re.compile(include) if include else None
    exclude_re = re.compile(exclude) if exclude else None

    out: List[Tuple[str, str]] = []
    for name in sorted(os.listdir(main_dir)):
        d = os.path.join(main_dir, name)
        if not os.path.isdir(d):
            continue
        if run_set is not None and name not in run_set:
            continue
        if include_re is not None and include_re.search(name) is None:
            continue
        if exclude_re is not None and exclude_re.search(name) is not None:
            continue
        if not _is_pipeline_run_dir(d):
            continue
        out.append((name, d))
    return out


def _supported_pipeline_flags() -> set[str]:
    """
    Extract supported `--flags` from `script/run_pipeline.py` source so we can
    safely ignore stale keys inside old `pipeline_args.json`.
    """
    path = os.path.join(_repo_root(), "script", "run_pipeline.py")
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
    except FileNotFoundError:
        return set()

    # Match add_argument('--flag' ...) and add_argument("--flag" ...)
    flags = set(re.findall(r"add_argument\(\s*['\"]--([a-zA-Z0-9_\-]+)['\"]", src))
    return flags


def _append_cli_arg(cmd: List[str], key: str, value: Any) -> None:
    """
    Convert a key/value pair from pipeline_args.json into CLI args.
    """
    flag = f"--{key}"
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    if value is None:
        return
    cmd.extend([flag, str(value)])


@dataclass(frozen=True)
class Job:
    run_name: str
    src_run_dir: str
    out_run_dir: str
    cmd: List[str]


def _build_job(
    *,
    run_name: str,
    src_run_dir: str,
    out_run_dir: str,
    gpu: int,
    samples: int,
    bins: int,
    no_baseline_cache: bool,
) -> Job:
    args_path = os.path.join(src_run_dir, "pipeline_args.json")
    args_obj = _read_json(args_path)

    supported = _supported_pipeline_flags()
    if not supported:
        raise RuntimeError("Failed to detect supported flags from script/run_pipeline.py")

    cmd: List[str] = [
        sys.executable,
        os.path.join(_repo_root(), "script", "run_pipeline.py"),
    ]

    # Reuse original args when possible (ignore output_dir + gpu, override below).
    for key, value in args_obj.items():
        if key in ("output_dir", "gpu"):
            continue
        if key not in supported:
            continue
        _append_cli_arg(cmd, key, value)

    # Hard overrides
    cmd.extend(["--gpu", str(int(gpu))])
    cmd.extend(["--output_dir", out_run_dir])
    cmd.append("--export_gaussian_samples")
    cmd.extend(["--gaussian_samples_num", str(int(samples))])
    cmd.extend(["--gaussian_plot_bins", str(int(bins))])
    if no_baseline_cache:
        cmd.append("--no_baseline_cache")

    return Job(run_name=run_name, src_run_dir=src_run_dir, out_run_dir=out_run_dir, cmd=cmd)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run per-Gaussian distribution exports for main_experiment runs")
    p.add_argument(
        "main_dir",
        type=str,
        help="A main_experiment_*/ directory (or a single run dir containing pipeline_args.json).",
    )
    p.add_argument("--gpu", type=int, default=0, help="GPU id passed to script/run_pipeline.py")
    p.add_argument("--runs", type=str, nargs="+", default=None, help="Explicit run subdir names to process.")
    p.add_argument("--include", type=str, default=None, help="Regex filter on run subdir names (include).")
    p.add_argument("--exclude", type=str, default=None, help="Regex filter on run subdir names (exclude).")
    p.add_argument(
        "--out_root",
        type=str,
        default=None,
        help="Output root directory (default: <main_dir>/_per_gaussian_<timestamp>/).",
    )
    p.add_argument("--samples", type=int, default=32, help="Number of samples for Gaussian export.")
    p.add_argument("--bins", type=int, default=80, help="Histogram bins for distribution plots.")
    p.add_argument(
        "--no_baseline_cache",
        action="store_true",
        help="Force rerun Phase 1 baseline attack (slower, but baseline export is sample-aligned).",
    )
    p.add_argument("--dry_run", action="store_true", help="Print commands without running.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    main_dir = os.path.abspath(os.path.expanduser(args.main_dir))
    run_dirs = _scan_run_dirs(main_dir, runs=args.runs, include=args.include, exclude=args.exclude)
    if not run_dirs:
        raise RuntimeError(f"No run dirs found under: {main_dir}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.out_root
    if out_root is None:
        out_root = os.path.join(main_dir, f"_per_gaussian_{ts}")
    out_root = os.path.abspath(os.path.expanduser(out_root))
    os.makedirs(out_root, exist_ok=True)

    env = _ensure_data_disk_env()
    # Also apply to current process (for optional re-plotting).
    os.environ.update(env)

    jobs: List[Job] = []
    for run_name, src_run_dir in run_dirs:
        out_run_dir = os.path.join(out_root, run_name)
        if os.path.exists(out_run_dir) and os.listdir(out_run_dir):
            raise FileExistsError(
                f"Output run dir already exists and is non-empty: {out_run_dir}\n"
                "Pick a new --out_root or delete the folder."
            )
        os.makedirs(out_run_dir, exist_ok=True)
        jobs.append(
            _build_job(
                run_name=run_name,
                src_run_dir=src_run_dir,
                out_run_dir=out_run_dir,
                gpu=int(args.gpu),
                samples=int(args.samples),
                bins=int(args.bins),
                no_baseline_cache=bool(args.no_baseline_cache),
            )
        )

    print(f"[MainGaussianDist] Runs: {len(jobs)}")
    print(f"[MainGaussianDist] Out root: {out_root}")

    index: List[Dict[str, Any]] = []
    for i, job in enumerate(jobs, start=1):
        print(f"\n[{i}/{len(jobs)}] {job.run_name}")
        print(f"  src: {job.src_run_dir}")
        print(f"  out: {job.out_run_dir}")
        print(f"  cmd: {' '.join(job.cmd)}")

        if args.dry_run:
            continue

        proc = subprocess.run(job.cmd, cwd=_repo_root(), env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"Pipeline failed for {job.run_name} (exit={proc.returncode})")

        # Optional: re-plot only baseline vs post-defense for clarity.
        try:
            from tools.gaussian_plotting import load_gaussian_export, plot_gaussian_distributions

            export_dir = os.path.join(job.out_run_dir, "gaussian_samples")
            b_path = os.path.join(export_dir, "baseline_attack.pt")
            p_path = os.path.join(export_dir, "postdefense_attack.pt")
            if os.path.exists(b_path) and os.path.exists(p_path):
                exports = [load_gaussian_export(b_path), load_gaussian_export(p_path)]
                out_png = os.path.join(job.out_run_dir, "gaussian_distributions_baseline_vs_postdef.png")
                out_json = os.path.join(
                    job.out_run_dir, "gaussian_distributions_baseline_vs_postdef_summary.json"
                )
                plot_gaussian_distributions(
                    exports,
                    save_path=out_png,
                    summary_path=out_json,
                    title="Gaussian Distributions (baseline vs post-defense attack)",
                    bins=int(args.bins),
                )
        except Exception:
            pass

        index.append(
            {
                "run_name": job.run_name,
                "src_run_dir": job.src_run_dir,
                "out_run_dir": job.out_run_dir,
                "gaussian_plot": os.path.join(job.out_run_dir, "gaussian_distributions.png"),
                "gaussian_summary": os.path.join(job.out_run_dir, "gaussian_distributions_summary.json"),
                "gaussian_plot_baseline_vs_postdef": os.path.join(
                    job.out_run_dir, "gaussian_distributions_baseline_vs_postdef.png"
                ),
                "gaussian_summary_baseline_vs_postdef": os.path.join(
                    job.out_run_dir, "gaussian_distributions_baseline_vs_postdef_summary.json"
                ),
                "exports_dir": os.path.join(job.out_run_dir, "gaussian_samples"),
            }
        )

    index_path = os.path.join(out_root, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": datetime.now().isoformat(),
                "main_dir": main_dir,
                "out_root": out_root,
                "runs": index,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n[MainGaussianDist] index: {index_path}")


if __name__ == "__main__":
    main()
