#!/usr/bin/env python3
"""
Print a fixed-count attack step report from metrics.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


sys.path.insert(0, _repo_root())

from tools import build_dual_attack_step_report, format_dual_attack_step_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Print attack step report from metrics.json")
    parser.add_argument("--metrics", type=str, required=True, help="Path to metrics.json")
    parser.add_argument(
        "--checkpoints",
        type=int,
        default=5,
        help="Number of evenly spaced checkpoints to report",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="both",
        choices=("both", "baseline", "postdefense"),
        help="Which attack phase(s) to print",
    )
    args = parser.parse_args()

    with open(args.metrics, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    config = metrics.get("config") or {}
    total_steps = config.get("attack_steps")
    report = build_dual_attack_step_report(
        metrics.get("baseline_attack"),
        metrics.get("postdefense_attack"),
        total_steps=total_steps,
        num_checkpoints=args.checkpoints,
    )
    print(format_dual_attack_step_report(report, phase=args.phase))


if __name__ == "__main__":
    main()
