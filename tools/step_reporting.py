"""
Utilities for summarizing attack histories at evenly spaced step checkpoints.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def build_requested_steps(total_steps: int | None, num_checkpoints: int = 5) -> List[int]:
    """Build evenly spaced checkpoint steps, always including the final step."""
    total = _safe_int(total_steps)
    if total is None or total <= 0:
        return []

    num = max(1, min(_safe_int(num_checkpoints) or 5, total))
    requested = [int(math.ceil(total * i / num)) for i in range(1, num + 1)]
    requested[-1] = total

    normalized: List[int] = []
    prev = 0
    for step in requested:
        step = max(step, prev + 1)
        step = min(step, total)
        normalized.append(step)
        prev = step
    return normalized


def _normalize_history(step_history: Any) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not isinstance(step_history, list):
        return entries

    for entry in step_history:
        if not isinstance(entry, dict):
            continue
        step = _safe_int(entry.get("step"))
        if step is None:
            continue
        normalized = dict(entry)
        normalized["step"] = step
        entries.append(normalized)

    entries.sort(key=lambda item: item["step"])
    return entries


def _infer_total_steps(*histories: Any) -> Optional[int]:
    last_steps: List[int] = []
    for history in histories:
        entries = _normalize_history(history)
        if entries:
            last_steps.append(entries[-1]["step"])
    if not last_steps:
        return None
    return max(last_steps)


def _select_entry(entries: List[Dict[str, Any]], requested_step: int) -> Optional[Dict[str, Any]]:
    if not entries:
        return None

    return min(
        entries,
        key=lambda entry: (
            abs(entry["step"] - requested_step),
            entry["step"] < requested_step,
            entry["step"],
        ),
    )


def build_attack_step_report(
    step_history: Any,
    *,
    total_steps: int | None = None,
    num_checkpoints: int = 5,
) -> Dict[str, Any]:
    """Build a checkpoint report for one attack history."""
    entries = _normalize_history(step_history)
    if total_steps is None:
        total_steps = _infer_total_steps(entries)
    total_steps = _safe_int(total_steps)
    requested_steps = build_requested_steps(total_steps, num_checkpoints)

    checkpoints: List[Dict[str, Any]] = []
    for requested_step in requested_steps:
        entry = _select_entry(entries, requested_step)
        if entry is None:
            checkpoints.append(
                {
                    "requested_step": requested_step,
                    "actual_step": None,
                    "epoch": None,
                    "loss": None,
                    "lpips": None,
                    "masked_lpips": None,
                    "masked_psnr": None,
                    "source_psnr": None,
                    "source_lpips": None,
                }
            )
            continue

        checkpoints.append(
            {
                "requested_step": requested_step,
                "actual_step": entry.get("step"),
                "epoch": _safe_int(entry.get("epoch")),
                "loss": _safe_float(entry.get("loss")),
                "lpips": _safe_float(entry.get("lpips")),
                "masked_lpips": _safe_float(entry.get("masked_lpips")),
                "masked_psnr": _safe_float(entry.get("masked_psnr")),
                "source_psnr": _safe_float(entry.get("source_psnr")),
                "source_lpips": _safe_float(entry.get("source_lpips")),
            }
        )

    return {
        "total_steps": total_steps,
        "num_checkpoints": len(requested_steps),
        "requested_steps": requested_steps,
        "checkpoints": checkpoints,
    }


def build_dual_attack_step_report(
    baseline_history: Any,
    postdefense_history: Any,
    *,
    total_steps: int | None = None,
    num_checkpoints: int = 5,
) -> Dict[str, Any]:
    """Build a combined baseline/post-defense checkpoint report."""
    if total_steps is None:
        total_steps = _infer_total_steps(baseline_history, postdefense_history)

    baseline = build_attack_step_report(
        baseline_history, total_steps=total_steps, num_checkpoints=num_checkpoints
    )
    postdefense = build_attack_step_report(
        postdefense_history, total_steps=total_steps, num_checkpoints=num_checkpoints
    )

    return {
        "total_steps": baseline.get("total_steps") or postdefense.get("total_steps"),
        "num_checkpoints": max(
            baseline.get("num_checkpoints", 0), postdefense.get("num_checkpoints", 0)
        ),
        "requested_steps": baseline.get("requested_steps") or postdefense.get("requested_steps") or [],
        "baseline": baseline.get("checkpoints", []),
        "postdefense": postdefense.get("checkpoints", []),
    }


def _format_value(value: Any, ndigits: int = 4) -> str:
    number = _safe_float(value)
    if number is None:
        return "NA"
    return f"{number:.{ndigits}f}"


def _format_phase_line(label: str, entry: Dict[str, Any]) -> str:
    actual_step = entry.get("actual_step")
    if actual_step is None:
        return f"  {label:<20} (无可用 step 记录)"

    return (
        f"  {label:<20} actual_step={actual_step:<4} "
        f"masked_psnr={_format_value(entry.get('masked_psnr'), 2)} "
        f"masked_lpips={_format_value(entry.get('masked_lpips'), 4)} "
        f"source_psnr={_format_value(entry.get('source_psnr'), 2)} "
        f"source_lpips={_format_value(entry.get('source_lpips'), 4)}"
    )


def format_dual_attack_step_report(
    report: Dict[str, Any],
    *,
    phase: str = "both",
) -> str:
    """Render the combined checkpoint report as plain text."""
    phase = (phase or "both").lower().strip()
    if phase not in {"both", "baseline", "postdefense"}:
        raise ValueError(f"Unsupported phase: {phase}")

    requested_steps = report.get("requested_steps") or []
    total_steps = report.get("total_steps")
    baseline = report.get("baseline") or []
    postdefense = report.get("postdefense") or []

    header = f"Attack step report: {len(requested_steps)} checkpoints"
    if total_steps is not None:
        header += f" (total_steps={total_steps})"

    lines = [header]
    if not requested_steps:
        lines.append("  (无可用 attack step 历史)")
        return "\n".join(lines)

    for idx, requested_step in enumerate(requested_steps):
        lines.append(f"Step {requested_step}:")
        if phase in {"both", "baseline"}:
            entry = baseline[idx] if idx < len(baseline) else {}
            lines.append(_format_phase_line("Baseline Attack", entry))
        if phase in {"both", "postdefense"}:
            entry = postdefense[idx] if idx < len(postdefense) else {}
            lines.append(_format_phase_line("Post-Defense Attack", entry))

    return "\n".join(lines)
