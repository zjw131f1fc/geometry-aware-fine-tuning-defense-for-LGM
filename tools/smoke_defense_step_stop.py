"""
Smoke check: step-based defense should not overshoot.

This script does NOT instantiate the full defense pipeline/model.
It only exercises the batch-planning logic in `DefenseTrainer.train_epoch`
using a tiny dummy model/optimizer and dummy loaders.

Run:
  python tools/smoke_defense_step_stop.py
"""

from __future__ import annotations

import os
import importlib.util
import random
import sys

# Some environments set invalid OMP_NUM_THREADS; normalize before importing torch to avoid noisy libgomp warnings.
_omp = os.environ.get("OMP_NUM_THREADS")
if _omp is not None and not _omp.isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"

import torch

# Ensure the repo root is importable when running as a script.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_DEFENSE_TRAINER_PATH = os.path.join(_REPO_ROOT, "training", "defense_trainer.py")
_spec = importlib.util.spec_from_file_location("_defense_trainer_smoke", _DEFENSE_TRAINER_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Failed to load defense_trainer module from: {_DEFENSE_TRAINER_PATH}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
DefenseTrainer = _mod.DefenseTrainer


class _DummyLoader:
    def __init__(self, n: int):
        self._n = int(n)

    def __len__(self) -> int:
        return self._n

    def __iter__(self):
        # yield distinct objects to avoid any accidental in-place reuse assumptions
        for i in range(self._n):
            yield {"i": i}


class _DummyModelMgr:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1)


class _FakeTrainer:
    """
    Minimal object that provides the attributes/methods `DefenseTrainer.train_epoch` expects.
    """

    def __init__(self, target_batches: int, source_batches: int = 1):
        self.model_mgr = _DummyModelMgr()
        self.optimizer = torch.optim.SGD(self.model_mgr.model.parameters(), lr=0.0)
        self.config = {"training": {"gradient_clip": 0}}

        self.source_loader = _DummyLoader(source_batches)
        self.target_loader = _DummyLoader(target_batches)

        self.source_ratio = 0.0  # deterministic: always use target
        self.gradient_accumulation_steps = 4

    def train_step(self, batch, is_target_data: bool):
        # Return a numeric loss, no backward needed for this smoke check.
        # (optimizer.step() is still called; lr=0.0 ensures no mutation.)
        _ = (batch, is_target_data)
        return {"loss": 1.0}


def _assert_eq(got, expected, msg: str):
    if got != expected:
        raise AssertionError(f"{msg}: got={got}, expected={expected}")


def main() -> None:
    # Make random branch deterministic (even though source_ratio=0.0).
    random.seed(0)
    torch.manual_seed(0)

    # Simulate last epoch: target loader has 83 batches, but only 12 steps remain.
    fake = _FakeTrainer(target_batches=83)
    start_step = 2988
    max_steps = 3000
    metrics, end_step = DefenseTrainer.train_epoch(
        fake, epoch=37, global_step=start_step, max_steps=max_steps
    )
    _assert_eq(end_step, max_steps, "train_epoch should stop exactly at max_steps")
    if "loss" not in metrics:
        raise AssertionError("metrics should include 'loss' when batches > 0")

    # If already at max, planned_batches becomes 0 and global_step should not advance.
    fake2 = _FakeTrainer(target_batches=83)
    metrics2, end_step2 = DefenseTrainer.train_epoch(
        fake2, epoch=1, global_step=max_steps, max_steps=max_steps
    )
    _assert_eq(end_step2, max_steps, "global_step should not advance when max is reached")
    _assert_eq(metrics2, {}, "metrics should be empty when no batches are run")

    print("OK: defense step stop smoke check passed.")


if __name__ == "__main__":
    main()
