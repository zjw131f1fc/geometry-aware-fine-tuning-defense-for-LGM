"""
Tools for 3D Defense project.
"""

from .batch_renderer import ObjaverseBatchRenderer
from .utils import (
    set_seed, get_base_model, prepare_lgm_data,
    BASELINE_CACHE_DIR, compute_baseline_hash, compute_defense_hash,
    load_baseline_cache, save_baseline_cache, copy_cached_renders,
)
from .plotting import plot_pipeline_results
from . import model_registry

__all__ = [
    "ObjaverseBatchRenderer", "set_seed", "get_base_model", "model_registry",
    "prepare_lgm_data", "BASELINE_CACHE_DIR", "compute_baseline_hash",
    "compute_defense_hash",
    "load_baseline_cache", "save_baseline_cache", "copy_cached_renders",
    "plot_pipeline_results",
]
