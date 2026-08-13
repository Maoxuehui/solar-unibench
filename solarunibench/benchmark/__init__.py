"""E8 unified benchmark package.

Physics-time-aligned, top-journal-standard benchmark: a fixed F=4 input block
across all PV/GHI datasets, known-future ``X_future``, and daytime-masked
Forecast-Skill evaluation.

Task 1 exposes:
* :class:`DatasetConfig` and the :data:`DATASET_CONFIG` matrix
  (:mod:`solarunibench.benchmark.config`);
* :func:`add_clearness_aligned` — the clearness-index-aligned feature that
  forces F=4 (:mod:`solarunibench.benchmark.features`).
"""
from __future__ import annotations

from .config import DATASET_CONFIG, DatasetConfig, physical_horizons_min
from .features import add_clearness_aligned

__all__ = [
    "DATASET_CONFIG",
    "DatasetConfig",
    "physical_horizons_min",
    "add_clearness_aligned",
]
