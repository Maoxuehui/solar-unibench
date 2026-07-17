"""Forecast baseline models.

* :class:`SmartPersistenceModel` — clear-sky-scaled physical skill reference
  (spec §10). Target-agnostic: works for PV power and GHI alike.
* :class:`LightGBMModel` — per-horizon LightGBM tabular baseline
  (direct multi-step regression on flattened past + future features).
"""
from .lightgbm_model import LightGBMModel
from .smart_persistence import SmartPersistenceModel

__all__ = ["SmartPersistenceModel", "LightGBMModel"]
