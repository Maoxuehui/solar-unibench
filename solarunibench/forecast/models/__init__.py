"""Forecast baseline models.

* :class:`SmartPersistenceModel` — clear-sky-scaled physical skill reference
  (spec §10). Target-agnostic: works for PV power and GHI alike.
* :class:`LightGBMModel` — per-horizon LightGBM tabular baseline
  (direct multi-step regression on flattened past + future features).
* :class:`MLPModel` — single multi-output sklearn ``MLPRegressor`` baseline
  (configurable; ``skippd_paper`` preset reproduces Nie et al. 2020).
"""
from .lightgbm_model import LightGBMModel
from .mlp_model import MLPModel
from .smart_persistence import SmartPersistenceModel

__all__ = ["SmartPersistenceModel", "LightGBMModel", "MLPModel"]
