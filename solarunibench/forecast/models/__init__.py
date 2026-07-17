"""Forecast baseline models.

* :class:`SmartPersistenceModel` — clear-sky-scaled physical skill reference
  (spec §10). Target-agnostic: works for PV power and GHI alike.
"""
from .smart_persistence import SmartPersistenceModel

__all__ = ["SmartPersistenceModel"]
