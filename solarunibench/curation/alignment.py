"""Eq. (1) — spatial & timezone inversion by clear-sky shape matching.

Canonical homes of the physics: ``operators.reverse_geolocate`` (grid-search
lat/lon against an analytic clear-sky curve) and ``operators.geo_recover``
(multi-day clear-sky / TSR recovery used by the E1 blind test). This module
is the paper-facing seam: one name per equation, timezone derived from the
recovered longitude.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..operators.geo_recover import (
    recover_geo_clearsky_global,
    recover_geo_tsr,
)
from ..operators.reverse_geolocate import reverse_geolocate

__all__ = ["recover_geolocation", "recover_tsr_alignment",
           "reverse_geolocate", "recover_geo_clearsky_global"]


def recover_geolocation(
    times: pd.DatetimeIndex,
    power: np.ndarray,
    lat_grid: np.ndarray | None = None,
    lon_grid: np.ndarray | None = None,
) -> dict:
    """Recover (latitude, longitude, timezone) — Eq. (1).

    Matches the normalized daytime shape of ``power`` to analytic clear-sky
    curves over a lat/lon grid; timezone follows from solar noon at the
    recovered longitude (fixed UTC offset).

    Returns ``{"latitude", "longitude", "timezone", "details"}``.
    """
    out = reverse_geolocate(times, np.asarray(power, dtype=float),
                            lat_grid=lat_grid, lon_grid=lon_grid)
    if out["latitude"] is None:
        return {**out, "timezone": None}
    out["timezone"] = f"UTC{int(round(out['longitude'] / 15.0)):+03d}"
    return out


def recover_tsr_alignment(times: pd.DatetimeIndex,
                          tsr_hourly: np.ndarray) -> dict:
    """Time-series-of-record alignment recovery (multi-day variant, Eq. 1)."""
    return recover_geo_tsr(times, tsr_hourly)
