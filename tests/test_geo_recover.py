"""Synthetic tests for the geo-recovery operators (Methods 1 and 5).

These build self-consistent hourly series at a KNOWN (lat, lon) and assert each
method recovers it within tolerance. No real data is used here -- the real-data
cross-check lives in tests/test_gefcom_geo_integration.py (RUN_SLOW-gated).
"""
import numpy as np
import pandas as pd

from solarunibench.operators import solar_geometry as sg
from solarunibench.operators.clearsky import clearsky_ghi_simple
from solarunibench.operators.geo_recover import (
    recover_geo_tsr,
    recover_geo_pvlib,
    _pvlib_clearsky_ghi,
)


def _synth_tsr_cumulative(times, lat, lon):
    """Build an hourly TSR series whose daily total follows toa_daily_irradiation
    and whose hourly increments follow a clear-sky GHI bell peaking at solar noon.

    TSR is constructed as the within-day cumulative sum of GHI, mimicking an
    ECMWF-style accumulated field that resets each calendar day. As a result:
      * ``resample('1D').sum()`` is proportional to the true daily TOA total (-> lat),
      * ``diff()`` returns the GHI bell whose peak is the solar-noon UTC hour (-> lon).
    """
    doy = times.dayofyear.to_numpy().astype(float)
    shape = np.maximum(clearsky_ghi_simple(times, lat, lon), 0.0)
    # per-hour daily TOA total and per-day shape sum (broadcast across each day)
    daily_toa_total = sg.toa_daily_irradiation(doy, lat)
    shape_day_sum = pd.Series(shape, index=times).groupby(times.dayofyear).transform("sum").to_numpy()
    scale = np.where(shape_day_sum > 1e-6, daily_toa_total / shape_day_sum, 0.0)
    ghi = shape * scale
    # within-calendar-day cumulative sum (resets at midnight)
    s_ghi = pd.Series(ghi, index=times)
    tsr = s_ghi.groupby(s_ghi.index.date).cumsum()
    return tsr.to_numpy(dtype=float)


def test_recover_geo_tsr_recovers_known_site():
    true_lat, true_lon = -30.0, 145.0  # Australia
    times = pd.date_range("2013-01-01", periods=24 * 365, freq="1h", tz="UTC")
    tsr = _synth_tsr_cumulative(times, true_lat, true_lon)
    res = recover_geo_tsr(times, tsr)
    assert res["latitude"] is not None and res["longitude"] is not None
    assert abs(res["latitude"] - true_lat) <= 5.0, res["details"]
    assert abs(res["longitude"] - true_lon) <= 15.0, res["details"]
    # hemisphere must be correct
    assert res["details"]["hemisphere"] == "south"
    assert res["latitude"] < 0


def test_recover_geo_pvlib_recovers_known_site():
    true_lat, true_lon = -30.0, 145.0  # both on the default 5-deg grid
    times = pd.date_range("2013-01-01", periods=24 * 120, freq="1h", tz="UTC")
    # Synthesize POWER as the normalized pvlib clear-sky GHI at the true site
    # (same model the search uses, so recovery is near-exact).
    cs = _pvlib_clearsky_ghi(times, true_lat, true_lon)
    power = cs / (cs.max() + 1e-9)
    res = recover_geo_pvlib(times, power, sample_size=800)
    assert res["latitude"] is not None and res["longitude"] is not None
    # coarse 5-deg grid -> allow one grid cell of slack
    assert abs(res["latitude"] - true_lat) <= 6.0, res["details"]
    assert abs(res["longitude"] - true_lon) <= 6.0, res["details"]
