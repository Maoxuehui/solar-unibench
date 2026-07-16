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
    recover_geo_clearsky_global,
    _detect_mirror_pairs,
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


# ===========================================================================
# Method 5 (global): recover_geo_clearsky_global -- top-N + mirror detection
# ===========================================================================

def test_recover_geo_clearsky_global_recovers_known_site_and_returns_topn():
    """Clean full-year POWER -> exact recovery, ranked top-N, no false mirror."""
    true_lat, true_lon = -30.0, 145.0  # Australia; both on the 5-deg grid
    times = pd.date_range("2013-01-01", periods=24 * 365, freq="1h", tz="UTC")
    # POWER = normalized pvlib Ineichen clear-sky GHI at the true site.
    cs = _pvlib_clearsky_ghi(times, true_lat, true_lon)
    power = cs / (cs.max() + 1e-9)
    # Fast unit-test grid (clean data is unambiguous; 5-deg + 800 samples ~ seconds).
    lat5 = np.arange(-60.0, 60.0001, 5.0)
    lon5 = np.arange(-180.0, 180.0001, 5.0)
    res = recover_geo_clearsky_global(times, power, sample_size=800,
                                      lat_grid=lat5, lon_grid=lon5, top_n=10)
    assert res["latitude"] is not None and res["longitude"] is not None
    d = res["details"]
    assert d["method"] == "global_clearsky"
    # Best candidate is the true site (same model -> near-exact, MSE ~ 0).
    assert abs(res["latitude"] - true_lat) <= 6.0, d
    assert abs(res["longitude"] - true_lon) <= 6.0, d
    # top-N is a ranked list of {lat, lon, mse}, ascending in MSE, bounded by top_n.
    top = d["top_n"]
    assert 1 <= len(top) <= 10
    mses = [c["mse"] for c in top]
    assert mses == sorted(mses)
    assert all(set(c.keys()) >= {"lat", "lon", "mse"} for c in top)
    # The true site must appear within the top-N.
    assert any(abs(c["lat"] - true_lat) <= 6.0 and abs(c["lon"] - true_lon) <= 6.0
               for c in top)
    # Clean, full-year data carries an unambiguous seasonal amplitude signal, so the
    # N/S mirror is NOT expected here (the hemisphere ambiguity is a cloudy-PV
    # phenomenon -- see test_detect_mirror_pairs_flags_hemisphere_ambiguity and the
    # real-data RUN_SLOW integration test).
    assert d["has_mirror"] is False
    assert d["mirror_pairs"] == []


def test_recover_geo_clearsky_global_topn_contains_true_or_mirror():
    """The true site OR its N/S mirror must be within the returned top-N.

    This is the load-bearing correctness assertion for the global method: even when
    the hemisphere is ambiguous, the true latitude band and longitude must survive in
    the candidate set so a downstream disambiguator (TSR) can pick the side.
    """
    true_lat, true_lon = -30.0, 145.0
    times = pd.date_range("2013-01-01", periods=24 * 365, freq="1h", tz="UTC")
    cs = _pvlib_clearsky_ghi(times, true_lat, true_lon)
    power = cs / (cs.max() + 1e-9)
    lat5 = np.arange(-60.0, 60.0001, 5.0)
    lon5 = np.arange(-180.0, 180.0001, 5.0)
    res = recover_geo_clearsky_global(times, power, sample_size=800,
                                      lat_grid=lat5, lon_grid=lon5, top_n=10)
    top = res["details"]["top_n"]
    true_hit = any(abs(c["lat"] - true_lat) <= 6.0 and abs(c["lon"] - true_lon) <= 6.0
                   for c in top)
    mirror_hit = any(abs(c["lat"] - (-true_lat)) <= 6.0 and abs(c["lon"] - true_lon) <= 6.0
                     for c in top)
    assert true_hit or mirror_hit, top


def test_detect_mirror_pairs_flags_hemisphere_ambiguity():
    """Mirror detection fires when N/S-symmetric candidates co-occur (when present)."""
    # Two cells at lon 145 with opposite-sign, equal-magnitude lat -> one mirror pair.
    cands_with_mirror = [
        {"lat": -30.0, "lon": 145.0, "mse": 0.010},
        {"lat": 30.0, "lon": 145.0, "mse": 0.012},   # mirror of the above
        {"lat": -35.0, "lon": 145.0, "mse": 0.030},  # also mirrors +30 (within tol)
        {"lat": 20.0, "lon": -50.0, "mse": 0.050},   # no mirror here
    ]
    pairs = _detect_mirror_pairs(cands_with_mirror)
    assert len(pairs) >= 1
    # The (-30, 145) / (+30, 145) pair must be flagged.
    assert any(abs(p["lon"] - 145.0) < 1e-6 and
               abs(p["lat_south"] - (-30.0)) < 1e-6 and
               abs(p["lat_north"] - 30.0) < 1e-6 for p in pairs)

    # Same-hemisphere candidates -> no pairs.
    same_hemi = [
        {"lat": -30.0, "lon": 145.0, "mse": 0.01},
        {"lat": -35.0, "lon": 145.0, "mse": 0.02},
    ]
    assert _detect_mirror_pairs(same_hemi) == []

    # Opposite hemispheres but different longitude -> no pair.
    diff_lon = [
        {"lat": -30.0, "lon": 145.0, "mse": 0.01},
        {"lat": 30.0, "lon": 100.0, "mse": 0.02},
    ]
    assert _detect_mirror_pairs(diff_lon) == []

    # Opposite hemispheres, same lon, but very different |lat| -> no pair.
    diff_mag = [
        {"lat": -10.0, "lon": 145.0, "mse": 0.01},
        {"lat": 50.0, "lon": 145.0, "mse": 0.02},
    ]
    assert _detect_mirror_pairs(diff_mag) == []


def test_recover_geo_clearsky_global_skips_insufficient_data():
    """A handful of daytime samples short-circuits (returns None coord, skipped)."""
    times = pd.date_range("2013-06-01 06:00", periods=12, freq="1h", tz="UTC")
    power = np.clip(np.random.default_rng(0).random(12), 0.0, 1.0)
    res = recover_geo_clearsky_global(times, power)
    assert res["latitude"] is None and res["longitude"] is None
    assert res["details"]["status"].startswith("skipped")
    assert res["details"]["top_n"] == []
