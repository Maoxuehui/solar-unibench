"""Geo-recovery operators (primary).

Two methods ported from the reference ``geo_location.py``:

* **Method 1 -- ``recover_geo_tsr``**: TSR / VAR178 analytical recovery. Latitude is
  recovered from the summer/winter daily-TOA ratio (bisection + grid refinement) and
  longitude from the diurnal solar-noon UTC hour of the TSR increments. Fast,
  analytical, no external deps. This is the *primary* recovery used by the GEFCom
  adapter.

* **Method 5 -- ``recover_geo_pvlib``**: pvlib Ineichen clear-sky 2-D grid search
  over (lat, lon) minimizing peak-normalized MSE vs POWER. Used as an independent
  cross-check.

Both reuse the pure solar-geometry helpers in :mod:`solarunibench.operators.solar_geometry`
(``toa_daily_irradiation``, ``solar_noon_to_longitude``, ...). The simplified
clear-sky match (Method 3) is retained in :mod:`solarunibench.operators.reverse_geolocate`
as a documented baseline.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import solar_geometry as sg

__all__ = ["recover_geo_tsr", "recover_geo_pvlib"]


# ==================================================================
# Method 1: TSR (VAR178) analytical recovery
# ==================================================================

def _summer_winter_ratio_to_latitude(ratio: float) -> float:
    """Solve |latitude| from the summer/winter daily-TOA ratio via bisection.

    R(latitude) = H_TOA(NH-summer, lat) / H_TOA(NH-winter, lat) is monotonically
    increasing in |latitude| (equator ~1.0, mid-lat ~2-3, polar -> inf). We search
    [0, 80] degrees. The ratio is normalized to >= 1 first, so the returned value
    is the absolute latitude; the caller applies the hemisphere sign.
    """
    if ratio < 1.0:
        ratio = 1.0 / ratio
    lat_lo, lat_hi = 0.0, 80.0
    # NH-calendar summer / winter day-of-year anchors.
    summer_doys = np.array([152.0, 182.0, 213.0])   # ~Jun/Jul/Aug
    winter_doys = np.array([335.0, 1.0, 335.0])     # ~Dec/Jan/Dec
    for _ in range(100):  # bisection -> machine precision
        lat_mid = (lat_lo + lat_hi) / 2.0
        h_summer = sg.toa_daily_irradiation(summer_doys, lat_mid).mean()
        h_winter = sg.toa_daily_irradiation(winter_doys, lat_mid).mean()
        ratio_mid = 1e6 if h_winter < 1e-3 else float(h_summer / h_winter)
        if ratio_mid < ratio:
            lat_lo = lat_mid
        else:
            lat_hi = lat_mid
    return (lat_lo + lat_hi) / 2.0


def _refine_latitude_tsr(tsr_daily: pd.Series, hemisphere: str = "auto") -> tuple[float, float]:
    """Refine latitude by 1-D grid search against theoretical TOA daily irradiation.

    For each candidate |lat| (signed by hemisphere) we compute the theoretical TOA
    daily irradiation, fit a single median scale to the observed daily TSR sums, and
    score the MSE. Returns (best_latitude, best_mse).
    """
    if hemisphere == "auto":
        summer = tsr_daily[tsr_daily.index.month.isin([6, 7, 8])].mean()
        winter = tsr_daily[tsr_daily.index.month.isin([12, 1, 2])].mean()
        hemisphere = "north" if summer > winter else "south"

    unique_doys = sorted(tsr_daily.index.dayofyear.unique())
    doy_arr = np.array(unique_doys, dtype=float)
    obs_values = np.array([tsr_daily[tsr_daily.index.dayofyear == d].mean() for d in unique_doys])

    best_lat, best_mse = None, np.inf
    for lat_abs in np.arange(0.1, 75.0, 0.1):
        lat = lat_abs if hemisphere == "north" else -lat_abs
        theory = sg.toa_daily_irradiation(doy_arr, lat)
        valid = theory > 1e-3
        if valid.sum() < 10:
            continue
        scale = np.median(obs_values[valid] / theory[valid])
        if scale < 1e-6 or scale > 100:
            continue
        mse = float(np.mean((obs_values[valid] - theory[valid] * scale) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_lat = float(lat)
    return best_lat, best_mse


def recover_geo_tsr(times: pd.DatetimeIndex, tsr_hourly: np.ndarray) -> dict:
    """Method 1: recover ``{latitude, longitude, details}`` from TSR (VAR178).

    TSR is top-of-atmosphere net solar radiation: pure geometry, unaffected by
    cloud/aerosol. Latitude follows the summer/winter daily-sum ratio (analytical
    estimate + grid refinement); longitude follows the UTC hour of the diurnal TSR
    increment peak (solar noon).

    Parameters
    ----------
    times : hourly UTC timestamps (tz-aware recommended).
    tsr_hourly : VAR178 (TSR) values aligned to ``times``. ECMWF accumulated fields
        are handled via first-differencing (hourly increments -> diurnal profile).
    """
    tsr = pd.Series(np.asarray(tsr_hourly, dtype=float), index=pd.DatetimeIndex(times)).sort_index()

    # --- latitude ---
    tsr_daily = tsr.resample("1D").sum()
    summer_mean = tsr_daily[tsr_daily.index.month.isin([6, 7, 8])].mean()
    winter_mean = tsr_daily[tsr_daily.index.month.isin([12, 1, 2])].mean()
    ratio = 1e6 if winter_mean < 1e-3 else float(summer_mean / winter_mean)
    lat_abs = _summer_winter_ratio_to_latitude(ratio)
    hemisphere = "north" if summer_mean > winter_mean else "south"
    lat_analytic = lat_abs if hemisphere == "north" else -lat_abs
    lat_refined, refine_mse = _refine_latitude_tsr(tsr_daily, hemisphere)
    latitude = lat_refined if lat_refined is not None else lat_analytic

    # --- longitude (diurnal peak UTC hour -> solar noon -> longitude) ---
    # First-difference to remove the ECMWF accumulation ramp; positive increments
    # trace the daytime diurnal profile, whose peak is local solar noon.
    tsr_diff = tsr.diff()
    daytime = tsr_diff[tsr_diff > 0]
    if len(daytime) > 0:
        hourly_profile = daytime.groupby(daytime.index.hour).mean()
        peak_utc_hour = float(hourly_profile.idxmax())
    else:
        hourly_profile = tsr.groupby(tsr.index.hour).mean()
        peak_utc_hour = float(hourly_profile.idxmax()) if len(hourly_profile) else 12.0
    longitude = float(sg.solar_noon_to_longitude(peak_utc_hour))

    return {
        "latitude": float(latitude),
        "longitude": longitude,
        "details": {
            "method": "TSR (VAR178) analytical",
            "summer_winter_ratio": ratio,
            "hemisphere": hemisphere,
            "lat_analytic": float(lat_analytic),
            "lat_refined": float(lat_refined) if lat_refined is not None else None,
            "refine_mse": float(refine_mse),
            "peak_utc_hour": peak_utc_hour,
        },
    }


# ==================================================================
# Method 5: pvlib Ineichen clear-sky grid search
# ==================================================================

def _pvlib_clearsky_ghi(times: pd.DatetimeIndex, lat: float, lon: float,
                        linke_turbidity: float = 3.0) -> np.ndarray:
    """Fast pvlib Ineichen clear-sky GHI with a fixed Linke turbidity.

    Using a fixed turbidity avoids the per-call ``lookup_linke_turbidity`` overhead
    (and any network/cache dependence), making the grid search tractable. The shape
    -- which is what the (peak-normalized) MSE match is sensitive to -- is preserved.
    """
    import pvlib
    solpos = pvlib.solarposition.get_solarposition(times, lat, lon)
    app_zen = solpos["apparent_zenith"]
    am_rel = pvlib.atmosphere.get_relative_airmass(app_zen)
    am_abs = pvlib.atmosphere.get_absolute_airmass(am_rel)
    cs = pvlib.clearsky.ineichen(app_zen, am_abs, linke_turbidity=linke_turbidity,
                                 altitude=0.0)
    return cs["ghi"].to_numpy(dtype=float)


def recover_geo_pvlib(times: pd.DatetimeIndex, power: np.ndarray, sample_size: int = 800,
                      lat_grid=None, lon_grid=None, offsets=None) -> dict:
    """Method 5: recover ``{latitude, longitude, details}`` via pvlib clear-sky match.

    Samples POWER (uniform stride to ~``sample_size`` points), estimates a timezone
    guess from the median sunrise UTC hour, then 2-D grid-searches (lat, lon)
    minimizing peak-normalized MSE between POWER and the pvlib Ineichen clear-sky GHI.

    Parameters
    ----------
    times : hourly UTC timestamps.
    power : POWER (pv output, unit-less/normalized) aligned to ``times``.
    sample_size : cap on the number of points used (uniform stride subsampling).
    lat_grid / lon_grid : search grids. Defaults are coarse (5 deg step) so the
        search finishes in reasonable time on real data.
    offsets : vestigial in the reference (the offset loop did not alter the
        clearsky computation); kept for API compatibility and reported in details.

    Notes
    -----
    The reference ``infer_geo_from_pvlib`` looped over UTC offsets, but the offset
    never entered the ``Location.get_clearsky`` call (the same UTC timestamps and
    ``lon`` were used regardless), so it was a 5x no-op. We therefore run a single
    2-D (lat, lon) pass -- mathematically identical, ~5x faster.
    """
    power = pd.Series(np.asarray(power, dtype=float), index=pd.DatetimeIndex(times)).sort_index()

    # Uniform subsample to bound runtime (preserves seasonal coverage).
    if len(power) > sample_size:
        step = max(1, len(power) // sample_size)
        power = power.iloc[::step][:sample_size]
    times_s = power.index
    pvals = power.to_numpy()
    daytime = pvals > 0.01

    # A clearsky-shape fit needs many daytime samples; short-circuit otherwise
    # (also keeps tiny synthetic fixtures fast).
    if int(daytime.sum()) < 50:
        return {
            "latitude": None, "longitude": None,
            "details": {
                "method": "pvlib clear-sky grid search",
                "status": "skipped: insufficient daytime samples",
                "n_daytime": int(daytime.sum()),
                "sunrise_median_utc": float("nan"),
                "sample_size": int(len(power)),
            },
        }

    # Timezone guess from median sunrise hour (purely informational).
    sunrise_hours = []
    for _date, group in power.groupby(power.index.date):
        day = group[group > 0.01]
        if len(day) > 0:
            sunrise_hours.append(day.index[0].hour + day.index[0].minute / 60.0)
    sunrise_median = float(np.median(sunrise_hours)) if sunrise_hours else 12.0
    tz_guess = 6.0 - sunrise_median

    if lat_grid is None:
        lat_grid = np.arange(-60, 61, 5.0)
    if lon_grid is None:
        lon_grid = np.arange(-180, 181, 5.0)
    if offsets is None:
        offsets = [tz_guess + d for d in (-2.0, -1.0, 0.0, 1.0, 2.0)]

    best = None  # (mse, lat, lon, scale)
    for lat in lat_grid:
        lat = float(lat)
        for lon in lon_grid:
            lon = float(lon)
            try:
                cs_ghi = _pvlib_clearsky_ghi(times_s, lat, lon)
            except Exception:
                continue
            mask = daytime & (cs_ghi > 1.0)
            if mask.sum() < 50:
                continue
            scale = float(pvals[mask].max() / (cs_ghi[mask].max() + 1e-9))
            mse = float(np.mean((pvals[mask] - cs_ghi[mask] * scale) ** 2))
            if best is None or mse < best[0]:
                best = (mse, lat, lon, scale)

    if best is None:
        return {
            "latitude": None, "longitude": None,
            "details": {
                "method": "pvlib clear-sky grid search",
                "status": "no valid solution",
                "tz_offset_guess": tz_guess,
                "sunrise_median_utc": sunrise_median,
                "sample_size": int(len(power)),
            },
        }

    mse, lat, lon, scale = best
    return {
        "latitude": float(lat),
        "longitude": float(lon),
        "details": {
            "method": "pvlib clear-sky grid search",
            "best_mse": mse,
            "best_scale": scale,
            "tz_offset_guess": tz_guess,
            "best_offset": offsets[2] if len(offsets) == 5 else tz_guess,
            "sunrise_median_utc": sunrise_median,
            "sample_size": int(len(power)),
            "lat_step": float(np.diff(lat_grid)[0]) if len(lat_grid) > 1 else 5.0,
            "lon_step": float(np.diff(lon_grid)[0]) if len(lon_grid) > 1 else 5.0,
        },
    }
