"""Geo-recovery operators (primary).

Three complementary methods ported/extended from the reference ``geo_location.py``:

* **Method 1 -- ``recover_geo_tsr``**: TSR / VAR178 analytical recovery. Latitude is
  recovered from the summer/winter daily-TOA ratio (bisection + grid refinement) and
  longitude from the diurnal solar-noon UTC hour of the TSR increments. Fast,
  analytical, no external deps. Preferred when NWP ``VAR178`` is available; also used
  to *disambiguate* the clear-sky N/S mirror pairs.

* **Method 5 (global) -- ``recover_geo_clearsky_global``**: the DEFAULT recovery and
  the only option when only POWER is available. Faithful port of the reference
  ``infer_geo_from_pvlib`` (sunrise-median -> timezone guess; pvlib Ineichen-Perez
  clear-sky GHI; gamma peak-normalization ``scale = power.max()/cs_ghi.max()``; MSE
  minimization), globalized to ``[-60, 60] x [-180, 180]``. Returns the top-N
  candidates and flags N/S mirror pairs (the inherent hemisphere ambiguity of
  clear-sky MSE on cloudy PV).

The clear-sky method reuses :func:`_pvlib_clearsky_ghi`. The TSR method reuses the
pure solar-geometry helpers in :mod:`solarunibench.operators.solar_geometry`
(``toa_daily_irradiation``, ``solar_noon_to_longitude``, ...). The simplified
clear-sky match (Method 3) is retained in :mod:`solarunibench.operators.reverse_geolocate`
as a documented baseline.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import solar_geometry as sg

__all__ = ["recover_geo_tsr", "recover_geo_clearsky_global"]


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
# Shared pvlib Ineichen clear-sky helper (used by the global method below)
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


# ==================================================================
# Method 5 (global): pvlib Ineichen clear-sky global grid + top-N + mirror detection
# ==================================================================

def _detect_mirror_pairs(candidates, lat_tol: float = 5.0, lon_tol: float = 2.0):
    """Find N/S-symmetric candidate pairs that need external disambiguation.

    Two candidates form a mirror pair when they sit at (near) the same longitude and
    their latitudes are opposite in sign but close in magnitude (e.g. (-30, 145) and
    (+30, 145)). On cloudy PV the seasonal amplitude signal is corrupted, so both
    hemispheres can fit the clear-sky shape equally well -- this is the inherent
    hemisphere ambiguity of clear-sky MSE matching.

    Parameters
    ----------
    candidates : list of dicts with keys ``lat``, ``lon``, ``mse``.
    lat_tol : max tolerated difference ``||lat_a| - |lat_b||`` in degrees.
    lon_tol : max tolerated longitude difference in degrees.

    Returns
    -------
    list of ``{lon, lat_north, lat_south, mse_north, mse_south}`` (one per pair,
    deduplicated, north lat >= 0).
    """
    pairs = []
    seen = set()
    n = len(candidates)
    for i in range(n):
        a = candidates[i]
        for j in range(i + 1, n):
            b = candidates[j]
            if a["lat"] == 0 or b["lat"] == 0:
                continue
            # opposite hemispheres
            if (a["lat"] > 0) == (b["lat"] > 0):
                continue
            if abs(a["lon"] - b["lon"]) > lon_tol:
                continue
            if abs(abs(a["lat"]) - abs(b["lat"])) > lat_tol:
                continue
            north, south = (a, b) if a["lat"] > 0 else (b, a)
            lon_mid = round((north["lon"] + south["lon"]) / 2.0, 4)
            lat_mag = round(0.5 * (abs(north["lat"]) + abs(south["lat"])), 4)
            key = (lat_mag, lon_mid)
            if key in seen:
                continue
            seen.add(key)
            pairs.append({
                "lon": lon_mid,
                "lat_north": float(north["lat"]),
                "lat_south": float(south["lat"]),
                "mse_north": float(north["mse"]),
                "mse_south": float(south["mse"]),
            })
    return pairs


def recover_geo_clearsky_global(times: pd.DatetimeIndex, power: np.ndarray,
                                 sample_size: int = 2400, lat_grid=None, lon_grid=None,
                                 offsets=None, top_n: int = 10, return_grid: bool = False) -> dict:
    """Method 5 (global): DEFAULT clear-sky recovery with top-N + mirror detection.

    Faithful port of the reference ``infer_geo_from_pvlib`` (sunrise-median -> timezone
    guess; pvlib Ineichen-Perez clear-sky GHI; gamma peak-normalization
    ``scale = power.max()/cs_ghi.max()``; MSE minimization over a (lat, lon) grid),
    then globalized and extended:

    * **Global grid** ``[-60, 60] x [-180, 180]`` (default 3 deg step -- a prior
      diagnostic found finer grids miss *worse* on cloudy PV; the resolution is not
      the limiting factor, the hemisphere ambiguity is).
    * **Top-N candidates** returned ranked by MSE (not just the single best).
    * **N/S mirror detection** -- flags hemisphere-symmetric candidate pairs that need
      external (e.g. TSR) disambiguation.

    Parameters
    ----------
    times : hourly UTC timestamps.
    power : POWER (pv output, unit-less/normalized) aligned to ``times``.
    sample_size : cap on the number of points used (uniform stride subsampling,
        preserving seasonal coverage -- matches the reference's "均匀采样"). Default
        2400 per the reference notebook.
    lat_grid / lon_grid : search grids. Defaults are the global 3 deg grid.
    offsets : vestigial in the reference (the offset loop never entered the
        ``get_clearsky`` call); kept for API compatibility and reported in details.
    top_n : number of best candidates to return.

    Returns
    -------
    dict ``{latitude, longitude, details}`` where ``latitude/longitude`` are the
    best (lowest-MSE) candidate, and ``details`` holds:
        top_n        : list of ``{lat, lon, mse}`` ranked by MSE (best first),
        mirror_pairs : list of N/S-symmetric pairs within top_n,
        method       : ``"global_clearsky"``,
        plus diagnostic fields (sunrise_median_utc, tz guess, sample_size, ...).

    Notes
    -----
    The reference ``infer_geo_from_pvlib`` / ``infer_geo_global`` looped over UTC
    offsets, but the offset never entered the ``Location.get_clearsky`` call (the same
    UTC timestamps and ``lon`` were used regardless), so it was a 5x no-op. We run a
    single 2-D (lat, lon) pass -- mathematically identical, ~5x faster.
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
                "method": "global_clearsky",
                "status": "skipped: insufficient daytime samples",
                "n_daytime": int(daytime.sum()),
                "sunrise_median_utc": float("nan"),
                "sample_size": int(len(power)),
                "top_n": [],
                "mirror_pairs": [],
            },
        }

    # Timezone guess from median sunrise hour (purely informational; preserved from
    # Method 5). Sunrise near UTC 6:00 -> tz ~ 0; near UTC 20:00 -> east ~ +10 (AUS).
    sunrise_hours = []
    for _date, group in power.groupby(power.index.date):
        day = group[group > 0.01]
        if len(day) > 0:
            sunrise_hours.append(day.index[0].hour + day.index[0].minute / 60.0)
    sunrise_median = float(np.median(sunrise_hours)) if sunrise_hours else 12.0
    tz_guess = 6.0 - sunrise_median

    # Global grid (reference was 1.5 deg; 3 deg keeps the cloudy-PV search tractable
    # without changing the science -- resolution is not the limiting factor).
    if lat_grid is None:
        lat_grid = np.arange(-60.0, 60.0001, 3.0)
    if lon_grid is None:
        lon_grid = np.arange(-180.0, 180.0001, 3.0)
    if offsets is None:
        offsets = [tz_guess + d for d in (-2.0, -1.0, 0.0, 1.0, 2.0)]

    candidates = []
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
            candidates.append({"lat": lat, "lon": lon, "mse": mse, "scale": scale})

    if not candidates:
        return {
            "latitude": None, "longitude": None,
            "details": {
                "method": "global_clearsky",
                "status": "no valid solution",
                "tz_offset_guess": tz_guess,
                "sunrise_median_utc": sunrise_median,
                "sample_size": int(len(power)),
                "top_n": [],
                "mirror_pairs": [],
            },
        }

    # Rank by MSE ascending and keep the top-N.
    candidates.sort(key=lambda c: c["mse"])
    top = candidates[:top_n]
    mirror_pairs = _detect_mirror_pairs(top)

    best = top[0]
    return {
        "latitude": float(best["lat"]),
        "longitude": float(best["lon"]),
        "details": {
            "method": "global_clearsky",
            "best_mse": float(best["mse"]),
            "best_scale": float(best["scale"]),
            "tz_offset_guess": tz_guess,
            "best_offset": offsets[2] if len(offsets) == 5 else tz_guess,
            "sunrise_median_utc": sunrise_median,
            "sample_size": int(len(power)),
            "lat_step": float(np.diff(lat_grid)[0]) if len(lat_grid) > 1 else 3.0,
            "lon_step": float(np.diff(lon_grid)[0]) if len(lon_grid) > 1 else 3.0,
            "top_n": [{"lat": c["lat"], "lon": c["lon"], "mse": c["mse"]} for c in top],
            "mirror_pairs": mirror_pairs,
            "has_mirror": bool(mirror_pairs),
            # Full candidate grid (for 2-D residual contours); only on request
            # to avoid bloating the default return payload.
            "grid": ([{"lat": c["lat"], "lon": c["lon"], "mse": c["mse"]} for c in candidates]
                     if return_grid else []),
        },
    }
