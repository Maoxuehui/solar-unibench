#!/usr/bin/env python
"""Geo-validation experiments (Task A phase-shift ablation + Task B Folsom blind inversion).

Two honest, ESSD-style data-quality experiments on the clear-sky geo-recovery operator
(:func:`solarunibench.operators.geo_recover.recover_geo_clearsky_global`):

Task A — phase / timezone perturbation ablation (``results_geo_ablation_phase_shift.csv``)
    For each dataset in {GEFCom2014 (hourly PV), Folsom (hourly-resampled GHI)} and each
    time shift dt in [-60, -30, 0, +30, +60] min, shift the timestamps and run the
    clear-sky global recovery on a 6 deg grid. Records the recovered (lat, lon), the
    best MSE, the sunrise/tz diagnostics, and the mirror-pair count. The recovered
    LONGITUDE is expected to drift ~15 deg per 60 min dt (solar-noon -> longitude),
    which is the clean, interpretable phase signal the clear-sky method does pick up.

    Folsom-only companions:
      * clearness-index corruption: recompute ghi_clearsky at the TRUE coords + the
        SHIFTED timestamps (phase-misaligned with the measured ghi) and record the
        fraction of non-finite / unphysical (>1.2) kt = ghi_actual / ghi_clearsky.
      * downstream LightGBM skill decay: a bounded 1h-ahead forecast (lookback=12,
        horizon=1 hourly) where the perturbation corrupts the cos_zenith/ghi_clearsky
        features; train LightGBM (n_estimators=80) and score daytime skill vs Smart
        Persistence. Skill should peak at dt=0 and decay both sides (the "avalanche
        parabola").

Task B — Folsom blind inversion + mirror ambiguity (``results_folsom_blind_inversion.csv``)
    Run the clear-sky recovery at 3 deg resolution with ``return_grid=True`` on Folsom
    hourly ghi_actual, compute the great-circle (haversine) error vs the known coords
    (lat 38.651, lon -121.169), record the mirror pairs, and dump the full grid MSE to
    ``folsom_grid_mse.csv`` for the 2D residual contour. Honest negative result: on
    cloudy GHI the clear-sky shape match is hemisphere-ambiguous, so the recovery is
    expected to miss by a large margin — this is the disclosed limitation, not a hidden
    failure.

Run:
    python scripts/geo_validation_experiments.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from solarunibench.loader import SolarUniBenchLoader
from solarunibench.operators.geo_recover import recover_geo_clearsky_global, _pvlib_clearsky_ghi
from solarunibench.operators.clearsky import compute_astronomy
from solarunibench.forecast.core import EndogenousFeatureEngineer
from solarunibench.forecast.models.lightgbm_model import LightGBMModel
from solarunibench.forecast.models.smart_persistence import SmartPersistenceModel
from solarunibench.forecast.metrics import daytime_metrics

DATA_ROOT = REPO / "data" / "standardized"
OUT = REPO / "figures_full" / "results"
OUT.mkdir(parents=True, exist_ok=True)

# Known Folsom truth (stored coords in metadata; pinned here for the haversine error).
FOLSOM_TRUE_LAT = 38.651
FOLSOM_TRUE_LON = -121.169

# Bounded chunk sizes. A 90-day CONTIGUOUS hourly chunk preserves the diurnal cycle the
# longitude recovery depends on (the operator's uniform stride subsample aliases the
# daily signal away on multi-year series -- see experiments/e1_geo.py).
_CHUNK_HOURS = 90 * 24
_DT_GRID = [-60, -30, 0, 30, 60]
_LAT_GRID6 = np.arange(-60.0, 60.0001, 6.0)
_LON_GRID6 = np.arange(-180.0, 180.0001, 6.0)
_LAT_GRID3 = np.arange(-60.0, 60.0001, 3.0)
_LON_GRID3 = np.arange(-180.0, 180.0001, 3.0)

# LightGBM downstream (Folsom). Bounded to ~5e3 hourly rows per the task brief.
_LGB_LOOKBACK = 12
_LGB_HORIZON = 1
_LGB_CHUNK = 5200
_LGB_TRAIN_FRAC = 0.8
_LGB_FEATURES = ["ghi_clearsky", "cos_zenith", "clearness_index"]
_EPS = 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return float(2.0 * r * np.arcsin(np.sqrt(a)))


def _naive_utc(series: pd.Series) -> pd.DatetimeIndex:
    ts = pd.to_datetime(series)
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return pd.DatetimeIndex(ts)


def _log(msg):
    print(f"[geo-val] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def _folsom_hourly(loader, n_hours=_CHUNK_HOURS):
    """Hourly-resampled Folsom ghi_actual, bounded to a contiguous chunk.

    Returns (times_naive_utc, ghi_actual_array) over the first ``n_hours`` hours.
    """
    df = loader.load_tabular("folsom")
    df = df[["timestamp_utc", "ghi_actual"]].dropna(subset=["ghi_actual"]).sort_values("timestamp_utc")
    times = _naive_utc(df["timestamp_utc"])
    s = pd.Series(df["ghi_actual"].to_numpy(dtype=float), index=times)
    hourly = s.resample("1h").mean().dropna()
    hourly = hourly.iloc[:n_hours]
    return hourly.index, hourly.to_numpy(dtype=float)


def _folsom_hourly_full(loader):
    """Full hourly Folsom ghi_actual (for the downstream skill chunk + kt metric)."""
    df = loader.load_tabular("folsom")
    df = df[["timestamp_utc", "ghi_actual"]].dropna(subset=["ghi_actual"]).sort_values("timestamp_utc")
    times = _naive_utc(df["timestamp_utc"])
    s = pd.Series(df["ghi_actual"].to_numpy(dtype=float), index=times)
    return s.resample("1h").mean().dropna()


def _gefcom_hourly(loader, zone="GEFCOM_ZONE_1", n_hours=_CHUNK_HOURS):
    """Hourly GEFCom PV (one zone), bounded to a contiguous chunk."""
    df = loader.load_tabular("gefcom2014")
    df = df[df["entity_id"] == zone].sort_values("timestamp_utc")
    times = _naive_utc(df["timestamp_utc"])
    s = pd.Series(df["pv_power_normalized"].to_numpy(dtype=float), index=times).dropna()
    s = s.iloc[:n_hours]
    return s.index, s.to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Task A: phase-shift ablation (geo recovery)
# ---------------------------------------------------------------------------

def _geo_recover_shifted(times, power, dt_min, lat_grid, lon_grid, sample_size):
    shifted = pd.DatetimeIndex(times) + pd.DateOffset(minutes=dt_min)
    res = recover_geo_clearsky_global(
        shifted, power,
        sample_size=sample_size,
        lat_grid=lat_grid, lon_grid=lon_grid,
        top_n=10,
    )
    d = res.get("details", {})
    return {
        "dt_min": dt_min,
        "recovered_lat": res.get("latitude"),
        "recovered_lon": res.get("longitude"),
        "best_mse": d.get("best_mse", float("nan")),
        "sunrise_median_utc": d.get("sunrise_median_utc", float("nan")),
        "tz_offset_guess": d.get("tz_offset_guess", float("nan")),
        "n_mirror_pairs": len(d.get("mirror_pairs", [])),
    }


def task_a_geo(loader):
    rows = []
    folsom_times, folsom_pow = _folsom_hourly(loader)
    gef_times, gef_pow = _gefcom_hourly(loader)
    for dt in _DT_GRID:
        for ds, times, power in [("gefcom2014", gef_times, gef_pow),
                                 ("folsom", folsom_times, folsom_pow)]:
            t0 = time.time()
            row = _geo_recover_shifted(
                times, power, dt, _LAT_GRID6, _LON_GRID6, sample_size=len(power),
            )
            row["dataset"] = ds
            rows.append(row)
            _log(f"geo  ds={ds:12s} dt={dt:+4d}  -> lat={row['recovered_lat']} "
                 f"lon={row['recovered_lon']}  mse={row['best_mse']:.4g}  "
                 f"({time.time()-t0:.1f}s)")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Task A: Folsom clearness-index corruption metric
# ---------------------------------------------------------------------------

def task_a_kt_corruption(loader):
    """Recompute ghi_clearsky at TRUE coords + SHIFTED hourly times; score kt corruption.

    The phase shift misaligns the clear-sky envelope with the measured ghi, so kt
    departs from its physical [0, ~1.1] band. Two honest fractions are recorded:
      * kt_nonfinite_frac -- fraction of inf/nan (computed WITHOUT an eps floor, so a
        nonzero ghi_actual over a zero (shifted-to-night) clear-sky yields inf).
      * kt_unphysical_frac -- fraction of finite kt exceeding 1.2 (the dominant
        corruption signature once an eps floor is applied, as in the feature pipeline).
    """
    hs = _folsom_hourly_full(loader).iloc[:_CHUNK_HOURS]
    times = hs.index
    ghi_actual = hs.to_numpy(dtype=float)
    rows = []
    for dt in _DT_GRID:
        shifted = pd.DatetimeIndex(times) + pd.DateOffset(minutes=dt)
        ghi_cs = _pvlib_clearsky_ghi(shifted, FOLSOM_TRUE_LAT, FOLSOM_TRUE_LON)
        with np.errstate(divide="ignore", invalid="ignore"):
            kt_raw = ghi_actual / ghi_cs                       # no floor -> inf/nan possible
            kt_floored = ghi_actual / np.maximum(ghi_cs, _EPS)  # eps floor -> finite but huge
        nonfinite = ~np.isfinite(kt_raw)
        unphysical = np.isfinite(kt_floored) & (kt_floored > 1.2)
        rows.append({
            "dt_min": dt,
            "kt_nonfinite_frac": float(nonfinite.mean()),
            "kt_unphysical_frac": float(unphysical.mean()),
            "kt_mean": float(np.nanmean(np.where(nonfinite, np.nan, kt_raw))),
            "kt_median": float(np.nanmedian(np.where(nonfinite, np.nan, kt_raw))),
        })
        _log(f"kt   dt={dt:+4d}  nonfinite={rows[-1]['kt_nonfinite_frac']:.3f}  "
             f"unphysical={rows[-1]['kt_unphysical_frac']:.3f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Task A: Folsom downstream LightGBM skill decay (the "avalanche parabola")
# ---------------------------------------------------------------------------

def _build_lgb_frame(hourly_series, dt_min):
    """Hourly forecast frame with cos_zenith/ghi_clearsky/clearness_index recomputed
    at the SHIFTED timestamps (TRUE Folsom coords). The ghi_actual target is NOT
    shifted -- it keeps its real value, so the features become phase-misaligned with
    the target. At dt=0 the recompute is exact (no corruption).

    ``true_cos_zenith`` (computed at the UN-shifted timestamps) is returned alongside
    so the daytime *evaluator mask* can be held fixed across dt -- otherwise the
    corrupted cos_zenith would also corrupt the sample-selection mask, confounding the
    feature-corruption signal we are trying to isolate.
    """
    shifted = pd.DatetimeIndex(hourly_series.index) + pd.DateOffset(minutes=dt_min)
    ghi_actual = hourly_series.to_numpy(dtype=float)
    cz_shift = compute_astronomy(shifted, FOLSOM_TRUE_LAT, FOLSOM_TRUE_LON)["cos_zenith"].to_numpy()
    ghi_cs = _pvlib_clearsky_ghi(shifted, FOLSOM_TRUE_LAT, FOLSOM_TRUE_LON)
    kt = ghi_actual / np.maximum(ghi_cs, _EPS)
    df = pd.DataFrame({
        "timestamp_utc": pd.DatetimeIndex(hourly_series.index),  # windowing key (unshifted)
        "entity_id": "FOLSOM_STN",
        "ghi_actual": ghi_actual,
        "ghi_clearsky": ghi_cs.astype(float),
        "cos_zenith": cz_shift.astype(float),   # corrupted (shifted-time) cos_zenith
        "clearness_index": kt.astype(float),
    })
    # TRUE (un-shifted) cos_zenith at each row -- used ONLY for the daytime mask, so
    # the scored sample set is identical across dt.
    true_cz = compute_astronomy(
        pd.DatetimeIndex(hourly_series.index), FOLSOM_TRUE_LAT, FOLSOM_TRUE_LON
    )["cos_zenith"].to_numpy()
    return df, true_cz


def _lgb_skill_for_dt(hourly_series, dt_min, true_cz_test_windows):
    """``true_cz_test_windows`` -- the precomputed TRUE issue-time cos_zenith for the
    TEST windows (identical across dt). Caller builds it once from the dt=0 frame so
    the daytime mask is held fixed."""
    df, _ = _build_lgb_frame(hourly_series, dt_min)
    n = len(df)
    n_tr = int(n * _LGB_TRAIN_FRAC)
    df_tr = df.iloc[:n_tr].reset_index(drop=True)
    df_te = df.iloc[n_tr:].reset_index(drop=True)

    eng = EndogenousFeatureEngineer("ghi_actual", feature_cols=list(_LGB_FEATURES))
    xp_tr, y_tr, _, xf_tr, _, _ = eng.create_features(df_tr, _LGB_LOOKBACK, _LGB_HORIZON, 1)
    xp_te, y_te, _, xf_te, _, _ = eng.create_features(df_te, _LGB_LOOKBACK, _LGB_HORIZON, 1)

    lgbm = LightGBMModel(params={"n_estimators": 80, "n_jobs": 1, "verbose": -1, "random_state": 0})
    lgbm.fit(xp_tr, y_tr, None, xf_tr)
    pred = lgbm.predict(xp_te, None, xf_te)

    sp = SmartPersistenceModel()
    sp.fit(xp_tr, y_tr, None, xf_tr)
    sp_pred = sp.predict(xp_te, None, xf_te)

    # Score on the FIXED true-daytime mask (isolates feature corruption from mask
    # corruption). true_cz_test_windows has one entry per test window.
    m = daytime_metrics(y_te, pred, true_cz_test_windows, reference=sp_pred)
    sp_mae = float(np.nanmean(np.abs(y_te - sp_pred)[np.isfinite(y_te - sp_pred)]))
    return {
        "dt_min": dt_min,
        "skill": float(m.get("Daytime_skill", float("nan"))),
        "lgbm_daytime_mae": float(m.get("Daytime_MAE", float("nan"))),
        "sp_daytime_mae": sp_mae,
        "n_daytime": int(m.get("n_daytime", 0)),
        "n_train_windows": int(xp_tr.shape[0]),
        "n_test_windows": int(xp_te.shape[0]),
    }


def task_a_lgb_skill(loader):
    hs = _folsom_hourly_full(loader).iloc[:_LGB_CHUNK]
    # Build the dt=0 frame once to harvest the TRUE issue-time cos_zenith per TEST
    # window -- the fixed daytime mask reused for every dt.
    df0, true_cz = _build_lgb_frame(hs, 0)
    n = len(df0)
    n_tr = int(n * _LGB_TRAIN_FRAC)
    df_te0 = df0.iloc[n_tr:].reset_index(drop=True)
    eng0 = EndogenousFeatureEngineer("ghi_actual", feature_cols=list(_LGB_FEATURES))
    _, _, _, _, cz_te0, _ = eng0.create_features(df_te0, _LGB_LOOKBACK, _LGB_HORIZON, 1)
    # cz_te0 comes from the dt=0 (true) cos_zenith column -> the fixed mask.
    true_cz_test_windows = cz_te0
    rows = []
    for dt in _DT_GRID:
        t0 = time.time()
        row = _lgb_skill_for_dt(hs, dt, true_cz_test_windows)
        rows.append(row)
        _log(f"lgb  dt={dt:+4d}  skill={row['skill']:+.4f}  "
             f"({time.time()-t0:.1f}s, n_day={row['n_daytime']})")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Task B: Folsom blind inversion (3 deg grid, full grid dump)
# ---------------------------------------------------------------------------

def task_b_blind(loader):
    times, power = _folsom_hourly(loader)
    t0 = time.time()
    res = recover_geo_clearsky_global(
        times, power,
        sample_size=len(power),
        lat_grid=_LAT_GRID3, lon_grid=_LON_GRID3,
        top_n=10, return_grid=True,
    )
    d = res.get("details", {})
    rec_lat, rec_lon = res.get("latitude"), res.get("longitude")
    err = _haversine_km(FOLSOM_TRUE_LAT, FOLSOM_TRUE_LON, rec_lat, rec_lon) if rec_lat is not None else float("nan")
    mirror = d.get("mirror_pairs", [])
    _log(f"blind recovered=({rec_lat}, {rec_lon})  truth=({FOLSOM_TRUE_LAT}, {FOLSOM_TRUE_LON})  "
         f"err={err:.0f} km  mse={d.get('best_mse')}  mirrors={len(mirror)}  "
         f"({time.time()-t0:.1f}s)")

    # Full-grid dump for the 2D residual contour.
    grid = d.get("grid", [])
    if grid:
        pd.DataFrame(grid).to_csv(OUT / "folsom_grid_mse.csv", index=False)
        _log(f"wrote folsom_grid_mse.csv ({len(grid)} grid points)")

    summary = {
        "dataset": "folsom",
        "target": "ghi_actual",
        "true_lat": FOLSOM_TRUE_LAT,
        "true_lon": FOLSOM_TRUE_LON,
        "recovered_lat": rec_lat,
        "recovered_lon": rec_lon,
        "best_mse": float(d.get("best_mse", float("nan"))),
        "great_circle_error_km": float(err),
        "sunrise_median_utc": float(d.get("sunrise_median_utc", float("nan"))),
        "tz_offset_guess": float(d.get("tz_offset_guess", float("nan"))),
        "n_mirror_pairs": len(mirror),
        "mirror_pairs": mirror,
    }
    return pd.DataFrame([summary])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    loader = SolarUniBenchLoader(str(DATA_ROOT))

    _log("=== Task A: geo phase-shift ablation ===")
    geo = task_a_geo(loader)

    _log("=== Task A: Folsom kt corruption ===")
    kt = task_a_kt_corruption(loader)

    _log("=== Task A: Folsom LightGBM skill decay ===")
    skill = task_a_lgb_skill(loader)

    _log("=== Task B: Folsom blind inversion (3 deg) ===")
    blind = task_b_blind(loader)

    # Merge the Folsom Task-A rows (geo + kt + skill) on dt_min into one ablation CSV.
    folsom_geo = geo[geo.dataset == "folsom"].drop(columns=["dataset"]).rename(
        columns={"recovered_lat": "folsom_recovered_lat", "recovered_lon": "folsom_recovered_lon"}
    )
    gefcom_geo = geo[geo.dataset == "gefcom2014"].drop(columns=["dataset"]).rename(
        columns={"recovered_lat": "gefcom_recovered_lat", "recovered_lon": "gefcom_recovered_lon"}
    )
    ablation = (
        folsom_geo.merge(gefcom_geo, on="dt_min", how="outer", suffixes=("_folsom", "_gefcom"))
                  .merge(kt, on="dt_min", how="outer")
                  .merge(skill, on="dt_min", how="outer")
    )
    # Keep the per-dataset mirror counts and sunrise diagnostics distinct.
    for col in ["best_mse", "sunrise_median_utc", "tz_offset_guess", "n_mirror_pairs"]:
        if f"{col}_folsom" not in ablation.columns:
            ablation = ablation.rename(columns={col: f"{col}_folsom"})
    ablation_path = OUT / "results_geo_ablation_phase_shift.csv"
    ablation.to_csv(ablation_path, index=False)
    _log(f"wrote {ablation_path} ({len(ablation)} rows)")

    blind_path = OUT / "results_folsom_blind_inversion.csv"
    # mirror_pairs is a list -> serialize to a compact string for CSV.
    blind["mirror_pairs_json"] = blind["mirror_pairs"].apply(
        lambda v: "" if not v else str(v)
    )
    blind.drop(columns=["mirror_pairs"]).to_csv(blind_path, index=False)
    _log(f"wrote {blind_path}")

    # Headline numbers to stdout for the report.
    folsom_dt0 = blind.iloc[0]
    _log("HEADLINE: Folsom blind great-circle error = %.0f km"
         % folsom_dt0["great_circle_error_km"])
    skill_dt0 = skill.loc[skill.dt_min == 0, "skill"].iloc[0]
    skill_min = skill["skill"].min()
    _log("HEADLINE: LightGBM skill drop vs dt=0 = %.4f (dt=0 skill=%.4f, min skill=%.4f)"
         % (skill_dt0 - skill_min, skill_dt0, skill_min))
    kt0 = kt.loc[kt.dt_min == 0, "kt_nonfinite_frac"].iloc[0]
    kt_max = kt["kt_nonfinite_frac"].max()
    _log("HEADLINE: kt non-finite fraction range = [%.3f (dt=0), %.3f (max)]" % (kt0, kt_max))


if __name__ == "__main__":
    main()
