"""Folsom clock phase-shift sweep (E10) — modality-wise temporal misalignment response.

Quantifies how a KNOWN clock offset in each input modality degrades a fixed
60-min-ahead GHI forecast on Folsom (1-min cadence, 2014-01 → 2016-12).

Updated spec (user, 2026-08):
* Modalities swept ONE AT A TIME (others pinned at Delta = 0):
    TSI — total-sky-imager features (local imaging modality; Folsom carries no
          GOES channel — the sky camera IS the imaging modality)
    MET — surface meteorology (obs_*)
    NWP — NAM forecasts (nwp_*; inherently day-ahead -> the control arm)
* ASTRONOMICAL ANCHORS LOCKED at Delta = 0 in every cell: cos_zenith(t) and
  ghi_clearsky(t) always describe the CURRENT physical time t (plus the
  endogenous GHI lag). They are never shifted, for any modality, any Delta.
* Target: ground-truth GHI at t + 60 min.
* Omega = [-120, -90, -60, -30, 0, +30, +60, +90, +120] min.
* Daylight mask computed ONCE from the UN-SHIFTED t (cos_zenith > 0.05) and
  frozen; feature shifts are index-only re-reads -> the validation sample
  count is IDENTICAL across all 27 cells.
* Split by calendar months of the 36-month record (no shuffle):
    train      months 1-24   (2014-01 .. 2015-12)
    validation months 25-30  (2016-01 .. 2016-06)   <- sweep here
    test       months 31-36  (2016-07 .. 2016-12)   locked, untouched
* Negative Delta: the sample at t reads the feature stream at t+Delta, i.e.
  the model is handed |Delta|-minute-OLD information (staleness). Positive
  Delta hands it future information (leakage).
* Feature assembly uses strict index-aligned concatenation (join='inner'
  semantics via a single master DatetimeIndex); the target vector and the
  daylight mask are sliced from the SAME frozen index — they cannot move.

Outputs:
  figures_full/results/results_folsom_phase_sweep.csv
  figures_full/folsom_phase_response.png   (two panels: RMSE | MAE)
  stdout: markdown table of best Delta-hat per modality

Run: conda run -n solar_uni python scripts/folsom_phase_sweep.py [--fast]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from solarunibench.loader import SolarUniBenchLoader  # noqa: E402

# ===========================================================================
# PARAMETER BLOCK
# ===========================================================================
TARGET_H_MIN = 60
OMEGA_MIN = [-120, -90, -60, -30, 0, 30, 60, 90, 120]
DAYTIME_ZENITH = 0.05
CADENCE_MIN = 1
RANDOM_STATE = 0
LGBM_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=63,
                   subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
                   random_state=RANDOM_STATE, verbose=-1)

# month blocks over the 36-month record (1-based month ordinals)
TRAIN_MONTHS = (1, 24)
VAL_MONTHS = (25, 30)
TEST_MONTHS = (31, 36)          # locked — never touched by the sweep

MODALITY_COLS = {
    "TSI": ["sky_frame_count", "sky_minute_gap"],
    "MET": ["obs_air_temp", "obs_relhum", "obs_press", "obs_windsp", "obs_winddir"],
    "NWP": ["nwp_dwsw", "nwp_cloud_cover", "nwp_precipitation", "nwp_pressure",
            "nwp_wind-u", "nwp_wind-v", "nwp_temperature", "nwp_rel_humidity"],
}
ANCHOR_COLS = ["ghi_lag", "cos_zenith", "ghi_clearsky"]   # Delta = 0, always

OUT_DIR = _REPO_ROOT / "figures_full" / "results"
FIG_PATH = _REPO_ROOT / "figures_full" / "folsom_phase_response.png"


# ===========================================================================
# Data assembly
# ===========================================================================
def load_base() -> pd.DataFrame:
    """Master frame on a tz-naive 1-min DatetimeIndex (single source of truth)."""
    loader = SolarUniBenchLoader(str(_REPO_ROOT / "data" / "standardized"))
    df = loader.load_tabular("folsom").sort_values("timestamp_utc").reset_index(drop=True)
    ts = pd.DatetimeIndex(pd.to_datetime(df["timestamp_utc"])).tz_localize(None)

    out = pd.DataFrame(index=ts)
    out["ghi"] = df["ghi_actual"].to_numpy(dtype=float)
    h_steps = TARGET_H_MIN // CADENCE_MIN
    out["y"] = out["ghi"].shift(-h_steps)                 # target at t+60
    out["ghi_lag"] = out["ghi"]                           # anchor
    out["cos_zenith"] = df["cos_zenith"].to_numpy(dtype=float)      # anchor (RAW t)
    out["ghi_clearsky"] = df["ghi_clearsky"].to_numpy(dtype=float)  # anchor (RAW t)
    for c in MODALITY_COLS["MET"] + MODALITY_COLS["NWP"]:
        out[c] = df[c].to_numpy(dtype=float)
    return out


def attach_tsi(base: pd.DataFrame) -> pd.DataFrame:
    """Sky-camera (TSI) per-minute statistics from the materialized stores.

    Two features per 1-min slot: frame count (camera availability/activeness)
    and minute-gap since the previous frame (temporal cadence behaviour).
    Both are strictly derived from the stored image timestamps.
    """
    frames = []
    for year in (2014, 2015, 2016):
        import zarr

        p = (_REPO_ROOT / "data" / "standardized" / "folsom"
             / f"sky_images_{year}.zarr")
        if not p.exists():
            continue
        g = zarr.open(str(p), mode="r")
        raw = g["images_timestamps"][:]
        ts = pd.DatetimeIndex(pd.to_datetime(list(raw))).tz_localize(None) \
            if getattr(raw, "dtype", None) != object else pd.DatetimeIndex(
                pd.to_datetime([str(x) for x in raw])).tz_localize(None)
        frames.append(pd.Series(1, index=ts, name="hit"))
    img = pd.concat(frames).sort_index()
    per_min = img.groupby(img.index.floor("min")).size().rename("sky_frame_count")
    gaps = img.index.to_series().diff().dt.total_seconds().fillna(60.0)
    gap_per_min = gaps.groupby(img.index.floor("min")).mean().rename("sky_minute_gap")

    base = base.join(per_min, how="inner")
    base = base.join(gap_per_min, how="inner")
    # inner-join semantics verified: index is the master 1-min grid
    base["sky_frame_count"] = base["sky_frame_count"].fillna(0.0)
    base["sky_minute_gap"] = base["sky_minute_gap"].fillna(60.0)
    return base


# ===========================================================================
# Frozen month-block split (mask on RAW t)
# ===========================================================================
def month_split(base: pd.DataFrame):
    """Positional index arrays for train/val/test from calendar month blocks."""
    first = base.index[0]
    month_ord = ((base.index.year - first.year) * 12 + (base.index.month - first.month)) + 1
    daylight = (base["cos_zenith"] > DAYTIME_ZENITH) & base["y"].notna()

    def block(lo, hi):
        return base.index[(month_ord >= lo) & (month_ord <= hi) & daylight]

    tr = block(*TRAIN_MONTHS)
    va = block(*VAL_MONTHS)
    te = block(*TEST_MONTHS)
    return tr, va, te


# ===========================================================================
# Feature-level shift (index-only; anchors and target never move)
# ===========================================================================
def build_matrix(base: pd.DataFrame, group: str, delta_min: int) -> pd.DataFrame:
    steps = int(round(delta_min / CADENCE_MIN))
    mod_cols = MODALITY_COLS[group]
    shifted = base[mod_cols].shift(steps) if steps else base[mod_cols]
    # strict index alignment: everything comes from the same master grid
    X = base[ANCHOR_COLS].join(shifted, how="inner")
    assert len(X) == len(base), "index drift in feature assembly"
    return X


# ===========================================================================
# Sweep
# ===========================================================================
def run(fast: bool = False) -> int:
    import lightgbm as lgb

    params = dict(LGBM_PARAMS)
    if fast:
        params["n_estimators"] = 80

    base = attach_tsi(load_base())
    tr, va, te = month_split(base)
    print(f"Folsom 36-month split (daylight, y available): "
          f"train={len(tr):,}  val={len(va):,}  test={len(te):,} (locked)",
          flush=True)

    y = base["y"]
    y_tr = y.loc[tr].to_numpy(dtype=float)
    y_va = y.loc[va].to_numpy(dtype=float)

    rows = []
    for group in ("TSI", "MET", "NWP"):
        for delta in OMEGA_MIN:
            X = build_matrix(base, group, delta)
            Xtr = X.loc[tr].to_numpy(dtype=np.float32)
            Xva = X.loc[va].to_numpy(dtype=np.float32)
            model = lgb.LGBMRegressor(**params)
            model.fit(Xtr, y_tr)
            pred = model.predict(Xva)
            rmse = float(np.sqrt(np.mean((pred - y_va) ** 2)))
            mae = float(np.mean(np.abs(pred - y_va)))
            rows.append({"modality": group, "delta_min": delta,
                         "rmse": rmse, "mae": mae, "n_val": int(len(y_va))})
            print(f"  {group:<4} Δ={delta:+4d} min  RMSE={rmse:7.2f}  MAE={mae:7.2f}",
                  flush=True)
    res = pd.DataFrame(rows)
    assert res.n_val.nunique() == 1, "validation sample count drifted!"

    for m in ("rmse", "mae"):
        base_vals = res[res.delta_min == 0].set_index("modality")[m]
        res[f"d{m}_pct"] = res.apply(
            lambda r: (r[m] / base_vals[r.modality] - 1.0) * 100.0, axis=1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv = OUT_DIR / "results_folsom_phase_sweep.csv"
    res.to_csv(csv, index=False)
    print(f"\nwrote {csv}", flush=True)

    print("\n| Modality | best Δ̂ (min) | RMSE @Δ̂ | ΔRMSE% | MAE @Δ̂ | ΔMAE% |")
    print("|---|---|---|---|---|---|")
    for g, d in res.groupby("modality"):
        b = d.loc[d.rmse.idxmin()]
        print(f"| {g} | {int(b.delta_min):+d} | {b.rmse:.2f} | {b.drmse_pct:+.2f}% "
              f"| {b.mae:.2f} | {b.dmae_pct:+.2f}% |")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"TSI": "#56B4E9", "MET": "#E69F00", "NWP": "#009E73"}
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4))
    for ax, metric, ylab in ((axes[0], "rmse", "Validation RMSE (W m$^{-2}$)"),
                             (axes[1], "mae", "Validation MAE (W m$^{-2}$)")):
        for g, d in res.groupby("modality"):
            d = d.sort_values("delta_min")
            ax.plot(d.delta_min, d[metric], marker="o", markersize=4.5,
                    linewidth=1.4, color=colors[g], label=g)
        ax.axvline(0, color="0.6", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Clock phase shift Δ (min)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_title("(a) RMSE response", loc="left", fontsize=10)
    axes[1].set_title("(b) MAE response", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
    print(f"wrote {FIG_PATH}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="reduced n_estimators")
    args = ap.parse_args()
    return run(fast=args.fast)


if __name__ == "__main__":
    sys.exit(main())
