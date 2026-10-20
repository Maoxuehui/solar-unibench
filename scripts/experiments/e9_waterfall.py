"""Phase 3 — Governance waterfall ablation (Folsom).

A unified model (LightGBM — the crossover criterion's "simplest non-trivial"
model; DLinear was the earlier probe) is trained at each governance level on the SAME train
rows, scored on a FIXED cleaned test set (clean ground truth). Each step adds
one governance operator; MAE should DECREASE as governance accumulates — the
per-operator contribution to the headline "governance > complexity" gain.

Steps (accumulating):
  raw            -> baseline (no governance)
  + night        -> zero target where cos_zenith <= 0 (drop night sensor noise)
  + range_clamp  -> clip irradiance >= 0, PV to [0, capacity] (physical bounds)
  + utc_align    -> regularize to a clean UTC cadence (dedup + sort)
  + clearsky     -> add ghi_clearsky + cos_zenith as anchor features

Output: ``figures_full/results/results_e9_waterfall.csv`` + printed table.
"""
import os

os.environ.setdefault("SUB_FULL_CONFIG", "1")
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from solarunibench.forecast.core import EndogenousFeatureEngineer
from solarunibench.forecast.models import LightGBMModel
from solarunibench.forecast.scaling import PhysicalFeatureScaler
from solarunibench.pipeline.adapters.folsom import FolsomPipeline
from solarunibench.operators.quality import apply_quality_audit
_REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = Path(os.environ.get(
    "SOLAR_RAW_ROOT",
    "D:/monica/workspace/solar_benchmark_dataset"))

DATAROOT = str(RAW_ROOT)
TARGET = "ghi_actual"
LOOKBACK, HORIZON, STRIDE = 12, 1, 12
DAYTIME = 0.05
RAW_LIMIT = None  # load FULL raw (all 3 years), then uniform-subsample below


def build_raw() -> pd.DataFrame:
    tmp = tempfile.mkdtemp(prefix="folsom_wf_", dir=tempfile.gettempdir())
    FolsomPipeline(
        raw_data_dir=f"{DATAROOT}/Folsom_data", output_dir=tmp, dataset_name="fw",
    ).run_pipeline(raw=True, limit=RAW_LIMIT)
    return pd.read_parquet(f"{tmp}/fw/time_series.parquet")


# -- governance steps (each returns a new frame; cumulative) ----------------
def s_raw(df):
    return df


def s_night(df):
    df = df.copy()
    night = df["cos_zenith"] <= 0
    for c in (TARGET, "pv_power_normalized"):
        if c in df and df[c].notna().any():
            df.loc[night, c] = 0.0
    return df


def s_clamp(df):
    df = df.copy()
    if TARGET in df:
        df[TARGET] = df[TARGET].clip(lower=0.0)
    if "pv_power_normalized" in df and df["pv_power_normalized"].notna().any():
        df["pv_power_normalized"] = df["pv_power_normalized"].clip(lower=0.0, upper=1.0)
    return df


def s_utc(df):
    return (
        df.sort_values(["entity_id", "timestamp_utc"], kind="mergesort")
          .drop_duplicates(["entity_id", "timestamp_utc"]).reset_index(drop=True)
    )


STEPS = [
    ("raw", s_raw, ()),
    ("+night", s_night, ()),
    ("+range_clamp", s_clamp, ()),
    ("+utc_align", s_utc, ()),
    ("+clearsky_features", s_raw, ("ghi_clearsky", "cos_zenith")),
]


def build_windows(df, feature_cols):
    eng = EndogenousFeatureEngineer(TARGET, feature_cols=list(feature_cols))
    X, Y, *_ = eng.create_features(df, LOOKBACK, HORIZON, STRIDE)
    return X, Y


def scaled_windows(train_df, test_df, fcols):
    """Fit PhysicalFeatureScaler on TRAIN only (zero leakage), transform both,
    then build daytime windows. Returns (Xtr, Ytr, Xte, Yte)."""
    scaler = PhysicalFeatureScaler().fit(train_df)
    tr = scaler.transform(train_df)
    te = scaler.transform(test_df)
    Xtr, Ytr = build_windows(tr, fcols)
    Xte, Yte = build_windows(te, fcols)
    day = daytime_idx(te)  # daytime mask on the (sliced) test frame
    return Xtr, Ytr, Xte[day], Yte[day]


def daytime_idx(df):
    """Row-aligned boolean mask of daytime rows (cos_zenith > threshold), on the
    ISSUE timestamp of each window (lookback-th row of each sliding window)."""
    # windows start at 0..n-lookback-horizon; issue time is at start+lookback
    cz = df["cos_zenith"].to_numpy()
    last_start = len(df) - LOOKBACK - HORIZON
    issue = np.array([cz[s + LOOKBACK] for s in range(0, last_start + 1, STRIDE)])
    return issue > DAYTIME


def main() -> int:
    print("Building Folsom raw + fully-cleaned reference...", flush=True)
    raw = build_raw()
    # Uniform-stride subsample across the full loaded range (was implicit
    # head(N) via RAW_LIMIT). 20000 rows spanning multiple seasons, not just Jan.
    TARGET = 20000
    if len(raw) > TARGET:
        stride = len(raw) / TARGET
        idx = np.arange(0, len(raw), stride).round().astype(np.intp)[:TARGET]
        raw = raw.iloc[idx].reset_index(drop=True)
        print(f"  uniform subsampled: {len(raw)} rows "
              f"({pd.to_datetime(raw['timestamp_utc']).min()} -> "
              f"{pd.to_datetime(raw['timestamp_utc']).max()})", flush=True)
    cleaned = apply_quality_audit(raw.copy())
    # Fixed temporal split (per-entity 70/30) on the row-aligned raw/cleaned.
    n = len(raw)
    n_tr = int(n * 0.70)
    test_idx = np.arange(n_tr, n)
    cleaned_test = cleaned.iloc[test_idx].reset_index(drop=True)

    rows = []
    cum = raw.copy()
    for label, fn, fcols in STEPS:
        if fn is not s_raw or label == "raw":
            cum = fn(cum)  # apply the governance step cumulatively (raw is identity)
        # Train = governed TRAIN slice; Test = FIXED cleaned test. Scaler fit on
        # TRAIN only (zero leakage), applied to both — without it the clearsky
        # features (~1000 W/m^2) blow up the linear model.
        train = cum.iloc[:n_tr].reset_index(drop=True)
        Xtr, Ytr, Xte, Yte = scaled_windows(train, cleaned_test, fcols)
        if len(Ytr) < 8 or len(Yte) < 8:
            print(f"  {label}: too few windows (train={len(Ytr)} test={len(Yte)})", flush=True)
            continue
        # LightGBM probe (DLinear removed from the display set 2026-08): the
        # SAME "simplest non-trivial" model the crossover criterion uses, so
        # the waterfall's per-operator contributions are consistent with the
        # headline E9 evidence. FULL preset (SUB_FULL_CONFIG=1) n_estimators.
        from solarunibench.experiments._config import LGBM_PARAMS

        m = LightGBMModel(params=LGBM_PARAMS)
        m.fit(Xtr, Ytr)
        pred = m.predict(Xte)
        mae = float(np.mean(np.abs(np.asarray(pred) - np.asarray(Yte))))
        rows.append({"step": label, "feature_cols": ",".join(fcols) or "(target_only)",
                     "daytime_mae": mae, "n_train": int(len(Ytr)), "n_test": int(len(Yte))})
        print(f"  {label:>20} | features={fcols or 'target'} | MAE={mae:.4f}", flush=True)
        del m
        try:
            import torch as _t
            if _t.cuda.is_available():
                _t.cuda.empty_cache()
        except Exception:
            pass

    out = Path("figures_full/results/results_e9_waterfall.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
