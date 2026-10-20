"""Phase 3 — Temporal-granularity sensitivity (Folsom).

Folsom tabular is 1-minute. Resampling to coarser cadences (5 min, 15 min, 1 h)
weakens Smart Persistence (the target moves more between samples), so the
skill denominator grows and deep models' relative skill should flip from
negative (fine cadence, SP near-perfect) toward positive (coarse cadence).

For each granularity: build windows, score Smart Persistence (MAE) and the model probe (LightGBM; was DLinear),
report MAE_SP, MAE_model, and skill = 1 - MAE_model/MAE_SP.
Output: ``figures_full/results/results_e9_granularity.csv``.
"""
import os

os.environ.setdefault("SUB_FULL_CONFIG", "1")
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from solarunibench.forecast.core import EndogenousFeatureEngineer
from solarunibench.forecast.models import LightGBMModel, SmartPersistenceModel
from solarunibench.forecast.scaling import PhysicalFeatureScaler
from solarunibench.loader import SolarUniBenchLoader
_REPO_ROOT = Path(__file__).resolve().parents[2]

TARGET = "ghi_actual"
LOOKBACK, HORIZON, STRIDE = 12, 1, 12
DAYTIME = 0.05
GRANULARITIES = ["1min", "5min", "15min", "1h"]


def mae(p, y):
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(y))))


def resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample to `freq` per entity (mean of the target + clearsky anchors)."""
    cols = ["timestamp_utc", "entity_id", TARGET, "ghi_clearsky", "cos_zenith"]
    have = [c for c in cols if c in df.columns]
    d = df[have].copy()
    d["_ts"] = pd.to_datetime(d["timestamp_utc"])
    out = (
        d.set_index("_ts")
         .groupby("entity_id")
         .resample(freq)
         .mean(numeric_only=True)
         .reset_index()
    )
    out = out.rename(columns={"_ts": "timestamp_utc"})
    # drop rows that lost the entity (resample edge); keep daytime finite
    out = out.dropna(subset=[TARGET, "cos_zenith"]).reset_index(drop=True)
    return out


def daytime_mask(df):
    cz = df["cos_zenith"].to_numpy()
    last_start = len(df) - LOOKBACK - HORIZON
    if last_start < 1:
        return np.array([], dtype=bool)
    issue = np.array([cz[s + LOOKBACK] for s in range(0, last_start + 1, STRIDE)])
    return issue > DAYTIME


def eval_granularity(df: pd.DataFrame, freq: str) -> dict:
    """Score SmartPersistence + the model probe at one granularity, MAE in PHYSICAL W/m^2.

    SP is computed on the UNSCALED test frame (physical ghi). The probe trains on
    scaled features (for stability) but its predictions + the target are inverted
    (xIRRADIANCE_MAX) so its MAE is also in W/m^2 — comparable across granularities
    (the previous per-granularity scaling made MAEs non-comparable).
    """
    n = len(df)
    if n < 200:
        return {"granularity": freq, "n_rows": n, "status": "too_few_rows"}
    n_tr = int(n * 0.70)
    train, test = df.iloc[:n_tr].reset_index(drop=True), df.iloc[n_tr:].reset_index(drop=True)
    scaler = PhysicalFeatureScaler().fit(train)
    tr, te = scaler.transform(train), scaler.transform(test)

    eng = EndogenousFeatureEngineer(TARGET, feature_cols=["ghi_clearsky", "cos_zenith"])
    Xtr, Ytr, *_ = eng.create_features(tr, LOOKBACK, HORIZON, STRIDE)
    Xte, Yte, *_ = eng.create_features(te, LOOKBACK, HORIZON, STRIDE)
    day = daytime_mask(te)
    if day.sum() < 8 or len(Ytr) < 8:
        return {"granularity": freq, "n_rows": n, "status": "too_few_daytime_windows"}
    Xte_d, Yte_d = Xte[day], Yte[day]

    # SmartPersistence on the UNSCALED test (physical W/m^2). Clear-sky-scaled
    # persistence: kappa = target_now/clearsky_now, propagated by clearsky_horizon.
    tgt = test[TARGET].to_numpy()
    gcs = test["ghi_clearsky"].to_numpy()
    last_start = len(test) - LOOKBACK - HORIZON
    idx = list(range(0, last_start + 1, STRIDE))
    eps = 1e-6
    kappa = np.array([tgt[s + LOOKBACK - 1] for s in idx]) / np.maximum(
        np.array([gcs[s + LOOKBACK - 1] for s in idx]), eps)
    sp_pred = np.maximum(
        kappa * np.array([gcs[min(s + LOOKBACK + HORIZON - 1, len(test) - 1)] for s in idx]), 0.0)
    sp_pred_d = sp_pred[day]
    # Physical horizon target (W/m^2) at the same windows.
    y_phys_d = np.array([tgt[min(s + LOOKBACK + HORIZON - 1, len(test) - 1)] for s in idx])[day]
    mae_sp = mae(sp_pred_d, y_phys_d)

    # Probe model on scaled data; invert predictions + target to physical W/m^2.
    irr_max = getattr(PhysicalFeatureScaler(), "irradiance_max", 1500)
    from solarunibench.experiments._config import LGBM_PARAMS
    m = LightGBMModel(params=LGBM_PARAMS)  # probe swapped DLinear -> LightGBM (2026-08)
    m.fit(Xtr, Ytr)
    pred_phys = m.predict(Xte_d) * irr_max
    mae_m = mae(pred_phys, y_phys_d)
    skill = 1.0 - mae_m / mae_sp if mae_sp > 0 else float("nan")
    # Dual metric: normalized MAE = physical MAE / max(GHI) (capacity-style).
    max_ghi = float(np.nanmax(tgt)) if np.isfinite(tgt).any() else 1.0
    max_ghi = max(max_ghi, 1.0)
    del m
    try:
        import torch as _t
        if _t.cuda.is_available():
            _t.cuda.empty_cache()
    except Exception:
        pass
    return {
        "granularity": freq, "n_rows": int(n), "n_daytime_test": int(day.sum()),
        "mae_sp_wm2": mae_sp, "mae_model_wm2": mae_m,
        "nmae_sp": mae_sp / max_ghi, "nmae_model": mae_m / max_ghi,
        "skill_model_vs_sp": skill, "max_ghi_wm2": max_ghi,
        "status": "ran",
    }


def main() -> int:
    loader = SolarUniBenchLoader(str(_REPO_ROOT / "data" / "standardized"))
    df = loader.load_tabular("folsom")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    # Uniform-stride subsample across the FULL 3-year range (was .head(43200)
    # which only covered Jan 2014). Keeps 43200 rows but spans 2014-2016.
    if len(df) > 43200:
        stride = len(df) / 43200
        idx = np.arange(0, len(df), stride).round().astype(np.intp)[:43200]
        df = df.iloc[idx].reset_index(drop=True)
    print(f"Folsom base: {len(df)} rows @ native cadence", flush=True)

    rows = []
    for freq in GRANULARITIES:
        d = df if freq == "1min" else resample(df, freq)
        r = eval_granularity(d, freq)
        rows.append(r)
        if r.get("status") == "ran":
            print(f"  {freq:>4}: n={r['n_rows']} SP={r['mae_sp_wm2']:.1f} W/m^2 "
                  f"(nMAE {r['nmae_sp']:.3f}) | model={r['mae_model_wm2']:.1f} W/m^2 "
                  f"(nMAE {r['nmae_model']:.3f}) | skill={r['skill_model_vs_sp']:+.3f}",
                  flush=True)
        else:
            print(f"  {freq:>4}: {r.get('status')}", flush=True)

    out = Path("figures_full/results/results_e9_granularity.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
