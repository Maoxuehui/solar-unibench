"""E10b — Folsom aligned governance waterfall (REAL measurements, 4 stages).

Extends the existing E9 waterfall (Raw -> +Hygiene -> +Clear-sky, LightGBM
probe) with a TRUE intermediate stage: multi-modal phase rectification.
Nothing is pre-assumed — every stage's marginal MAE is measured on the same
frozen windows; the stage contributions sum to whatever they actually sum to.

Stages (cumulative, same grid/scaler/split as scripts/e9_waterfall.py):
  1. raw              absolute raw irradiance sequence, target-lag only
  2. +hygiene         label-side governance (+night, +clamp, +utc) — the
                      stable baseline (MAE ~0.0542 in the sibling experiment)
  3. +phase_rect      MET-stream clock rectification: the surface-met block is
                      re-read at t + delta (delta = +30 min, the E10 sweep's
                      measured optimum for the Folsom surface-met stream);
                      anchors (cos_zenith, clear-sky) stay at t
  4. +clear-sky       deterministic clear-sky + solar-geometry features on
                      top of the rectified baseline

Also re-emits the sibling 3-stage waterfall on the identical frame for a
clean side-by-side (alignment contribution = stage-3 delta measured HERE).

Outputs:
  figures_full/results/results_e10b_aligned_waterfall.csv
  figures_full/fig_e10b_aligned_waterfall.png (waterfall, both variants)
  stdout markdown summary

Run: conda run -n solar_uni python scripts/e10b_aligned_waterfall.py
"""
from __future__ import annotations

import os

os.environ.setdefault("SUB_FULL_CONFIG", "1")

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from solarunibench.forecast.core import EndogenousFeatureEngineer
from solarunibench.forecast.models import LightGBMModel
from solarunibench.forecast.scaling import PhysicalFeatureScaler
from solarunibench.operators.quality import apply_quality_audit
from solarunibench.pipeline.adapters.folsom import FolsomPipeline

DATAROOT = "D:/monica/workspace/solar_benchmark_dataset"
TARGET = "ghi_actual"
LOOKBACK, HORIZON, STRIDE = 12, 1, 12
DAYTIME = 0.05
SUBSAMPLE = 20000
PHASE_DELTA_MIN = 30            # E10 sweep optimum for the surface-MET stream
MET_COLS = ["obs_air_temp", "obs_relhum", "obs_press", "obs_windsp", "obs_winddir"]


def build_raw() -> pd.DataFrame:
    tmp = tempfile.mkdtemp(prefix="folsom_wf2_", dir="D:/tmp")
    FolsomPipeline(
        raw_data_dir=f"{DATAROOT}/Folsom_data", output_dir=tmp, dataset_name="fw2",
    ).run_pipeline(raw=True, limit=None)
    return pd.read_parquet(f"{tmp}/fw2/time_series.parquet")


def s_hygiene(df: pd.DataFrame) -> pd.DataFrame:
    """+night +clamp +utc — the label-side governance bundle (single stage)."""
    df = df.copy()
    night = df["cos_zenith"] <= 0
    for c in (TARGET, "pv_power_normalized"):
        if c in df and df[c].notna().any():
            df.loc[night, c] = 0.0
    df[TARGET] = df[TARGET].clip(lower=0.0)
    return (df.sort_values(["entity_id", "timestamp_utc"], kind="mergesort")
              .drop_duplicates(["entity_id", "timestamp_utc"])
              .reset_index(drop=True))


def phase_rectify(df: pd.DataFrame, delta_min: int) -> pd.DataFrame:
    """Re-read the MET block at t + delta (index-only shift on the sorted grid).

    Anchors (cos_zenith, clear-sky, the target itself) remain at physical
    time t — only the MET stream moves, exactly as in the E10 sweep.
    """
    df = df.copy()
    steps = int(delta_min)  # 1-min cadence -> 1 step per minute
    df[MET_COLS] = df[MET_COLS].shift(steps)
    return df


def build_windows(df, feature_cols):
    eng = EndogenousFeatureEngineer(TARGET, feature_cols=list(feature_cols))
    X, Y, *_ = eng.create_features(df, LOOKBACK, HORIZON, STRIDE)
    return X, Y


def daytime_idx(df):
    cz = df["cos_zenith"].to_numpy()
    last_start = len(df) - LOOKBACK - HORIZON
    issue = np.array([cz[s + LOOKBACK] for s in range(0, last_start + 1, STRIDE)])
    return issue > DAYTIME


def eval_stage(train_df, test_df, fcols) -> dict:
    """Fit/eval LightGBM on the given feature columns; return daytime MAE."""
    scaler = PhysicalFeatureScaler().fit(train_df)
    tr, te = scaler.transform(train_df), scaler.transform(test_df)
    Xtr, Ytr = build_windows(tr, fcols)
    Xte, Yte = build_windows(te, fcols)
    day = daytime_idx(te)
    Xte_d, Yte_d = Xte[day], Yte[day]
    m = LightGBMModel(params={"n_estimators": 500, "learning_rate": 0.05,
                              "num_leaves": 63, "subsample": 0.9,
                              "colsample_bytree": 0.9, "n_jobs": -1,
                              "random_state": 0, "verbose": -1})
    m.fit(Xtr, Ytr)
    pred = m.predict(Xte_d)
    return {"mae": float(np.mean(np.abs(np.asarray(pred) - np.asarray(Yte_d)))),
            "n_test": int(len(Yte_d))}


def main() -> int:
    print("Building Folsom raw (full range, uniform subsample)...", flush=True)
    raw = build_raw()
    if len(raw) > SUBSAMPLE:
        stride = len(raw) / SUBSAMPLE
        idx = np.arange(0, len(raw), stride).round().astype(np.intp)[:SUBSAMPLE]
        raw = raw.iloc[idx].reset_index(drop=True)
        print(f"  subsampled to {len(raw)} rows", flush=True)

    # governance frame + fixed chronological split, IDENTICAL for all stages
    gov = s_hygiene(raw.copy())
    cleaned_ref = apply_quality_audit(raw.copy())   # clean-ground-truth test labels
    n = len(gov)
    n_tr = int(n * 0.70)
    test_slice = slice(n_tr, n)

    # cumulative frames
    frames = {
        "raw": raw,
        "+hygiene": gov,
        "+phase_rect": phase_rectify(gov.copy(), PHASE_DELTA_MIN),
    }
    # The MET stream is a FEATURE in every stage (like the E10 sweep) so the
    # phase shift at stage 3 actually bites; clear-sky adds the anchors on top.
    met_feats = tuple(MET_COLS)
    feature_sets = {
        "raw": met_feats,
        "+hygiene": met_feats,
        "+phase_rect": met_feats,
        "+clear_sky": met_feats + ("ghi_clearsky", "cos_zenith"),
    }
    # For the clear-sky stage the frame is the PHASE-RECTIFIED one (stage 4
    # stacks on stage 3); for the sibling waterfall (no phase stage) it stacks
    # on hygiene directly — measured separately below.

    rows = []
    # --- aligned 4-stage cascade -------------------------------------------
    prev_mae = None
    cum = None
    for stage in ("raw", "+hygiene", "+phase_rect", "+clear_sky"):
        frame = frames.get(stage, frames["+phase_rect"])  # clear_sky reuses #3
        fcols = feature_sets[stage]
        train = frame.iloc[:n_tr].reset_index(drop=True)
        test = cleaned_ref.iloc[test_slice].reset_index(drop=True)
        r = eval_stage(train, test, fcols)
        r.update({"stage": stage, "waterfall": "aligned_4stage",
                  "delta_mae_pct": None if prev_mae is None
                  else (r["mae"] / prev_mae - 1.0) * 100.0})
        rows.append(r)
        prev_mae = r["mae"]
        print(f"  [aligned] {stage:<14} MAE={r['mae']:.4f}"
              f"  ({'' if prev_mae is None else ''})", flush=True)

    # --- sibling 3-stage (hygiene -> clear-sky directly) --------------------
    prev_mae = None
    for stage in ("raw", "+hygiene", "+clear_sky"):
        frame = frames.get(stage, frames["+hygiene"])
        fcols = feature_sets[stage]
        train = frame.iloc[:n_tr].reset_index(drop=True)
        test = cleaned_ref.iloc[test_slice].reset_index(drop=True)
        r = eval_stage(train, test, fcols)
        r.update({"stage": stage, "waterfall": "sibling_3stage",
                  "delta_mae_pct": None if prev_mae is None
                  else (r["mae"] / prev_mae - 1.0) * 100.0})
        rows.append(r)
        prev_mae = r["mae"]
        print(f"  [sibling ] {stage:<14} MAE={r['mae']:.4f}", flush=True)

    res = pd.DataFrame(rows)[["waterfall", "stage", "mae", "delta_mae_pct", "n_test"]]
    out = Path("figures_full/results/results_e10b_aligned_waterfall.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"\nwrote {out}", flush=True)

    a = res[res.waterfall == "aligned_4stage"].set_index("stage")["mae"]
    print("\n| waterfall | raw | +hygiene | +phase_rect | +clear_sky | net |")
    print("|---|---|---|---|---|---|")
    print(f"| aligned 4-stage | {a['raw']:.4f} | {a['+hygiene']:.4f} "
          f"| {a['+phase_rect']:.4f} | {a['+clear_sky']:.4f} "
          f"| {(a['+clear_sky']/a['raw']-1)*100:+.1f}% |")
    print(f"| sibling 3-stage | {a['raw']:.4f} | {a['+hygiene']:.4f} | — "
          f"| {res[(res.waterfall=='sibling_3stage')&(res.stage=='+clear_sky')].mae.iloc[0]:.4f} "
          f"| {(res[(res.waterfall=='sibling_3stage')&(res.stage=='+clear_sky')].mae.iloc[0]/a['raw']-1)*100:+.1f}% |")

    # ---------------- figure -------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from solarunibench.viz.style import set_style

    set_style("light")
    stages = ["Raw", "+Hygiene", "+Phase\nRectification", "+Clear-sky"]
    mae = [a[s] for s in ("raw", "+hygiene", "+phase_rect", "+clear_sky")]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    colors = ["#08306b", "#2171b5", "#D55E00", "#009E73"]
    bars = ax.bar(np.arange(4), mae, width=0.62, facecolor=colors, edgecolor="none")
    span = max(mae) - min(mae)
    for i, v in enumerate(mae):
        ax.text(i, v + 0.012 * span, f"{v:.4f}", ha="center", va="bottom",
                fontsize=9, color="black")
    for i in range(1, 4):
        pct = (mae[i] / mae[i - 1] - 1) * 100
        ax.annotate("", xy=(i, mae[i]), xytext=(i - 1, mae[i - 1]),
                    arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.0,
                                    connectionstyle="arc3,rad=-0.25"))
        ax.text(i - 0.5, max(mae[i], mae[i - 1]) + 0.07 * span, f"{pct:+.1f}%",
                ha="center", fontsize=8.5, color="#C04040")
    ax.set_xticks(np.arange(4), stages, fontsize=9)
    ax.set_ylabel("Daytime MAE (LightGBM, scaled)")
    ax.set_ylim(0, max(mae) + 0.22 * span)
    ax.set_title("Folsom aligned governance waterfall — "
                 "phase rectification measured in-line", fontsize=10, loc="left")
    fig.tight_layout()
    fig_path = Path("figures_full/fig_e10b_aligned_waterfall.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"wrote {fig_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
