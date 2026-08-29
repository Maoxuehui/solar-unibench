"""E10b-MET — Folsom governance waterfall, MET-only variant (no phase shift).

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
  figures_full/results/results_e10b_waterfall_met.csv
  figures_full/fig_e10b_waterfall_met.png (waterfall, both variants)
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
    # v5 — MET-INFORMED BASELINE: target-lag + the 5 raw MET columns are in
    # the feature set from Stage 1 (no baseline-shift artifacts; every later
    # delta is measured on the fully-informed base). Counterfactual branch:
    # +30 min MET shift applied on top of Stage 2 (sub-sampled discrete grid),
    # then clear-sky layered on the distorted space.
    MET_F = tuple(MET_COLS)
    feature_sets = {
        "base": MET_F,
        "+hygiene": MET_F,
        "+clear_sky": MET_F + ("ghi_clearsky", "cos_zenith"),
    }

    rows = []

    def run_cell(waterfall: str, stage: str, frame: pd.DataFrame, fcols,
                 ref_mae) -> float:
        train = frame.iloc[:n_tr].reset_index(drop=True)
        test = cleaned_ref.iloc[test_slice].reset_index(drop=True)
        r = eval_stage(train, test, fcols)
        r.update({"stage": stage, "waterfall": waterfall,
                  "delta_mae_pct": (None if ref_mae is None
                                    else (r["mae"] / ref_mae - 1.0) * 100.0)})
        rows.append(r)
        return r["mae"]

    # --- MAIN 3-stage pipeline (MET-informed baseline) -----------------------
    base_mae = run_cell("main_3stage", "base", raw, feature_sets["base"], None)
    print(f"  [main ] base          MAE={base_mae:.4f}", flush=True)
    hyg_mae = run_cell("main_3stage", "+hygiene", gov,
                       feature_sets["+hygiene"], base_mae)
    print(f"  [main ] +hygiene      MAE={hyg_mae:.4f}", flush=True)
    cs_mae = run_cell("main_3stage", "+clear_sky", gov,
                      feature_sets["+clear_sky"], hyg_mae)
    print(f"  [main ] +clear_sky    MAE={cs_mae:.4f}", flush=True)

    # --- COUNTERFACTUAL stress branch: RETIRED in this variant (user: this
    # experiment excludes the phase-shift alignment). The main 3-stage
    # MET-informed pipeline above is the complete result.

    res = pd.DataFrame(rows)[["waterfall", "stage", "mae", "delta_mae_pct", "n_test"]]
    out = Path("figures_full/results/results_e10b_waterfall_met.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"\nwrote {out}", flush=True)

    m = dict(base=base_mae, hyg=hyg_mae, cs=cs_mae)
    print("")
    print("| branch | stage | MAE | delta% |")
    print("|---|---|---|---|")
    print(f"| main | base (MET-informed) | {m['base']:.4f} | — |")
    print(f"| main | +hygiene | {m['hyg']:.4f} | {(m['hyg']/m['base']-1)*100:+.2f}% |")
    print(f"| main | +clear_sky | {m['cs']:.4f} | {(m['cs']/m['hyg']-1)*100:+.2f}% |")
    print(f"| main | NET | {m['cs']:.4f} | {(m['cs']/m['base']-1)*100:+.2f}% |")

    # ---------------- figure -------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from solarunibench.viz.style import set_style

    set_style("light")
    # MET-only 3-stage waterfall (phase-shift counterfactual excluded here).
    # ESSD pass: raw unrounded metrics from the run (no hardcoded numbers),
    # 5-decimal bar labels so the -0.15% hygiene no-op reads 0.05304 ->
    # 0.05296, no embedded title (caption owns the description), gentler
    # arcs + generous top headroom so no text or curve touches the boundary.
    stages = ["Base\n(Target-lag + 5 MET)", "+Hygiene", "+Clear-sky"]
    main_mae = [base_mae, hyg_mae, cs_mae]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    colors = ["#08306b", "#2171b5", "#009E73"]
    x = np.arange(3)
    ax.bar(x, main_mae, width=0.55, facecolor=colors, edgecolor="none")
    span = max(main_mae) - min(main_mae)
    for xi, v in zip(x, main_mae):
        ax.text(xi, v + 0.12 * span, f"{v:.5f}", ha="center", va="bottom",
                fontsize=9, color="black")
    for i in (1, 2):
        pct = (main_mae[i] / main_mae[i - 1] - 1) * 100
        ax.annotate("", xy=(i, main_mae[i]), xytext=(i - 1, main_mae[i - 1]),
                    arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.0,
                                    connectionstyle="arc3,rad=-0.15"))
        ax.text(i - 0.5, max(main_mae[i], main_mae[i - 1]) + 0.28 * span,
                f"{pct:+.2f}%", ha="center", va="bottom", fontsize=8.5,
                color="#C04040")
    ax.set_xticks(x, stages, fontsize=9)
    ax.set_ylabel("Daytime MAE (LightGBM, scaled)")
    ax.set_ylim(0, max(main_mae) + 0.42 * span)   # breathing room, no title
    fig.tight_layout()
    fig_path = Path("figures_full/fig_e10b_waterfall_met.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {fig_path} (+ .pdf)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
