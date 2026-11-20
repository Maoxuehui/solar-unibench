"""E10b-contiguous — Folsom governance waterfall on the E8-aligned protocol.

Branch companion to scripts/experiments/e9_contiguous.py: reruns the
3-stage progressive ablation (Raw -> +Hygiene -> +Clear-sky) for fig08
panel (b) under the SAME strict full-data contiguous regime as panel (a):

  * full Folsom record (no subsampling — the legacy e10b_waterfall_met.py
    used a 20k-row uniform subsample and L=12/H=1/stride=12 rows),
  * frozen E8 splits (test governed windows as the scoring set),
  * L=720 / H=60 / stride=30 (12 h -> 1 h physical), strict per-window
    contiguity, daytime mask cos_zenith > 0.05,
  * LightGBM probe (E8-FULL preset), scored on the governed test against
    the same daytime MAE basis as the legacy waterfall for comparability
    of the RELATIVE stage deltas.

Stages (cumulative, target identical across stages — only features grow):
  1. raw          target-lag only (anchors dropped, ffill/0 NaN policy)
  2. +hygiene     label-side governance applied to the target
                  (night-zero cos_zenith<=0, clip lower 0) — mirrors
                  operators.quality.apply_quality_audit's label actions
                  on a target-only feature set
  3. +clear_sky   adds the deterministic anchors as FEATURES
                  [cos_zenith, ghi_clearsky, clearness_index_aligned]
                  (E8's exact governed feature block)

Outputs:
  results/results_e10b_waterfall_contiguous.csv
  stdout markdown summary

Run (conda env solar_uni):
    python scripts/experiments/e10b_waterfall_contiguous.py [--quick]
"""
from __future__ import annotations

import os

os.environ.setdefault("SUB_FULL_CONFIG", "1")

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

sys.path.insert(0, str(_REPO_ROOT / "scripts" / "experiments"))

from e9_contiguous import (  # noqa: E402
    _DAYTIME,
    ContiguousEngineer,
    run_tabular_arm,
)
from solarunibench.benchmark.evaluator import evaluate_predictions  # noqa: E402
from solarunibench.benchmark.features import add_clearness_aligned  # noqa: E402
from solarunibench.forecast.models.lightgbm_model import (  # noqa: E402
    LightGBMModel,
)
from solarunibench.forecast.models.smart_persistence import (  # noqa: E402
    SmartPersistenceModel,
)
from solarunibench.loader import SolarUniBenchLoader  # noqa: E402

OUT = _REPO_ROOT / "results" / "results_e10b_waterfall_contiguous.csv"
DS, TARGET = "folsom", "ghi_actual"
LB, HZ = 720, 60
STRIDE = 30                      # E8 FULL map for 1-min data
DT = 60.0
ANCHORS = ("ghi_clearsky", "cos_zenith", "clearness_index",
           "clearness_index_aligned", "dni_clearsky", "dhi_clearsky")


def s_hygiene(df: pd.DataFrame) -> pd.DataFrame:
    """Label-side governance on the target only (night-zero + clip >= 0).

    ``cos_zenith`` is the governance OPERATOR's input (not a model feature:
    this stage still runs the target-only engineer) — merged in by caller.
    """
    df = df.copy()
    night = df["cos_zenith"] <= 0
    df.loc[night, TARGET] = 0.0
    df[TARGET] = df[TARGET].clip(lower=0.0)
    return df


def _with_cols(raw: pd.DataFrame, gov: pd.DataFrame,
               cols: list[str]) -> pd.DataFrame:
    """Merge deterministic anchor columns from the governed frame onto the
    raw frame by (entity_id, timestamp_utc) — anchors are astronomy-derived
    and identical across arms; only the TARGET differs."""
    add = gov[["entity_id", "timestamp_utc", *cols]]
    return raw.merge(add, on=["entity_id", "timestamp_utc"], how="inner")


def main() -> int:
    quick = "--quick" in sys.argv
    stride = LB + HZ if quick else STRIDE
    loader = SolarUniBenchLoader(str(_REPO_ROOT / "data" / "standardized"))

    gov_tr = add_clearness_aligned(loader.load_tabular(DS, "train"), TARGET)
    gov_te = add_clearness_aligned(loader.load_tabular(DS, "test"), TARGET)

    # Raw tabular frame via the e9 helper (full record, governed splits;
    # anchors dropped there, re-merged here per stage as the design demands).
    from e9_contiguous import raw_frames

    raw_tr, _ = raw_frames(loader, DS, TARGET)
    if quick:
        gov_tr = gov_tr.groupby("entity_id", group_keys=False).tail(3000)
        gov_te = gov_te.groupby("entity_id", group_keys=False).tail(1200)
        raw_tr = raw_tr.groupby("entity_id", group_keys=False).tail(3000)

    raw_hyg = _with_cols(raw_tr, gov_tr, ["cos_zenith"])       # stage 2 input
    raw_csk = _with_cols(raw_tr, gov_tr,
                         ["cos_zenith", "ghi_clearsky",
                          "clearness_index_aligned"])           # stage 3 input
    stages = [
        ("raw", raw_tr, False),
        ("+hygiene", s_hygiene(raw_hyg), False),
        ("+clear_sky", raw_csk, True),   # anchors as FEATURES (E8 block)
    ]

    # Shared SP reference + daytime basis on the governed test.
    res_sp = run_tabular_arm(DS, TARGET, gov_tr, gov_te,
                             SmartPersistenceModel(), LB, HZ, stride, DT, True)
    eng = ContiguousEngineer(TARGET, feature_cols=["cos_zenith",
                                                    "ghi_clearsky"],
                             dt_seconds=DT)
    _, _, _, _, cz, _ = eng.create_features(gov_te, LB, HZ, stride)
    sp_met = evaluate_predictions(res_sp["y_true"], res_sp["pred"], cz,
                                  res_sp["pred"], threshold=_DAYTIME)
    print(f"[SP] daytime MAE={sp_met['MAE']:.5f}  "
          f"windows={len(res_sp['y_true'])}", flush=True)

    rows = []
    prev = None
    for name, train_frame, governed in stages:
        model = LightGBMModel(params={"n_estimators": 30, "verbose": -1,
                                      "n_jobs": -1, "random_state": 0}) \
            if quick else LightGBMModel()
        res = run_tabular_arm(DS, TARGET, train_frame, gov_te, model,
                              LB, HZ, stride, DT, governed)
        met = evaluate_predictions(res_sp["y_true"], res["pred"], cz,
                                   res_sp["pred"], threshold=_DAYTIME)
        delta = None if prev is None else (met["MAE"] / prev - 1.0) * 100
        rows.append({"stage": name, "mae": met["MAE"],
                     "rmse": met["RMSE"],
                     "skill_mae_pct": met["Skill_MAE_pct"],
                     "delta_mae_pct": delta,
                     "n_daytime": met["n_daytime"]})
        prev = met["MAE"]
        print(f"  {name:12s} MAE={met['MAE']:.5f} "
              f"({'' if delta is None else f'{delta:+.2f}%'}) "
              f"skill={met['Skill_MAE_pct']:+.1f}%", flush=True)

    out_dir = OUT.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    out = OUT if not quick else OUT.with_name(
        "results_e10b_waterfall_contiguous_quick.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
