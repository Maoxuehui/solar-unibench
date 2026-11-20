"""Patch-up: fill the two raw/CrossViViT cells missed by the in-flight run.

The full chain ran with pre-fix resize_frames (hard 4-channel check), so
skippd and folsom raw/CrossViViT (RGB 3-channel arm) failed. This reruns
ONLY those cells with the fixed code, merges them into
results/results_e9_contiguous.csv, recomputes chi and the markdown summary,
and re-renders fig08.

Run: python scripts/experiments/e9_patch_raw_crossvivit.py
"""
from __future__ import annotations

import os

os.environ.setdefault("SUB_FULL_CONFIG", "1")

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from e9_contiguous import (  # noqa: E402
    OUT_CELLS,
    OUT_CHI,
    OUT_MD,
    ROW_PARAMS,
    FULL_STRIDE,
    _expand_night_grid,
    _row,
    compute_chi,
    governed_frames,
    raw_frames,
    run_crossvivit_arm,
    write_markdown,
)
from solarunibench.benchmark.config import DATASET_CONFIG  # noqa: E402
from solarunibench.loader import SolarUniBenchLoader  # noqa: E402


def main() -> int:
    loader = SolarUniBenchLoader(str(_REPO / "data" / "standardized"))
    df = pd.read_csv(OUT_CELLS)
    for ds in ("skippd", "folsom"):
        cfg = DATASET_CONFIG[ds]
        target = cfg.target
        lb, hz = ROW_PARAMS[ds]
        st = FULL_STRIDE[cfg.sampling_mins]
        dt = cfg.sampling_mins * 60.0
        print(f"\n=== patch {ds} raw/CrossViViT ===", flush=True)
        gov_tr, gov_te = governed_frames(loader, ds, target)
        raw_tr, _ = raw_frames(loader, ds, target)
        gov_tr = _expand_night_grid(gov_tr, target, ds, True)
        gov_te = _expand_night_grid(gov_te, target, ds, True)
        raw_tr = _expand_night_grid(raw_tr, target, ds, False)
        y, pred, cz, sp = run_crossvivit_arm(
            loader, ds, target, raw_tr, gov_te, lb, hz, st, dt,
            governed=False)
        from solarunibench.benchmark.evaluator import evaluate_predictions

        met = evaluate_predictions(y, pred, cz, sp)
        print(f"  raw/CrossViViT MAE={met['MAE']:.5f} "
              f"SkillMAE={met['Skill_MAE_pct']:+.1f}% "
              f"n_day={met['n_daytime']}", flush=True)
        m = (df.dataset == ds) & (df.arm == "raw") & (df.model == "CrossViViT")
        for k, v in (("MAE", met["MAE"]), ("RMSE", met["RMSE"]),
                     ("Skill_MAE_pct", met["Skill_MAE_pct"]),
                     ("Skill_RMSE_pct", met["Skill_RMSE_pct"]),
                     ("n_daytime", met["n_daytime"])):
            df.loc[m, k] = v
    df.to_csv(OUT_CELLS, index=False)
    chi = compute_chi(df)
    chi.to_csv(OUT_CHI, index=False)
    write_markdown(df, chi)
    print(f"\npatched {OUT_CELLS}; rewrote {OUT_CHI} + {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
