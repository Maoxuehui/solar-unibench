"""E8 Task 6 — real-data integration run (RUN_SLOW-gated).

The headline deliverable of Task 6: run the full 4-dataset x 3-horizon x 9-model
grid on the REAL standardized packages, assert the runner returns finite Daytime
Forecast Skill (%) numbers, :func:`save_results` writes the multi-index CSV, and
PRINT the skill table for the record (the printed table is the headline
artifact; pytest ``-s`` surfaces it).

Gating
------
``RUN_SLOW=1`` is required because the test trains models. Without it the test
is skipped (matches the E1-E8 RUN_SLOW convention).

Tractability
------------
Two knobs keep this tractable on a laptop, per the plan's "TINY configs for
tractability; full-config is a later run" note:

* The deep models are built with :func:`build_models(tiny=True, device="cpu")`
  — 1 epoch per deep model, 10 MLP iterations, deterministic CPU (auto-selecting
  CUDA on a host with a broken GPU would make the tiny run fail).
* A :class:`_BoundedLoader` tails each (dataset, split, entity) frame to its
  last ``E8_BOUND`` rows (default 8000) before windowing. The 1-min datasets
  (folsom/skippd) have 0.4-1.6M rows and building all stride-1 windows on them
  would OOM; the tail is REAL data (just a bounded evaluation window). Override
  with ``E8_BOUND=0`` for the full-scale run.

Resilience
----------
Each dataset runs in its own try/except so one slow/OOMing cell cannot block the
whole grid (per the Task 6 brief). The assertions tolerate a failed dataset as
long as enough cells come back finite (the framework-level property under test).
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Force CPU for the torch deep models BEFORE they import torch (defense in depth
# on top of build_models(device="cpu")); keeps tiny mode deterministic on hosts
# whose CUDA driver is broken.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from solarunibench.benchmark.config import DATASET_CONFIG  # noqa: E402
from solarunibench.benchmark.run_e8 import build_models, run_benchmark  # noqa: E402
from solarunibench.benchmark.summary import (  # noqa: E402
    MODELS,
    build_results_table,
    save_results,
)

slow = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW"),
    reason="set RUN_SLOW=1 to run the real-data E8 integration test (slow: trains models)",
)

_REAL_DATA_ROOT = "D:/monica/workspace/solar-unibench/data/standardized"
_ALL_DATASETS = list(DATASET_CONFIG)
_IMAGE_DATASETS = {ds for ds, cfg in DATASET_CONFIG.items() if cfg.image_dataset}
# Tail rows per (split, entity). 8000 keeps the 1-min datasets tractable on a
# laptop; the tail is real data. Override with E8_BOUND=0 for the full run.
_DEFAULT_BOUND = 8000


class _BoundedLoader:
    """Wrap ``SolarUniBenchLoader``; tail-bound each split per entity.

    Mirrors ``scripts/run_e8_unified.py``'s ``BoundedLoader`` so the test and
    the one-command script exercise identical data paths.
    """

    def __init__(self, inner, bound: int):
        self._inner = inner
        self._bound = bound

    def load_tabular(self, dataset: str, split: str | None = None) -> pd.DataFrame:
        df = self._inner.load_tabular(dataset, split)
        if self._bound and self._bound > 0 and len(df) > self._bound:
            df = (
                df.sort_values(["entity_id", "timestamp_utc"])
                .groupby("entity_id", sort=False)
                .tail(self._bound)
                .reset_index(drop=True)
            )
        return df


def _bound_from_env() -> int:
    raw = os.environ.get("E8_BOUND")
    if raw is None:
        return _DEFAULT_BOUND
    try:
        return int(raw)
    except ValueError:
        return _DEFAULT_BOUND


@slow
def test_e8_real_data_full_grid(tmp_path):
    """Run the full real-data grid; assert finite metrics, CSV write, print table."""
    from solarunibench.loader import SolarUniBenchLoader

    bound = _bound_from_env()
    loader = _BoundedLoader(SolarUniBenchLoader(_REAL_DATA_ROOT), bound)
    models = build_models(tiny=True, device="cpu")

    # Run each dataset in its own try/except so a single OOM/slow cell cannot
    # block the grid (Task 6 brief: "don't let one cell block the whole grid").
    rows: list[dict] = []
    seen: set[str] = set()
    for ds in _ALL_DATASETS:
        try:
            rows.extend(
                run_benchmark(loader, datasets=[ds], models=models, tiny=True)
            )
            seen.add(ds)
        except Exception as e:  # noqa: BLE001 - report and continue
            print(
                f"[FAIL] {ds}: {type(e).__name__}: {e}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc()

    # 1. Non-empty grid with at least one dataset represented.
    assert isinstance(rows, list) and len(rows) > 0, "benchmark produced no rows"
    df = pd.DataFrame(rows)

    # 2. SmartPersistence is the Skill%==0 reference and appears in every cell.
    sp = df[df["model"] == "SmartPersistence"]
    assert len(sp) > 0, "SmartPersistence reference missing"
    assert (sp["Skill_MAE_pct"].abs() < 1e-9).all(), (
        "SmartPersistence Skill% must be 0 (parity with itself)"
    )

    # 3. CrossViViT only on image datasets (the multimodal gating).
    cv = df[df["model"] == "CrossViViT"]
    if len(cv):
        assert set(cv["dataset"].unique()) <= _IMAGE_DATASETS, (
            f"CrossViViT appeared on non-image dataset: {set(cv['dataset'].unique())}"
        )

    # 4. The overwhelming majority of cells must report finite MAE/Skill (guards
    #    against a systematic break of the harness). Deep models may gracefully
    #    NaN on an individual tiny cell, so we assert >= 80% finite, not all.
    finite_frac = df["Skill_MAE_pct"].apply(np.isfinite).mean()
    assert finite_frac >= 0.8, f"only {finite_frac:.0%} of cells finite"
    for _, r in df.iterrows():
        if np.isfinite(r["Skill_MAE_pct"]):
            assert np.isfinite(r["MAE"]) and r["MAE"] >= 0
            assert r["n_daytime"] > 0

    # 5. save_results writes the multi-index CSV (+ .tex side-effect) under a
    #    tmp/figures/results dir.
    out_dir = tmp_path / "figures" / "results"
    paths = save_results(rows, out_dir, metric="Skill_MAE_pct")
    assert Path(paths["csv"]).exists() and Path(paths["csv"]).stat().st_size > 0
    csv_df = pd.read_csv(paths["csv"], index_col=[0, 1, 2])
    # One row per (dataset, horizon) that produced results.
    assert len(csv_df) >= len(seen)  # at least one horizon per dataset seen

    # 6. The headline deliverable: print the skill table (pytest -s surfaces it).
    table = build_results_table(rows, metric="Skill_MAE_pct")
    print("\n=== E8 unified benchmark — Daytime Forecast Skill (% MAE vs SmartPersistence) ===")
    print("=== rows: Dataset x Horizon | cols: 9 models | sign: 0=parity, +=better, -=worse ===")
    with pd.option_context("display.float_format", lambda v: f"{v:7.2f}"):
        print(table.to_string())

    # Sanity: every model column in the canonical battery is either present or
    # NaN (CrossViViT NaN on non-image datasets that ran).
    assert set(MODELS) >= set(table.columns)


@slow
def test_e8_real_data_no_mutation(tmp_path):
    """The runner must not mutate the loader's real-data frames (purity)."""
    from solarunibench.loader import SolarUniBenchLoader

    bound = _bound_from_env()
    inner = SolarUniBenchLoader(_REAL_DATA_ROOT)
    # Snapshot the (bounded) tabular content of one small dataset before/after.
    ds = "gefcom2014"
    before = _BoundedLoader(inner, bound).load_tabular(ds, "test").copy(deep=True)
    loader = _BoundedLoader(inner, bound)
    models = [m for m in build_models(tiny=True, device="cpu") if m.name == "LightGBM"]
    run_benchmark(loader, datasets=[ds], horizons=[1], models=models, tiny=True)
    after = _BoundedLoader(inner, bound).load_tabular(ds, "test")
    pd.testing.assert_frame_equal(before, after)
