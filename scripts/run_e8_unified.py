#!/usr/bin/env python
"""E8 unified benchmark — one-command reproducible real-data run.

Runs the full 4-dataset x 3-horizon x 9-model grid on the REAL standardized
packages and prints the Daytime Forecast Skill (%) table, with the multi-index
CSV (+ LaTeX) summary written under ``figures/results/``.

This is the Task 6 deliverable: the real-data integration run that exercises
the whole E8 harness end-to-end (config + F=4 clearness feature + daytime-masked
Skill% evaluator) on the four standardized datasets.

Tractability
------------
Two knobs keep the first table tractable on a laptop (per the plan's "TINY
configs for tractability; full-config is a later run" note):

* ``--tiny`` (default) shrinks the deep-model epoch budget to 1 and the MLP
  iteration cap to 10 via :func:`build_models(tiny=True, ...)`.
* ``--bound N`` (default 8000) tails each (dataset, split, entity) frame to its
  last *N* rows before windowing. The four standardized datasets range from 59K
  to 1.55M rows; building all stride-1 windows on the full frames would OOM on
  the 1-min datasets (folsom/skippd). The tail is REAL data — just a bounded
  evaluation window — so the skill numbers are real (if noisier than the
  full-config run; absolute skill will improve with the later large-epoch run).
  Pass ``--bound 0`` to disable bounding (full-scale run).

The deep (torch) models are forced onto CPU (``device="cpu"``) for determinism:
``tiny=True`` exists precisely for tractable/deterministic first tables, and
auto-selecting CUDA on a host whose GPU kernel is broken would make tiny mode
fail. ``--full`` (full-config, later run) keeps the same CPU pin so the two
runs are comparable.

Resilience
----------
Each dataset is run in its own try/except so one slow/OOMing cell cannot block
the whole grid (per the Task 6 brief: "don't let one cell block the whole
grid"); a failed dataset is reported and excluded from the printed table.

Usage
-----
::

    python scripts/run_e8_unified.py                  # tiny + bound=8000 (default)
    python scripts/run_e8_unified.py --bound 0        # full-scale data run
    python scripts/run_e8_unified.py --full           # full-config epochs
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

# Force the torch deep models onto CPU BEFORE torch is imported by the model
# modules below (defense in depth on top of build_models(device="cpu")); on a
# host with a broken CUDA driver this keeps tiny mode deterministic.
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# Make the in-repo ``solarunibench`` package importable when the script is run
# directly (``python scripts/run_e8_unified.py`` puts ``scripts/`` — not the repo
# root — on sys.path). No-op when the package is already installed.
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from solarunibench.benchmark.config import DATASET_CONFIG  # noqa: E402
from solarunibench.benchmark.run_e8 import build_models, run_benchmark  # noqa: E402
from solarunibench.benchmark.summary import (  # noqa: E402
    build_results_table,
    save_results,
)
from solarunibench.loader import SolarUniBenchLoader  # noqa: E402

_DATA_ROOT = "D:/monica/workspace/solar-unibench/data/standardized"
_OUT_DIR = "D:/monica/workspace/solar-unibench/figures/results"
# Headline metric for the printed table + saved CSV (sign: 0 = parity with
# SmartPersistence, + = better, - = worse; already in percent units).
_METRIC = "Skill_MAE_pct"


class BoundedLoader:
    """Wrap a :class:`SolarUniBenchLoader`; tail-bound each split per entity.

    Returns the last ``bound`` rows of every (dataset, split, entity) group so
    windowing stays within-entity (the feature engineer is entity-aware). Rows
    are re-sorted by ``(entity_id, timestamp_utc)`` before the tail so the
    subset is the most-recent contiguous block per entity. ``bound <= 0`` or a
    frame smaller than ``bound`` is returned untouched (full data).
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root", default=_DATA_ROOT, help="standardized data root dir"
    )
    p.add_argument(
        "--out-dir", default=_OUT_DIR, help="where to write e8_summary.csv/.tex"
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="full-config epochs (default: tiny=1 epoch deep / 10 iter MLP)",
    )
    p.add_argument(
        "--bound",
        type=int,
        default=8000,
        help="tail rows per (split, entity); 0 disables bounding (default 8000)",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_CONFIG),
        help="subset of datasets (default: all 4)",
    )
    args = p.parse_args()

    tiny = not args.full
    loader = BoundedLoader(SolarUniBenchLoader(args.data_root), args.bound)
    models = build_models(tiny=tiny, device="cpu")

    print(f"E8 unified benchmark: datasets={args.datasets} tiny={tiny} "
          f"bound={args.bound} device=cpu", flush=True)

    rows: list[dict] = []
    for ds in args.datasets:
        t0 = time.time()
        try:
            ds_rows = run_benchmark(loader, datasets=[ds], models=models, tiny=tiny)
            rows.extend(ds_rows)
            print(
                f"  [OK]   {ds:11s} {len(ds_rows):3d} rows in {time.time()-t0:6.1f}s",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001 - report + continue (resilience)
            print(
                f"  [FAIL] {ds:11s} {type(e).__name__}: {e} "
                f"(after {time.time()-t0:.1f}s)",
                flush=True,
            )
            traceback.print_exc()

    if not rows:
        print("\nNo rows produced — every dataset failed. Aborting.", file=sys.stderr)
        return 1

    # Persist the multi-index CSV (+ LaTeX) summary.
    paths = save_results(rows, args.out_dir, metric=_METRIC)
    print(f"\nWrote: {paths['csv']}")
    print(f"       {paths['tex']}")

    # Headline deliverable: the printed skill table.
    table = build_results_table(rows, metric=_METRIC)
    print(f"\n=== E8 unified benchmark — Daytime Forecast Skill (% MAE vs SmartPersistence) ===")
    print(f"=== rows: Dataset x Horizon | cols: 9 models | sign: 0=parity, +=better, -=worse ===")
    # Skill_MAE_pct is already in percent units; format to 2 dp. Missing cells
    # (e.g. CrossViViT on non-image datasets) render as NaN.
    with pd.option_context("display.float_format", lambda v: f"{v:7.2f}"):
        print(table.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
