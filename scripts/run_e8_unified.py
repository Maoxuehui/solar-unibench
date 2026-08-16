#!/usr/bin/env python
"""E8 unified benchmark — one-command reproducible real-data run.

Runs the full 4-dataset x 3-horizon x 9-model grid on the REAL standardized
packages and prints the Daytime Forecast Skill (%) table, with the multi-index
CSV (+ LaTeX) summary written under ``figures/results/``.

This is the Task 6 deliverable: the real-data integration run that exercises
the whole E8 harness end-to-end (config + F=4 clearness feature + daytime-masked
Skill% evaluator) on the four standardized datasets.

Two modes
---------
* ``--tiny`` (default): tractable first table. Deep models at 1 epoch, MLP at
  10 iterations, ``device="cpu"`` for determinism, and each (split, entity)
  frame tailed to its last ``--bound N`` rows (default 8000) so stride-1
  windowing of the 1-min datasets stays in memory.
* ``--full`` (the E8 FULL-config upgrade): trains the deep models for 50 epochs
  with patience=7 val-based early stopping, AdamW + CosineAnnealingLR,
  effective batch 64 (CrossViViT accumulates 4 micro-batches of 16 for 24GB at
  256x256), and ``device="cuda"`` on the FULL multi-year data range. To keep
  stride-1 windowing of 1.55M rows x lookback-240 tractable, FULL mode windows
  with a per-sampling-rate stride (1-min → 30, 30-min/1-h → 1) — full SEASONAL
  coverage at a tractable ~50k windows, not every stride-1 window. With a FULL
  stride the ``--bound`` tail defaults to 0 (all rows). Each cell is wrapped in
  a CUDA→CPU fallback so a transient CUDA error degrades instead of killing the
  grid.

Resilience
----------
Each dataset is run in its own try/except so one slow/OOMing cell cannot block
the whole grid (per the Task 6 brief: "don't let one cell block the whole
grid"); a failed dataset is reported and excluded from the printed table.

Usage
-----
::

    python scripts/run_e8_unified.py                  # tiny + bound=8000 (default)
    python scripts/run_e8_unified.py --bound 0        # tiny, full-scale data
    python scripts/run_e8_unified.py --full           # FULL config on CUDA
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

# NOTE: the tiny run pins device="cpu" explicitly via build_models(device=...),
# so we do NOT blanket-disable CUDA here (that would break the FULL GPU run).
# ``run_with_cuda_fallback`` (per-cell) is the safety net for transient errors.

# Make the in-repo ``solarunibench`` package importable when the script is run
# directly (``python scripts/run_e8_unified.py`` puts ``scripts/`` — not the repo
# root — on sys.path). No-op when the package is already installed.
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from solarunibench.benchmark.config import DATASET_CONFIG  # noqa: E402
from solarunibench.benchmark.run_e8 import (  # noqa: E402
    build_models,
    probe_cuda,
    run_benchmark,
)
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
        help="FULL config: 50 epochs + patience=7 early stop + AdamW+Cosine + "
        "effective batch 64 (CrossViViT accum 4) + CUDA + full-data stride "
        "(default: tiny=1 epoch deep / 10 iter MLP on CPU)",
    )
    p.add_argument(
        "--bound",
        type=int,
        default=None,
        help="tail rows per (split, entity); 0 disables bounding. "
        "Default: 8000 (tiny) / 0 (full)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="torch device override (default: cpu for tiny, cuda for full)",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_CONFIG),
        help="subset of datasets (default: all 4)",
    )
    args = p.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    tiny = not args.full
    # FULL stride makes full-data windowing tractable → no tail bound needed;
    # tiny still tails to 8000 rows for a fast deterministic table.
    bound = args.bound if args.bound is not None else (0 if args.full else 8000)
    device = args.device or ("cuda" if args.full else "cpu")

    cuda_ok = probe_cuda(device) if device.startswith("cuda") else False
    if args.full:
        if device.startswith("cuda") and not cuda_ok:
            print(
                f"WARNING: --full requested device={device} but the CUDA probe "
                f"failed; falling back to cpu for this run.",
                flush=True,
            )
            device = "cpu"
        print(
            f"  FULL config: epochs=50 patience=7 batch=64 (CrossViViT accum=4) "
            f"AdamW(lr=1e-3,wd=1e-4)+Cosine stride=full-per-dataset "
            f"CUDA={'OK' if cuda_ok else 'n/a'}",
            flush=True,
        )

    loader = BoundedLoader(SolarUniBenchLoader(args.data_root), bound)
    models = build_models(tiny=tiny, full=args.full, device=device)

    print(
        f"E8 unified benchmark: datasets={args.datasets} "
        f"mode={'FULL' if args.full else 'tiny'} bound={bound} device={device}",
        flush=True,
    )

    rows: list[dict] = []
    # Per-dataset checkpoint: write each dataset's rows to disk the moment it
    # completes so an interrupted run (kill/OOM/crash) can resume from here
    # instead of losing the whole grid. On relaunch, datasets already present
    # in the checkpoint are reused as-is.
    ckpt = Path(args.out_dir) / "e8_full_checkpoint.csv"
    done: set[str] = set()
    if ckpt.exists():
        try:
            prev = pd.read_csv(ckpt)
            done = set(prev["dataset"].unique())
            print(f"Checkpoint found: {sorted(done)} — resuming around them.", flush=True)
        except Exception as e:  # noqa: BLE001 - corrupt checkpoint -> rerun all
            print(f"WARNING: could not read checkpoint ({e}); ignoring it.", flush=True)
    for ds in args.datasets:
        if ds in done:
            ds_rows = prev[prev["dataset"] == ds].to_dict("records")
            rows.extend(ds_rows)
            print(
                f"  [SKIP] {ds:11s} {len(ds_rows):3d} rows reused from checkpoint",
                flush=True,
            )
            continue
        t0 = time.time()
        try:
            ds_rows = run_benchmark(
                loader, datasets=[ds], models=models, tiny=tiny, full=args.full
            )
            rows.extend(ds_rows)
            # Checkpoint immediately (append; header only on first write).
            pd.DataFrame(ds_rows).to_csv(
                ckpt, mode="a", header=not ckpt.exists(), index=False
            )
            print(
                f"  [OK]   {ds:11s} {len(ds_rows):3d} rows in {time.time()-t0:6.1f}s "
                f"(checkpointed)",
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
