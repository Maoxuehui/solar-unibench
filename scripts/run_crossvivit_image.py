#!/usr/bin/env python
"""E8 add-on — one-command CrossViViT IMAGE benchmark run (folsom + skippd).

Runs the image-enabled CrossViViT variant on the REAL standardized packages
(sky images attached to fit/predict) at the image datasets' horizons [5,10,15]
and writes ``figures/results/e8_crossvivit_image.csv`` — SEPARATE from the
main tabular-only grid (whose ``e8_summary.csv`` rows are untouched).

FULL config by default: epochs=50 patience=7 AdamW+Cosine, effective batch 64
(CrossViViT micro 16 + accum 4), device cuda, stride-30 windowing of the 1-min
datasets, image lookback subsampled to one frame per 30 min (8 frames over the
4-h lookback; the model's image tokenizer pools over frames).

Usage
-----
::

    python scripts/run_crossvivit_image.py                     # full config, cuda
    python scripts/run_crossvivit_image.py --datasets skippd   # subset
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make the in-repo ``solarunibench`` package importable when run directly.
import sys as _sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from solarunibench.benchmark.crossvivit_image import (  # noqa: E402
    run_crossvivit_image_benchmark,
    save_crossvivit_image_results,
)
from solarunibench.benchmark.run_e8 import probe_cuda  # noqa: E402
from solarunibench.loader import SolarUniBenchLoader  # noqa: E402

_DATA_ROOT = "D:/monica/workspace/solar-unibench/data/standardized"
_OUT_DIR = "D:/monica/workspace/solar-unibench/figures/results"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=_DATA_ROOT)
    p.add_argument("--out-dir", default=_OUT_DIR)
    p.add_argument("--datasets", nargs="+", default=["folsom", "skippd"])
    p.add_argument(
        "--device", default="cuda",
        help="torch device (default cuda; auto-falls back to cpu if the probe fails)",
    )
    args = p.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not probe_cuda(args.device):
        print(f"WARNING: CUDA probe failed on {args.device}; falling back to cpu.",
              flush=True)
        args.device = "cpu"

    from solarunibench.forecast.models import CrossViViTModel
    from solarunibench.benchmark.run_e8 import (
        _FULL_ACCUM_CROSSVIVIT, _FULL_BATCH, _FULL_EPOCHS, _FULL_LR,
        _FULL_PATIENCE, _FULL_VAL_FRACTION, _FULL_WD,
    )
    model = CrossViViTModel(
        batch_size=_FULL_BATCH,
        accum_steps=_FULL_ACCUM_CROSSVIVIT,
        epochs=_FULL_EPOCHS,
        lr=_FULL_LR,
        weight_decay=_FULL_WD,
        patience=_FULL_PATIENCE,
        val_fraction=_FULL_VAL_FRACTION,
        device=args.device,
    )

    print(
        f"CrossViViT-image benchmark: datasets={args.datasets} horizons=[5,10,15] "
        f"FULL config (epochs={_FULL_EPOCHS} patience={_FULL_PATIENCE} "
        f"batch={_FULL_BATCH}+accum{_FULL_ACCUM_CROSSVIVIT}) device={args.device}",
        flush=True,
    )
    loader = SolarUniBenchLoader(args.data_root)
    rows: list[dict] = []
    for ds in args.datasets:
        t0 = time.time()
        try:
            ds_rows = run_crossvivit_image_benchmark(
                loader, datasets=[ds], model=model, data_root=args.data_root
            )
            rows.extend(ds_rows)
            for r in ds_rows:
                print(
                    f"  [OK] {ds}/h{r['horizon']:>2}: MAE={r['MAE']:.5f} "
                    f"Skill_MAE={r['Skill_MAE_pct']:+7.2f}% "
                    f"Skill_RMSE={r['Skill_RMSE_pct']:+7.2f}% "
                    f"n_daytime={r['n_daytime']} ({time.time()-t0:6.1f}s)",
                    flush=True,
                )
            if not ds_rows:
                print(f"  [WARN] {ds}: no image-covered windows (no rows)", flush=True)
        except Exception as e:  # noqa: BLE001 - report + continue (resilience)
            print(f"  [FAIL] {ds}: {type(e).__name__}: {e}", flush=True)
            import traceback

            traceback.print_exc()

    if not rows:
        print("\nNo rows produced. Aborting.", file=sys.stderr)
        return 1
    path = save_crossvivit_image_results(rows, args.out_dir)
    print(f"\nWrote: {path}")
    import pandas as pd

    with pd.option_context("display.float_format", lambda v: f"{v:7.2f}"):
        print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
