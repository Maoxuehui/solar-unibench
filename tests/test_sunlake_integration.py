"""Slow real-data integration test for the SunLake adapter (RUN_SLOW-gated).

Runs SunLakePipeline.run_pipeline() on the real pulled SunLake tabular parquet
(6 BSRN stations). Skips with a clear message if the pull hasn't been run.
"""
import os
import time

import pandas as pd
import pytest

from solarunibench.pipeline.adapters.sunlake import SunLakePipeline

DATA_ROOT = os.environ.get("SOLAR_UNIBENCH_DATA_ROOT",
                           "D:/monica/workspace/solar_benchmark_dataset")
SUNLAKE_DIR = os.path.join(DATA_ROOT, "SunLake_data")
slow = pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="set RUN_SLOW=1")


@slow
def test_real_sunlake_builds(tmp_path):
    tabular = os.path.join(SUNLAKE_DIR, "sunlake_tabular.parquet")
    if not os.path.exists(tabular):
        pytest.skip(
            "sunlake_tabular.parquet not found — run "
            "`conda run -n solar_uni python scripts/pull_sunlake_tabular.py` first"
        )
    p = SunLakePipeline(raw_data_dir=SUNLAKE_DIR, output_dir=str(tmp_path),
                        dataset_name="sunlake")
    t0 = time.perf_counter()
    m = p.run_pipeline()
    dt = time.perf_counter() - t0
    assert m["n_entities"] == 6  # CAB, CNR, IZA, PAL, PAY, TAM
    ts = pd.read_parquet(tmp_path / "sunlake" / "time_series.parquet")
    # SunLake is irradiance-only: GHI must be present and carry real data
    assert "ghi_actual" in ts.columns and ts["ghi_actual"].notna().any()
    print(f"\n[SunLake integration] rows={len(ts)} entities={m['n_entities']} "
          f"build_seconds={dt:.1f}")
