"""Slow real-data integration test for the Folsom adapter (RUN_SLOW-gated).

Runs FolsomPipeline.run_pipeline() on the REAL Folsom dataset and asserts the
manifest entity count, a plausible real row count, and clearsky presence.
Folsom is the largest dataset (~1.5M rows); allow a minute or two.
"""
import os
import time

import pandas as pd
import pytest

from solarunibench.pipeline.adapters.folsom import FolsomPipeline

DATA_ROOT = os.environ.get("SOLAR_UNIBENCH_DATA_ROOT",
                           "D:/monica/workspace/solar_benchmark_dataset")
slow = pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="set RUN_SLOW=1")


@slow
def test_real_folsom_builds(tmp_path):
    p = FolsomPipeline(raw_data_dir=f"{DATA_ROOT}/Folsom_data",
                       output_dir=str(tmp_path), dataset_name="folsom")
    t0 = time.perf_counter()
    m = p.run_pipeline()
    dt = time.perf_counter() - t0
    assert m["n_entities"] == 1
    ts = pd.read_parquet(tmp_path / "folsom" / "time_series.parquet")
    assert len(ts) > 100_000  # ~1.5M rows on the real Folsom archive
    assert "ghi_clearsky" in ts.columns and ts["ghi_clearsky"].notna().any()
    # UTC-alignment sanity: at the timestamp where ghi_actual is maximal (solar
    # noon), the sun must be high — cos_zenith > 0.5. If the stamps were local
    # time misinterpreted as UTC, solar noon would land far from max ghi_actual.
    noon_zen = ts.loc[ts["ghi_actual"].idxmax(), "cos_zenith"]
    assert noon_zen > 0.5, f"Folsom UTC misalignment: max-ghi_actual cos_zenith={noon_zen} (stamps may be local, not UTC)"
    # surface timing in the report without asserting a hard bound
    print(f"\n[Folsom integration] rows={len(ts)} entities={m['n_entities']} "
          f"noon_cos_zenith={noon_zen:.4f} build_seconds={dt:.1f}")
