"""Slow real-data integration test for the SKIPPD adapter (RUN_SLOW-gated).

Runs SKIPPDPipeline.run_pipeline() on the real SKIPPD archive and asserts the
manifest entity count, PV normalization bounds, and NWP column presence.
"""
import os
import time

import pandas as pd
import pytest

from solarunibench.pipeline.adapters.skippd import SKIPPDPipeline

DATA_ROOT = os.environ.get("SOLAR_UNIBENCH_DATA_ROOT",
                           "D:/monica/workspace/solar_benchmark_dataset")
slow = pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="set RUN_SLOW=1")


# Previously xfailed: the real SKIPPD GFS `time_forecast` column is object-dtype
# where each cell is a 2-element datetime64[us] ndarray (valid forecast time +
# sentinel). The adapter now extracts [0] and localizes to UTC, so the GFS join
# resolves on real data and this is a hard assertion.


@slow
def test_real_skippd_builds(tmp_path):
    p = SKIPPDPipeline(raw_data_dir=f"{DATA_ROOT}/SKIPPD_all",
                       output_dir=str(tmp_path), dataset_name="skippd")
    t0 = time.perf_counter()
    m = p.run_pipeline()
    dt = time.perf_counter() - t0
    assert m["n_entities"] == 1
    ts = pd.read_parquet(tmp_path / "skippd" / "time_series.parquet")
    pv = ts["pv_power_normalized"].dropna()
    assert pv.between(0.0, 1.0).all()  # normalized by capacity 30.1
    # at least one namespaced NWP column (ERA5 and/or GFS) is present
    assert any(c.startswith("nwp_") for c in ts.columns)
    # GFS-specific NWP columns must be present on real data — these variable
    # names (downward_shortwave_radiation_flux, total_cloud_cover_entire_atmosphere)
    # are GFS-only, so a GFS->ERA5-only regression would fail here.
    gfs_cols = [c for c in ts.columns
                if c.startswith("nwp_downward_shortwave_radiation_flux")
                or c.startswith("nwp_total_cloud_cover_entire_atmosphere")]
    assert len(gfs_cols) > 0, "GFS NWP columns missing — GFS join regressed"
    print(f"\n[SKIPPD integration] rows={len(ts)} entities={m['n_entities']} "
          f"pv_max={pv.max():.4f} build_seconds={dt:.1f} "
          f"nwp_cols={sum(c.startswith('nwp_') for c in ts.columns)} "
          f"gfs_cols={len(gfs_cols)}")
