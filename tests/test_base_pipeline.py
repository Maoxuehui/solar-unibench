import pandas as pd, numpy as np, json
from pathlib import Path
from solarunibench.pipeline.base import BaseSolarPipeline


class _ToyAdapter(BaseSolarPipeline):
    def load_and_standardize_raw(self):
        from tests.conftest import synthetic_gefcom
        return synthetic_gefcom(n_hours=24, n_zones=1)


def test_run_pipeline_writes_conformant_artifacts(tmp_path):
    p = _ToyAdapter(raw_data_dir=".", output_dir=str(tmp_path), dataset_name="toy")
    manifest = p.run_pipeline()
    out = Path(tmp_path) / "toy"
    assert (out / "metadata.parquet").exists()
    assert (out / "time_series.parquet").exists()
    assert (out / "manifest.json").exists()
    ts = pd.read_parquet(out / "time_series.parquet")
    for col in ["timestamp_utc", "entity_id", "ghi_clearsky", "cos_zenith",
                "clearness_index", "quality_flag"]:
        assert col in ts.columns
    # The base owns the physics columns: its computed clear-sky must land in the
    # contract column (not be all-NA from an _x/_y merge collision), daytime
    # clearness must be derived, and no suffixed columns may survive.
    assert not ts["ghi_clearsky"].isna().all()
    assert ts["clearness_index"].notna().any()
    assert not any(c.endswith("_x") or c.endswith("_y") for c in ts.columns)
    assert manifest["dataset"] == "toy"


def test_raw_snapshot_is_pre_governance(tmp_path):
    p = _ToyAdapter(raw_data_dir=".", output_dir=str(tmp_path), dataset_name="toy")
    p.run_pipeline(raw=True)
    out = Path(tmp_path) / "toy"
    ts = pd.read_parquet(out / "time_series.parquet")
    # raw frame = standardized but BEFORE quality audit: clearness_index/quality_flag are NA.
    assert ts["quality_flag"].isna().all()
    assert ts["clearness_index"].isna().all()
