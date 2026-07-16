"""Contract conformance for the SunLake adapter (fast, synthetic fixture).

Builds a fixture package via SunLakePipeline.run_pipeline(), reloads the parquet,
and asserts validate_metadata / validate_time_series pass plus the Honest-NaN
invariant for SunLake's absent modality (PV): pv_power_normalized is all-NaN.
"""
import pandas as pd
from solarunibench.contract import validate_metadata, validate_time_series
from solarunibench.pipeline.adapters.sunlake import SunLakePipeline
from tests.test_sunlake_adapter import _write_sunlake_fixture


def test_sunlake_package_satisfies_contract(tmp_path):
    _write_sunlake_fixture(tmp_path)
    SunLakePipeline(raw_data_dir=str(tmp_path), output_dir=str(tmp_path / "out"),
                    dataset_name="sunlake").run_pipeline()
    out = tmp_path / "out" / "sunlake"
    meta = pd.read_parquet(out / "metadata.parquet")
    ts = pd.read_parquet(out / "time_series.parquet")
    validate_metadata(meta)
    validate_time_series(ts)
    assert meta["dataset_source"].eq("sunlake").all()
    assert not meta["geo_inferred"].iloc[0]  # station coords known (robust to parquet bool round-trip)
    assert ts["quality_flag"].isin([0, 1, 2, 3]).all()
    # SunLake = irradiance-only (no PV sensor) -> pv_power_normalized must be
    # entirely NaN (Honest NaN), never fabricated night-zeros.
    assert ts["pv_power_normalized"].isna().all()
