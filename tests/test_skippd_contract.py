"""Contract conformance for the SKIPPD adapter (fast, synthetic fixture).

Builds a fixture package via SKIPPDPipeline.run_pipeline(), reloads the parquet,
and asserts validate_metadata / validate_time_series pass plus the Honest-NaN
invariant for SKIPPD's absent modality (ground irradiance): ghi_actual is all-NaN.
"""
import pandas as pd
from solarunibench.contract import validate_metadata, validate_time_series
from solarunibench.pipeline.adapters.skippd import SKIPPDPipeline
from tests.test_skippd_adapter import _write_skippd_fixture


def test_skippd_package_satisfies_contract(tmp_path):
    _write_skippd_fixture(tmp_path)
    SKIPPDPipeline(raw_data_dir=str(tmp_path / "SKIPPD_all"),
                   output_dir=str(tmp_path / "out"), dataset_name="skippd").run_pipeline()
    out = tmp_path / "out" / "skippd"
    meta = pd.read_parquet(out / "metadata.parquet")
    ts = pd.read_parquet(out / "time_series.parquet")
    validate_metadata(meta)
    validate_time_series(ts)
    assert meta["dataset_source"].eq("skippd").all()
    assert not meta["geo_inferred"].iloc[0]  # known coords (robust to parquet bool round-trip)
    assert ts["quality_flag"].isin([0, 1, 2, 3]).all()
    # SKIPPD = PV-only (no ground irradiance sensor) -> ghi_actual must be
    # entirely NaN (Honest NaN), never fabricated.
    assert ts["ghi_actual"].isna().all()
