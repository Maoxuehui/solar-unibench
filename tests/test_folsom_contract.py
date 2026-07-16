"""Contract conformance for the Folsom adapter (fast, synthetic fixture).

Builds a fixture package via FolsomPipeline.run_pipeline(), reloads the parquet,
and asserts validate_metadata / validate_time_series pass plus the Honest-NaN
invariant for Folsom's absent modality (PV): pv_power_normalized is all-NaN.
"""
import pandas as pd
from solarunibench.contract import validate_metadata, validate_time_series
from solarunibench.pipeline.adapters.folsom import FolsomPipeline
from tests.test_folsom_adapter import _write_folsom_fixture


def test_folsom_package_satisfies_contract(tmp_path):
    _write_folsom_fixture(tmp_path)
    FolsomPipeline(raw_data_dir=str(tmp_path / "Folsom_data"),
                   output_dir=str(tmp_path / "out"), dataset_name="folsom").run_pipeline()
    out = tmp_path / "out" / "folsom"
    meta = pd.read_parquet(out / "metadata.parquet")
    ts = pd.read_parquet(out / "time_series.parquet")
    validate_metadata(meta)
    validate_time_series(ts)
    assert meta["dataset_source"].eq("folsom").all()
    assert not meta["geo_inferred"].iloc[0]  # known coords, not inferred (robust to parquet bool round-trip)
    assert ts["quality_flag"].isin([0, 1, 2, 3]).all()
    # Folsom = irradiance-only (no PV sensor) -> pv_power_normalized must be
    # entirely NaN (Honest NaN), never fabricated night-zeros.
    assert ts["pv_power_normalized"].isna().all()
