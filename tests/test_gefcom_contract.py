import pandas as pd
from solarunibench.contract import validate_metadata, validate_time_series
from solarunibench.pipeline.adapters.gefcom import GEFCom2014Pipeline


def _build(tmp_path):
    from tests.test_gefcom_adapter import _write_fixture
    _write_fixture(tmp_path)
    p = GEFCom2014Pipeline(raw_data_dir=str(tmp_path / "GEFCom2014_data"),
                           output_dir=str(tmp_path / "out"), dataset_name="gefcom2014")
    p.run_pipeline()
    return tmp_path / "out" / "gefcom2014"


def test_gefcom_package_satisfies_contract(tmp_path):
    out = _build(tmp_path)
    meta = pd.read_parquet(out / "metadata.parquet")
    ts = pd.read_parquet(out / "time_series.parquet")
    validate_metadata(meta)
    validate_time_series(ts)
    assert meta["dataset_source"].eq("gefcom2014").all()
    assert meta["geo_inferred"].all()  # all GEFCom coords are recovered
    assert ts["quality_flag"].isin([0, 1, 2, 3]).all()
    # GEFCom is PV-only (no ground irradiance) → ghi_actual and clearness_index
    # must be entirely NaN (Honest NaN), never fabricated night-zeros.
    assert ts["ghi_actual"].isna().all()
    assert ts["clearness_index"].isna().all()
