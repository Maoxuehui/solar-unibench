import pandas as pd
from pathlib import Path
from solarunibench.pipeline.adapters.sunlake import SunLakePipeline


def _write_sunlake_fixture(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    ts = pd.date_range("2018-06-01 00:00", periods=5, freq="30min", tz="UTC")
    pd.DataFrame({"entity_id": ["SUNLAKE_CAB"] * 5, "timestamp_utc": ts,
                  "ghi_actual": [0.0, 50.0, 600.0, 800.0, 20.0],
                  "dhi_actual": [0.0, 40.0, 120.0, 150.0, 30.0],
                  "dni_actual": [0.0, 30.0, 500.0, 700.0, 10.0]}
                 ).to_parquet(root / "sunlake_tabular.parquet", index=False)
    pd.DataFrame([{"entity_id": "SUNLAKE_CAB", "station": "CAB", "latitude": 51.968,
                   "longitude": 4.928, "elevation_meters": 0.0}]
                ).to_parquet(root / "sunlake_stations.parquet", index=False)


def test_sunlake_loads_stations_and_ghi(tmp_path):
    _write_sunlake_fixture(tmp_path)
    p = SunLakePipeline(raw_data_dir=str(tmp_path), output_dir=str(tmp_path / "out"), dataset_name="sunlake")
    df_meta, df_ts = p.load_and_standardize_raw()
    assert "SUNLAKE_CAB" in set(df_meta["entity_id"])
    assert df_meta["geo_inferred"].iloc[0] is False
    assert df_ts["timestamp_utc"].dt.tz is not None
    assert {"ghi_actual", "dni_actual", "dhi_actual"} <= set(df_ts.columns)
    assert df_ts["pv_power_normalized"].isna().all()  # no PV
