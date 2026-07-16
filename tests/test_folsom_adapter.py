import pandas as pd
from pathlib import Path
from solarunibench.pipeline.adapters.folsom import FolsomPipeline


def _write_folsom_fixture(root: Path):
    d = root / "Folsom_data"
    d.mkdir(parents=True)
    # Span local night and day so the quality audit's night path is exercised.
    # Folsom is ~UTC-7; 2014-06-01 06:00 UTC ≈ local night, 14:00-18:00 ≈ day.
    ts = pd.date_range("2014-06-01 06:00", periods=6, freq="2h")  # UTC-naive
    stamp = ts.strftime("%Y-%m-%d %H:%M:%S")
    pd.DataFrame({"timeStamp": stamp, "ghi": [0, 100, 600, 900, 300, 0],
                  "dni": [0, 50, 500, 800, 200, 0], "dhi": [0, 60, 120, 150, 100, 0]}
                 ).to_csv(d / "Folsom_irradiance.csv", index=False)
    pd.DataFrame({"timeStamp": stamp, "air_temp": [10, 12, 15, 18, 16, 11],
                  "relhum": [60, 58, 55, 50, 52, 59], "press": [1010, 1010, 1011, 1011, 1010, 1010],
                  "windsp": [1.5, 1.8, 2.0, 2.2, 1.9, 1.6], "winddir": [10, 20, 30, 40, 30, 20],
                  "max_windsp": [3, 3, 4, 4, 3, 3], "precipitation": [0, 0, 0, 0, 0, 0]}
                 ).to_csv(d / "Folsom_weather.csv", index=False)
    # NAM: line 1 is a comment, header on line 2
    with open(d / "Folsom_NAM_lat38.6_lon-121.2.csv", "w") as f:
        f.write("12Z NAM forecast for Folsom Lat=38.6 Lon=-121.2\n")
        f.write("reftime,valtime,dwsw,cloud_cover,precipitation,pressure,wind-u,wind-v,temperature,rel_humidity\n")
        for s in stamp:
            f.write(f"2014-06-01 12:00:00,{s},500.0,20.0,0.0,100674.0,-1.8,-0.5,293.0,43.0\n")


def test_folsom_loads_one_entity_utc_and_namespaces_nwp(tmp_path):
    _write_folsom_fixture(tmp_path)
    p = FolsomPipeline(raw_data_dir=str(tmp_path / "Folsom_data"),
                       output_dir=str(tmp_path / "out"), dataset_name="folsom")
    df_meta, df_ts = p.load_and_standardize_raw()
    assert df_meta["entity_id"].nunique() == 1
    assert df_meta["geo_inferred"].iloc[0] is False
    assert df_ts["timestamp_utc"].dt.tz is not None  # UTC-aware
    assert {"ghi_actual", "dni_actual", "dhi_actual"} <= set(df_ts.columns)
    assert "pv_power_normalized" in df_ts.columns and df_ts["pv_power_normalized"].isna().all()
    assert any(c.startswith("nwp_") for c in df_ts.columns)  # NAM forecast -> nwp_*
    assert any(c.startswith("obs_") for c in df_ts.columns)  # ground weather -> obs_*


def test_folsom_run_pipeline_pv_stays_all_nan_including_night(tmp_path):
    # End-to-end: on the production run_pipeline(raw=False) path, the quality
    # audit must NOT fabricate 0.0 for the PV-less modality at night. The fixture
    # now spans local night, so this locks the Honest-NaN invariant for real.
    _write_folsom_fixture(tmp_path)
    p = FolsomPipeline(raw_data_dir=str(tmp_path / "Folsom_data"),
                       output_dir=str(tmp_path / "out"), dataset_name="folsom")
    p.run_pipeline(raw=False)
    ts = pd.read_parquet(tmp_path / "out" / "folsom" / "time_series.parquet")
    # Folsom has no PV sensor → pv_power_normalized must be entirely NaN,
    # including any night rows (no fabricated zeros).
    assert "pv_power_normalized" in ts.columns
    assert ts["pv_power_normalized"].isna().all(), "PV-less dataset must stay all-NaN end-to-end"
    # Sanity: the fixture does span night (cos_zenith <= 0), so the invariant is
    # actually exercised — not vacuously true on an all-daytime frame.
    assert (ts["cos_zenith"] <= 0).any(), "fixture must include a night row to exercise the path"
