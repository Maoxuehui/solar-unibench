import pandas as pd
from pathlib import Path
from solarunibench.pipeline.adapters.folsom import FolsomPipeline


def _write_folsom_fixture(root: Path):
    d = root / "Folsom_data"
    d.mkdir(parents=True)
    ts = pd.date_range("2014-06-01 14:00", periods=6, freq="1min")  # UTC-naive
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
    assert any(c.startswith("nwp_") for c in df_ts.columns)  # NAM namespaced
