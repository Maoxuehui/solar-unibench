import pandas as pd, numpy as np
from pathlib import Path
from solarunibench.pipeline.adapters.gefcom import GEFCom2014Pipeline


def _write_fixture(root: Path):
    solar = root / "GEFCom2014_data" / "Solar" / "Task 15"
    solar.mkdir(parents=True)
    times = pd.date_range("2013-06-01 06:00", periods=12, freq="1h")
    stamp = times.strftime("%Y%m%d %H:%M")
    rows_p, rows_t = [], []
    for z in [1, 2, 3]:
        for s in stamp:
            power = float(np.clip(np.random.rand(), 0, 1))
            rows_t.append({"ZONEID": z, "TIMESTAMP": s, "POWER": power})
            rows_p.append({"ZONEID": z, "TIMESTAMP": s, "VAR178": 1e6, "VAR169": 5e5,
                           "VAR78": 0, "VAR79": 0, "VAR134": 1e5, "VAR157": 50, "VAR164": 0.5,
                           "VAR165": 1, "VAR166": 1, "VAR167": 290, "VAR175": 3e5, "VAR228": 0,
                           "POWER": power})
    pd.DataFrame(rows_t).to_csv(solar / "train15.csv", index=False)
    pd.DataFrame(rows_p).to_csv(solar / "predictors15.csv", index=False)


def test_gefcom_loads_three_zones_and_parses_timestamps(tmp_path):
    _write_fixture(tmp_path)
    p = GEFCom2014Pipeline(raw_data_dir=str(tmp_path / "GEFCom2014_data"),
                           output_dir=str(tmp_path / "out"), dataset_name="gefcom2014")
    df_meta, df_ts = p.load_and_standardize_raw()
    assert df_meta["entity_id"].nunique() == 3
    assert {"pv_power_normalized", "timestamp_utc", "entity_id"} <= set(df_ts.columns)
    assert df_ts["timestamp_utc"].dt.tz is not None  # UTC-aware
    assert any(c.startswith("nwp_") for c in df_ts.columns)  # NWP namespaced
