import numpy as np
import pandas as pd
from pathlib import Path
from solarunibench.pipeline.adapters.skippd import SKIPPDPipeline


def _write_skippd_fixture(root: Path):
    base = root / "SKIPPD_all"
    for split in ("train", "test"):
        for cfg in ("labels", "ERA5", "GFS"):
            (base / cfg / split).mkdir(parents=True)
    pacific = pd.date_range("2017-06-01 12:00", periods=5, freq="1min", tz="US/Pacific")
    # labels: raw pv (peak ~29.6)
    pd.DataFrame({"time": pacific, "pv": [0.0, 10.0, 25.0, 29.0, 5.0]}
                 ).to_parquet(base / "labels" / "train" / "0000.parquet")
    pd.DataFrame({"time": pacific, "pv": [0.0, 10.0, 25.0, 29.0, 5.0]}
                 ).to_parquet(base / "labels" / "test" / "0000.parquet")
    # ERA5: join on valid_time (hourly, tz-aware scalar US/Pacific)
    era5_t = pd.date_range("2017-06-01 12:00", periods=2, freq="1h", tz="US/Pacific")
    pd.DataFrame({"valid_time": era5_t, "ssrd_nn1": [5e6, 5.5e6], "t2m_nn1": [290.0, 291.0]}
                 ).to_parquet(base / "ERA5" / "train" / "0000.parquet")
    pd.DataFrame({"valid_time": era5_t, "ssrd_nn1": [5e6, 5.5e6], "t2m_nn1": [290.0, 291.0]}
                 ).to_parquet(base / "ERA5" / "test" / "0000.parquet")
    # GFS: time_forecast mirrors real SKIPPD — object-dtype column of 2-element
    # datetime64[us] ndarrays (valid forecast time [0] = UTC-naive; [1] = sentinel
    # epoch+1us to discard). Valid times align with the labels' UTC stamps so the
    # join actually resolves a row (12:00 / 13:00 US/Pacific summer = 19:00 / 20:00 UTC).
    gfs_t = [
        np.array([np.datetime64("2017-06-01T19:00"),
                  np.datetime64("1970-01-01T00:00:00.000001")], dtype="datetime64[us]")
        for _ in range(2)
    ]
    gfs_t[1][0] = np.datetime64("2017-06-01T20:00")
    gfs_df = pd.DataFrame({"time_forecast": gfs_t,
                           "downward_shortwave_radiation_flux_cubic": [800.0, 810.0]})
    gfs_df.to_parquet(base / "GFS" / "train" / "0000.parquet")
    gfs_df.to_parquet(base / "GFS" / "test" / "0000.parquet")


def test_skippd_normalizes_pv_and_converts_pacific_to_utc(tmp_path):
    _write_skippd_fixture(tmp_path)
    p = SKIPPDPipeline(raw_data_dir=str(tmp_path / "SKIPPD_all"),
                       output_dir=str(tmp_path / "out"), dataset_name="skippd")
    df_meta, df_ts = p.load_and_standardize_raw()
    assert df_ts["timestamp_utc"].dt.tz is not None
    # Pacific -> UTC: the fixture's first stamp is 12:00 US/Pacific (summer DST,
    # UTC-7), which must land at 19:00 UTC. Locks the tz conversion, not just
    # tz-presence.
    assert df_ts["timestamp_utc"].iloc[0].hour == 19
    # pv normalized to [0,1] by capacity 30.1
    assert df_ts["pv_power_normalized"].max() <= 1.0 + 1e-6
    assert df_ts["pv_power_normalized"].min() >= 0.0
    # Pin the PV max value: fixture peaks at 29.0 raw / 30.1 capacity = 0.9634...
    # (guards against both a wrong capacity constant and a missing normalization).
    assert abs(df_ts["pv_power_normalized"].max() - 29.0 / 30.1) < 1e-6
    # ghi_actual absent (Honest NaN) — SKIPPD has no ground irradiance
    assert "ghi_actual" in df_ts.columns and df_ts["ghi_actual"].isna().all()
    # ERA5 namespaced
    assert "nwp_ssrd_nn1" in df_ts.columns
    # GFS namespaced: the 2-element datetime64[us] arrays in time_forecast must be
    # parsed (first element = valid forecast time, UTC-naive) into a joinable UTC
    # stamp. The fixture's first valid time (19:00 UTC) aligns with a labels stamp,
    # so the merge must resolve the value we wrote (800.0). Train/test carry
    # duplicate stamps; the adapter now dedups both labels and NWP frames before
    # merging, so the join is 1:1 — assert the resolved value is present.
    gfs_col = "nwp_downward_shortwave_radiation_flux_cubic"
    assert gfs_col in df_ts.columns
    gfs_at_19 = df_ts.loc[
        df_ts["timestamp_utc"] == pd.Timestamp("2017-06-01 19:00", tz="UTC"), gfs_col
    ]
    assert (gfs_at_19 == 800.0).any()
