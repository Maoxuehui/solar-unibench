"""Shared synthetic fixtures. No real data lives in the test suite."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def synthetic_gefcom(n_hours: int = 48, n_zones: int = 3, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (df_meta, df_ts) mimicking the GEFCom contract for unit tests.

    df_ts columns: timestamp_utc, entity_id, ghi_actual, pv_power_normalized, ghi_clearsky.
    df_meta columns: entity_id, dataset_source, latitude, longitude, elevation_meters,
                     local_timezone, surface_albedo, capacity_kw, slope, azimuth, geo_inferred.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2013-06-01 06:00", tz="UTC")
    times = pd.date_range(start, periods=n_hours, freq="1h", tz="UTC")
    # A toy clear-sky-ish bell curve over daytime hours
    hod = times.hour.to_numpy()
    bell = np.clip(np.sin(np.pi * (hod - 6) / 13.0), 0.0, None)
    meta_rows, ts_rows = [], []
    for z in range(1, n_zones + 1):
        eid = f"GEFCOM_ZONE_{z}"
        meta_rows.append({
            "entity_id": eid, "dataset_source": "gefcom2014",
            "latitude": -30.0 - z, "longitude": 140.0 + z, "elevation_meters": 50.0,
            "local_timezone": "Australia/Brisbane", "surface_albedo": 0.2,
            "capacity_kw": float("nan"), "slope": float("nan"), "azimuth": float("nan"),
            "geo_inferred": True,
        })
        ghi = bell * 900.0 + rng.normal(0, 10, n_hours)
        pv = np.clip(bell * 0.9 + rng.normal(0, 0.02, n_hours), 0, 1)
        for i, t in enumerate(times):
            ts_rows.append({
                "timestamp_utc": t, "entity_id": eid,
                "ghi_actual": float(ghi[i]), "pv_power_normalized": float(pv[i]),
                "ghi_clearsky": float(bell[i] * 950.0),
            })
    df_meta = pd.DataFrame(meta_rows)
    df_ts = pd.DataFrame(ts_rows)
    return df_meta, df_ts
