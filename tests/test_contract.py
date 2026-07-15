import pandas as pd
import numpy as np
import pytest
from solarunibench.contract import (
    validate_metadata, validate_time_series, ContractError, META_COLUMNS, TS_COLUMNS,
)


def _good_meta():
    return pd.DataFrame([{
        "entity_id": "Z1", "dataset_source": "gefcom2014", "latitude": -30.0,
        "longitude": 140.0, "elevation_meters": 50.0, "local_timezone": "UTC",
        "surface_albedo": 0.2, "capacity_kw": np.nan, "slope": np.nan, "azimuth": np.nan,
        "geo_inferred": True,
    }])


def _good_ts():
    return pd.DataFrame([{
        "timestamp_utc": pd.Timestamp("2013-06-01 06:00", tz="UTC"), "entity_id": "Z1",
        "ghi_actual": 500.0, "dni_actual": np.nan, "dhi_actual": np.nan,
        "pv_power_normalized": 0.5, "ghi_clearsky": 600.0, "dni_clearsky": np.nan,
        "dhi_clearsky": np.nan, "cos_zenith": 0.5, "solar_azimuth": 90.0,
        "clearness_index": 0.83, "quality_flag": 0,
    }])


def test_good_frames_pass():
    validate_metadata(_good_meta())
    validate_time_series(_good_ts())


def test_missing_meta_column_raises():
    df = _good_meta().drop(columns=["latitude"])
    with pytest.raises(ContractError):
        validate_metadata(df)


def test_missing_ts_column_raises():
    df = _good_ts().drop(columns=["clearness_index"])
    with pytest.raises(ContractError):
        validate_time_series(df)


def test_quality_flag_domain():
    df = _good_ts()
    df.loc[0, "quality_flag"] = 9
    with pytest.raises(ContractError):
        validate_time_series(df)


def test_contract_constants_present():
    assert "entity_id" in META_COLUMNS
    assert "clearness_index" in TS_COLUMNS
