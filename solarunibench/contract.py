"""Data contract: the single source of truth for standardized packages."""
import pandas as pd


class ContractError(ValueError):
    """Raised when a frame violates the Solar-UniBench data contract."""


META_COLUMNS = [
    "entity_id", "dataset_source", "latitude", "longitude", "elevation_meters",
    "local_timezone", "surface_albedo", "capacity_kw", "slope", "azimuth", "geo_inferred",
]

TS_COLUMNS = [
    "timestamp_utc", "entity_id",
    "ghi_actual", "dni_actual", "dhi_actual", "pv_power_normalized",
    "ghi_clearsky", "dni_clearsky", "dhi_clearsky",
    "cos_zenith", "solar_azimuth", "clearness_index", "quality_flag",
]

META_DTYPES = {
    "entity_id": "string", "dataset_source": "string", "latitude": "float32",
    "longitude": "float32", "elevation_meters": "float32", "local_timezone": "string",
    "surface_albedo": "float32", "capacity_kw": "float32", "slope": "float32",
    "azimuth": "float32", "geo_inferred": "bool",
}

TS_DTYPES = {
    "timestamp_utc": "datetime64[ns, UTC]", "entity_id": "string",
    "ghi_actual": "float32", "dni_actual": "float32", "dhi_actual": "float32",
    "pv_power_normalized": "float32", "ghi_clearsky": "float32", "dni_clearsky": "float32",
    "dhi_clearsky": "float32", "cos_zenith": "float32", "solar_azimuth": "float32",
    "clearness_index": "float32", "quality_flag": "int8",
}

VALID_QUALITY_FLAGS = {0, 1, 2, 3}


def validate_metadata(df: pd.DataFrame) -> None:
    missing = [c for c in META_COLUMNS if c not in df.columns]
    if missing:
        raise ContractError(f"metadata missing columns: {missing}")


def validate_time_series(df: pd.DataFrame) -> None:
    missing = [c for c in TS_COLUMNS if c not in df.columns]
    if missing:
        raise ContractError(f"time_series missing columns: {missing}")
    if df.duplicated(subset=["timestamp_utc", "entity_id"]).any():
        raise ContractError("time_series has duplicate (timestamp_utc, entity_id) keys")
    if "quality_flag" in df.columns:
        bad = set(df["quality_flag"].dropna().unique()) - VALID_QUALITY_FLAGS
        if bad:
            raise ContractError(f"invalid quality_flag values: {bad}")
