"""Clear-sky geo-recovery: grid-search lat/lon by POWER-vs-clearsky shape match."""
import numpy as np
import pandas as pd
from .clearsky import clearsky_ghi_simple


def reverse_geolocate(times: pd.DatetimeIndex, power: np.ndarray,
                      lat_grid: np.ndarray = None, lon_grid: np.ndarray = None) -> dict:
    """Recover (lat, lon) by matching normalized POWER to the clear-sky GHI shape.

    MSE is computed on peak-normalized daytime values (POWER is unit-less, so only
    shape information is used). This is the geo *recovery* method.
    """
    power = np.asarray(power, dtype=float)
    if lat_grid is None:
        lat_grid = np.arange(-60, 61, 2.0)
    if lon_grid is None:
        lon_grid = np.arange(-180, 181, 2.0)

    daytime = power > 0.01
    best_lat, best_lon, best_mse = None, None, np.inf
    for lat in lat_grid:
        for lon in lon_grid:
            cs = clearsky_ghi_simple(times, lat, lon)
            mask = daytime & (cs > 1.0)
            if mask.sum() < 100:
                continue
            scale = power[mask].max() / (cs[mask].max() + 1e-9)
            mse = float(np.mean((power[mask] - cs[mask] * scale) ** 2))
            if mse < best_mse:
                best_mse, best_lat, best_lon = mse, float(lat), float(lon)
    return {
        "latitude": best_lat, "longitude": best_lon,
        "details": {"best_mse": best_mse, "method": "clearsky_shape_match"},
    }
