"""TSR (VAR178) validation of recovered coordinates. TOA radiation is pure geometry."""
import numpy as np
import pandas as pd
from . import solar_geometry as sg


def _r_squared(obs: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(pred) & (pred > 1e-6)
    if mask.sum() < 5:
        return float("nan")
    o, p = obs[mask], pred[mask]
    scale = np.median(o / p)
    ss_res = np.sum((o - p * scale) ** 2)
    ss_tot = np.sum((o - o.mean()) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def validate_with_tsr(times: pd.DatetimeIndex, tsr_hourly: np.ndarray,
                      lat: float, lon: float) -> dict:
    """Validate recovered (lat, lon) against observed top-of-atmosphere radiation.

    Latitude: R^2/RMSE of daily TSR vs theoretical TOA daily irradiation at `lat`.
    Longitude: solar-noon UTC hour implied by TSR's diurnal peak vs `lon`.
    """
    s = pd.Series(np.asarray(tsr_hourly, dtype=float), index=times)
    daily = s.resample("1D").sum()
    doys = daily.index.dayofyear.to_numpy().astype(float)
    obs = daily.to_numpy()
    theory = sg.toa_daily_irradiation(doys, lat)
    valid = theory > 1e-3
    r2 = _r_squared(obs[valid], theory[valid])
    rmse = float(np.sqrt(np.nanmean((obs[valid] - theory[valid]) ** 2))) if valid.sum() else float("nan")

    # Longitude check via diurnal peak hour of positive TSR increments
    diff = s.diff()
    daytime = diff[diff > 0]
    peak_hour = float(daytime.groupby(daytime.index.hour).mean().idxmax()) if len(daytime) else 12.0
    lon_implied = sg.solar_noon_to_longitude(peak_hour)
    return {
        "r2_latitude": r2, "rmse_latitude": rmse,
        "longitude_check_hour": peak_hour, "longitude_implied": lon_implied,
        "details": {"recovered_longitude": lon, "n_days": int(valid.sum())},
    }
