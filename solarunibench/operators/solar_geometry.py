"""Pure solar-geometry helpers (no external deps). Ported from geo_location.py."""
import numpy as np
import pandas as pd

SOLAR_CONSTANT = 1367.0  # W/m^2


def day_of_year(times: pd.DatetimeIndex) -> np.ndarray:
    return times.dayofyear.values.astype(float)


def solar_declination(doy: np.ndarray) -> np.ndarray:
    """Cooper equation: declination (radians)."""
    return np.deg2rad(23.45) * np.sin(2 * np.pi * (284 + doy) / 365.0)


def eccentricity_correction(doy: np.ndarray) -> np.ndarray:
    return 1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)


def sunset_hour_angle(lat_rad: float, declination: np.ndarray) -> np.ndarray:
    cos_omega_s = -np.tan(lat_rad) * np.tan(declination)
    cos_omega_s = np.clip(cos_omega_s, -1.0, 1.0)
    return np.arccos(cos_omega_s)


def toa_daily_irradiation(doy: np.ndarray, lat_deg: float) -> np.ndarray:
    """Daily top-of-atmosphere irradiation (J/m^2). Pure geometry (no atmosphere)."""
    lat_rad = np.deg2rad(lat_deg)
    delta = solar_declination(doy)
    dr = eccentricity_correction(doy)
    omega_s = sunset_hour_angle(lat_rad, delta)
    term = omega_s * np.sin(lat_rad) * np.sin(delta) + np.cos(lat_rad) * np.cos(delta) * np.sin(omega_s)
    return (24.0 / np.pi) * SOLAR_CONSTANT * dr * term * 3600.0


def solar_noon_to_longitude(peak_utc_hour: float) -> float:
    """Longitude (deg, east positive) from the UTC hour of solar noon."""
    return (12.0 - peak_utc_hour) * 15.0
