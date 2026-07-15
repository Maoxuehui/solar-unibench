"""Clear-sky + solar-position operators (pvlib) and a simplified Ineichen for grid search."""
import numpy as np
import pandas as pd
import pvlib
from . import solar_geometry as sg


def compute_astronomy(times: pd.DatetimeIndex, lat: float, lon: float) -> pd.DataFrame:
    solpos = pvlib.solarposition.get_solarposition(times, lat, lon)
    cos_zenith = np.maximum(0.0, np.cos(np.radians(solpos["zenith"].values))).astype(np.float32)
    az = solpos["azimuth"].values.astype(np.float32)
    return pd.DataFrame({"cos_zenith": cos_zenith, "solar_azimuth": az}, index=times)


def clearsky_ineichen(times: pd.DatetimeIndex, lat: float, lon: float,
                      elevation: float = 0.0) -> pd.DataFrame:
    """pvlib Ineichen clear-sky. NOTE: correct API (methodology's appointment_time= was wrong)."""
    solpos = pvlib.solarposition.get_solarposition(times, lat, lon, altitude=elevation)
    app_zen = solpos["apparent_zenith"]
    try:
        # Linke turbidity climatology (needs internet on first use; falls back below).
        lt = pvlib.clearsky.lookup_linke_turbidity(times, lat, lon)
    except Exception:
        lt = 3.0  # global clean-atmosphere default
    # pvlib >=0.10 made airmass_absolute a required arg and removed interp_turbidity;
    # compute it explicitly (matches the v0.15.2 ineichen signature).
    airmass_rel = pvlib.atmosphere.get_relative_airmass(app_zen)
    airmass_abs = pvlib.atmosphere.get_absolute_airmass(airmass_rel)
    cs = pvlib.clearsky.ineichen(app_zen, airmass_abs, linke_turbidity=lt,
                                 altitude=elevation)
    return pd.DataFrame({
        "ghi_clearsky": cs["ghi"].astype(np.float32),
        "dni_clearsky": cs["dni"].astype(np.float32),
        "dhi_clearsky": cs["dhi"].astype(np.float32),
    }, index=times)


def clearsky_ghi_simple(times: pd.DatetimeIndex, lat_deg: float, lon_deg: float,
                        turbidity: float = 2.0) -> np.ndarray:
    """Dependency-light Ineichen GHI (ported from geo_location._clearsky_ghi). For grid search only."""
    doy = sg.day_of_year(times)
    delta = sg.solar_declination(doy)
    dr = sg.eccentricity_correction(doy)
    utc_hours = times.hour.to_numpy() + times.minute.to_numpy() / 60.0
    local_solar_time = utc_hours + lon_deg / 15.0
    hour_angle = np.deg2rad(15.0 * (local_solar_time - 12.0))
    lat_rad = np.deg2rad(lat_deg)
    cos_zenith = np.sin(lat_rad) * np.sin(delta) + np.cos(lat_rad) * np.cos(delta) * np.cos(hour_angle)
    cos_zenith = np.clip(cos_zenith, 0.0, 1.0)
    zenith = np.rad2deg(np.arccos(cos_zenith))
    am = 1.0 / (cos_zenith + 0.50572 * (96.07995 - zenith) ** -1.6364)
    am = np.clip(am, 0.0, 40.0)
    fh1 = np.exp(-lat_deg / 8000.0)
    fh2 = np.exp(-lat_deg / 8000.0)
    cg1 = 5.09e-5 * (turbidity ** 2) + 3.16e-3 * turbidity + 0.079
    cg2 = 1.03e-7 * (turbidity ** 3) - 3.73e-5 * (turbidity ** 2) + 5.45e-3 * turbidity + 0.109
    i0 = sg.SOLAR_CONSTANT * dr
    ghi = cg1 * i0 * cos_zenith * np.exp(-cg2 * am * (fh1 + fh2 * turbidity))
    return np.maximum(ghi, 0.0)
