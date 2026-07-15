import numpy as np
import pandas as pd
from solarunibench.operators import solar_geometry as sg


def test_declination_summer_solstice():
    # DOY 172 ~ June solstice -> declination ~ +23.45 deg
    dec = sg.solar_declination(np.array([172.0]))
    assert abs(np.rad2deg(dec[0]) - 23.45) < 0.6


def test_eccentricity_in_expected_range():
    dr = sg.eccentricity_correction(np.arange(1, 366, dtype=float))
    assert dr.min() > 0.96 and dr.max() < 1.04


def test_sunset_hour_angle_equator():
    # At the equator, year-round sunset hour angle ~ pi/2 (12h day)
    omega = sg.sunset_hour_angle(np.deg2rad(0.0), sg.solar_declination(np.array([172.0])))
    assert abs(omega[0] - np.pi / 2) < 0.05


def test_toa_daily_irradiation_monotonic_in_latitude():
    doy = np.array([172.0])
    h_eq = sg.toa_daily_irradiation(doy, 0.0)[0]
    h_mid = sg.toa_daily_irradiation(doy, 45.0)[0]
    # June solstice: equator receives less daily TOA than 45N (long day)
    assert h_mid > h_eq


def test_solar_noon_to_longitude():
    assert abs(sg.solar_noon_to_longitude(12.0) - 0.0) < 1e-9
    assert abs(sg.solar_noon_to_longitude(10.0) - 30.0) < 1e-9  # east
    assert abs(sg.solar_noon_to_longitude(14.0) - (-30.0)) < 1e-9  # west
