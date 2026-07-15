import numpy as np
import pandas as pd
from solarunibench.operators import solar_geometry as sg, tsr_validation as tv


def test_tsr_validation_high_r2_for_correct_lat():
    true_lat = 40.0
    times = pd.date_range("2013-01-01", periods=24 * 365, freq="1h", tz="UTC")
    # Build an hourly TSR whose daily sum follows theoretical TOA at true_lat (scaled).
    doy = times.dayofyear.to_numpy()
    daily_toa = sg.toa_daily_irradiation(doy.astype(float), true_lat)
    # spread daily total uniformly-ish across day hours (toy, monotonic in TOA is enough)
    tsr = np.where(daily_toa > 0, daily_toa / 12.0, 0.0)
    res = tv.validate_with_tsr(times, tsr, lat=true_lat, lon=0.0)
    assert res["r2_latitude"] > 0.90


def test_tsr_validation_low_r2_for_wrong_lat():
    true_lat = 40.0
    times = pd.date_range("2013-01-01", periods=24 * 365, freq="1h", tz="UTC")
    doy = times.dayofyear.to_numpy()
    daily_toa = sg.toa_daily_irradiation(doy.astype(float), true_lat)
    tsr = np.where(daily_toa > 0, daily_toa / 12.0, 0.0)
    res = tv.validate_with_tsr(times, tsr, lat=10.0, lon=0.0)  # wrong latitude
    assert res["r2_latitude"] < 0.90
