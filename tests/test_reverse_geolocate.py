import numpy as np
import pandas as pd
from solarunibench.operators import clearsky, reverse_geolocate


def test_recovers_known_site_within_tolerance():
    true_lat, true_lon = 35.0, -120.0
    times = pd.date_range("2013-03-01", periods=24 * 60, freq="1h", tz="UTC")
    # Synthesize a POWER shape from the simple clear-sky model at the true site.
    shape = clearsky.clearsky_ghi_simple(times, true_lat, true_lon)
    power = shape / (shape.max() + 1e-9)  # normalized like GEFCom POWER
    res = reverse_geolocate.reverse_geolocate(times, power,
                                              lat_grid=np.arange(20, 51, 2.0),
                                              lon_grid=np.arange(-140, -100, 2.0))
    assert abs(res["latitude"] - true_lat) <= 4.0
    assert abs(res["longitude"] - true_lon) <= 6.0
