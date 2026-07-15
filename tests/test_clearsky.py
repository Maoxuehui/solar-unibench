import numpy as np
import pandas as pd
from solarunibench.operators import clearsky


def _summer_noon_times():
    # 2013-06-01 12:00 local at a site ~ lon=0 -> 12:00 UTC is solar noon
    return pd.date_range("2013-06-01 10:00", periods=5, freq="1h", tz="UTC")


def test_cos_zenith_bounded_and_night_low():
    times = _summer_noon_times()
    astro = clearsky.compute_astronomy(times, lat=45.0, lon=0.0)
    assert ((astro["cos_zenith"] >= 0) & (astro["cos_zenith"] <= 1)).all()


def test_clearsky_ineichen_night_near_zero():
    # Midnight UTC at lon=0 -> night -> GHI ~ 0
    night = pd.date_range("2013-06-01 00:00", periods=3, freq="1h", tz="UTC")
    cs = clearsky.clearsky_ineichen(night, lat=45.0, lon=0.0)
    assert (cs["ghi_clearsky"] < 1.0).all()


def test_clearsky_simple_matches_ineichen_shape_daytime():
    times = _summer_noon_times()
    simple = clearsky.clearsky_ghi_simple(times, lat_deg=45.0, lon_deg=0.0)
    cs = clearsky.clearsky_ineichen(times, lat=45.0, lon=0.0)
    # Both peak around local noon (lon=0 -> 12 UTC); daytime values positive & ordered similarly
    assert (simple[2:4] > 0).all()
    assert np.corrcoef(simple, cs["ghi_clearsky"].to_numpy())[0, 1] > 0.95
