import numpy as np
import pandas as pd
from solarunibench.operators import quality


def _df():
    return pd.DataFrame({
        "cos_zenith": [0.0, 0.5, 0.5, 0.5],
        "ghi_actual": [10.0, 600.0, 1500.0, 400.0],   # night, good, >clearsky(outlier), good
        "ghi_clearsky": [0.0, 800.0, 800.0, 800.0],
        "pv_power_normalized": [0.05, 0.5, 1.2, 0.4],
    })


def test_night_zeroed_and_flagged():
    out = quality.apply_quality_audit(_df(), capacity=1.0)
    assert out.loc[0, "ghi_actual"] == 0.0
    assert out.loc[0, "pv_power_normalized"] == 0.0
    assert out.loc[0, "quality_flag"] == 3  # night
    assert out.loc[0, "clearness_index"] == 0.0


def test_clearness_index_bounded():
    out = quality.apply_quality_audit(_df(), capacity=1.0)
    assert out["clearness_index"].between(0.0, 1.5).all()
    # row 1: ghi_actual=600, ghi_clearsky=800 -> 0.75
    assert abs(out.loc[1, "clearness_index"] - 0.75) < 0.01


def test_pv_clamped_to_capacity():
    out = quality.apply_quality_audit(_df(), capacity=1.0)
    assert out["pv_power_normalized"].max() <= 1.0
