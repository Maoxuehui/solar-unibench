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


def test_missing_modality_stays_all_nan_at_night():
    # PV-less dataset (e.g. Folsom): pv_power_normalized is all-NA. The quality
    # audit must NOT fabricate 0.0 at night for a sensor that doesn't exist
    # (Honest NaN). The night row must still be flagged.
    df = pd.DataFrame({
        "cos_zenith": [-0.5, 0.5, 0.5],
        "ghi_actual": [np.nan, 600.0, 400.0],
        "ghi_clearsky": [0.0, 800.0, 800.0],
        "pv_power_normalized": [np.nan, np.nan, np.nan],  # PV-less dataset
    })
    out = quality.apply_quality_audit(df, capacity=1.0)
    assert out["pv_power_normalized"].isna().all()  # no fabricated night zero
    assert out.loc[0, "quality_flag"] == 3  # night still flagged


def test_all_na_ghi_modality_keeps_clearness_nan_at_night():
    # A dataset with no irradiance sensor (ghi_actual entirely NA) must not have
    # clearness_index fabricated to 0.0 at night — the ratio is undefined and
    # must stay NaN (Honest NaN for a missing irradiance modality).
    df = pd.DataFrame({
        "cos_zenith": [-0.5, 0.5, 0.5],
        "ghi_actual": [np.nan, np.nan, np.nan],  # missing irradiance modality
        "ghi_clearsky": [0.0, 800.0, 800.0],
        "pv_power_normalized": [np.nan, np.nan, np.nan],
    })
    out = quality.apply_quality_audit(df, capacity=1.0)
    # ghi_actual stays all-NaN (clamp skipped), clearness stays NaN at night.
    assert out["ghi_actual"].isna().all()
    assert pd.isna(out.loc[0, "clearness_index"])


def test_present_modality_at_night_is_still_zeroed():
    # The normal case still works: a present (non-NA) PV value at night IS zeroed.
    df = pd.DataFrame({
        "cos_zenith": [-0.5, 0.5, 0.5],
        "ghi_actual": [np.nan, 600.0, 400.0],
        "ghi_clearsky": [0.0, 800.0, 800.0],
        "pv_power_normalized": [0.2, 0.5, 0.4],  # PV present, night row has a value
    })
    out = quality.apply_quality_audit(df, capacity=1.0)
    assert out.loc[0, "pv_power_normalized"] == 0.0  # night zeroed
    assert out.loc[0, "quality_flag"] == 3


def test_object_dtype_all_na_ghi_does_not_crash():
    # An object-dtype all-NA `ghi_actual` column (a missing irradiance modality
    # filled with pd.NA) must NOT crash the clearness-index computation. The
    # unguarded `np.clip(...).astype("float32")` raised TypeError on NAType.
    # Guarded path: clearness_index is all-NaN (Honest NaN for absent modality).
    df = pd.DataFrame({
        "cos_zenith": [-0.5, 0.5, 0.5],
        "ghi_actual": pd.Series([pd.NA, pd.NA, pd.NA], dtype=object),  # object all-NA
        "ghi_clearsky": [0.0, 800.0, 800.0],
        "pv_power_normalized": [np.nan, np.nan, np.nan],
    })
    out = quality.apply_quality_audit(df, capacity=1.0)
    assert "clearness_index" in out.columns
    assert out["clearness_index"].isna().all()  # no crash, all-NaN result
    assert out["ghi_actual"].isna().all()  # clamp skipped for missing modality
