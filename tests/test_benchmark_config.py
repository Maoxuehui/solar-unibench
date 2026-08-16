"""E8 unified benchmark — Task 1 tests.

Covers the per-dataset physics-time config map and the
``clearness_index_aligned`` feature that forces a uniform F=4 input block
(``[target, cos_zenith, ghi_clearsky, clearness_index_aligned]``) across all
PV/GHI datasets via :class:`EndogenousFeatureEngineer`.

Design notes
------------
* The clearness proxy must be **NaN-safe** (Honest NaN): missing measurements
  propagate as NaN, never fabricated. PV datasets additionally define the
  proxy as ``0 where ghi_clearsky == 0`` (night → 0, no div-by-zero).
* Physical-time horizons: the step counts in ``horizons_steps`` multiplied by
  ``sampling_mins`` must equal ``physical_horizons_min`` for every dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solarunibench.benchmark.config import (
    DATASET_CONFIG,
    DatasetConfig,
    physical_horizons_min,
)
from solarunibench.benchmark.features import add_clearness_aligned
from solarunibench.forecast.core import EndogenousFeatureEngineer

# Target column names (single source of truth, mirrors the data contract).
_GHI_TARGET = "ghi_actual"
_PV_TARGET = "pv_power_normalized"


# -- config matrix -----------------------------------------------------------


@pytest.mark.parametrize(
    "name,target,sampling,lookback,horizons,image",
    [
        ("folsom", _GHI_TARGET, 1, 240, [5, 10, 15], True),
        ("skippd", _PV_TARGET, 1, 240, [5, 10, 15], True),
        ("sunlake", _GHI_TARGET, 30, 24, [1, 2, 3], False),
        ("gefcom2014", _PV_TARGET, 60, 24, [1, 2, 3], False),
    ],
)
def test_dataset_config_matrix(
    name: str,
    target: str,
    sampling: int,
    lookback: int,
    horizons: list[int],
    image: bool,
) -> None:
    cfg = DATASET_CONFIG[name]
    assert isinstance(cfg, DatasetConfig)
    assert cfg.target == target
    assert cfg.sampling_mins == sampling
    assert cfg.lookback_steps == lookback
    assert cfg.horizons_steps == horizons
    assert cfg.image_dataset is image


def test_dataset_config_covers_exactly_four_datasets() -> None:
    assert set(DATASET_CONFIG) == {"folsom", "skippd", "sunlake", "gefcom2014"}


def test_horizons_steps_match_physical_time() -> None:
    """horizons_steps[i] * sampling_mins == physical_horizons_min[i]."""
    for name, cfg in DATASET_CONFIG.items():
        phys = physical_horizons_min(name)
        derived = [h * cfg.sampling_mins for h in cfg.horizons_steps]
        assert derived == phys, f"{name}: {derived} != physical {phys}"


@pytest.mark.parametrize(
    "name,expected",
    [
        ("folsom", [5, 10, 15]),
        ("skippd", [5, 10, 15]),
        ("sunlake", [30, 60, 90]),
        ("gefcom2014", [60, 120, 180]),
    ],
)
def test_physical_horizons_min(name: str, expected: list[int]) -> None:
    assert physical_horizons_min(name) == expected


def test_lookback_is_four_hours_physical_or_24h_gefcom() -> None:
    """Lookback window is 4h (folsom/skippd/sunlake) or 24h (gefcom2014)."""
    expected_phys = {
        "folsom": 240,
        "skippd": 240,
        "sunlake": 720,  # 24 steps * 30 min
        "gefcom2014": 1440,  # 24 steps * 60 min
    }
    for name, mins in expected_phys.items():
        cfg = DATASET_CONFIG[name]
        assert cfg.lookback_steps * cfg.sampling_mins == mins, name


# -- clearness_index_aligned: GHI datasets -----------------------------------


def _toy_ghi_df() -> pd.DataFrame:
    """Deterministic GHI frame covering day, night, over/under-clearsky, NaN."""
    return pd.DataFrame(
        {
            "ghi_actual": [500.0, 2000.0, 0.0, np.nan, 800.0],
            "ghi_clearsky": [1000.0, 1000.0, 0.0, 1000.0, np.nan],
        }
    )


def test_clearness_aligned_ghi_is_real_ratio() -> None:
    df = add_clearness_aligned(_toy_ghi_df(), _GHI_TARGET)
    a = df["clearness_index_aligned"].to_numpy()
    # 500/1000 = 0.5 (real clearness ratio, not fabricated).
    assert np.isclose(a[0], 0.5)
    # 2000/1000 = 2.0 → clipped to 1.5.
    assert np.isclose(a[1], 1.5)
    # night: 0/eps → 0 (no inf / no div-by-zero).
    assert np.isclose(a[2], 0.0)
    # NaN target must propagate (Honest NaN — never fabricated).
    assert np.isnan(a[3])
    # NaN clearsky must propagate.
    assert np.isnan(a[4])


def test_clearness_aligned_ghi_column_added_others_untouched() -> None:
    df0 = _toy_ghi_df()
    df = add_clearness_aligned(df0, _GHI_TARGET)
    assert "clearness_index_aligned" in df.columns
    # Original columns unchanged (NaN-aware comparison).
    pd.testing.assert_series_equal(
        df["ghi_actual"], df0["ghi_actual"], check_names=False
    )
    pd.testing.assert_series_equal(
        df["ghi_clearsky"], df0["ghi_clearsky"], check_names=False
    )


# -- clearness_index_aligned: PV datasets (proxy) ----------------------------


def _toy_pv_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            _PV_TARGET: [0.5, 0.9, 0.5, np.nan],
            "ghi_clearsky": [1000.0, 0.0, np.nan, 1000.0],
        }
    )


def test_clearness_aligned_pv_proxy_zero_when_clearsky_zero() -> None:
    df = add_clearness_aligned(_toy_pv_df(), _PV_TARGET)
    a = df["clearness_index_aligned"].to_numpy()
    # Day: target/clearsky proxy (0.5/1000 = 5e-4), clipped to [0, 1.5].
    assert np.isclose(a[0], 0.0005)
    # Night (clearsky == 0): defined as 0 regardless of target (no div-by-zero).
    assert a[1] == 0.0
    # NaN clearsky propagates (NaN-safe, not fabricated).
    assert np.isnan(a[2])
    # NaN target propagates.
    assert np.isnan(a[3])


def test_clearness_aligned_pv_proxy_never_negative_or_above_cap() -> None:
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            _PV_TARGET: rng.uniform(-0.2, 2.0, n),  # include negatives/noise
            "ghi_clearsky": rng.uniform(1.0, 1000.0, n),
        }
    )
    a = add_clearness_aligned(df, _PV_TARGET)["clearness_index_aligned"].to_numpy()
    finite = a[np.isfinite(a)]
    assert (finite >= 0.0).all() and (finite <= 1.5).all()


def test_clearness_aligned_no_fabrication_when_target_nan_all() -> None:
    """All-NaN target column stays all-NaN (must not fabricate zeros)."""
    df = pd.DataFrame(
        {_PV_TARGET: [np.nan, np.nan, np.nan], "ghi_clearsky": [1000.0, 500.0, 0.0]}
    )
    a = add_clearness_aligned(df, _PV_TARGET)["clearness_index_aligned"].to_numpy()
    # PV rule forces 0 where clearsky==0, so only the last row is defined (0).
    assert np.isnan(a[0]) and np.isnan(a[1])
    assert a[2] == 0.0  # night rule still applies even with NaN target


# -- F=4 end-to-end via EndogenousFeatureEngineer ----------------------------


def _frame_for_engineer(target: str) -> pd.DataFrame:
    """Build a frame with the columns the engineer + feature need."""
    n = 40
    t = pd.date_range("2021-06-01", periods=n, freq="1h", tz="UTC")
    bell = np.clip(np.sin(np.pi * np.arange(n) / 20.0), 0.0, None)
    if target == _GHI_TARGET:
        ghi_actual = bell * 700.0
    else:
        ghi_actual = bell * 700.0  # present but not the target
    pv = np.clip(bell * 0.9, 0.0, 1.0)
    df = pd.DataFrame(
        {
            "timestamp_utc": t,
            "entity_id": "E1",
            _GHI_TARGET: ghi_actual,
            _PV_TARGET: pv,
            "ghi_clearsky": bell * 950.0,
            "cos_zenith": np.clip(bell, 0.0, 1.0),
        }
    )
    return add_clearness_aligned(df, target)


@pytest.mark.parametrize("target", [_GHI_TARGET, _PV_TARGET])
def test_engineer_yields_four_input_channels(target: str) -> None:
    """F=4: [target, cos_zenith, ghi_clearsky, clearness_index_aligned]."""
    df = _frame_for_engineer(target)
    fe = EndogenousFeatureEngineer(
        target,
        feature_cols=["cos_zenith", "ghi_clearsky", "clearness_index_aligned"],
    )
    Xp, Y, _T, Xf, _cz, _img = fe.create_features(df, lookback=6, horizon=3, stride=1)
    assert Xp.ndim == 3
    assert Xp.shape[2] == 4, f"expected F=4, got {Xp.shape[2]} for target={target}"
    assert Y.shape[1] == 3
    # known-future X_future still [ghi_clearsky, cos_zenith] = 2 channels.
    assert Xf is not None and Xf.shape[2] == 2
    # No NaNs leaked into the design matrix for this clean synthetic frame.
    assert np.isfinite(Xp).all(), f"NaNs in X_past for target={target}"


def test_clearness_aligned_missing_required_column_raises() -> None:
    """No ghi_clearsky → ValueError, not a fabricated column."""
    df = pd.DataFrame({_GHI_TARGET: [1.0, 2.0]})
    with pytest.raises(ValueError):
        add_clearness_aligned(df, _GHI_TARGET)
