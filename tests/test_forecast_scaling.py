"""Tests for :class:`PhysicalFeatureScaler` — global, train-fit, zero-leakage.

Design contract (see `.superpowers/sdd/bi-task-2-brief.md`):

* **Irradiance columns** (regex ``^(ghi|dni|dhi)``) are divided by a physical
  constant ``irradiance_max`` (default 1500.0 W/m^2, the solar-constant ceiling).
  No fit is needed → no possibility of leakage; nighttime 0.0 is preserved
  exactly (``0 / 1500 == 0``).
* **NWP columns** (regex ``^nwp_``) are z-scored per column with a
  :class:`sklearn.preprocessing.StandardScaler` **fit on TRAIN μ/σ only**;
  NaNs are skipped at fit time (NaN-safe).
* **Everything else** (``cos_zenith`` ∈ [0,1], ``clearness_index`` ∈ [0,1],
  ``pv_power_normalized`` target ∈ [0,1], ``timestamp_utc``, ``entity_id``,
  ``split``) is passed through unchanged.

The scaler is also verified to be a graceful no-op when no irradiance/NWP
columns are present (the existing forecast core tests use a synthetic df
without those columns and must not break).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from solarunibench.forecast.core import (
    EndogenousFeatureEngineer,
    PipelineConfig,
    SolarForecastPipeline,
    ForecastModel,
)
from solarunibench.forecast.scaling import PhysicalFeatureScaler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _train_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """A frame with irradiance + NWP + passthrough columns.

    ``ghi_clearsky`` peaks at 1100 W/m^2 (above the [0,1000] target range that
    broke the unscaled MLP); ``nwp_VAR78`` is an exogenous NWP channel with a
    non-trivial train distribution (mean≈50, std≈29).
    """
    rng = np.random.default_rng(seed)
    t = pd.date_range("2020-06-01", periods=n, freq="1h", tz="UTC")
    bell = np.clip(np.sin(np.pi * (t.hour.to_numpy() - 6) / 13.0), 0.0, None)
    # Chronological 80/20 train/test split so the pipeline's _load_split finds
    # both halves. The direct scaler tests call fit_transform on the whole df
    # and are agnostic to the split values.
    split = np.array(["train"] * n, dtype=object)
    split[int(n * 0.8):] = "test"
    return pd.DataFrame(
        {
            "timestamp_utc": t,
            "entity_id": "E1",
            # Irradiance: peaks at 1100 → after /1500 peaks at 0.7333.
            "ghi_clearsky": bell * 1100.0,
            "ghi_actual": bell * 1000.0 * rng.uniform(0.7, 1.0, n),
            # NWP exog: mean≈50, std≈29.
            "nwp_VAR78": rng.uniform(0.0, 100.0, n),
            # Passthrough columns (must be byte-for-byte unchanged).
            "cos_zenith": np.linspace(0.0, 0.9, n),
            "clearness_index": np.linspace(0.0, 0.8, n),
            "pv_power_normalized": np.clip(bell * 0.9 + rng.normal(0, 0.02, n), 0, 1),
            "split": split,
        }
    )


def _test_df_different_distribution(n: int = 50, seed: int = 1) -> pd.DataFrame:
    """A test frame whose NWP distribution DIFFERS from train.

    Used by the leakage test: a correctly-fit scaler must scale test using
    TRAIN μ/σ, producing a transformed test mean far from 0.
    """
    rng = np.random.default_rng(seed)
    t = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp_utc": t,
            "entity_id": "E1",
            "ghi_clearsky": np.linspace(0, 900, n),
            "nwp_VAR78": rng.uniform(190.0, 210.0, n),  # mean≈200, very different
            "cos_zenith": np.linspace(0.1, 0.5, n),
            "pv_power_normalized": np.linspace(0, 0.5, n),
            "split": "test",
        }
    )


# ---------------------------------------------------------------------------
# Step 1: fit_transform contract
# ---------------------------------------------------------------------------

def test_fit_transform_scales_irradiance_by_physical_constant():
    """Irradiance cols divided by 1500 (constant); train max 1100 → 0.7333."""
    df = _train_df()
    out = PhysicalFeatureScaler().fit_transform(df)
    # Train ghi_clearsky peaks below 1100 → scaled peaks below 1100/1500.
    assert out["ghi_clearsky"].max() <= 1100.0 / 1500.0 + 1e-9
    # Exact division: the entire column is original / 1500.
    np.testing.assert_allclose(
        out["ghi_clearsky"].to_numpy(), df["ghi_clearsky"].to_numpy() / 1500.0
    )
    # Nighttime 0.0 preserved exactly.
    assert (out.loc[df["ghi_clearsky"] == 0.0, "ghi_clearsky"] == 0.0).all()


def test_fit_transform_zscores_nwp_on_train_stats():
    """NWP col is z-scored using TRAIN μ/σ → train mean≈0, std≈1 after fit_transform."""
    df = _train_df()
    out = PhysicalFeatureScaler().fit_transform(df)
    np.testing.assert_allclose(out["nwp_VAR78"].mean(), 0.0, atol=1e-9)
    np.testing.assert_allclose(out["nwp_VAR78"].std(ddof=0), 1.0, atol=1e-9)


def test_fit_transform_preserves_passthrough_columns():
    """cos_zenith, clearness_index and the target are unchanged."""
    df = _train_df()
    out = PhysicalFeatureScaler().fit_transform(df)
    for col in ("cos_zenith", "clearness_index", "pv_power_normalized"):
        np.testing.assert_array_equal(out[col].to_numpy(), df[col].to_numpy())


def test_fit_transform_preserves_metadata_columns():
    """timestamp_utc / entity_id / split are passed through unchanged."""
    df = _train_df()
    out = PhysicalFeatureScaler().fit_transform(df)
    for col in ("timestamp_utc", "entity_id", "split"):
        pd.testing.assert_series_equal(out[col], df[col], check_names=True)


# ---------------------------------------------------------------------------
# Step 1: leakage — test uses TRAIN μ/σ, not test's own
# ---------------------------------------------------------------------------

def test_transform_uses_train_stats_no_leakage():
    """Fit on train; transform a different test df → TRAIN μ/σ are used."""
    train = _train_df()
    test = _test_df_different_distribution()
    scaler = PhysicalFeatureScaler().fit(train)
    out_test = scaler.transform(test)

    train_mean = train["nwp_VAR78"].mean()
    train_std = train["nwp_VAR78"].std(ddof=0)
    expected = (test["nwp_VAR78"] - train_mean) / train_std
    np.testing.assert_allclose(out_test["nwp_VAR78"].to_numpy(), expected.to_numpy())

    # If test's OWN stats were used (leak), the transformed test mean would be ≈0.
    # With train stats it is (test_mean - train_mean) / train_std ≈ 5.2 — far from 0.
    assert abs(out_test["nwp_VAR78"].mean()) > 1.0, (
        "transform appears to use test statistics (leakage): "
        f"transformed test mean={out_test['nwp_VAR78'].mean():.4f} should be far from 0"
    )


def test_irradiance_scaling_is_fit_invariant():
    """Irradiance ÷ constant: fit() must not change transform output (no fit state)."""
    df = _train_df()
    a = PhysicalFeatureScaler().fit(df).transform(df)
    # A fresh scaler with NO fit at all must produce the same irradiance scaling.
    b = PhysicalFeatureScaler().transform(df)
    np.testing.assert_allclose(a["ghi_clearsky"].to_numpy(), b["ghi_clearsky"].to_numpy())


# ---------------------------------------------------------------------------
# Step 1: passthrough (no irradiance / no NWP cols → no-op)
# ---------------------------------------------------------------------------

def test_passthrough_when_no_irradiance_or_nwp_columns():
    """Existing forecast-core tests use a df without irradiance/NWP cols → no-op."""
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC"),
            "entity_id": "E1",
            "pv_power_normalized": rng.uniform(0, 1, n),
            "cos_zenith": rng.uniform(0, 1, n),
            "split": "train",
        }
    )
    out = PhysicalFeatureScaler().fit_transform(df)
    # Frame is unchanged (same columns, same values, same dtypes).
    pd.testing.assert_frame_equal(out, df)


def test_custom_irradiance_max_changes_ceiling():
    """irradiance_max is injectable (ablation axis)."""
    df = _train_df()
    out = PhysicalFeatureScaler(irradiance_max=1100.0).fit_transform(df)
    np.testing.assert_allclose(
        out["ghi_clearsky"].to_numpy(), df["ghi_clearsky"].to_numpy() / 1100.0
    )


# ---------------------------------------------------------------------------
# Pipeline wiring: scaler is the pipeline default + injectable
# ---------------------------------------------------------------------------

class _NoOpModel(ForecastModel):
    """Records the design matrix it sees so the test can assert on scaling."""

    def fit(self, X_past, Y, times=None, X_future=None, *, images=None):
        self.seen_X = X_past
        return None

    def predict(self, X_past, times=None, X_future=None, *, images=None):
        return np.zeros((len(X_past), 3), np.float32)

    @property
    def name(self) -> str:
        return "noop"


def test_pipeline_default_scaler_applies_before_feature_engineering():
    """Default scaler scales the raw frame before ``create_features`` sees it."""
    df = _train_df()
    fe = EndogenousFeatureEngineer(
        target="pv_power_normalized", feature_cols=["ghi_clearsky", "nwp_VAR78"]
    )
    model = _NoOpModel()
    cfg = PipelineConfig(
        dataset="toy", target="pv_power_normalized", lookback=6, horizon=3, stride=6
    )
    SolarForecastPipeline(cfg, fe, model).run(df_provider=lambda c: df)

    # ghi_clearsky channel is X_past[:, :, 1]; after scaling it must be ≤ 1100/1500.
    ghi_channel = model.seen_X[:, :, 1]
    assert ghi_channel.max() <= 1100.0 / 1500.0 + 1e-9
    # nwp_VAR78 channel is X_past[:, :, 2]; after z-score it is O(1).
    nwp_channel = model.seen_X[:, :, 2]
    assert np.isfinite(nwp_channel).all()
    assert abs(nwp_channel.mean()) < 1.0


def test_pipeline_scaler_is_injectable_for_ablation():
    """A no-op injected scaler leaves irradiance unscaled (raw ablation axis)."""
    df = _train_df()

    class _RawScaler:
        def fit(self, df_):
            return self

        def transform(self, df_):
            return df_

    fe = EndogenousFeatureEngineer(
        target="pv_power_normalized", feature_cols=["ghi_clearsky"]
    )
    model = _NoOpModel()
    cfg = PipelineConfig(
        dataset="toy", target="pv_power_normalized", lookback=6, horizon=3, stride=6
    )
    SolarForecastPipeline(cfg, fe, model, scaler=_RawScaler()).run(
        df_provider=lambda c: df
    )
    # Raw (unscaled) ghi_clearsky still in [0, 1100].
    ghi_channel = model.seen_X[:, :, 1]
    assert ghi_channel.max() > 1000.0  # unscaled → preserves W/m^2 magnitude


# ---------------------------------------------------------------------------
# NaN-safety (documented behavior)
# ---------------------------------------------------------------------------

def test_nwp_fit_is_nan_safe():
    """NaNs in the NWP column at fit time are skipped (don't poison μ/σ)."""
    n = 100
    rng = np.random.default_rng(0)
    vals = rng.uniform(0.0, 100.0, n)
    df = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC"),
            "entity_id": "E1",
            "nwp_VAR78": vals,
            "pv_power_normalized": rng.uniform(0, 1, n),
            "split": "train",
        }
    )
    df.loc[df.index[:20], "nwp_VAR78"] = np.nan  # 20 NaNs at the top
    out = PhysicalFeatureScaler().fit_transform(df)
    # Fit ignored NaN → mean/std of the FINITE transformed rows ≈ 0 / 1.
    finite = out["nwp_VAR78"].dropna()
    np.testing.assert_allclose(finite.mean(), 0.0, atol=1e-9)
    np.testing.assert_allclose(finite.std(ddof=0), 1.0, atol=1e-9)
    # NaN positions preserved (transform must not fabricate values).
    assert out["nwp_VAR78"].isna().sum() == 20
