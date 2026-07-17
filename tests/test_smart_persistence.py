"""Tests for the Smart Persistence physical skill reference baseline.

Spec §10: clear-sky-scaled persistence. The model is target-agnostic — it works
for PV power and GHI alike by estimating a per-sample clearness ratio
``κ = target_now / clearsky_now`` and propagating it across the horizon via the
future clear-sky column (``X_future[:, :, 0]``).
"""
import numpy as np
import pytest

from solarunibench.forecast.core import ForecastModel
from solarunibench.forecast.models.smart_persistence import SmartPersistenceModel


def _toy_windows(n=8, horizon=4, lookback=3, kappa=0.5, csky_level=800.0):
    """Build synthetic (X_past, X_future) where target_now = kappa * csky_now.

    ``X_past`` has a single channel (the target) — that is all SmartPersistence
    consumes. ``X_future`` carries two channels ``[ghi_clearsky, cos_zenith]``
    per the EndogenousFeatureEngineer contract; the second is unused here.
    """
    rng = np.arange(n, dtype=float)
    # Future clear-sky varies across horizon so we can verify scaling per-step.
    csky_future = (csky_level * (0.5 + 0.1 * np.add.outer(rng, np.arange(horizon)) / horizon))
    # csky_now proxy = first future step (matches predict's X_future[:, 0, 0]).
    csky_now = csky_future[:, 0]
    target_now = kappa * csky_now
    X_past = np.repeat(target_now[:, None], lookback, axis=1)[..., None]
    cos_zen = np.full((n, horizon), 0.5)
    X_future = np.stack([csky_future, cos_zen], axis=-1)
    return X_past, X_future


def test_smart_persistence_is_forecast_model():
    m = SmartPersistenceModel()
    assert isinstance(m, ForecastModel)
    assert m.name == "SmartPersistence"


def test_recovers_constant_kappa_daylight():
    """target tracks κ·clearsky with κ=0.5 → recovered pred ≈ 0.5·csky_future."""
    kappa = 0.5
    X_past, X_future = _toy_windows(kappa=kappa)
    model = SmartPersistenceModel()
    model.fit(X_past, Y=None, X_future=X_future)
    pred = model.predict(X_past, X_future=X_future)

    assert pred.shape == (X_past.shape[0], X_future.shape[1])
    expected = kappa * X_future[:, :, 0]
    np.testing.assert_allclose(pred, expected, rtol=1e-6, atol=1e-6)


def test_clip_nonnegative_at_night():
    """When target_now = 0 (night) the forecast must be ≥ 0, never negative."""
    X_past, X_future = _toy_windows(kappa=0.0)
    # Plant a negative future clear-sky value to exercise the floor.
    X_future[:, :, 0] = np.where(X_future[:, :, 0] > 0, X_future[:, :, 0], -1.0)
    # Make future clearsky negative at a couple of slots; csky_now stays positive.
    X_future[0, 0, 0] = 100.0
    model = SmartPersistenceModel()
    pred = model.predict(X_past, X_future=X_future)
    assert (pred >= 0).all()


def test_capacity_clipping():
    """When κ·csky_future exceeds ``capacity``, pred is clipped to capacity."""
    X_past, X_future = _toy_windows(kappa=0.9, csky_level=1000.0)
    model = SmartPersistenceModel(capacity=400.0)
    pred = model.predict(X_past, X_future=X_future)
    assert (pred <= 400.0 + 1e-9).all()
    # And the floor still holds.
    assert (pred >= 0).all()


def test_xfuture_none_falls_back_to_persistence():
    """No clear-sky context → plain persistence: target_now repeated over horizon."""
    X_past, _ = _toy_windows(kappa=0.5)
    target_now = X_past[:, -1, 0]
    horizon = 4
    model = SmartPersistenceModel(horizon=horizon)
    pred = model.predict(X_past, X_future=None)
    assert pred.shape == (X_past.shape[0], horizon)
    np.testing.assert_allclose(pred, np.repeat(target_now[:, None], horizon, axis=1))


def test_fit_accepts_minimal_args():
    """fit must accept the ABC signature without requiring Y/X_future."""
    X_past, X_future = _toy_windows()
    SmartPersistenceModel().fit(X_past)  # no kwargs — should not raise
