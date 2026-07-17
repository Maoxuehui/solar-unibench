"""Tests for the LightGBM tabular baseline (per-horizon direct multi-step).

The standard solar-forecasting tabular recipe: flatten ``X_past[n, lb, Fp]``
(and optional ``X_future[n, horizon, Ff]``) into one 2-D design matrix and fit
one :class:`lightgbm.LGBMRegressor` per horizon step. ``predict`` stacks the
per-step predictions into ``(n, horizon)``.
"""
import numpy as np
import pytest

from solarunibench.forecast.core import ForecastModel
from solarunibench.forecast.models.lightgbm_model import LightGBMModel


def _linear_windows(n=400, lookback=6, horizon=4, seed=0):
    """Synthetic windows where ``Y`` is a near-linear function of past features.

    ``Y[:, h] = 0.8 * X_past[:, -1, 1] + 0.5*h + noise`` — the predictive
    signal lives in the SECOND past channel (an exog), not in the last target,
    so plain persistence (repeat of ``X_past[:, -1, 0]``) is a weak baseline
    that LightGBM should beat decisively.
    """
    rng = np.random.default_rng(seed)
    Fp = 2
    X_past = rng.uniform(0.0, 10.0, size=(n, lookback, Fp))
    last_exog = X_past[:, -1, 1]
    H = np.arange(horizon, dtype=float)
    Y = (
        last_exog[:, None] * 0.8
        + 0.5 * H[None, :]
        + rng.normal(scale=0.1, size=(n, horizon))
    )
    return X_past, Y


def _mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def test_lightgbm_is_forecast_model():
    m = LightGBMModel()
    assert isinstance(m, ForecastModel)
    assert m.name == "LightGBM"


def test_custom_name():
    assert LightGBMModel(name="LGBM-Custom").name == "LGBM-Custom"


def test_predict_shape_and_finite():
    X_past, Y = _linear_windows()
    n_test = 50
    X_tr, X_te = X_past[:-n_test], X_past[-n_test:]
    Y_tr, _ = Y[:-n_test], Y[-n_test:]
    model = LightGBMModel()
    model.fit(X_tr, Y_tr)
    pred = model.predict(X_te)
    assert pred.shape == (n_test, Y.shape[1])
    assert np.all(np.isfinite(pred))


def test_beats_persistence_mae():
    """On synthetic near-linear y, LightGBM MAE must be < persistence MAE."""
    X_past, Y = _linear_windows(n=600, seed=1)
    n_test = 120
    X_tr, X_te = X_past[:-n_test], X_past[-n_test:]
    Y_tr, Y_te = Y[:-n_test], Y[-n_test:]
    model = LightGBMModel()
    model.fit(X_tr, Y_tr)
    pred = model.predict(X_te)
    horizon = Y.shape[1]
    persist = np.repeat(X_te[:, -1, 0][:, None], horizon, axis=1)
    assert _mae(pred, Y_te) < _mae(persist, Y_te)


def test_handles_xfuture_concat():
    """X_future is concatenated into the design matrix; its signal is learnable."""
    rng = np.random.default_rng(2)
    n, lb, hz = 300, 4, 3
    X_past = rng.uniform(0.0, 1.0, size=(n, lb, 2))
    signal = rng.uniform(0.0, 10.0, size=(n, hz))
    X_future = signal[:, :, None]  # single informative future channel
    Y = signal + rng.normal(scale=0.05, size=(n, hz))
    n_test = 60
    tr = slice(0, n - n_test)
    te = slice(n - n_test, n)
    model = LightGBMModel()
    model.fit(X_past[tr], Y[tr], X_future=X_future[tr])
    pred = model.predict(X_past[te], X_future=X_future[te])
    assert pred.shape == (n_test, hz)
    persist = np.repeat(X_past[te, -1, 0][:, None], hz, axis=1)
    assert _mae(pred, Y[te]) < _mae(persist, Y[te])


def test_xfuture_none_works():
    X_past, Y = _linear_windows(n=200)
    model = LightGBMModel()
    model.fit(X_past, Y, X_future=None)
    pred = model.predict(X_past, X_future=None)
    assert pred.shape == (200, Y.shape[1])


def test_predict_before_fit_raises():
    """Calling predict before fit is a programmer error, not a silent failure."""
    X_past, _ = _linear_windows(n=10)
    with pytest.raises(RuntimeError):
        LightGBMModel().predict(X_past)
