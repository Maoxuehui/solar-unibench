"""Tests for the configurable MLP baseline (sklearn MLPRegressor).

A single multi-output :class:`sklearn.neural_network.MLPRegressor` is fit on the
flattened ``X_past[n, lb, Fp]`` (concatenated with the optional future-context
block ``X_future[n, horizon, Ff]``) and predicts the whole horizon vector at
once — ``predict`` returns shape ``(n, horizon)``.

The ``skippd_paper`` preset replicates Nie et al. 2020 (a single hidden layer of
7 logistic units); the default ``(128, 64)`` + ReLU is the in-house baseline.
"""
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from solarunibench.forecast.core import ForecastModel
from solarunibench.forecast.models.mlp_model import MLPModel


def _linear_windows(n=400, lookback=6, horizon=4, seed=0):
    """Synthetic windows where ``Y`` is a near-linear function of past features.

    ``Y[:, h] = 0.8 * X_past[:, -1, 1] + 0.5*h + noise`` — the predictive
    signal lives in the SECOND past channel (an exog), not in the last target.
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


# Small max_iter keeps the synthetic-data tests fast; we don't care about
# convergence on the toy problem, only that fit/predict round-trips.
_SMALL_MLP = {"max_iter": 50, "random_state": 0}


def test_mlp_is_forecast_model():
    m = MLPModel(**_SMALL_MLP)
    assert isinstance(m, ForecastModel)
    assert m.name == "MLP"


def test_custom_name():
    assert MLPModel(name="MLP-Custom", **_SMALL_MLP).name == "MLP-Custom"


def test_default_config_values():
    """Defaults are the documented in-house baseline: (128, 64) + relu."""
    m = MLPModel(**_SMALL_MLP)
    assert m.hidden_layer_sizes == (128, 64)
    assert m.activation == "relu"


def test_predict_shape_and_finite():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        X_past, Y = _linear_windows()
        n_test = 50
        X_tr, X_te = X_past[:-n_test], X_past[-n_test:]
        Y_tr, _ = Y[:-n_test], Y[-n_test:]
        model = MLPModel(**_SMALL_MLP)
        model.fit(X_tr, Y_tr)
        pred = model.predict(X_te)
    assert pred.shape == (n_test, Y.shape[1])
    assert np.all(np.isfinite(pred))


def test_default_config_finite_mae():
    """Default (128, 64)+relu config trains and produces a finite MAE."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        X_past, Y = _linear_windows(n=400)
        n_test = 80
        X_tr, X_te = X_past[:-n_test], X_past[-n_test:]
        Y_tr, Y_te = Y[:-n_test], Y[-n_test:]
        model = MLPModel(max_iter=200, random_state=0)
        model.fit(X_tr, Y_tr)
        pred = model.predict(X_te)
    assert np.isfinite(_mae(pred, Y_te))


def test_skippd_paper_preset_overrides_explicit_args():
    """preset='skippd_paper' wins over explicit hidden/activation kwargs."""
    m = MLPModel(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        preset="skippd_paper",
        **_SMALL_MLP,
    )
    assert m.hidden_layer_sizes == (7,)
    assert m.activation == "logistic"


def test_skippd_paper_preset_trains_and_shape():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        X_past, Y = _linear_windows(n=300, horizon=4)
        n_test = 40
        X_tr, X_te = X_past[:-n_test], X_past[-n_test:]
        Y_tr, _ = Y[:-n_test], Y[-n_test:]
        model = MLPModel(preset="skippd_paper", max_iter=200, random_state=0)
        model.fit(X_tr, Y_tr)
        pred = model.predict(X_te)
    assert pred.shape == (n_test, Y.shape[1])
    assert np.all(np.isfinite(pred))


def test_handles_xfuture_concat():
    """X_future is concatenated into the design matrix; its signal is learnable."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        rng = np.random.default_rng(2)
        n, lb, hz = 300, 4, 3
        X_past = rng.uniform(0.0, 1.0, size=(n, lb, 2))
        signal = rng.uniform(0.0, 10.0, size=(n, hz))
        X_future = signal[:, :, None]
        Y = signal + rng.normal(scale=0.05, size=(n, hz))
        n_test = 60
        tr = slice(0, n - n_test)
        te = slice(n - n_test, n)
        model = MLPModel(preset="skippd_paper", max_iter=500, random_state=0)
        model.fit(X_past[tr], Y[tr], X_future=X_future[tr])
        pred = model.predict(X_past[te], X_future=X_future[te])
    assert pred.shape == (n_test, hz)
    persist = np.repeat(X_past[te, -1, 0][:, None], hz, axis=1)
    assert _mae(pred, Y[te]) < _mae(persist, Y[te])


def test_xfuture_none_works():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        X_past, Y = _linear_windows(n=200)
        model = MLPModel(**_SMALL_MLP)
        model.fit(X_past, Y, X_future=None)
        pred = model.predict(X_past, X_future=None)
    assert pred.shape == (200, Y.shape[1])


def test_predict_before_fit_raises():
    """Calling predict before fit is a programmer error, not a silent failure."""
    X_past, _ = _linear_windows(n=10)
    with pytest.raises(RuntimeError):
        MLPModel(**_SMALL_MLP).predict(X_past)
