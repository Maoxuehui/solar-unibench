"""Tests for the LSTM sequence baseline (PyTorch).

:class:`LSTMModel` subclasses :class:`ForecastModel` and consumes the past
sequence block ``X_past[n, lb, Fp]`` directly (batch, seq, feat) — the natural
shape for an ``nn.LSTM``. A linear head maps the last LSTM hidden state to the
full forecast horizon, so ``predict`` returns ``(n, horizon)`` in one shot.

The fast suite uses a TINY config (``hidden=8``, ``epochs=2``, CPU) so it stays
quick; the "learns a temporal pattern" test bumps ``epochs`` on a small synthetic
problem where a constant (input-free) baseline is easily beaten.
"""
import numpy as np
import pytest
import torch

from solarunibench.forecast.core import ForecastModel
from solarunibench.forecast.models.lstm_model import LSTMModel

# TINY config: keeps the shape / finite / determinism tests sub-second on CPU.
_TINY: dict = {"hidden_size": 8, "num_layers": 2, "epochs": 2, "batch_size": 32, "device": "cpu"}


def _temporal_windows(n=400, lookback=8, horizon=4, seed=0):
    """Synthetic windows whose target is a learnable function of recent lags.

    ``Y[:, h] = mean(X_past[:, -3:, 0]) + 0.1*h + noise`` — the predictive signal
    is the mean of the last three target lags, which a recurrent net can extract
    but an input-free constant baseline cannot. Channel 0 is the target; channel
    1 is an uninformative exog (ensures the model uses >1 feature without confusing it).
    """
    rng = np.random.default_rng(seed)
    Fp = 2
    X_past = rng.uniform(0.0, 1.0, size=(n, lookback, Fp))
    last3_mean = X_past[:, -3:, 0].mean(axis=1)
    H = np.arange(horizon, dtype=float)
    Y = last3_mean[:, None] + 0.1 * H[None, :] + rng.normal(scale=0.02, size=(n, horizon))
    return X_past, Y


def _mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def _constant_baseline(Y_tr: np.ndarray, Y_te: np.ndarray) -> np.ndarray:
    """Input-free baseline: broadcast the per-horizon training mean."""
    return np.repeat(Y_tr.mean(axis=0)[None, :], Y_te.shape[0], axis=0)


# -- ABC / identity ----------------------------------------------------------


def test_lstm_is_forecast_model():
    m = LSTMModel(**_TINY)
    assert isinstance(m, ForecastModel)
    assert m.name == "LSTM"


def test_custom_name():
    assert LSTMModel(name="LSTM-Custom", **_TINY).name == "LSTM-Custom"


def test_default_config_values():
    """Defaults match the documented Phase 6b baseline."""
    m = LSTMModel()
    assert m.hidden_size == 64
    assert m.num_layers == 2
    assert m.dropout == 0.2
    assert m.lr == 1e-3
    assert m.epochs == 8
    assert m.batch_size == 32


# -- shape / finiteness ------------------------------------------------------


def test_predict_shape_and_finite():
    X_past, Y = _temporal_windows()
    n_test = 50
    X_tr, X_te = X_past[:-n_test], X_past[-n_test:]
    Y_tr, _ = Y[:-n_test], Y[-n_test:]
    model = LSTMModel(**_TINY)
    model.fit(X_tr, Y_tr)
    pred = model.predict(X_te)
    assert pred.shape == (n_test, Y.shape[1])
    assert pred.dtype == np.float32
    assert np.all(np.isfinite(pred))


def test_xfuture_none_works():
    """X_future is accepted (ABC signature) but not required by the LSTM."""
    X_past, Y = _temporal_windows(n=200)
    model = LSTMModel(**_TINY)
    model.fit(X_past, Y, X_future=None)
    pred = model.predict(X_past, X_future=None)
    assert pred.shape == (200, Y.shape[1])


# -- learning (beats input-free baseline) ------------------------------------


def test_learns_temporal_pattern_beats_constant():
    """A trained LSTM must beat an input-free constant baseline on a learnable
    temporal pattern — i.e. it actually uses the sequence, not just the mean."""
    X_past, Y = _temporal_windows(n=400, lookback=8, horizon=4)
    n_test = 80
    X_tr, X_te = X_past[:-n_test], X_past[-n_test:]
    Y_tr, Y_te = Y[:-n_test], Y[-n_test:]
    # epochs bumped on a tiny net (still sub-second on CPU) so the pattern is
    # reliably learned; hidden kept small (TINY spirit).
    model = LSTMModel(hidden_size=16, num_layers=2, epochs=20, batch_size=32,
                      lr=1e-3, device="cpu", seed=0)
    model.fit(X_tr, Y_tr)
    pred = model.predict(X_te)
    baseline = _constant_baseline(Y_tr, Y_te)
    assert _mae(pred, Y_te) < _mae(baseline, Y_te)


# -- determinism -------------------------------------------------------------


def test_determinism_same_seed_same_output():
    """Two independent fits with the same seed produce identical predictions."""
    X_past, Y = _temporal_windows(n=200)
    m1 = LSTMModel(**_TINY, seed=0)
    m1.fit(X_past, Y)
    p1 = m1.predict(X_past)
    m2 = LSTMModel(**_TINY, seed=0)
    m2.fit(X_past, Y)
    p2 = m2.predict(X_past)
    assert np.array_equal(p1, p2)


# -- robustness --------------------------------------------------------------


def test_handles_nan_inputs():
    """NaN lookback features must not crash or emit non-finite predictions
    (features are sanitized to 0 — Honest NaN, no silent fabrication of Y)."""
    rng = np.random.default_rng(3)
    X_past, Y = _temporal_windows(n=200)
    X_past_nan = X_past.copy()
    mask = rng.random(X_past_nan.shape) < 0.1
    X_past_nan[mask] = np.nan
    model = LSTMModel(**_TINY)
    model.fit(X_past_nan, Y)
    pred = model.predict(X_past_nan)
    assert pred.shape == (200, Y.shape[1])
    assert np.all(np.isfinite(pred))


def test_predict_before_fit_raises():
    """Calling predict before fit is a programmer error, not a silent failure."""
    X_past, _ = _temporal_windows(n=10)
    with pytest.raises(RuntimeError):
        LSTMModel(**_TINY).predict(X_past)
