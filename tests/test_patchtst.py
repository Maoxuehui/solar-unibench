"""Tests for the PatchTST-style Transformer baseline (PyTorch).

:class:`PatchTSTModel` subclasses :class:`ForecastModel` and patch-tokenizes the
past sequence block ``X_past[n, lb, Fp]`` along the LOOKBACK axis, per channel
(channel-independence, Nie et al. ICLR 2023 — "A Time Series is Worth 64 Words").
Patches are linearly projected (+positional), run through a
:class:`torch.nn.TransformerEncoder`, pooled, and a linear head emits the full
forecast horizon so ``predict`` returns ``(n, horizon)`` in one forward pass.

The fast suite uses a TINY config (``d_model=16``, ``depth=1``, ``patch_len=2``,
``epochs=2``, CPU) so it stays sub-second; the "learns a temporal pattern" test
bumps ``epochs`` on a small synthetic problem where a constant (input-free)
baseline is easily beaten. The ``variant="vanilla"`` flag (tokens = each
lookback step, no patching) is exercised as a drop-in TS-Transformer.
"""
import numpy as np
import pytest
import torch

from solarunibench.forecast.core import ForecastModel
from solarunibench.forecast.models.patchtst_model import PatchTSTModel

# TINY config: keeps the shape / finite / determinism tests sub-second on CPU.
_TINY: dict = {
    "d_model": 16,
    "n_heads": 4,
    "depth": 1,
    "patch_len": 2,
    "stride": 2,
    "epochs": 2,
    "batch_size": 32,
    "device": "cpu",
}


def _temporal_windows(n=400, lookback=8, horizon=4, seed=0):
    """Synthetic windows whose target is a learnable function of recent lags.

    ``Y[:, h] = mean(X_past[:, -3:, 0]) + 0.1*h + noise`` — the predictive signal
    is the mean of the last three target lags, which a Transformer over patches
    can extract but an input-free constant baseline cannot. Channel 0 is the
    target; channel 1 is an uninformative exog (ensures the model handles >1
    feature channel — the channel-independence path is exercised).
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


def test_patchtst_is_forecast_model():
    m = PatchTSTModel(**_TINY)
    assert isinstance(m, ForecastModel)
    assert m.name == "PatchTST"


def test_custom_name():
    assert PatchTSTModel(name="PatchTST-Custom", **_TINY).name == "PatchTST-Custom"


def test_default_config_values():
    """Defaults match the documented Phase 6b baseline + the spec config."""
    m = PatchTSTModel()
    assert m.patch_len == 4
    assert m.stride == 2
    assert m.d_model == 64
    assert m.n_heads == 4
    assert m.depth == 2
    assert m.dropout == 0.1
    assert m.lr == 1e-3
    assert m.epochs == 8
    assert m.batch_size == 32
    assert m.variant == "patchtst"


def test_default_variant_is_patchtst():
    """The default variant is patchtst (patch-tokenization along lookback)."""
    assert PatchTSTModel().variant == "patchtst"


# -- shape / finiteness ------------------------------------------------------


def test_predict_shape_and_finite():
    X_past, Y = _temporal_windows()
    n_test = 50
    X_tr, X_te = X_past[:-n_test], X_past[-n_test:]
    Y_tr, _ = Y[:-n_test], Y[-n_test:]
    model = PatchTSTModel(**_TINY)
    model.fit(X_tr, Y_tr)
    pred = model.predict(X_te)
    assert pred.shape == (n_test, Y.shape[1])
    assert pred.dtype == np.float32
    assert np.all(np.isfinite(pred))


def test_xfuture_none_works():
    """X_future is accepted (ABC signature) but not required by PatchTST."""
    X_past, Y = _temporal_windows(n=200)
    model = PatchTSTModel(**_TINY)
    model.fit(X_past, Y, X_future=None)
    pred = model.predict(X_past, X_future=None)
    assert pred.shape == (200, Y.shape[1])


# -- learning (beats input-free baseline) ------------------------------------


def test_learns_temporal_pattern_beats_constant():
    """A trained PatchTST must beat an input-free constant baseline on a
    learnable temporal pattern — i.e. it actually uses the patched lookback
    sequence, not just the mean."""
    X_past, Y = _temporal_windows(n=400, lookback=8, horizon=4)
    n_test = 80
    X_tr, X_te = X_past[:-n_test], X_past[-n_test:]
    Y_tr, Y_te = Y[:-n_test], Y[-n_test:]
    # epochs bumped on a tiny net (still sub-second on CPU) so the pattern is
    # reliably learned; d_model kept small (TINY spirit).
    model = PatchTSTModel(
        d_model=32, n_heads=4, depth=2, patch_len=2, stride=2,
        epochs=25, batch_size=32, lr=1e-3, device="cpu", seed=0,
    )
    model.fit(X_tr, Y_tr)
    pred = model.predict(X_te)
    baseline = _constant_baseline(Y_tr, Y_te)
    assert _mae(pred, Y_te) < _mae(baseline, Y_te)


# -- determinism -------------------------------------------------------------


def test_determinism_same_seed_same_output():
    """Two independent fits with the same seed produce identical predictions."""
    X_past, Y = _temporal_windows(n=200)
    m1 = PatchTSTModel(**_TINY, seed=0)
    m1.fit(X_past, Y)
    p1 = m1.predict(X_past)
    m2 = PatchTSTModel(**_TINY, seed=0)
    m2.fit(X_past, Y)
    p2 = m2.predict(X_past)
    assert np.array_equal(p1, p2)


# -- variant flag ------------------------------------------------------------


def test_vanilla_variant_constructs_and_predicts():
    """variant='vanilla' disables patching (tokens = each lookback step) and
    behaves as a drop-in vanilla TS-Transformer — shape + finite."""
    X_past, Y = _temporal_windows(n=200, lookback=8, horizon=4)
    model = PatchTSTModel(variant="vanilla", **_TINY)
    assert model.variant == "vanilla"
    model.fit(X_past, Y)
    pred = model.predict(X_past)
    assert pred.shape == (200, Y.shape[1])
    assert np.all(np.isfinite(pred))


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        PatchTSTModel(variant="bogus", **_TINY)


# -- robustness --------------------------------------------------------------


def test_handles_nan_inputs():
    """NaN lookback features must not crash or emit non-finite predictions
    (features are sanitized to 0 — Honest NaN, no silent fabrication of Y)."""
    rng = np.random.default_rng(3)
    X_past, Y = _temporal_windows(n=200)
    X_past_nan = X_past.copy()
    mask = rng.random(X_past_nan.shape) < 0.1
    X_past_nan[mask] = np.nan
    model = PatchTSTModel(**_TINY)
    model.fit(X_past_nan, Y)
    pred = model.predict(X_past_nan)
    assert pred.shape == (200, Y.shape[1])
    assert np.all(np.isfinite(pred))


def test_predict_before_fit_raises():
    """Calling predict before fit is a programmer error, not a silent failure."""
    X_past, _ = _temporal_windows(n=10)
    with pytest.raises(RuntimeError):
        PatchTSTModel(**_TINY).predict(X_past)


def test_patchtst_variant_lookback_too_short_raises():
    """If lookback < patch_len the patchtst variant cannot form even one patch —
    a clear ValueError beats a silent miscompile inside the encoder."""
    X_past, Y = _temporal_windows(n=40, lookback=3, horizon=2)
    with pytest.raises(ValueError):
        PatchTSTModel(patch_len=4, stride=2, **{k: v for k, v in _TINY.items()
                                                if k not in ("patch_len", "stride")}).fit(X_past, Y)


# -- channel-dependent (CD) variant ------------------------------------------


def test_patchtst_cd_variant_constructs_and_predicts():
    """variant='patchtst_cd' (channel-dependent: each patch spans ALL channels so
    cross-channel mixing happens in the projection) constructs, fits, and
    predicts the right shape — distinct from the CI default."""
    X_past, Y = _temporal_windows(n=200, lookback=8, horizon=4)
    model = PatchTSTModel(variant="patchtst_cd", **_TINY)
    assert model.variant == "patchtst_cd"
    model.fit(X_past, Y)
    pred = model.predict(X_past)
    assert pred.shape == (200, Y.shape[1])
    assert pred.dtype == np.float32
    assert np.all(np.isfinite(pred))


def test_cd_variant_distinct_from_ci():
    """The CD variant (cross-channel mixing in the projection) must produce
    DIFFERENT predictions than the CI default on the same data+seed — otherwise
    the two code paths are not actually distinct."""
    X_past, Y = _temporal_windows(n=120, lookback=8, horizon=4)
    ci = PatchTSTModel(variant="patchtst", **_TINY, seed=0)
    ci.fit(X_past, Y)
    p_ci = ci.predict(X_past)
    cd = PatchTSTModel(variant="patchtst_cd", **_TINY, seed=0)
    cd.fit(X_past, Y)
    p_cd = cd.predict(X_past)
    assert p_ci.shape == p_cd.shape == (120, Y.shape[1])
    assert not np.allclose(p_ci, p_cd), "CD variant must differ from CI variant"


def test_cd_variant_learns_temporal_pattern():
    """The CD variant also learns the synthetic temporal pattern (beats an
    input-free constant baseline) — confirms the CD path is a competent
    forecaster, not just shape-correct."""
    X_past, Y = _temporal_windows(n=400, lookback=8, horizon=4)
    n_test = 80
    X_tr, X_te = X_past[:-n_test], X_past[-n_test:]
    Y_tr, Y_te = Y[:-n_test], Y[-n_test:]
    model = PatchTSTModel(
        variant="patchtst_cd", d_model=32, n_heads=4, depth=2, patch_len=2,
        stride=2, epochs=25, batch_size=32, lr=1e-3, device="cpu", seed=0,
    )
    model.fit(X_tr, Y_tr)
    pred = model.predict(X_te)
    baseline = _constant_baseline(Y_tr, Y_te)
    assert _mae(pred, Y_te) < _mae(baseline, Y_te)


def test_cd_variant_lookback_too_short_raises():
    """The CD variant patches along the lookback too, so lookback < patch_len is
    a clear ValueError (mirrors the CI patchtst guard)."""
    X_past, Y = _temporal_windows(n=40, lookback=3, horizon=2)
    with pytest.raises(ValueError):
        PatchTSTModel(
            variant="patchtst_cd", patch_len=4, stride=2,
            **{k: v for k, v in _TINY.items() if k not in ("patch_len", "stride")},
        ).fit(X_past, Y)
