"""Tests for the CrossViViT multimodal baseline (PyTorch).

:class:`CrossViViTModel` (Boussif et al., NeurIPS 2023 — adapted) subclasses
:class:`ForecastModel` and FUSES two modalities:

* **Tabular branch** — patch-tokenizes ``X_past[n, lb, Fp]`` along the lookback
  axis (PatchTST-style) → ``d_model`` tabular tokens.
* **Image branch** — the per-window sky-image cube ``images[n, lb, 4, H, W]`` is
  resized to ``img_size`` and conv-patch-embedded → ``d_model`` image tokens.
* **Bidirectional cross-attention** — tabular tokens attend to image tokens AND
  vice-versa; the pooled fused representation feeds a linear horizon head.

When ``images`` is ``None`` the image branch degrades to the tabular tokens
(cross-attention collapses to self-attention), so the model still runs on
purely tabular datasets (PV / GEFCom) — graceful fallback.

This is the SMALL runnable variant (``d_model=64``, depth=2, img_size=32). The
full 145M CrossViViT (large RoPE) is a Phase-8 training-budget decision (spec
§15); the fast suite uses a TINY config (``d_model=16``, depth=1, img_size=8,
``epochs=2``) on synthetic data.
"""
import numpy as np
import pytest
import torch

from solarunibench.forecast.core import ForecastModel
from solarunibench.forecast.models.crossvivit_model import CrossViViTModel

# TINY config: keeps the shape / finite / determinism tests sub-second on CPU.
_TINY: dict = {
    "d_model": 16,
    "n_heads": 4,
    "depth": 1,
    "patch_len": 2,
    "img_size": 8,
    "img_patch": 4,
    "epochs": 3,
    "batch_size": 32,
    "device": "cpu",
}


def _windows_with_images(n=300, lb=4, Fp=2, horizon=3, img_hw=16, seed=0):
    """Synthetic windows whose TARGET IS DRIVEN BY THE IMAGE CONTENT.

    Each sample carries a scalar ``s`` (uniform[0,1]) encoded as the value of
    every sky-image pixel (all RGB channels); the tabular block ``X_past`` is
    INDEPENDENT random noise (no signal for ``Y``). The target is a linear
    function of ``s``::

        Y[:, h] = 2.0 * s + 0.2 * h + tiny noise

    So a model that reads the images can nail ``Y``; a tabular-only model
    (images=None) cannot beat the per-horizon mean. ``img_hw`` defaults larger
    than ``img_size`` so the resize path is exercised inside the model.
    """
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.0, 1.0, size=(n,))  # the image-encoded signal
    X_past = rng.uniform(0.0, 1.0, size=(n, lb, Fp)).astype(np.float32)  # uninformative
    # images: constant per sample = s on RGB channels, 0 on BRR. (n, lb, 4, H, W)
    base = np.zeros((n, lb, 4, img_hw, img_hw), dtype=np.float32)
    base[:, :, :3] = s[:, None, None, None, None]
    images = base
    H = np.arange(horizon, dtype=np.float32)
    Y = (2.0 * s[:, None] + 0.2 * H[None, :] + rng.normal(scale=0.01, size=(n, horizon))).astype(np.float32)
    return X_past, Y, images


def _mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def _constant_baseline(Y_tr: np.ndarray, Y_te: np.ndarray) -> np.ndarray:
    """Input-free baseline: broadcast the per-horizon training mean."""
    return np.repeat(Y_tr.mean(axis=0)[None, :], Y_te.shape[0], axis=0)


# -- ABC / identity ----------------------------------------------------------


def test_crossvivit_is_forecast_model():
    m = CrossViViTModel(**_TINY)
    assert isinstance(m, ForecastModel)
    assert m.name == "CrossViViT"


def test_custom_name():
    assert CrossViViTModel(name="CrossViViT-Custom", **_TINY).name == "CrossViViT-Custom"


def test_default_config_values():
    """Defaults match the documented Phase 6b small-variant baseline."""
    m = CrossViViTModel()
    assert m.d_model == 64
    assert m.n_heads == 4
    assert m.depth == 2
    assert m.patch_len == 4
    assert m.img_size == 32
    assert m.epochs == 8
    assert m.batch_size == 16
    assert m.variant == "crossvivit"


# -- shape / finiteness (with images) ---------------------------------------


def test_predict_shape_and_finite_with_images():
    X_past, Y, images = _windows_with_images()
    n_test = 40
    X_tr, X_te = X_past[:-n_test], X_past[-n_test:]
    Y_tr, _ = Y[:-n_test], Y[-n_test:]
    img_tr, img_te = images[:-n_test], images[-n_test:]
    model = CrossViViTModel(**_TINY)
    model.fit(X_tr, Y_tr, images=img_tr)
    pred = model.predict(X_te, images=img_te)
    assert pred.shape == (n_test, Y.shape[1])
    assert pred.dtype == np.float32
    assert np.all(np.isfinite(pred))


# -- graceful fallback (images=None) ----------------------------------------


def test_images_none_fallback_shape_and_finite():
    """images=None must still produce (n, horizon) finite output (tabular path)."""
    X_past, Y, _ = _windows_with_images()
    model = CrossViViTModel(**_TINY)
    model.fit(X_past, Y, images=None)
    pred = model.predict(X_past, images=None)
    assert pred.shape == X_past.shape[:1] + (Y.shape[1],)
    assert np.all(np.isfinite(pred))


def test_fit_without_images_predict_with_images_still_finite():
    """A model fit tabular-only must not crash if images are later supplied
    (it ignores the untrained image branch — finite output, graceful)."""
    X_past, Y, images = _windows_with_images(n=80)
    model = CrossViViTModel(**_TINY)
    model.fit(X_past, Y, images=None)
    pred = model.predict(X_past, images=images)
    assert pred.shape == (80, Y.shape[1])
    assert np.all(np.isfinite(pred))


# -- learning: image branch carries signal ----------------------------------


def test_image_branch_beats_tabular_only():
    """The full (tabular+image) model must beat a tabular-only (images=None)
    fit on a problem whose target is driven by image content — i.e. the cross-
    attention image branch actually carries signal an input-free / tabular-only
    model cannot extract."""
    X_past, Y, images = _windows_with_images(n=500, lb=4, horizon=3, img_hw=16)
    n_test = 100
    X_tr, X_te = X_past[:-n_test], X_past[-n_test:]
    Y_tr, Y_te = Y[:-n_test], Y[-n_test:]
    img_tr, img_te = images[:-n_test], images[-n_test:]

    # Full multimodal model — bigger/tiny-tuned so it reliably learns the
    # image-encoded scalar in a few epochs (still sub-second on CPU).
    full = CrossViViTModel(
        d_model=32, n_heads=4, depth=1, patch_len=2, img_size=8, img_patch=4,
        epochs=40, batch_size=64, lr=1e-2, device="cpu", seed=0,
    )
    full.fit(X_tr, Y_tr, images=img_tr)
    full_pred = full.predict(X_te, images=img_te)

    # Tabular-only ablation (same architecture, images=None).
    tab = CrossViViTModel(
        d_model=32, n_heads=4, depth=1, patch_len=2, img_size=8, img_patch=4,
        epochs=40, batch_size=64, lr=1e-2, device="cpu", seed=0,
    )
    tab.fit(X_tr, Y_tr, images=None)
    tab_pred = tab.predict(X_te, images=None)

    baseline = _constant_baseline(Y_tr, Y_te)
    full_mae = _mae(full_pred, Y_te)
    tab_mae = _mae(tab_pred, Y_te)
    base_mae = _mae(baseline, Y_te)

    # The image-branch model beats BOTH the tabular-only ablation and the
    # input-free constant baseline — the images carry signal X_past cannot.
    assert full_mae < tab_mae
    assert full_mae < base_mae
    # And it learned the target well (Y ≈ 2s + 0.2h, s∈[0,1] → range ≈ [0, 2.4]).
    assert full_mae < 0.25


# -- determinism ------------------------------------------------------------


def test_determinism_same_seed_same_output():
    """Two independent fits with the same seed produce identical predictions."""
    X_past, Y, images = _windows_with_images(n=120)
    m1 = CrossViViTModel(**_TINY, seed=0)
    m1.fit(X_past, Y, images=images)
    p1 = m1.predict(X_past, images=images)
    m2 = CrossViViTModel(**_TINY, seed=0)
    m2.fit(X_past, Y, images=images)
    p2 = m2.predict(X_past, images=images)
    assert np.array_equal(p1, p2)


# -- robustness -------------------------------------------------------------


def test_handles_nan_tabular():
    """NaN lookback features must not crash or emit non-finite predictions."""
    rng = np.random.default_rng(3)
    X_past, Y, images = _windows_with_images(n=120)
    X_past_nan = X_past.copy()
    mask = rng.random(X_past_nan.shape) < 0.1
    X_past_nan[mask] = np.nan
    model = CrossViViTModel(**_TINY)
    model.fit(X_past_nan, Y, images=images)
    pred = model.predict(X_past_nan, images=images)
    assert pred.shape == (120, Y.shape[1])
    assert np.all(np.isfinite(pred))


def test_predict_before_fit_raises():
    X_past, _, images = _windows_with_images(n=16)
    with pytest.raises(RuntimeError):
        CrossViViTModel(**_TINY).predict(X_past, images=images)
