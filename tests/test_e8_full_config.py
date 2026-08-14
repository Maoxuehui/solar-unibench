"""E8 FULL-config upgrade tests — AdamW+Cosine, early stopping, val split,
gradient accumulation, CUDA fallback, and the FULL model presets.

These tests use a TINY-but-FULL-shaped config (tiny data, small epoch/patience
budgets) to verify the FULL-config *behavior* fast, without a GPU or a long run.
They exercise :func:`fit_loop` directly (for optimizer/scheduler/early-stop/
accumulation) and the runner helpers (for the CUDA fallback + presets).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
from torch import nn

from solarunibench.benchmark.run_e8 import (
    _build_full_models,
    _full_stride,
    build_models,
    run_with_cuda_fallback,
)
from solarunibench.forecast.models import CrossViViTModel, LSTMModel, PatchTSTModel
from solarunibench.forecast.models._torch_base import _eval_mae, fit_loop


# -- tiny helpers ------------------------------------------------------------


class _LinearNet(nn.Module):
    """Minimal ``(n, d) -> (n, h)`` net for direct fit_loop tests."""

    def __init__(self, d: int, h: int, hidden: int = 32, depth: int = 1):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(d, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, h)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _toy_tensors(n: int = 40, d: int = 4, h: int = 1, seed: int = 0):
    """Deterministic ``(Xt, Yt)`` on CPU for direct fit_loop tests."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    w = rng.standard_normal((d, h)).astype(np.float32)
    Y = X @ w + 0.05 * rng.standard_normal((n, h)).astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(Y)


# ---------------------------------------------------------------------------
# 1. Optimizer + LR scheduler
# ---------------------------------------------------------------------------


def test_fit_loop_uses_adamw_and_cosine_scheduler() -> None:
    """fit_loop reports AdamW + CosineAnnealingLR in a FULL-shaped config.

    The scheduler engages on the FULL signal (patience), so this uses a tiny
    patience to turn it on while staying fast.
    """
    Xt, Yt = _toy_tensors(n=24, seed=0)
    net = _LinearNet(d=4, h=1)
    diag = fit_loop(
        net,
        Xt,
        Yt,
        epochs=3,
        batch_size=8,
        lr=1e-3,
        loss_fn=nn.L1Loss(),
        weight_decay=1e-4,
        patience=2,           # FULL signal → CosineAnnealingLR engages
        val_fraction=0.2,
    )
    assert diag["optimizer"] == "AdamW"
    assert diag["scheduler"] == "CosineAnnealingLR"


def test_fit_loop_default_keeps_constant_lr() -> None:
    """Non-FULL callers (patience=None) keep a CONSTANT lr (old Adam dynamics)."""
    Xt, Yt = _toy_tensors(n=24, seed=0)
    net = _LinearNet(d=4, h=1)
    diag = fit_loop(
        net, Xt, Yt, epochs=2, batch_size=8, lr=1e-3, loss_fn=nn.L1Loss()
    )
    assert diag["optimizer"] == "AdamW"
    assert diag["scheduler"] == "none"


# ---------------------------------------------------------------------------
# 2. Early stopping triggers + restores best weights
# ---------------------------------------------------------------------------


def test_early_stopping_triggers_and_restores_best() -> None:
    """A high-capacity net overfitting a tiny train set halts early and the
    final net matches the best-val-MAE weights (best-weight restore)."""
    # Tiny train (16) + larger val (40): the net memorizes train noise, val MAE
    # rises after a few epochs → early stopping with patience=3.
    n_train, n_val, d, h = 16, 40, 8, 1
    rng = np.random.default_rng(7)
    Xt = torch.from_numpy(rng.standard_normal((n_train + n_val, d)).astype(np.float32))
    # Train targets carry noise the net will memorize; val targets share the
    # linear signal but a DIFFERENT noise draw → val worsens as train overfits.
    w = rng.standard_normal((d, h)).astype(np.float32)
    Yt = torch.from_numpy(
        (Xt.numpy() @ w + 0.3 * rng.standard_normal((n_train + n_val, h))).astype(np.float32)
    )

    net = _LinearNet(d=d, h=h, hidden=128, depth=3)
    diag = fit_loop(
        net,
        Xt,
        Yt,
        epochs=80,
        batch_size=16,
        lr=5e-3,
        loss_fn=nn.L1Loss(),
        weight_decay=0.0,
        patience=3,
        val_fraction=n_val / (n_train + n_val),
        seed=0,
    )
    assert diag["used_val"] is True
    assert diag["stopped_early"] is True, "val should have worsened → early stop"
    assert diag["epochs_run"] < diag["epochs_requested"]

    # Best-weight restore: the final net's val MAE must equal the recorded best
    # (if the LAST — worse — epoch's weights had been kept, this would diverge).
    val_start = diag["val_start"]
    val_tensors = (Xt[val_start:],)
    final_val_mae = _eval_mae(net, val_tensors, Yt[val_start:], micro_batch=16)
    assert abs(final_val_mae - diag["best_val_mae"]) < 1e-5


# ---------------------------------------------------------------------------
# 3. Validation split is chronological (the LAST val_fraction of time-sorted windows)
# ---------------------------------------------------------------------------


def test_val_split_is_chronological_last_portion() -> None:
    """The val holdout is the LAST ``val_fraction`` rows, not the first.

    Construct Yt with the last 2 rows = 100 and the first 8 = 0; a net trained
    only on the first 8 predicts ~0, so if val = LAST 2 the val MAE is ~100
    (large), and if val were the FIRST 2 it would be ~0. Large → chronological.
    """
    n, d = 10, 2
    Xt = torch.from_numpy(np.random.default_rng(1).standard_normal((n, d)).astype(np.float32))
    Yt = np.zeros((n, 1), dtype=np.float32)
    Yt[-2:] = 100.0  # the LAST 2 targets are huge
    Yt = torch.from_numpy(Yt)

    net = _LinearNet(d=d, h=1)
    diag = fit_loop(
        net,
        Xt,
        Yt,
        epochs=5,
        batch_size=4,
        lr=1e-2,
        loss_fn=nn.L1Loss(),
        patience=None,           # no early stop; we just want the val split
        val_fraction=0.2,
        seed=0,
    )
    # Structural: val is the last 2 of 10.
    assert diag["n_train"] == 8
    assert diag["n_val"] == 2
    assert diag["val_start"] == 8
    # Behavioral: val MAE reflects the huge LAST-2 targets (net predicts ~0),
    # proving val = the chronological LAST portion, not the first.
    assert diag["best_val_mae"] > 50.0


# ---------------------------------------------------------------------------
# 4. Gradient accumulation (CrossViViT multimodal tuple path)
# ---------------------------------------------------------------------------


def test_gradient_accumulation_multimodal_tuple() -> None:
    """accum_steps splits the effective batch into micro-batches (tuple input).

    Uses the CrossViViT multimodal shape (Xt, images) so the tuple indexing path
    inside fit_loop is exercised, and asserts the micro-batch size = batch // accum.
    """
    n, lb, fp, h = 12, 8, 2, 1
    rng = np.random.default_rng(3)
    Xt = torch.from_numpy(rng.standard_normal((n, lb, fp)).astype(np.float32))
    # Tiny image cube (n, lb, 4, 8, 8) — the cross-attention net accepts it.
    imgs = torch.from_numpy(rng.standard_normal((n, lb, 4, 8, 8)).astype(np.float32))
    Yt = torch.from_numpy(rng.standard_normal((n, h)).astype(np.float32))

    net = CrossViViTModel(
        d_model=16, n_heads=2, depth=1, patch_len=4, img_size=8, img_patch=4,
        epochs=1, batch_size=8, accum_steps=4, device="cpu",
    )
    # Build the underlying net manually (fit_loop needs an nn.Module + tensors).
    net._seed(0)
    net._net = __import__(
        "solarunibench.forecast.models.crossvivit_model", fromlist=["_CrossViViTNet"]
    )._CrossViViTNet(
        lookback=lb, n_channels=fp, horizon=h, patch_len=4, stride=4,
        img_size=8, img_patch=4, in_channels=4, d_model=16, n_heads=2, depth=1,
        dropout=0.0,
    ).to("cpu")

    diag = fit_loop(
        net._net,
        (Xt, imgs),          # multimodal tuple → cross-attention path
        Yt,
        epochs=1,
        batch_size=8,
        lr=1e-3,
        loss_fn=nn.L1Loss(),
        accum_steps=4,
    )
    assert diag["accum_steps"] == 4
    assert diag["micro_batch_size"] == 2  # 8 // 4


def test_crossvivit_model_accumulation_end_to_end() -> None:
    """The CrossViViT model's own fit() runs the accumulation path end-to-end."""
    n, lb, fp, h = 10, 8, 2, 3
    rng = np.random.default_rng(5)
    X = rng.standard_normal((n, lb, fp)).astype(np.float32)
    Y = rng.standard_normal((n, h)).astype(np.float32)
    imgs = rng.standard_normal((n, lb, 4, 8, 8)).astype(np.float32)

    model = CrossViViTModel(
        d_model=16, n_heads=2, depth=1, patch_len=4, img_size=8, img_patch=4,
        epochs=1, batch_size=8, accum_steps=4, patience=2, val_fraction=0.2,
        device="cpu",
    )
    model.fit(X, Y, images=imgs)  # should not raise
    pred = model.predict(X, images=imgs)
    assert pred.shape == (n, h)


# ---------------------------------------------------------------------------
# 5. CUDA auto-fallback
# ---------------------------------------------------------------------------


class _FakeModel:
    """Fake model: fn raises a CUDA error while device=='cuda', succeeds on CPU."""

    name = "FakeCUDA"

    def __init__(self):
        self.device = torch.device("cuda")
        self.fit_calls = 0

    def fit(self):
        self.fit_calls += 1
        if self.device.type == "cuda":
            raise RuntimeError("CUDA out of memory")
        return "ok-cpu"


def test_cuda_fallback_retries_on_cpu() -> None:
    """A CUDA RuntimeError triggers a single CPU retry (model.device flipped)."""
    model = _FakeModel()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = run_with_cuda_fallback(lambda m: m.fit(), model)
    assert result == "ok-cpu"
    assert model.device.type == "cpu"
    assert model.fit_calls == 2  # one cuda attempt + one cpu retry
    assert any(issubclass(x.category, RuntimeWarning) for x in w)


def test_cuda_fallback_reraises_non_cuda_errors() -> None:
    """A non-CUDA RuntimeError propagates unchanged (no silent retry)."""

    def boom(_m):
        raise RuntimeError("shape mismatch in linear")

    model = _FakeModel()
    with pytest.raises(RuntimeError, match="shape mismatch"):
        run_with_cuda_fallback(boom, model)
    assert model.device.type == "cuda"  # unchanged


# ---------------------------------------------------------------------------
# 6. FULL-config presets
# ---------------------------------------------------------------------------


def test_build_models_full_sets_full_presets() -> None:
    """build_models(full=True) wires every FULL preset into the deep models."""
    models = build_models(full=True, device="cpu")
    by_name = {m.name: m for m in models}

    lstm = by_name["LSTM"]
    assert lstm.epochs == 50
    assert lstm.patience == 7
    assert lstm.weight_decay == 1e-4
    assert lstm.val_fraction == 0.15
    assert lstm.batch_size == 64

    # CrossViViT accumulates 4 micro-batches of 16 (effective 64) for 24GB.
    cvv = by_name["CrossViViT"]
    assert cvv.batch_size == 64
    assert cvv.accum_steps == 4

    # Both PatchTST variants present and FULL-configured.
    assert isinstance(by_name["PatchTST-CI"], PatchTSTModel)
    assert by_name["PatchTST-CI"].epochs == 50
    assert by_name["PatchTST-CD"].variant == "patchtst_cd"


def test_full_stride_per_sampling_rate() -> None:
    """FULL-mode windowing stride: 1-min data → 30, 30-min/1-h → 1."""
    assert _full_stride(1) == 30
    assert _full_stride(30) == 1
    assert _full_stride(60) == 1


def test_build_full_models_returns_eight() -> None:
    """The FULL battery still carries 8 non-reference models."""
    assert len(_build_full_models(device="cpu")) == 8
