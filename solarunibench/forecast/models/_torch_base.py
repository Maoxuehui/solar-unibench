"""Shared torch helpers + a generic training loop for torch ForecastModels.

The LSTM and PatchTST baselines (and any future torch :class:`ForecastModel`)
duplicated three static helpers — ``_resolve_device`` / ``_seed`` / ``_sanitize``
— plus an identical shuffled-minibatch optimizer loop. This module DRYs that up:

* :class:`TorchForecastMixin` — mix into a :class:`ForecastModel` that holds a
  torch ``nn.Module`` to inherit device resolution, deterministic seeding, and
  NaN/inf sanitization (Honest NaN imputation to 0).
* :func:`fit_loop` — the generic ``AdamW`` + ``CosineAnnealingLR`` +
  shuffled-minibatch training loop the deep baselines share; takes the net +
  already-on-device tensors + the hyperparameters that differ per model. Supports
  chronological validation-based early stopping (with best-weight restore) and
  gradient accumulation for VRAM-tight full-config runs.
* :func:`train_kwargs` — collect the fit-loop hyperparameters from a model
  instance (with FULL-config defaults), so each model's ``fit`` call site is a
  single ``fit_loop(net, Xt, Yt, loss_fn=loss_fn, **train_kwargs(self))`` line.

Design notes
------------
* **Optimizer + scheduler.`` ``fit_loop`` always builds ``AdamW`` +
  ``CosineAnnealingLR(T_max=epochs)`` internally. With ``weight_decay=0`` AdamW
  reduces to the old plain-Adam update, so tiny/1-epoch callers are unaffected;
  the FULL config sets ``weight_decay=1e-4`` for real regularization.
* **Validation-based early stopping (opt-in).`` Enabled when ``patience is not
  None`` (or ``val_fraction > 0``): the LAST ``val_fraction`` of the training
  windows — already time-sorted by the feature engineer — is held out as a
  chronological validation set, val MAE is monitored each epoch, training halts
  after ``patience`` non-improving epochs, and the best-weight ``state_dict`` is
  restored at the end (so an overfit late epoch can never win).
* **Gradient accumulation.`` ``accum_steps > 1`` splits the effective
  ``batch_size`` into ``micro_batch = batch_size // accum_steps`` micro-batches
  whose gradients are summed (loss scaled by ``1/accum_steps``) before each
  ``optimizer.step()``. Used by the CrossViViT image path (256x256, lookback
  240) to keep effective batch 64 while fitting a 24GB GPU at micro-batch 16.
* **Tensors must already be on the right device.** The helpers do not move
  data; the caller moves once before the loop (and before ``predict``).
* **Determinism.** ``_seed`` forces cuDNN into deterministic mode and is meant
  to be called immediately before the network is constructed so init + training
  are reproducible from a single ``seed``.
"""
from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np
import torch
from torch import nn


class TorchForecastMixin:
    """Shared torch helpers for :class:`ForecastModel` implementations.

    A subclass is expected to set ``self.device`` (a :class:`torch.device`) in
    its constructor (typically via ``self._resolve_device(device)``). The mixin
    only contributes static helpers; it owns no state of its own.
    """

    # Declared for type checkers; concrete subclasses assign in ``__init__``.
    device: torch.device

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        """Resolve a device spec; ``None`` auto-selects CUDA when available."""
        if device is not None:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _seed(seed: int) -> None:
        """Seed Python/NumPy/Torch and force cuDNN into deterministic mode."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def _sanitize(t: torch.Tensor) -> torch.Tensor:
        """Replace NaN/inf lookback features with 0 (Honest NaN imputation)."""
        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def train_kwargs(obj: Any) -> dict:
    """Collect :func:`fit_loop` training kwargs from a model instance.

    Reads the standard hyperparameters every torch ForecastModel carries
    (``epochs`` / ``batch_size`` / ``lr``) plus the FULL-config additions
    (``weight_decay`` / ``patience`` / ``val_fraction`` / ``accum_steps``) with
    safe defaults, so the call site is a one-liner and models that pre-date the
    FULL config (or are constructed without the new kwargs) keep working.

    ``loss_fn`` is NOT included — it is caller-supplied (each model builds its
    own ``nn.L1Loss()``).
    """
    return {
        "epochs": getattr(obj, "epochs", 8),
        "batch_size": getattr(obj, "batch_size", 32),
        "lr": getattr(obj, "lr", 1e-3),
        "weight_decay": getattr(obj, "weight_decay", 0.0),
        "patience": getattr(obj, "patience", None),
        "val_fraction": getattr(obj, "val_fraction", 0.0),
        "accum_steps": getattr(obj, "accum_steps", 1),
    }


def _eval_mae(
    net: nn.Module,
    tensors: tuple[torch.Tensor, ...],
    Y: torch.Tensor,
    micro_batch: int,
) -> float:
    """Mean absolute error of ``net`` over ``(tensors, Y)`` in eval mode."""
    net.eval()
    n = tensors[0].shape[0]
    if n == 0:
        net.train()
        return float("inf")
    abs_sum = 0.0
    with torch.no_grad():
        for start in range(0, n, micro_batch):
            sl = slice(start, start + micro_batch)
            pred = net(*[t[sl] for t in tensors])
            abs_sum += torch.abs(pred - Y[sl]).sum().item()
    net.train()
    return abs_sum / max(1, Y.numel())


def fit_loop(
    net: nn.Module,
    Xt: torch.Tensor | tuple[torch.Tensor, ...],
    Yt: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    loss_fn: nn.Module,
    weight_decay: float = 0.0,
    patience: int | None = None,
    val_fraction: float = 0.0,
    accum_steps: int = 1,
    min_delta: float = 0.0,
    seed: int | None = None,
) -> dict:
    """Generic shuffled-minibatch ``AdamW`` + ``CosineAnnealingLR`` training loop.

    Parameters
    ----------
    net:
        The :class:`torch.nn.Module` to train (already moved to the device). Its
        ``forward`` is called as ``net(*batched_inputs)``.
    Xt:
        Either a single input tensor, or a tuple of input tensors that all share
        dim 0 (the sample axis). A tuple lets a multimodal net (e.g. CrossViViT)
        receive parallel modalities — each is indexed by the same minibatch row
        slice and passed positionally to ``net.forward``. All tensors must be on
        the device.
    Yt:
        Target tensor, already on the device, aligned on dim 0.
    epochs, batch_size, lr:
        Training hyperparameters. ``batch_size`` is the EFFECTIVE batch size;
        with ``accum_steps > 1`` it is split into ``batch_size // accum_steps``
        micro-batches accumulated before each optimizer step.
    loss_fn:
        Loss applied between ``net(...batched inputs...)`` and ``Yt[batch]``
        (MAE / L1 for the current baselines). It is scaled by ``1/accum_steps``
        for correct gradient accumulation.
    weight_decay:
        AdamW weight decay (FULL config: ``1e-4``; ``0.0`` → Adam-equivalent).
    patience:
        Early-stopping patience in epochs. ``None`` disables early stopping.
        When non-None, a chronological validation split is required (see
        ``val_fraction``) and the best-val-MAE weights are restored at the end.
    val_fraction:
        Fraction of training windows held out as a chronological validation set
        (the LAST ``val_fraction`` rows — windows are already time-sorted). When
        ``patience is not None`` but ``val_fraction <= 0``, defaults to ``0.15``.
        ``0.0`` + ``patience is None`` → train on all data (tiny-mode default).
    accum_steps:
        Number of gradient-accumulation micro-batches per effective batch.
        ``1`` → plain minibatching (no accumulation).
    min_delta:
        Minimum val-MAE improvement (absolute) to reset the patience counter.
    seed:
        Optional NumPy seed for the epoch-shuffle RNG (for direct-loop tests);
        ``None`` preserves the caller's RNG state (models already seed via
        :meth:`TorchForecastMixin._seed`).

    Returns
    -------
    dict
        Diagnostics: ``optimizer``/``scheduler`` names, ``epochs_run``,
        ``stopped_early``, ``best_val_mae`` (``None`` if no val), ``val_start``,
        ``n_train``, ``n_val``, ``micro_batch_size``, ``accum_steps``,
        ``used_val`` — enabling tests to assert the FULL-config behavior without
        poking at loop internals.
    """
    tensors = (Xt,) if isinstance(Xt, torch.Tensor) else tuple(Xt)
    if seed is not None:
        np.random.seed(seed)

    n = tensors[0].shape[0]
    epochs = max(1, int(epochs))
    accum_steps = max(1, int(accum_steps))
    # Effective batch_size split into accumulated micro-batches. max(1, ...) so
    # a tiny batch_size // accum_steps never collapses to 0.
    micro_batch = max(1, int(batch_size) // accum_steps)

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    # CosineAnnealingLR is a FULL-config feature: engage it alongside early
    # stopping (patience). Default/tiny callers (patience=None) keep a CONSTANT
    # learning rate — the old Adam training dynamics existing tests are tuned to.
    # AdamW with weight_decay=0 + constant LR is numerically identical to Adam,
    # so non-FULL callers are unaffected by the optimizer switch.
    use_scheduler = patience is not None
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if use_scheduler
        else None
    )

    # Chronological validation split (the feature engineer emits time-sorted
    # windows, so the LAST val_fraction rows are the most-recent holdout).
    used_val = (patience is not None) or val_fraction > 0
    if used_val:
        if patience is None:
            # No early stopping, but track best so a late overfit epoch can't win.
            patience = epochs
        if val_fraction <= 0:
            val_fraction = 0.15
        n_val = max(1, int(round(n * val_fraction)))
        if n_val >= n:  # never hold out all samples; fall back to 20%.
            n_val = max(1, n // 5)
        val_start = n - n_val
    else:
        n_val = 0
        val_start = n

    train_tensors = [t[:val_start] for t in tensors]
    train_Y = Yt[:val_start]
    val_tensors = [t[val_start:] for t in tensors]
    val_Y = Yt[val_start:]
    train_n = val_start
    idx = np.arange(train_n)

    best_val = float("inf")
    best_state: dict | None = None
    bad_epochs = 0
    stopped_early = False
    epochs_run = 0

    net.train()
    for epoch in range(epochs):
        epochs_run = epoch + 1
        np.random.shuffle(idx)
        optimizer.zero_grad()
        accum_count = 0
        for start in range(0, train_n, micro_batch):
            batch = idx[start : start + micro_batch]
            pred = net(*[t[batch] for t in train_tensors])
            # Scale by 1/accum_steps so the summed gradient matches a full
            # effective-batch update (standard gradient accumulation).
            loss = loss_fn(pred, train_Y[batch]) / accum_steps
            loss.backward()
            accum_count += 1
            if accum_count >= accum_steps:
                optimizer.step()
                optimizer.zero_grad()
                accum_count = 0
        if accum_count > 0:  # flush the trailing partial effective batch.
            optimizer.step()
            optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        if used_val:
            val_mae = _eval_mae(net, tuple(val_tensors), val_Y, micro_batch)
            if val_mae < best_val - min_delta:
                best_val = val_mae
                best_state = copy.deepcopy(net.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    stopped_early = True
                    break

    # Restore the best-validation weights so a late overfit epoch cannot win.
    if best_state is not None:
        net.load_state_dict(best_state)

    return {
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR" if use_scheduler else "none",
        "epochs_run": epochs_run,
        "epochs_requested": epochs,
        "stopped_early": stopped_early,
        "best_val_mae": best_val if used_val else None,
        "val_start": val_start,
        "n_train": val_start,
        "n_val": n_val,
        "micro_batch_size": micro_batch,
        "accum_steps": accum_steps,
        "used_val": used_val,
    }
