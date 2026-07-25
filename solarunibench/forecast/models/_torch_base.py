"""Shared torch helpers + a generic training loop for torch ForecastModels.

The LSTM and PatchTST baselines (and any future torch :class:`ForecastModel`)
duplicated three static helpers — ``_resolve_device`` / ``_seed`` / ``_sanitize``
— plus an identical shuffled-minibatch Adam loop. This module DRYs that up:

* :class:`TorchForecastMixin` — mix into a :class:`ForecastModel` that holds a
  torch ``nn.Module`` to inherit device resolution, deterministic seeding, and
  NaN/inf sanitization (Honest NaN imputation to 0).
* :func:`fit_loop` — the generic ``Adam`` + shuffled-minibatch training loop the
  deep baselines share; takes the net + already-on-device tensors + the four
  hyperparameters that differ per model.

Design notes
------------
* **Loss + optimizer are caller-supplied** (the loop is optimizer-agnostic in
  principle), but in practice every current caller uses ``Adam`` + MAE (L1),
  which is what makes a single shared loop correct. ``fit_loop`` builds the
  ``Adam`` optimizer internally so the caller does not repeat that line.
* **Tensors must already be on the right device.** The helpers do not move
  data; the caller moves once before the loop (and before ``predict``).
* **Determinism.** ``_seed`` forces cuDNN into deterministic mode and is meant
  to be called immediately before the network is constructed so init + training
  are reproducible from a single ``seed``.
"""
from __future__ import annotations

import random

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


def fit_loop(
    net: nn.Module,
    Xt: torch.Tensor | tuple[torch.Tensor, ...],
    Yt: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    loss_fn: nn.Module,
) -> None:
    """Generic shuffled-minibatch ``Adam`` training loop.

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
        Training hyperparameters (``batch_size`` larger than ``n`` = full-batch).
    loss_fn:
        Loss applied between ``net(...batched inputs...)`` and ``Yt[batch]``
        (MAE for the current baselines).
    """
    tensors = (Xt,) if isinstance(Xt, torch.Tensor) else tuple(Xt)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    n = tensors[0].shape[0]
    idx = np.arange(n)
    for _ in range(epochs):
        np.random.shuffle(idx)
        for start in range(0, n, batch_size):
            batch = idx[start : start + batch_size]
            optimizer.zero_grad()
            pred = net(*[t[batch] for t in tensors])
            loss = loss_fn(pred, Yt[batch])
            loss.backward()
            optimizer.step()
