"""PatchTST-style Transformer baseline — channel-independent patch forecaster.

A :class:`torch.nn.TransformerEncoder` reads the past feature block
``X_past[n, lb, Fp]`` AFTER patch-tokenization along the LOOKBACK axis, per
channel (channel-independence, Nie et al. ICLR 2023 — "A Time Series is Worth 64
Words"). Each channel's lookback sequence is unfolded into non-overlapping /
strided patches, linearly projected to ``d_model`` (+positional), run through a
``TransformerEncoder``, pooled, and a linear head maps the pooled embedding to
the full forecast horizon, so ``predict`` returns ``(n, horizon)`` in one pass.

Design notes
------------
* **Channel-independence (PatchTST key idea).** The ``(n, Fp)`` (sample, channel)
  pairs are merged into the batch dimension and all run through the SAME
  transformer with shared weights; channels are then averaged before the head.
  This is the PatchTST pattern (one model per variate, weights shared) and lets
  the net work for any ``Fp`` without a channel-mixing input projection.
* **Patching along lookback.** ``variant="patchtst"`` (channel-independent, the
  PatchTST default) unfolds each channel's ``lookback`` sequence into
  ``n_patches = (lookback - patch_len)//stride + 1`` tokens of length
  ``patch_len``; ``variant="patchtst_cd"`` (channel-dependent) patches across
  ALL channels together (each patch has width ``patch_len*Fp`` so the projection
  mixes variates — the natural antagonist to channel-independence, useful when
  cross-channel interactions carry signal); ``variant="vanilla"`` disables
  patching so each lookback step is its own token (a drop-in vanilla
  TS-Transformer per the spec's "swap to vanilla" flag).
* **Scaled features in, no re-scaling inside.** The pipeline's
  :class:`PhysicalFeatureScaler` (default-on) already applies irradiance/1500
  and train-fit NWP z-scoring BEFORE feature engineering, so the model receives
  scaled features and must not re-scale (would double-normalize).
* **Determinism.** ``fit`` seeds Python/NumPy/Torch (and cuDNN) from a fixed
  ``seed`` (default 0) right before building the network, so two fits on the
  same data with the same seed produce byte-identical predictions.
* **Honest NaN.** Missing lookback features are sanitized to 0 via
  :func:`torch.nan_to_num` so a stray NaN in the NWP block cannot poison
  training or emit non-finite forecasts; the target ``Y`` is never fabricated.
* **Device-aware.** ``device=None`` auto-selects CUDA when available and falls
  back to CPU otherwise; pass ``device="cpu"`` for deterministic tiny tests.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..core import ForecastModel
from ._torch_base import TorchForecastMixin, fit_loop

# Valid tokenization variants. ``patchtst`` patches along lookback (channel-
# independent); ``patchtst_cd`` is the channel-dependent variant (each patch
# spans ALL channels, so cross-channel mixing happens in the projection);
# ``vanilla`` treats each lookback step as a token (no patching — a plain
# TS-Transformer).
_VARIANTS: tuple[str, ...] = ("patchtst", "patchtst_cd", "vanilla")


class _PatchTSTNet(nn.Module):
    """Channel-independent PatchTST transformer + linear horizon head.

    Each ``(sample, channel)`` pair is a separate sequence in the batch
    (channel-independence, shared encoder weights). Patches (or single
    timesteps, for ``variant="vanilla"``) are projected to ``d_model``, summed
    with a learned positional embedding, encoded, mean-pooled over tokens, then
    averaged across channels, and a linear head emits the horizon vector.
    """

    def __init__(
        self,
        lookback: int,
        n_channels: int,
        horizon: int,
        patch_len: int,
        stride: int,
        d_model: int,
        n_heads: int,
        depth: int,
        dropout: float,
        variant: str,
    ):
        super().__init__()
        self.variant = variant
        self.patch_len = patch_len
        self.stride = stride

        if variant in ("patchtst", "patchtst_cd"):
            if lookback < patch_len:
                raise ValueError(
                    f"{variant} variant requires lookback >= patch_len "
                    f"(got lookback={lookback}, patch_len={patch_len}); "
                    f"use a smaller patch_len or variant='vanilla'."
                )
            # unfold over lookback: number of strided windows of width patch_len.
            self.n_tokens = (lookback - patch_len) // stride + 1
            # Channel-dependent (CD): each patch spans ALL channels (width
            # patch_len*Fp) so the projection mixes variates. Channel-independent
            # (CI, the PatchTST default): one patch per channel (width patch_len).
            in_dim = patch_len * n_channels if variant == "patchtst_cd" else patch_len
            self.proj = nn.Linear(in_dim, d_model)
        else:  # "vanilla" — one token per lookback step (token width 1).
            self.n_tokens = lookback
            self.proj = nn.Linear(1, d_model)

        # Learned positional embedding (one row per token). ``n_tokens`` is fixed
        # once the lookback is known at fit time, so this is constructed there.
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n, lookback, Fp)
        n, _, fp = x.shape
        if self.variant == "patchtst_cd":
            # Channel-dependent: each patch spans ALL channels (cross-channel
            # mixing happens in the projection). x.unfold(1, patch_len, stride)
            # -> (n, n_tokens, Fp, patch_len) -> (n, n_tokens, patch_len*Fp).
            tokens = x.unfold(1, self.patch_len, self.stride).permute(0, 1, 3, 2)
            tokens = tokens.reshape(n, self.n_tokens, self.patch_len * fp)
            h = self.proj(tokens) + self.pos           # (n, n_tokens, d_model)
            out = self.encoder(h).mean(dim=1)            # (n, d_model)
            return self.head(out)                        # (n, horizon)
        # Channel-independent (patchtst / vanilla): merge (n, Fp) into batch.
        xc = x.permute(0, 2, 1).reshape(n * fp, -1)  # (n*Fp, lookback)
        if self.variant == "patchtst":
            # (n*Fp, n_tokens, patch_len) — unfold along the lookback dim.
            tokens = xc.unfold(1, self.patch_len, self.stride)
        else:
            tokens = xc.unsqueeze(-1)  # (n*Fp, lookback, 1)
        h = self.proj(tokens)  # (n*Fp, n_tokens, d_model)
        h = h + self.pos
        out = self.encoder(h)  # (n*Fp, n_tokens, d_model)
        # Mean-pool over tokens, then average channels back into the sample axis.
        pooled = out.mean(dim=1)  # (n*Fp, d_model)
        pooled = pooled.reshape(n, fp, -1).mean(dim=1)  # (n, d_model)
        return self.head(pooled)  # (n, horizon)


class PatchTSTModel(TorchForecastMixin, ForecastModel):
    """PatchTST-style channel-independent Transformer forecaster.

    Parameters
    ----------
    d_model:
        Transformer hidden width (must be divisible by ``n_heads``).
    n_heads:
        Number of self-attention heads.
    patch_len:
        Patch length (tokens per patch) for ``variant="patchtst"``. Ignored by
        ``variant="vanilla"``.
    stride:
        Stride between patches along the lookback axis.
    depth:
        Number of :class:`torch.nn.TransformerEncoderLayer` blocks.
    dropout:
        Dropout inside the encoder layers (attention + FFN).
    lr:
        Adam learning rate.
    epochs:
        Number of training epochs over the fit window set.
    batch_size:
        Minibatch size (full-batch when larger than ``n``).
    device:
        ``"cuda"`` / ``"cpu"`` / a :class:`torch.device`; ``None`` auto-selects
        CUDA when available else CPU.
    name:
        Display name returned by the :attr:`name` property.
    variant:
        ``"patchtst"`` (default; channel-independent, patches the lookback per
        channel), ``"patchtst_cd"`` (channel-dependent — each patch spans ALL
        channels so cross-channel mixing happens in the projection; better when
        variate interactions matter, e.g. high-frequency cloud motion), or
        ``"vanilla"`` (each lookback step is a token — a plain TS-Transformer).
    seed:
        Seed for Python/NumPy/Torch RNG (and cuDNN determinism) applied at the
        start of :meth:`fit`; guarantees reproducible fits.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        patch_len: int = 4,
        stride: int = 2,
        depth: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 8,
        batch_size: int = 32,
        device: str | torch.device | None = None,
        name: str = "PatchTST",
        variant: str = "patchtst",
        seed: int = 0,
    ):
        if variant not in _VARIANTS:
            raise ValueError(
                f"unknown PatchTST variant '{variant}' (known: {list(_VARIANTS)})"
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.patch_len = patch_len
        self.stride = stride
        self.depth = depth
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = self._resolve_device(device)
        self._name = name
        self.variant = variant
        self.seed = seed
        self._net: _PatchTSTNet | None = None

    @property
    def name(self) -> str:
        return self._name

    # -- ForecastModel API ---------------------------------------------------

    def fit(
        self,
        X_past: np.ndarray,
        Y: np.ndarray,
        times: np.ndarray | None = None,
        X_future: np.ndarray | None = None,
        *,
        images: np.ndarray | None = None,
    ) -> None:
        """Train the PatchTST encoder + linear head with Adam + MAE (L1) loss.

        ``times`` / ``X_future`` / ``images`` are accepted only to honor the
        :class:`ForecastModel` signature; the transformer consumes the
        ``X_past`` sequence alone.
        """
        X = np.asarray(X_past, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(
                f"X_past must be 3-D (n, lookback, Fp); got shape {X.shape}"
            )
        Y = np.asarray(Y, dtype=np.float32)
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2-D (n, horizon); got shape {Y.shape}")
        n, lookback, fp = X.shape
        horizon = Y.shape[1]

        # Lookback-too-short is a configuration error for both patching variants
        # (patchtst + patchtst_cd); raise early with a clear message instead of
        # miscompiling the encoder.
        if self.variant in ("patchtst", "patchtst_cd") and lookback < self.patch_len:
            raise ValueError(
                f"{self.variant} variant requires lookback >= patch_len "
                f"(got lookback={lookback}, patch_len={self.patch_len}); "
                f"use a smaller patch_len or variant='vanilla'."
            )

        # Seed immediately before model construction so init + training are
        # reproducible (two fits, same seed -> identical predictions).
        self._seed(self.seed)
        self._net = _PatchTSTNet(
            lookback=lookback,
            n_channels=fp,
            horizon=horizon,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            n_heads=self.n_heads,
            depth=self.depth,
            dropout=self.dropout,
            variant=self.variant,
        ).to(self.device)
        loss_fn = nn.L1Loss()  # MAE — robust, matches the Phase 6b spec.

        Xt = self._sanitize(torch.from_numpy(X)).to(self.device)
        Yt = torch.from_numpy(Y).to(self.device)

        fit_loop(
            self._net,
            Xt,
            Yt,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            loss_fn=loss_fn,
        )
        return None

    def predict(
        self,
        X_past: np.ndarray,
        times: np.ndarray | None = None,
        X_future: np.ndarray | None = None,
        *,
        images: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return predictions of shape ``(n, horizon)`` as float32 numpy."""
        if self._net is None:
            raise RuntimeError(
                "PatchTSTModel.predict called before fit; "
                "call fit(X_past, Y, ...) first."
            )
        X = np.asarray(X_past, dtype=np.float32)
        Xt = self._sanitize(torch.from_numpy(X)).to(self.device)
        self._net.eval()
        with torch.no_grad():
            pred = self._net(Xt).cpu().numpy().astype(np.float32)
        return pred
