"""iTransformer baseline — attention across VARIABLE channels (inverted).

Liu et al., ICLR 2024 — "iTransformer: Inverted Transformers are Effective for
Time Series Forecasting". The key inversion vs. a vanilla time-series
Transformer: each **variable (channel)** is embedded into a single token (its
whole lookback series projected to ``d_model``), self-attention then runs
ACROSS CHANNELS (the token axis), modelling inter-variate relationships, and a
linear head maps each channel token back to the horizon. The per-channel
forecasts are averaged into the single ``(n, horizon)`` target — uniform with
how :class:`PatchTSTModel` / :class:`DLinearModel` pool their channel
representations.

This is the deliberate complement to PatchTST: PatchTST attends over TIME
patches (channel-independent); iTransformer attends over CHANNELS (variate
interactions). Putting both in the benchmark lets E8 separate "temporal-pattern
modelling" from "variate-interaction modelling".

Design notes
------------
* **Channel = token (the inversion).** ``X_past[n, lb, Fp]`` is transposed to
  ``(n, Fp, lb)`` and each channel's whole lookback is projected to one
  ``d_model`` token by a shared ``Linear(lb, d_model)``; the
  :class:`torch.nn.TransformerEncoder` then attends across the ``Fp`` token axis.
  A shared ``Linear(d_model, horizon)`` head emits a per-channel forecast and the
  channels are averaged.
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
from ._torch_base import TorchForecastMixin, fit_loop, train_kwargs


class _iTransformerNet(nn.Module):
    """Inverted transformer: per-channel tokens + cross-channel self-attention.

    Each channel's lookback series is embedded into ONE token (the inversion);
    a :class:`torch.nn.TransformerEncoder` attends across the channel-token axis
    (modelling inter-variate relationships); a shared linear head maps each
    channel token to the horizon and channels are averaged.
    """

    def __init__(
        self,
        lookback: int,
        n_channels: int,
        horizon: int,
        d_model: int,
        n_heads: int,
        depth: int,
        dropout: float,
    ):
        super().__init__()
        # Embed each channel's lookback series into one token (the inversion).
        self.channel_embed = nn.Linear(lookback, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        # Per-channel horizon forecast (shared head, averaged across channels).
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n, lookback, Fp) -> (n, Fp, lookback) -> embed -> (n, Fp, d_model).
        tokens = self.channel_embed(x.transpose(1, 2))  # (n, Fp, d_model)
        out = self.encoder(tokens)                       # (n, Fp, d_model) — cross-channel
        y = self.head(out)                               # (n, Fp, horizon)
        return y.mean(dim=1)                             # (n, horizon)


class iTransformerModel(TorchForecastMixin, ForecastModel):
    """iTransformer (inverted: attention across variable channels).

    Parameters
    ----------
    d_model:
        Transformer hidden width (must be divisible by ``n_heads``).
    n_heads:
        Number of self-attention heads across the channel-token axis.
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
    seed:
        Seed for Python/NumPy/Torch RNG (and cuDNN determinism) applied at the
        start of :meth:`fit`; guarantees reproducible fits.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        depth: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 8,
        batch_size: int = 32,
        weight_decay: float = 0.0,
        patience: int | None = None,
        val_fraction: float = 0.0,
        accum_steps: int = 1,
        device: str | torch.device | None = None,
        name: str = "iTransformer",
        seed: int = 0,
    ):
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.depth = depth
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        # FULL-config training additions (defaults preserve old tiny behavior).
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_fraction = val_fraction
        self.accum_steps = accum_steps
        self.device = self._resolve_device(device)
        self._name = name
        self.seed = seed
        self._net: _iTransformerNet | None = None

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
        """Train the inverted transformer with Adam + MAE (L1) loss.

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
        _, lookback, fp = X.shape
        horizon = Y.shape[1]

        # Seed immediately before model construction so init + training are
        # reproducible (two fits, same seed -> identical predictions).
        self._seed(self.seed)
        self._net = _iTransformerNet(
            lookback=lookback,
            n_channels=fp,
            horizon=horizon,
            d_model=self.d_model,
            n_heads=self.n_heads,
            depth=self.depth,
            dropout=self.dropout,
        ).to(self.device)
        loss_fn = nn.L1Loss()  # MAE — robust, matches the Phase 6b spec.

        Xt = self._sanitize(torch.from_numpy(X)).to(self.device)
        Yt = torch.from_numpy(Y).to(self.device)

        fit_loop(
            self._net,
            Xt,
            Yt,
            loss_fn=loss_fn,
            **train_kwargs(self),
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
                "iTransformerModel.predict called before fit; "
                "call fit(X_past, Y, ...) first."
            )
        X = np.asarray(X_past, dtype=np.float32)
        Xt = self._sanitize(torch.from_numpy(X)).to(self.device)
        self._net.eval()
        with torch.no_grad():
            pred = self._net(Xt).cpu().numpy().astype(np.float32)
        return pred
