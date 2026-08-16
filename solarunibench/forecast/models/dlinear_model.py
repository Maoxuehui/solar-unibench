"""DLinear baseline — trend/seasonal decomposition + two linear heads.

Zeng et al., AAAI 2023 — "Are Transformers Effective for Time Series
Forecasting?". A deliberately tiny model that is a strong Transformer
antagonist: the lookback is decomposed by a moving-average filter into a smooth
TREND and a zero-mean SEASONAL residual; each component is mapped to the
horizon by its own channel-**shared** linear layer and the two are summed:

``Y = W_seasonal · seasonal(X) + W_trend · trend(X)``

The per-channel forecasts are then averaged into the single ``(n, horizon)``
target — uniform with how :class:`PatchTSTModel` pools its channel-independent
representation. No attention, no nonlinearity: the value of DLinear for the
benchmark is precisely that it is the simplest competent baseline, so "does
complexity beat it?" is a fair question.

Design notes
------------
* **Channel-shared linear weights.** The two ``Linear(lookback, horizon)`` maps
  are shared across all ``Fp`` channels (the DLinear channel-independent design,
  mirroring PatchTST); the per-channel forecasts are averaged before the output.
* **Scaled features in, no re-scaling inside.** The pipeline's
  :class:`PhysicalFeatureScaler` (default-on) already applies irradiance/1500
  and train-fit NWP z-scoring BEFORE feature engineering, so the model receives
  scaled features and must not re-scale (would double-normalize).
* **Moving-average kernel is forced odd** so the trend keeps the lookback length
  under symmetric padding (no phase shift). The kernel auto-shrinks to the
  largest odd value ``<= lookback`` when the configured ``kernel_size`` would
  exceed the lookback, so DLinear stays runnable at short lookbacks.
* **Determinism.** ``fit`` seeds Python/NumPy/Torch (and cuDNN) from a fixed
  ``seed`` (default 0) right before building the network, so two fits on the
  same data with the same seed produce byte-identical predictions.
* **Honest NaN.** Missing lookback features are sanitized to 0 via
  :func:`torch.nan_to_num` (a documented, transparent imputation) so a stray NaN
  in the NWP block cannot poison training or emit non-finite forecasts; the
  target ``Y`` is never fabricated.
* **Device-aware.** ``device=None`` auto-selects CUDA when available and falls
  back to CPU otherwise; pass ``device="cpu"`` for deterministic tiny tests.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..core import ForecastModel
from ._torch_base import TorchForecastMixin, fit_loop, train_kwargs


def _odd_kernel(kernel_size: int, lookback: int) -> int:
    """Largest odd kernel ``<= min(kernel_size, lookback)`` (>= 1).

    Symmetric padding needs an odd window so the trend keeps the lookback length
    with no phase shift; clamping to ``lookback`` keeps DLinear runnable when the
    configured ``kernel_size`` exceeds a short lookback.
    """
    k = min(int(kernel_size), int(lookback))
    if k > 1 and k % 2 == 0:
        k -= 1
    return max(k, 1)


class _DLinearNet(nn.Module):
    """Trend/seasonal decomposition + two channel-shared linear horizon heads.

    A moving average (:class:`torch.nn.AvgPool1d` with symmetric padding and
    ``count_include_pad=False``) extracts the trend; the residual is the seasonal
    component. Each component is flattened per channel and mapped
    ``lookback -> horizon`` by a channel-shared linear; the two forecasts are
    summed and averaged across channels.
    """

    def __init__(self, lookback: int, horizon: int, kernel_size: int):
        super().__init__()
        # Force odd so symmetric padding preserves the lookback length exactly.
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            count_include_pad=False,
        )
        # One linear per component (lookback -> horizon), shared across channels.
        self.linear_trend = nn.Linear(lookback, horizon)
        self.linear_seasonal = nn.Linear(lookback, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n, lookback, Fp) -> decompose along the lookback axis.
        n, lb, fp = x.shape
        xt = x.transpose(1, 2)                    # (n, Fp, lookback)
        trend = self.avg(xt)                       # (n, Fp, lookback)
        seasonal = xt - trend                      # (n, Fp, lookback)
        yt = self.linear_trend(trend.reshape(n * fp, lb))        # (n*Fp, horizon)
        ys = self.linear_seasonal(seasonal.reshape(n * fp, lb))  # (n*Fp, horizon)
        y = (yt + ys).reshape(n, fp, -1).mean(dim=1)             # (n, horizon)
        return y


class DLinearModel(TorchForecastMixin, ForecastModel):
    """DLinear (trend + seasonal decomposition, channel-shared linear heads).

    Parameters
    ----------
    kernel_size:
        Moving-average window for trend extraction (forced odd; auto-shrinks to
        the largest odd value ``<= lookback`` when it exceeds the lookback).
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
        kernel_size: int = 25,
        lr: float = 1e-3,
        epochs: int = 8,
        batch_size: int = 32,
        weight_decay: float = 0.0,
        patience: int | None = None,
        val_fraction: float = 0.0,
        accum_steps: int = 1,
        device: str | torch.device | None = None,
        name: str = "DLinear",
        seed: int = 0,
    ):
        self.kernel_size = kernel_size
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
        self._net: _DLinearNet | None = None

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
        """Train the two linear heads with Adam + MAE (L1) loss.

        ``times`` / ``X_future`` / ``images`` are accepted only to honor the
        :class:`ForecastModel` signature; DLinear consumes the ``X_past``
        sequence alone.
        """
        X = np.asarray(X_past, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(
                f"X_past must be 3-D (n, lookback, Fp); got shape {X.shape}"
            )
        Y = np.asarray(Y, dtype=np.float32)
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2-D (n, horizon); got shape {Y.shape}")
        _, lookback, _ = X.shape
        horizon = Y.shape[1]

        # Auto-shrink the kernel so DLinear stays runnable at short lookbacks;
        # forced odd inside the net for symmetric padding (no phase shift).
        kernel = _odd_kernel(self.kernel_size, lookback)

        # Seed immediately before model construction so init + training are
        # reproducible (two fits, same seed -> identical predictions).
        self._seed(self.seed)
        self._net = _DLinearNet(
            lookback=lookback, horizon=horizon, kernel_size=kernel
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
                "DLinearModel.predict called before fit; "
                "call fit(X_past, Y, ...) first."
            )
        X = np.asarray(X_past, dtype=np.float32)
        Xt = self._sanitize(torch.from_numpy(X)).to(self.device)
        pred = (
            self._batched_eval_forward(self._net, Xt, self.batch_size)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        return pred
