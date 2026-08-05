"""DLinear baseline — trend/seasonal decomposition + two linear maps.

Zeng et al., AAAI 2023 — "Are Transformers Effective for Time Series
Forecasting?". A deliberately tiny model that is a strong Transformer
antagonist: the lookback is decomposed by a moving-average filter into a smooth
TREND and a zero-mean SEASONAL residual; each component is mapped to the
horizon by its own linear layer and the two are summed.

``Y = W_seasonal · seasonal(X) + W_trend · trend(X)``

Channel-independent in the DLinear sense: each channel's lookback is decomposed
and linearly mapped by channel-**shared** weights, then channel forecasts are
averaged into the single ``(n, horizon)`` target (mirrors how PatchTST pools its
channel-independent representation). No attention, no nonlinearity — the value
of DLinear for the benchmark is precisely that it is the simplest competent
baseline, so "does complexity beat it?" is a fair question.

Design notes
------------
* **Scaled features in, no re-scaling inside.** The pipeline's
  :class:`PhysicalFeatureScaler` has already normalized the block.
* **Moving-average kernel is forced odd** so the trend keeps the lookback
  length under symmetric padding (no phase shift).
* **Determinism + Honest NaN** match the other torch baselines (fixed seed +
  cuDNN determinism; NaN/inf sanitized to 0 so a stray NaN cannot poison).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..core import ForecastModel
from ._torch_base import TorchForecastMixin, fit_loop


class _DLinearNet(nn.Module):
    """Trend/seasonal decomposition + two channel-shared linear horizon heads."""

    def __init__(self, lookback: int, horizon: int, kernel_size: int):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1  # force odd → symmetric padding (no phase shift)
        self.lookback = lookback
        # AvgPool1d over the lookback axis; padding keeps the output length = lookback.
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size, stride=1,
            padding=(kernel_size - 1) // 2, count_include_pad=False,
        )
        # One linear per component (lookback -> horizon), shared across channels.
        self.linear_trend = nn.Linear(lookback, horizon)
        self.linear_seasonal = nn.Linear(lookback, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n, lookback, Fp) -> decompose along the lookback axis.
        n, lb, fp = x.shape
        xt = x.transpose(1, 2)                       # (n, Fp, lookback)
        trend = self.avg(xt)                          # (n, Fp, lookback)
        seasonal = xt - trend                         # (n, Fp, lookback)
        yt = self.linear_trend(trend.reshape(n * fp, lb))         # (n*Fp, horizon)
        ys = self.linear_seasonal(seasonal.reshape(n * fp, lb))   # (n*Fp, horizon)
        y = (yt + ys).reshape(n, fp, -1).mean(dim=1)              # (n, horizon)
        return y


class DLinearModel(TorchForecastMixin, ForecastModel):
    """DLinear (trend + seasonal decomposition, channel-shared linear heads).

    Parameters
    ----------
    kernel_size:
        Moving-average window for the trend extraction (forced odd).
    lr:
        Adam learning rate.
    epochs:
        Number of training epochs over the fit window set.
    batch_size:
        Minibatch size (full-batch when larger than ``n``).
    device:
        ``"cuda"`` / ``"cpu"`` / a :class:`torch.device`; ``None`` auto-selects.
    seed:
        Seed for Python/NumPy/Torch RNG (and cuDNN determinism).
    """

    def __init__(
        self,
        kernel_size: int = 25,
        lr: float = 1e-3,
        epochs: int = 8,
        batch_size: int = 32,
        device: str | torch.device | None = None,
        name: str = "DLinear",
        seed: int = 0,
    ):
        self.kernel_size = kernel_size
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = self._resolve_device(device)
        self._name = name
        self.seed = seed
        self._net: _DLinearNet | None = None

    @property
    def name(self) -> str:
        return self._name

    def fit(
        self,
        X_past: np.ndarray,
        Y: np.ndarray,
        times: np.ndarray | None = None,
        X_future: np.ndarray | None = None,
        *,
        images: np.ndarray | None = None,
    ) -> None:
        """Train the two linear heads with Adam + MAE (L1) loss."""
        X = np.asarray(X_past, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"X_past must be 3-D (n, lookback, Fp); got {X.shape}")
        Y = np.asarray(Y, dtype=np.float32)
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2-D (n, horizon); got {Y.shape}")
        n, lookback, _fp = X.shape
        horizon = Y.shape[1]
        # Lookback must be >= kernel for the moving average to be meaningful;
        # auto-shrink the kernel so DLinear stays runnable at short lookbacks.
        k = min(self.kernel_size, lookback if lookback % 2 else lookback - 1)
        if k < 1:
            k = 1
        self._seed(self.seed)
        self._net = _DLinearNet(lookback=lookback, horizon=horizon, kernel_size=k).to(self.device)
        loss_fn = nn.L1Loss()  # MAE
        Xt = self._sanitize(torch.from_numpy(X)).to(self.device)
        Yt = torch.from_numpy(Y).to(self.device)
        fit_loop(self._net, Xt, Yt, epochs=self.epochs, batch_size=self.batch_size,
                 lr=self.lr, loss_fn=loss_fn)
        return None

    def predict(
        self,
        X_past: np.ndarray,
        times: np.ndarray | None = None,
        X_future: np.ndarray | None = None,
        *,
        images: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("DLinearModel.predict called before fit.")
        X = np.asarray(X_past, dtype=np.float32)
        Xt = self._sanitize(torch.from_numpy(X)).to(self.device)
        self._net.eval()
        with torch.no_grad():
            return self._net(Xt).cpu().numpy().astype(np.float32)
