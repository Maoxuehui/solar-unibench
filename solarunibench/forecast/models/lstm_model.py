"""LSTM sequence baseline — direct multi-step recurrent forecaster.

A stacked :class:`torch.nn.LSTM` reads the past feature block
``X_past[n, lb, Fp]`` directly (it is already ``(batch, seq, feat)`` — the
natural shape for a recurrent net) and a single :class:`torch.nn.Linear` head
maps the LAST timestep's hidden state to the full forecast horizon, so
``predict`` returns ``(n, horizon)`` in one forward pass.

Design notes
------------
* **Sequence-native input.** Unlike the MLP/LightGBM baselines (which flatten
  ``lb*Fp``), the LSTM consumes the lookback axis as a sequence — this is the
  whole point of a recurrent baseline. ``X_future`` is accepted only to honor
  the :class:`ForecastModel` signature; it is not consumed (YAGNI — the
  sequence-to-horizon mapping is the baseline under test).
* **Scaled features in, no re-scaling inside.** The pipeline's
  :class:`PhysicalFeatureScaler` (default-on) already applies irradiance/1500
  and train-fit NWP z-scoring BEFORE feature engineering, so the LSTM receives
  scaled features and must not re-scale (would double-normalize / lose the
  physical anchoring).
* **Determinism.** ``fit`` seeds Python/NumPy/Torch (and cuDNN) from a fixed
  ``seed`` (default 0) right before building the network, so two fits on the
  same data with the same seed produce byte-identical predictions.
* **Honest NaN.** Missing lookback features are sanitized to 0 via
  :func:`torch.nan_to_num` (a documented, transparent imputation) so a stray
  NaN in the NWP block cannot poison training or emit non-finite forecasts; the
  target ``Y`` is never fabricated.
* **Device-aware.** ``device=None`` auto-selects CUDA when available and falls
  back to CPU otherwise; pass ``device="cpu"`` for deterministic tiny tests.
"""
from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn

from ..core import ForecastModel


class _LSTMNet(nn.Module):
    """Stacked LSTM + linear horizon head (last-timestep readout)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        horizon: int,
    ):
        super().__init__()
        # nn.LSTM only accepts ``dropout`` when num_layers > 1; otherwise it
        # raises. Gate it so a single-layer config stays valid.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)  # (n, lb, hidden)
        last = out[:, -1, :]  # (n, hidden) — last timestep's hidden state
        return self.head(last)  # (n, horizon)


class LSTMModel(ForecastModel):
    """Stacked-LSTM direct multi-step forecaster.

    Parameters
    ----------
    hidden_size:
        LSTM hidden state width.
    num_layers:
        Number of stacked LSTM layers (``dropout`` applies between them when >1).
    dropout:
        Dropout between LSTM layers (ignored when ``num_layers == 1``).
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
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 8,
        batch_size: int = 32,
        device: str | torch.device | None = None,
        name: str = "LSTM",
        seed: int = 0,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = self._resolve_device(device)
        self._name = name
        self.seed = seed
        self._net: _LSTMNet | None = None

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
    ) -> None:
        """Train the LSTM + linear head with Adam + MAE (L1) loss.

        ``times`` / ``X_future`` are accepted only to honor the
        :class:`ForecastModel` signature; the LSTM consumes the ``X_past``
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
        n, _, fp = X.shape
        horizon = Y.shape[1]

        # Seed immediately before model construction so init + training are
        # reproducible (two fits, same seed -> identical predictions).
        self._seed(self.seed)
        self._net = _LSTMNet(
            input_size=fp,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            horizon=horizon,
        ).to(self.device)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.L1Loss()  # MAE — robust, matches the Phase 6b spec.

        Xt = self._sanitize(torch.from_numpy(X)).to(self.device)
        Yt = torch.from_numpy(Y).to(self.device)

        self._net.train()
        idx = np.arange(n)
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.batch_size):
                batch = idx[start : start + self.batch_size]
                optimizer.zero_grad()
                pred = self._net(Xt[batch])
                loss = loss_fn(pred, Yt[batch])
                loss.backward()
                optimizer.step()
        return None

    def predict(
        self,
        X_past: np.ndarray,
        times: np.ndarray | None = None,
        X_future: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return predictions of shape ``(n, horizon)`` as float32 numpy."""
        if self._net is None:
            raise RuntimeError(
                "LSTMModel.predict called before fit; call fit(X_past, Y, ...) first."
            )
        X = np.asarray(X_past, dtype=np.float32)
        Xt = self._sanitize(torch.from_numpy(X)).to(self.device)
        self._net.eval()
        with torch.no_grad():
            pred = self._net(Xt).cpu().numpy().astype(np.float32)
        return pred

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
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
