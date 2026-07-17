"""Smart Persistence baseline — clear-sky-scaled physical skill reference.

Spec §10. Target-agnostic (works for PV power *and* GHI): the model estimates a
per-sample clearness ratio ``κ = target_now / clearsky_now`` at the forecast
issue time and propagates it across the horizon via the future clear-sky
column, so the day-ahead shape tracks the diurnal clear-sky curve rather than
the flat last-observed value.

Inputs (produced by :class:`EndogenousFeatureEngineer`):
* ``X_past[:, -1, 0]``  — target at the issue time (last lookback step, first
  channel; the target is always the FIRST past column).
* ``X_future[:, :, 0]`` — ``ghi_clearsky`` over the horizon (first future
  channel). ``X_future[:, 0, 0]`` is used as the "now" clear-sky proxy.

Physical bookkeeping:
* Predictions are floored at 0 (night / negative clearness should not yield
  negative power).
* If ``capacity`` is provided, predictions are also capped at ``capacity``.
* If ``X_future`` is ``None`` (no clear-sky context available) the model
  degrades gracefully to plain persistence — the last target value repeated
  across the horizon.

``fit`` is a no-op: per-sample κ is computed at predict time, which is the
standard formulation and removes the need for stable training-set κ estimates.
"""
from __future__ import annotations

import numpy as np

from ..core import ForecastModel

# Floor for the clear-sky denominator. Anything below this is treated as night
# (κ collapses to 0, so the forecast does too) rather than dividing by ~0.
_EPS = 1e-6


class SmartPersistenceModel(ForecastModel):
    """Clear-sky-scaled persistence skill reference.

    Parameters
    ----------
    capacity:
        Optional upper bound (e.g. PV plant nameplate / clear-sky irradiance
        ceiling). Predictions are clipped to ``[0, capacity]`` when set.
    epsilon:
        Clear-sky denominator floor; raise only if you need a wider night band.
    horizon:
        Forecast horizon to emit when ``X_future`` is ``None`` (persistence
        fallback). Defaults to the lookback length. The standardized benchmark
        always supplies ``X_future``, so this is only a defensive convenience.
    """

    def __init__(
        self,
        capacity: float | None = None,
        epsilon: float = _EPS,
        horizon: int | None = None,
    ):
        self.capacity = capacity
        self.epsilon = epsilon
        self.horizon = horizon

    # -- ForecastModel API ---------------------------------------------------

    def fit(
        self,
        X_past: np.ndarray,
        Y: np.ndarray | None = None,
        times: np.ndarray | None = None,
        X_future: np.ndarray | None = None,
    ) -> None:
        """No-op. Per-sample κ is computed at predict time (see module docstring).

        ``Y`` is accepted (and optional) only to honor the
        :class:`ForecastModel` signature; smart persistence does not require a
        training target.
        """
        return None

    def predict(
        self,
        X_past: np.ndarray,
        times: np.ndarray | None = None,
        X_future: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return clear-sky-scaled persistence forecasts of shape ``(n, horizon)``."""
        # Target at the issue time = last lookback step, first past channel.
        target_now = np.asarray(X_past[:, -1, 0], dtype=float)

        if X_future is None:
            # No clear-sky context: degrade to plain persistence.
            horizon = self.horizon if self.horizon is not None else X_past.shape[1]
            return self._clip(np.repeat(target_now[:, None], horizon, axis=1))

        csky_future = np.asarray(X_future[:, :, 0], dtype=float)
        csky_now = np.asarray(X_future[:, 0, 0], dtype=float)
        kappa = target_now / np.maximum(csky_now, self.epsilon)
        pred = kappa[:, None] * csky_future
        return self._clip(pred)

    @property
    def name(self) -> str:
        return "SmartPersistence"

    # -- internals -----------------------------------------------------------

    def _clip(self, pred: np.ndarray) -> np.ndarray:
        """Apply the physical floor (≥0) and, if set, the capacity ceiling."""
        np.maximum(pred, 0.0, out=pred)
        if self.capacity is not None:
            np.minimum(pred, self.capacity, out=pred)
        return pred
