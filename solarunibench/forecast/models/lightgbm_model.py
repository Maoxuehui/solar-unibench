"""LightGBM tabular baseline — direct per-horizon multi-output regression.

The standard solar-forecasting tabular recipe (direct multi-step forecasting):
flatten ``X_past[n, lb, Fp]`` together with the optional future-context block
``X_future[n, horizon, Ff]`` into one 2-D design matrix and fit one
:class:`lightgbm.LGBMRegressor` per horizon step. ``predict`` stacks the
per-step predictions into ``(n, horizon)``.

Design notes
------------
* **One model per horizon step.** A single regressor cannot natively emit a
  vector, and the per-step direct strategy consistently outperforms a
  recursive single-output fit on the day-ahead solar benchmark — it sidesteps
  error accumulation and lets each step learn its own feature mix.
* **NaN handling.** LightGBM handles missing feature values natively (it learns
  the best split direction for NaN at every node), so neither training nor
  prediction requires pre-imputation of missing past/future features.
* **Quiet by default.** ``verbose=-1`` is forced into the constructor params
  (caller may override) and LightGBM's UserWarnings are silenced inside
  ``fit``/``predict`` so pytest output stays clean.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from lightgbm import LGBMRegressor

from ..core import ForecastModel, _flatten_window_features

# Sensible defaults for the benchmark: quiet, deterministic, single-threaded
# (keeps tests reproducible; pass n_jobs=-1 via ``params`` for real training).
_DEFAULT_PARAMS: dict[str, Any] = {
    "verbose": -1,
    "n_estimators": 100,
    "n_jobs": 1,
    "random_state": 0,
}


class LightGBMModel(ForecastModel):
    """Per-horizon LightGBM tabular forecaster.

    Parameters
    ----------
    params:
        Optional dict of keyword arguments forwarded to every
        :class:`lightgbm.LGBMRegressor`. Merged over the quiet/deterministic
        defaults (see :data:`_DEFAULT_PARAMS`); pass ``verbose=0`` here to
        re-enable LightGBM's logging.
    name:
        Display name returned by the :attr:`name` property (used by the
        pipeline's result dict). Defaults to ``"LightGBM"``.
    """

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        name: str = "LightGBM",
    ):
        merged = dict(_DEFAULT_PARAMS)
        if params:
            merged.update(params)
        # Last-word safety: never emit LightGBM console chatter unless a caller
        # explicitly opts in by passing ``verbose>=0``.
        merged.setdefault("verbose", -1)
        self.params = merged
        self._name = name
        self._models: list[LGBMRegressor] = []

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
        """Fit one ``LGBMRegressor`` per horizon column of ``Y``.

        ``times`` / ``images`` are accepted only to honor the
        :class:`ForecastModel` signature; LightGBM is tabular-only.
        """
        X = _flatten_window_features(X_past, X_future)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2-D (n, horizon); got shape {Y.shape}")
        horizon = Y.shape[1]

        self._models = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            for h in range(horizon):
                est = LGBMRegressor(**self.params)
                est.fit(X, Y[:, h])
                self._models.append(est)
        return None

    def predict(
        self,
        X_past: np.ndarray,
        times: np.ndarray | None = None,
        X_future: np.ndarray | None = None,
        *,
        images: np.ndarray | None = None,
    ) -> np.ndarray:
        """Stack per-step predictions into ``(n, horizon)``."""
        if not self._models:
            raise RuntimeError(
                "LightGBMModel.predict called before fit; call fit(X_past, Y, ...) first."
            )
        X = _flatten_window_features(X_past, X_future)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            preds = np.column_stack([m.predict(X) for m in self._models])
        return preds
