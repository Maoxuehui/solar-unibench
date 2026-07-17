"""Configurable MLP baseline — single multi-output sklearn MLPRegressor.

A single :class:`sklearn.neural_network.MLPRegressor` is fit on the flattened
``X_past[n, lb, Fp]`` (concatenated with the optional future-context block
``X_future[n, horizon, Ff]``). ``MLPRegressor`` is multi-output natively, so
``predict`` returns the whole horizon vector at once — shape ``(n, horizon)``.

Design notes
------------
* **Configurable architecture.** ``hidden_layer_sizes`` and ``activation`` are
  constructor arguments; the default ``(128, 64)`` + ReLU is the in-house
  baseline.
* **``preset="skippd_paper"``** reproduces Nie et al. 2020 (a single hidden
  layer of 7 logistic units). A preset OVERRIDES any explicitly passed
  ``hidden_layer_sizes`` / ``activation`` — the published architecture is the
  whole point of selecting the preset, so an accidental kwarg should not silently
  diverge from it.
* **Shared flatten.** Past + future features are flattened by the same
  :func:`solarunibench.forecast.core._flatten_window_features` helper used by
  :class:`LightGBMModel`, so the two tabular baselines consume identical design
  matrices and can be compared without feature-layout confounding.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.neural_network import MLPRegressor

from ..core import ForecastModel, _flatten_window_features

# In-house baseline architecture (direct multi-step head, MLPRegressor emits a
# horizon-length vector natively).
_DEFAULT_HIDDEN_LAYER_SIZES: tuple[int, ...] = (128, 64)
_DEFAULT_ACTIVATION: str = "relu"

# Sensible deterministic defaults forwarded to every MLPRegressor; callers may
# override any of these (e.g. ``max_iter``, ``learning_rate_init``) via ``**kw``.
_DEFAULT_KW: dict[str, Any] = {"random_state": 0}

# Named architecture presets. A preset overrides explicit hidden_layer_sizes /
# activation kwargs (see class docstring).
_PRESETS: dict[str, dict[str, Any]] = {
    # Nie et al. 2020 (SKIPPD paper): one hidden layer of 7 logistic units.
    "skippd_paper": {"hidden_layer_sizes": (7,), "activation": "logistic"},
}


class MLPModel(ForecastModel):
    """Single multi-output :class:`MLPRegressor` forecaster.

    Parameters
    ----------
    hidden_layer_sizes:
        Forwarded to :class:`sklearn.neural_network.MLPRegressor`. Ignored when
        ``preset`` is set (the preset wins).
    activation:
        Forwarded to :class:`sklearn.neural_network.MLPRegressor`. Ignored when
        ``preset`` is set.
    preset:
        Optional named architecture preset. ``"skippd_paper"`` reproduces
        Nie et al. 2020 — ``hidden_layer_sizes=(7,)``, ``activation="logistic"``
        — overriding any explicit ``hidden_layer_sizes`` / ``activation``.
    name:
        Display name returned by the :attr:`name` property.
    **kw:
        Additional keyword arguments forwarded to
        :class:`sklearn.neural_network.MLPRegressor` (e.g. ``max_iter``,
        ``learning_rate_init``, ``alpha``).
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = _DEFAULT_HIDDEN_LAYER_SIZES,
        activation: str = _DEFAULT_ACTIVATION,
        preset: str | None = None,
        name: str = "MLP",
        **kw: Any,
    ):
        if preset is not None:
            if preset not in _PRESETS:
                raise ValueError(
                    f"unknown MLP preset '{preset}' "
                    f"(known: {sorted(_PRESETS)})"
                )
            cfg = _PRESETS[preset]
            hidden_layer_sizes = cfg["hidden_layer_sizes"]
            activation = cfg["activation"]
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.preset = preset
        self._name = name
        self._kw: dict[str, Any] = {**_DEFAULT_KW, **kw}
        self._model: MLPRegressor | None = None

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
        """Fit a single multi-output ``MLPRegressor`` on the flattened design.

        ``times`` is accepted only to honor the :class:`ForecastModel` signature;
        the MLP does not consume issue-time information here.
        """
        X = _flatten_window_features(X_past, X_future)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2-D (n, horizon); got shape {Y.shape}")

        self._model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            **self._kw,
        )
        self._model.fit(X, Y)
        return None

    def predict(
        self,
        X_past: np.ndarray,
        times: np.ndarray | None = None,
        X_future: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return predictions of shape ``(n, horizon)`` (multi-output)."""
        if self._model is None:
            raise RuntimeError(
                "MLPModel.predict called before fit; call fit(X_past, Y, ...) first."
            )
        X = _flatten_window_features(X_past, X_future)
        return self._model.predict(X)
