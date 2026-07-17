"""Forecast framework core.

Abstract base classes (:class:`FeatureEngineer`, :class:`ForecastModel`), the
:dataclass:`PipelineConfig`, a concrete :class:`EndogenousFeatureEngineer`
(sliding-window over a sorted, entity-grouped tabular frame), and the
:class:`SolarForecastPipeline` that ties loader -> features -> fit -> predict
together.

Design notes
------------
* ``create_features`` produces four blocks per the Phase 6a contract:
  ``X_past[n, lb, Fp]``, ``Y[n, horizon]``, ``times[n]`` and an optional future
  context block ``X_future[n, horizon, Ff]`` (carries future ``ghi_clearsky`` /
  ``cos_zenith`` so the physical Smart Persistence baseline and tabular models
  share one interface). ``X_future`` is ``None`` when those columns are absent.
* Windowing is entity-aware: rows are sorted by ``entity_id`` then
  ``timestamp_utc`` and grouped per entity, so a window never spans two entities.
* A missing target column raises :class:`ValueError` (Honest NaN) rather than
  silently fabricating data.
* The pipeline engineers train and test splits independently (the standard
  benchmark discipline: each split is evaluated with its own lookback context).
  When ``df_provider`` is ``None`` it falls back to
  :class:`solarunibench.loader.SolarUniBenchLoader`; otherwise the provider is
  invoked as ``df_provider(config)`` (the test hook).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..loader import SolarUniBenchLoader

# Repo-relative config path for the default ``output_root`` resolution.
_PATHS_YAML = Path(__file__).resolve().parents[2] / "configs" / "paths.yaml"

# Future-context columns shared by the physical Smart Persistence baseline and
# tabular models. X_future is built only when BOTH are present in the frame.
_FUTURE_COLS: tuple[str, ...] = ("ghi_clearsky", "cos_zenith")

# Split labels the pipeline recognizes.
_TRAIN_SPLIT = "train"
_TEST_SPLIT = "test"


def _default_data_root() -> str | None:
    """Return ``output_root`` from ``configs/paths.yaml`` (None if unreadable)."""
    try:
        cfg = yaml.safe_load(_PATHS_YAML.read_text())
    except (FileNotFoundError, OSError):
        return None
    root = cfg.get("output_root") if isinstance(cfg, dict) else None
    return root


@dataclass
class PipelineConfig:
    """Configuration consumed by :class:`SolarForecastPipeline`.

    ``target`` is per-dataset: ``pv_power_normalized`` for PV datasets
    (SKIPPD/GEFCom) or ``ghi_actual`` for irradiance datasets (Folsom/SunLake).
    """

    dataset: str
    target: str
    lookback: int = 24
    horizon: int = 24
    stride: int = 1
    feature_cols: list[str] | None = None
    capacity: float | None = None
    data_root: str | None = None


class FeatureEngineer(ABC):
    """Transform a tabular frame into windowed (X_past, Y, times, X_future)."""

    @abstractmethod
    def create_features(
        self, df: pd.DataFrame, lookback: int, horizon: int, stride: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Return ``(X_past[n, lb, Fp], Y[n, horizon], times[n], X_future | None)``.

        ``X_future`` has shape ``(n, horizon, Ff)`` when future-context columns
        are available, otherwise ``None``.
        """
        raise NotImplementedError


class ForecastModel(ABC):
    """Minimal model interface: fit on windows, predict a horizon vector."""

    @abstractmethod
    def fit(
        self,
        X_past: np.ndarray,
        Y: np.ndarray,
        times: np.ndarray | None = None,
        X_future: np.ndarray | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        X_past: np.ndarray,
        times: np.ndarray | None = None,
        X_future: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return predictions of shape ``(n, horizon)``."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


class EndogenousFeatureEngineer(FeatureEngineer):
    """Sliding-window engineer over ``[target] + feature_cols``.

    * ``X_past`` stacks ``[target, *feature_cols]`` across the lookback window.
    * ``Y`` is the target over the forecast horizon.
    * ``X_future`` stacks ``[ghi_clearsky, cos_zenith]`` across the horizon when
      both columns are present (else ``None``).
    """

    def __init__(self, target: str, feature_cols: list[str] | None = None):
        self.target = target
        self.feature_cols: list[str] = list(feature_cols) if feature_cols else []

    def create_features(
        self, df: pd.DataFrame, lookback: int, horizon: int, stride: int = 1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        if lookback <= 0 or horizon <= 0 or stride <= 0:
            raise ValueError(
                f"lookback, horizon, stride must be positive (got {lookback}, {horizon}, {stride})"
            )
        if self.target not in df.columns:
            raise ValueError(
                f"target column '{self.target}' is not present in the dataframe "
                f"(columns: {list(df.columns)})"
            )
        for required in ("timestamp_utc", "entity_id"):
            if required not in df.columns:
                raise ValueError(f"dataframe must contain a '{required}' column")

        # Past block = target first, then the feature_cols that actually exist.
        past_cols = [self.target] + [c for c in self.feature_cols if c in df.columns]
        have_future = all(c in df.columns for c in _FUTURE_COLS)
        future_cols = list(_FUTURE_COLS) if have_future else []

        xp_windows: list[np.ndarray] = []
        y_windows: list[np.ndarray] = []
        t_windows: list[np.ndarray] = []
        xf_windows: list[np.ndarray] = []

        df_sorted = df.sort_values(["entity_id", "timestamp_utc"], kind="mergesort")
        for _, g in df_sorted.groupby("entity_id", sort=False):
            g = g.reset_index(drop=True)
            n = len(g)
            past = g[past_cols].to_numpy(dtype=float)
            tgt = g[self.target].to_numpy(dtype=float)
            ts = _to_datetime64_ns(g["timestamp_utc"])
            fut = g[future_cols].to_numpy(dtype=float) if future_cols else None

            last_start = n - lookback - horizon
            for start in range(0, last_start + 1, stride):
                horizon_slice = slice(start + lookback, start + lookback + horizon)
                xp_windows.append(past[start : start + lookback])
                y_windows.append(tgt[horizon_slice])
                t_windows.append(ts[start + lookback])  # forecast-issue time
                if fut is not None:
                    xf_windows.append(fut[horizon_slice])

        n_w = len(xp_windows)
        n_past_feat = len(past_cols)
        if n_w:
            X_past = np.stack(xp_windows, axis=0)
            Y = np.stack(y_windows, axis=0)
        else:
            X_past = np.empty((0, lookback, n_past_feat), dtype=float)
            Y = np.empty((0, horizon), dtype=float)
        times = (
            np.stack(t_windows, axis=0)
            if n_w
            else np.empty((0,), dtype="datetime64[ns]")
        )
        if have_future:
            X_future = (
                np.stack(xf_windows, axis=0)
                if n_w
                else np.empty((0, horizon, len(_FUTURE_COLS)), dtype=float)
            )
        else:
            X_future = None
        return X_past, Y, times, X_future


def _flatten_window_features(
    X_past: np.ndarray, X_future: np.ndarray | None
) -> np.ndarray:
    """Flatten ``X_past[n, lb, Fp]`` (+ optional ``X_future[n, horizon, Ff]``) to 2-D.

    Result shape: ``(n, lb*Fp)`` when ``X_future`` is ``None``, otherwise
    ``(n, lb*Fp + horizon*Ff)`` with the future block concatenated on the right
    (column order: all past features, then all future features).

    Shared by the tabular baselines (LightGBM, MLP) so they consume identical
    design matrices — direct comparison of skill is not confounded by differing
    feature layouts.
    """
    n = X_past.shape[0]
    Xp = np.asarray(X_past, dtype=float).reshape(n, -1)
    if X_future is None:
        return Xp
    Xf = np.asarray(X_future, dtype=float).reshape(n, -1)
    return np.concatenate([Xp, Xf], axis=1)


def _to_datetime64_ns(series: pd.Series) -> np.ndarray:
    """Coerce a timestamp column to naive UTC ``datetime64[ns]`` (tz-stable)."""
    ts = pd.to_datetime(series)
    if ts.dt.tz is not None:
        # Drop tz after anchoring to UTC wall-clock; safe across pandas versions.
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return ts.to_numpy().astype("datetime64[ns]")


class SolarForecastPipeline:
    """Load -> engineer -> fit -> predict harness for solar forecasting."""

    def __init__(
        self,
        config: PipelineConfig,
        feature_engineer: FeatureEngineer,
        model: ForecastModel,
    ):
        self.config = config
        self.feature_engineer = feature_engineer
        self.model = model

    def run(self, df_provider=None) -> dict:
        """Execute the pipeline and return ``{pred, y_true, times, model_name}``.

        If ``df_provider`` is ``None``, tabular frames are loaded via
        :class:`SolarUniBenchLoader` from ``config.data_root`` (falling back to
        ``configs/paths.yaml``'s ``output_root``). Otherwise ``df_provider`` is
        called as ``df_provider(config)`` and is expected to return a frame that
        includes a ``split`` column (or a single split already filtered).
        """
        df_train = self._load_split(_TRAIN_SPLIT, df_provider)
        df_test = self._load_split(_TEST_SPLIT, df_provider)

        lb = self.config.lookback
        hz = self.config.horizon
        st = self.config.stride
        xp_tr, y_tr, t_tr, xf_tr = self.feature_engineer.create_features(df_train, lb, hz, st)
        xp_te, y_te, t_te, xf_te = self.feature_engineer.create_features(df_test, lb, hz, st)

        # Positional call so any conforming model works regardless of whether it
        # spells the future-context parameter ``X_future`` or ``Xf``.
        self.model.fit(xp_tr, y_tr, t_tr, xf_tr)
        pred = self.model.predict(xp_te, t_te, xf_te)
        return {
            "pred": pred,
            "y_true": y_te,
            "times": t_te,
            "model_name": self.model.name,
        }

    # -- internals ---------------------------------------------------------

    def _load_split(self, split: str, df_provider) -> pd.DataFrame:
        if df_provider is None:
            root = self._resolve_data_root()
            return SolarUniBenchLoader(root).load_tabular(self.config.dataset, split)
        full = df_provider(self.config)
        if "split" in full.columns:
            return full[full["split"] == split].reset_index(drop=True)
        # No split column: caller supplied a single pre-filtered frame.
        return full.reset_index(drop=True)

    def _resolve_data_root(self) -> str:
        root = self.config.data_root or _default_data_root()
        if not root:
            raise ValueError(
                "config.data_root is unset and configs/paths.yaml 'output_root' "
                "could not be resolved; pass data_root explicitly."
            )
        return root
