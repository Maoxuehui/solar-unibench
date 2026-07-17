"""Forecast framework core.

Abstract base classes (:class:`FeatureEngineer`, :class:`ForecastModel`), the
:dataclass:`PipelineConfig`, a concrete :class:`EndogenousFeatureEngineer`
(sliding-window over a sorted, entity-grouped tabular frame), and the
:class:`SolarForecastPipeline` that ties loader -> features -> fit -> predict
together.

Design notes
------------
* ``create_features`` produces five blocks per the Phase 6a + benchmark-integrity
  contract: ``X_past[n, lb, Fp]``, ``Y[n, horizon]``, ``times[n]``, an optional
  future context block ``X_future[n, horizon, Ff]`` (carries future
  ``ghi_clearsky`` / ``cos_zenith`` so the physical Smart Persistence baseline
  and tabular models share one interface), and an optional ``cos_zenith_issue``
  per-sample array used by the daytime-window evaluator.
  ``X_future``/``cos_zenith_issue`` are ``None`` when those columns are absent.
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
from .scaling import PhysicalFeatureScaler

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
    """Transform a tabular frame into windowed features for the pipeline.

    Returns the 6-tuple ``(X_past, Y, times, X_future, cos_zenith_issue,
    images)``:

    * ``X_past``  — ``(n, lb, Fp)`` past feature block (target first).
    * ``Y``       — ``(n, horizon)`` target over the horizon.
    * ``times``   — ``(n,)`` forecast-issue timestamps.
    * ``X_future``— ``(n, horizon, Ff)`` future-context block when the future
      columns are present, else ``None``.
    * ``cos_zenith_issue`` — ``(n,)`` issue-time ``cos_zenith`` per sample when
      ``cos_zenith`` is a column (used by the daytime-window evaluator), else
      ``None``. Always ``None`` when ``cos_zenith`` is absent.
    * ``images`` — ``(n, lb, 4, H, W)`` per-window sky/satellite image cube
      (float32) when image-attach is configured and images resolve for every
      window, else ``None``. ``None`` for tabular datasets / when image-attach
      is off — tabular models are unaffected (Phase 6b-T5 multimodal extension).
    """

    @abstractmethod
    def create_features(
        self, df: pd.DataFrame, lookback: int, horizon: int, stride: int
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        """Return ``(X_past, Y, times, X_future, cos_zenith_issue, images)``."""
        raise NotImplementedError


class ForecastModel(ABC):
    """Minimal model interface: fit on windows, predict a horizon vector.

    The optional keyword-only ``images`` argument (Phase 6b-T5, multimodal
    extension) carries the per-window sky/satellite image cube
    ``(n, lookback, 4, H, W)`` for image-aware models (e.g. CrossViViT). It
    defaults to ``None`` and is IGNORED by every tabular model — that keeps the
    interface backward compatible (the tabular baselines simply add
    ``*, images=None`` to their signatures and drop it).
    """

    @abstractmethod
    def fit(
        self,
        X_past: np.ndarray,
        Y: np.ndarray,
        times: np.ndarray | None = None,
        X_future: np.ndarray | None = None,
        *,
        images: np.ndarray | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        X_past: np.ndarray,
        times: np.ndarray | None = None,
        X_future: np.ndarray | None = None,
        *,
        images: np.ndarray | None = None,
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
    * ``cos_zenith_issue`` is the per-sample ``cos_zenith`` at the forecast-issue
      timestamp when ``cos_zenith`` is a column (else ``None``); feeds the
      daytime-window evaluator.
    * ``images`` — ``(n, lookback, 4, H, W)`` per-window sky-image cube when
      image-attach is configured (``return_images=True`` with an
      ``image_loader`` + ``image_dataset``) and every window's lookback resolves
      images; otherwise ``None`` (tabular models unaffected). Image-attach
      mirrors :class:`solarunibench.dataset.SolarUniBenchDataset`: exact-timestamp
      match preferred, nearest-neighbor within ``match_tolerance`` fallback.
    """

    def __init__(
        self,
        target: str,
        feature_cols: list[str] | None = None,
        *,
        return_images: bool = False,
        image_loader: object | None = None,
        image_dataset: str | None = None,
        match_tolerance: pd.Timedelta | str | None = None,
    ):
        self.target = target
        self.feature_cols: list[str] = list(feature_cols) if feature_cols else []
        # Multimodal image-attach (Phase 6b-T5). Default off → 6th return is
        # None and tabular models are unaffected. When ``return_images=True``,
        # ``image_loader`` + ``image_dataset`` MUST be supplied; the dataset's
        # sky-image cube + timestamps are attached per window.
        self.return_images = return_images
        self.image_loader = image_loader
        self.image_dataset = image_dataset
        self._match_tolerance = (
            pd.Timedelta(match_tolerance) if match_tolerance is not None else None
        )
        if return_images and (image_loader is None or image_dataset is None):
            raise ValueError(
                "return_images=True requires both image_loader and image_dataset "
                "(the sky-image cube source + dataset name)."
            )
        # Lazily built and cached on first create_features call.
        self._img_index: _ImageIndex | None = None
        self._img_cube = None

    def create_features(
        self, df: pd.DataFrame, lookback: int, horizon: int, stride: int = 1
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
    ]:
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
        # Issue-time cos_zenith per sample feeds the daytime-window evaluator.
        # None when cos_zenith is absent (graceful backward-compat).
        have_cz = "cos_zenith" in df.columns

        xp_windows: list[np.ndarray] = []
        y_windows: list[np.ndarray] = []
        t_windows: list[np.ndarray] = []
        xf_windows: list[np.ndarray] = []
        cz_windows: list[np.ndarray] = []
        # Per-window lookback timestamps — kept so the image-attach step can
        # resolve sky frames for each window (exact-TS / NN-within-tolerance).
        lb_ts_windows: list[np.ndarray] = []

        df_sorted = df.sort_values(["entity_id", "timestamp_utc"], kind="mergesort")
        for _, g in df_sorted.groupby("entity_id", sort=False):
            g = g.reset_index(drop=True)
            n = len(g)
            past = g[past_cols].to_numpy(dtype=float)
            tgt = g[self.target].to_numpy(dtype=float)
            ts = _to_datetime64_ns(g["timestamp_utc"])
            fut = g[future_cols].to_numpy(dtype=float) if future_cols else None
            cz_col = g["cos_zenith"].to_numpy(dtype=float) if have_cz else None

            last_start = n - lookback - horizon
            for start in range(0, last_start + 1, stride):
                horizon_slice = slice(start + lookback, start + lookback + horizon)
                xp_windows.append(past[start : start + lookback])
                y_windows.append(tgt[horizon_slice])
                t_windows.append(ts[start + lookback])  # forecast-issue time
                lb_ts_windows.append(ts[start : start + lookback])
                if fut is not None:
                    xf_windows.append(fut[horizon_slice])
                if cz_col is not None:
                    # cos_zenith at the forecast-issue timestamp (aligned with
                    # ``times``); last_start bounds guarantee index validity.
                    cz_windows.append(cz_col[start + lookback])

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
        if have_cz:
            cos_zenith_issue = (
                np.stack(cz_windows, axis=0).astype(float)
                if n_w
                else np.empty((0,), dtype=float)
            )
        else:
            cos_zenith_issue = None
        images = self._build_images(lb_ts_windows)
        return X_past, Y, times, X_future, cos_zenith_issue, images

    # -- image attach --------------------------------------------------------

    def _build_images(self, lb_ts_windows: list[np.ndarray]) -> np.ndarray | None:
        """Resolve and stack the per-window sky-image cube, or None.

        Returns ``(n, lookback, 4, H, W)`` float32 when image-attach is on AND
        every window's lookback timestamps resolve to image-cube rows (exact-TS
        match preferred, nearest-neighbor within ``match_tolerance`` fallback —
        mirroring :class:`solarunibench.dataset.SolarUniBenchDataset`). Returns
        ``None`` when image-attach is off, no image cube is available, or any
        window misses — the model then degrades gracefully to tabular-only.
        """
        if not self.return_images or not lb_ts_windows:
            return None
        if self._img_index is None:
            assert self.image_loader is not None and self.image_dataset is not None
            img_ts = self.image_loader.get_image_timestamps(self.image_dataset)
            self._img_cube = self.image_loader.get_tensor_cube(
                self.image_dataset, "sky_images"
            )
            if img_ts is None or self._img_cube is None:
                # No images for this dataset → tabular-only (Honest NaN).
                self._img_index = _ImageIndex(np.asarray([]), self._match_tolerance)
            else:
                self._img_index = _ImageIndex(
                    np.asarray(img_ts), self._match_tolerance
                )
        if self._img_cube is None:
            return None

        from ..imaging.load import load_sky_frame

        cubes: list[np.ndarray] = []
        for lb_ts in lb_ts_windows:
            rows = self._img_index.rows_for(lb_ts)
            if rows is None or len(rows) != len(lb_ts):
                # All-or-nothing: a single unresolved window drops the image
                # modality for the whole batch (models need a uniform array).
                return None
            cubes.append(np.stack([load_sky_frame(self._img_cube, r) for r in rows]))
        return np.stack(cubes, axis=0).astype(np.float32)


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


class _ImageIndex:
    """Resolve per-window lookback timestamps to sky-image-cube rows.

    Mirrors :class:`solarunibench.dataset.SolarUniBenchDataset`'s image-attach:
    an exact-timestamp match is preferred; only on a miss, and only when a
    ``tolerance`` is set, does it fall back to nearest-neighbor within the
    tolerance (vectorized via a sorted :class:`pandas.DatetimeIndex`). Built
    once from a dataset's image-timestamp array; ``rows_for`` resolves a whole
    lookback window in one call.

    Both sides (image timestamps and the tabular lookback timestamps) are
    canonicalized to **naive UTC floored to the second** before matching, so a
    match is robust to the adapter's tz-aware ``"+00:00"`` ISO form vs the
    tz-stripped ``datetime64[ns]`` the FeatureEngineer carries internally.
    """

    @staticmethod
    def _norm(t) -> pd.Timestamp:
        """Canonicalize any timestamp to naive UTC, floored to the second.

        ``str(t)`` first so numpy string scalars (image timestamp arrays) and
        numpy datetime64s both parse via :class:`pandas.Timestamp`.
        """
        ts = pd.Timestamp(str(t))
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts.floor("s")

    def __init__(self, img_ts: np.ndarray, tolerance: pd.Timedelta | None):
        self.tolerance = tolerance
        arr = np.asarray(img_ts)
        # Exact-match dict: canonical naive-UTC Timestamp -> cube row.
        self._exact = {self._norm(t): i for i, t in enumerate(arr)}
        if tolerance is not None and len(arr) > 0:
            decorated = sorted(
                ((i, self._norm(t)) for i, t in enumerate(arr)), key=lambda x: x[1]
            )
            self._order = np.array([i for i, _ in decorated], dtype=np.intp)
            self._sorted = pd.DatetimeIndex([t for _, t in decorated])
        else:
            self._order = None
            self._sorted = None

    def rows_for(self, lb_ts: np.ndarray) -> list[int] | None:
        """Return one cube row per lookback timestamp, or None on any miss."""
        lb = np.asarray(lb_ts)
        if len(lb) == 0:
            return None
        normed = [self._norm(t) for t in lb]
        rows = [self._exact.get(k) for k in normed]
        if all(r is not None for r in rows) and len(rows) == len(lb):
            return rows  # type: ignore[return-value]
        # Nearest-neighbor fallback (only when a tolerance is configured).
        if self.tolerance is None or self._sorted is None:
            return None
        targets = pd.DatetimeIndex(normed)
        positions = self._sorted.get_indexer(
            targets, method="nearest", tolerance=self.tolerance
        )
        if (positions < 0).any():
            return None
        return list(self._order[positions])


class SolarForecastPipeline:
    """Load -> scale -> engineer -> fit -> predict harness for solar forecasting.

    A :class:`PhysicalFeatureScaler` (default-on) is fit on the TRAIN frame
    only and applied to both TRAIN and TEST frames BEFORE feature engineering
    — this makes the irradiance ÷ 1500 physical-constant normalization and
    the train-fit NWP z-scoring global across the pipeline, fixing the MLP
    divergence caused by unscaled ``ghi_clearsky`` ∈ [0, ~1100] sitting next
    to a target ∈ [0, 1]. Pass ``scaler=`` explicitly to ablate (e.g. a no-op
    scaler recovers raw, unscaled features); pass ``None`` to use the default
    :class:`PhysicalFeatureScaler`.
    """

    def __init__(
        self,
        config: PipelineConfig,
        feature_engineer: FeatureEngineer,
        model: ForecastModel,
        scaler: PhysicalFeatureScaler | None = None,
    ):
        self.config = config
        self.feature_engineer = feature_engineer
        self.model = model
        # Default-on (fixes MLP); injectable for ablation (raw vs scaled).
        self.scaler = scaler if scaler is not None else PhysicalFeatureScaler()

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

        # Global physical scaling on raw frames BEFORE feature engineering.
        # Fit on TRAIN only (zero leakage) and apply the same stats to TEST.
        # When the frame has no irradiance/NWP cols this is an exact no-op.
        self.scaler.fit(df_train)
        df_train = self.scaler.transform(df_train)
        df_test = self.scaler.transform(df_test)

        lb = self.config.lookback
        hz = self.config.horizon
        st = self.config.stride
        xp_tr, y_tr, t_tr, xf_tr, _, img_tr = self.feature_engineer.create_features(
            df_train, lb, hz, st
        )
        (
            xp_te,
            y_te,
            t_te,
            xf_te,
            cz_issue_te,
            img_te,
        ) = self.feature_engineer.create_features(df_test, lb, hz, st)

        # Positional call so any conforming model works regardless of whether it
        # spells the future-context parameter ``X_future`` or ``Xf``. ``images``
        # is the Phase 6b-T5 multimodal extension: ``None`` for tabular datasets
        # (every model accepts and ignores it); the per-window sky-image cube for
        # image-aware models (CrossViViT).
        self.model.fit(xp_tr, y_tr, t_tr, xf_tr, images=img_tr)
        pred = self.model.predict(xp_te, t_te, xf_te, images=img_te)
        result = {
            "pred": pred,
            "y_true": y_te,
            "times": t_te,
            "model_name": self.model.name,
        }

        # Daytime-window anti-fraud metrics (cos_zenith > 0.05). Computed only
        # when the feature engineer exposed an issue-time cos_zenith per test
        # sample — graceful no-op otherwise (keeps legacy frames working).
        if cz_issue_te is not None:
            from .metrics import daytime_metrics

            result["daytime"] = daytime_metrics(y_te, pred, cz_issue_te)
        return result

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
