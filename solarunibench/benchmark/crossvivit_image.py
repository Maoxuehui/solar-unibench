"""E8 add-on — image-enabled CrossViViT benchmark (folsom / skippd).

The main E8 grid (:func:`solarunibench.benchmark.run_e8.run_benchmark`) runs
CrossViViT TABULAR-ONLY (its ``SolarForecastPipeline`` never attaches sky
images because the plain :class:`EndogenousFeatureEngineer` the runner builds
has ``return_images=False``). Those recorded results are kept as-is. This
module is the additive image-enabled sibling: a dedicated runner that attaches
the per-window sky-image cube to CrossViViT's fit/predict and scores it with
the SAME daytime-masked evaluator + SmartPersistence reference, so the Skill%
is directly comparable with the main grid.

Why not reuse ``EndogenousFeatureEngineer(return_images=True)``
--------------------------------------------------------------
Its image path materializes every window's FULL lookback of NATIVE-resolution
frames — ``(n, 240, 4, 256, 256)`` float32 ≈ 3.2 GB **per window** on folsom.
Two size reductions make the image path tractable without changing the model:

* **Decode-once cache.** Stride-30 folsom windows share most of their
  lookback; every distinct cube row is JPEG-decoded + BRR-computed ONCE and
  cached as a resized ``(4, img_size, img_size)`` float32 tile. Consecutive
  windows then reuse the cache (the sliding window moves by 30 minutes =
  30 frames; a 240-min lookback keeps ~87% overlap).
* **Lookback subsampling (``image_stride``).** CrossViViT's image tokenizer
  POOLS over the image-frame axis (mean over lookback — see
  ``_CrossViViTNet._img_tokens``), so the image sequence length is
  architecturally free. We take every ``image_stride``-th frame
  (lookback-anchored, i.e. frames ``lookback-1, lookback-1-s, ...`` down to 0):
  1-min cadence → stride 30 gives one frame per 30 min (8 frames over a 4-h
  lookback), which matches the cloud-evolution timescale at 5–15 min horizons.

Result shape per window: ``(lb_img, 4, img_size, img_size)`` float32 —
``(8, 4, 32, 32)`` ≈ 130 KB for folsom's ~31k train windows ≈ 4 GB RAM (fits
the 512 GB workstation; the GPU only ever sees a micro-batch of 16).

Window discipline
-----------------
Same as the main grid: stride-30 windowing of 1-min data (FULL mode), exact-ts
image match preferred with nearest-neighbour-within-tolerance fallback
(default 120 s; folsom images are 1-min cadence with ~0.15% gaps). A window
whose (subsampled) image lookback cannot be fully resolved is DROPPED — for
both the model AND the SmartPersistence reference — so the skill stays
apples-to-apples on the identical (daytime-masked) window subset.

folsom sky shards
-----------------
At this branch folsom ships PER-YEAR sky zarrs (``sky_images_2014.zarr`` ...
``2016``) and the loader does not concatenate them, so
:func:`run_crossvivit_image_benchmark` probes for the shards itself and wraps
them in :class:`MultiShardSkyCube` (a read-only concatenated view). A dataset
with a single ``sky_images.zarr`` (skippd) goes through the plain loader API.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd

from .config import DATASET_CONFIG
from .evaluator import evaluate_predictions
from .features import add_clearness_aligned
from .run_e8 import _FEATURE_COLS, _full_stride
from ..forecast.core import PipelineConfig, SolarForecastPipeline
from ..forecast.models import CrossViViTModel, SmartPersistenceModel
from ..forecast.scaling import PhysicalFeatureScaler
from ..imaging.load import load_sky_frame
from ..imaging.resize import resize_frames

__all__ = [
    "MultiShardSkyCube",
    "run_crossvivit_image_benchmark",
    "save_crossvivit_image_results",
]

# Model name recorded in the results CSV (distinct from tabular "CrossViViT").
MODEL_NAME = "CrossViViT-image"

# Default CSV written by :func:`save_crossvivit_image_results`.
RESULTS_FILENAME = "e8_crossvivit_image.csv"

# Default sky-image match tolerance. folsom/skippd images are ~1-min cadence;
# 120 s covers the odd dropped frame without bridging real data holes.
DEFAULT_TOLERANCE = "120s"

# Default image-lookback subsampling stride, in tabular sampling steps. On the
# 1-min image datasets this = one sky frame per 30 minutes (8 frames over the
# 240-step lookback), matching the FULL-mode windowing stride.
DEFAULT_IMAGE_STRIDE = 30


class _Loader(Protocol):
    """Structural type for the loader dependency (injectable for tests)."""

    def load_tabular(self, dataset: str, split: str | None = None) -> pd.DataFrame: ...

    def get_tensor_cube(self, dataset: str, kind: str) -> Any: ...

    def get_image_timestamps(self, dataset: str) -> np.ndarray | None: ...


# ---------------------------------------------------------------------------
# folsom per-year shard concatenation
# ---------------------------------------------------------------------------


class MultiShardSkyCube:
    """Read-only concatenated view over per-year sky-image cube shards.

    Wraps e.g. folsom's ``sky_images_2014.zarr`` ... ``sky_images_2016.zarr``
    ``images`` arrays behind one indexable ``len`` / ``dtype`` / ``__getitem__``
    interface (everything :func:`solarunibench.imaging.load.load_sky_frame`
    needs). Indexing routes to the owning shard with a running offset — no
    data is copied or reordered (shards are chronological).
    """

    def __init__(self, shards: list[Any]):
        if not shards:
            raise ValueError("MultiShardSkyCube needs at least one shard")
        self._shards = list(shards)
        self._offsets = np.cumsum([0] + [len(s) for s in self._shards])
        self.dtype = self._shards[0].dtype
        mismatched = [
            i for i, s in enumerate(self._shards[1:], start=1) if s.dtype != self.dtype
        ]
        if mismatched:
            raise ValueError(
                f"sky shard dtypes disagree (shard indices {mismatched} differ "
                f"from shard 0's {self.dtype}); refusing to concatenate."
            )

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def __getitem__(self, idx: int):
        if idx < 0:  # mirror plain-array negative indexing
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(f"sky-cube index {idx} out of range [0, {len(self)})")
        # Bisect the running offsets (few shards → linear scan is fine).
        for s_i in range(len(self._shards)):
            if idx < self._offsets[s_i + 1]:
                return self._shards[s_i][idx - self._offsets[s_i]]
        raise IndexError(idx)  # unreachable; guarded above


def _resolve_sky_cube(loader: _Loader, dataset: str, data_root: str | None) -> tuple[Any, np.ndarray | None]:
    """Return ``(cube, image_timestamps)`` for ``dataset`` (shards aware).

    Prefers the plain loader API (single ``sky_images.zarr``, e.g. skippd).
    When that yields no cube and ``data_root`` is given, probes for folsom-style
    PER-YEAR shards (``sky_images_*.zarr``) and concatenates them. Returns
    ``(None, None)`` when the dataset has no sky imagery at all.
    """
    cube = loader.get_tensor_cube(dataset, "sky_images")
    img_ts = loader.get_image_timestamps(dataset)
    if cube is not None and img_ts is not None:
        return cube, img_ts
    if data_root is None:
        return None, None
    import zarr

    shard_dir = Path(data_root) / dataset
    shard_paths = sorted(shard_dir.glob("sky_images_*.zarr"))
    if not shard_paths:
        return None, None
    try:
        groups = [zarr.open(str(p), mode="r") for p in shard_paths]
        cubes = [g["images"] for g in groups]
        stamps = np.concatenate(
            [np.asarray(g["images_timestamps"]) for g in groups]
        )
    except KeyError:
        return None, None
    return MultiShardSkyCube(cubes), stamps


# ---------------------------------------------------------------------------
# Image-attaching feature engineer
# ---------------------------------------------------------------------------


class _ImageIndex:
    """Resolve tabular lookback timestamps to sky-cube rows.

    Same matching discipline as ``EndogenousFeatureEngineer``/``SolarUniBenchDataset``:
    exact (second-floored, naive-UTC) match preferred; nearest-neighbour within
    ``tolerance`` on a miss. Built once per (dataset, split) from the cube's
    timestamp array.
    """

    @staticmethod
    def _norm(t) -> pd.Timestamp:
        ts = pd.Timestamp(str(t))
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts.floor("s")

    def __init__(self, img_ts: np.ndarray, tolerance: pd.Timedelta | None):
        self.tolerance = tolerance
        arr = np.asarray(img_ts)
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
        """One cube row per lookback timestamp, or None on any miss."""
        lb = np.asarray(lb_ts)
        if len(lb) == 0:
            return None
        normed = [self._norm(t) for t in lb]
        rows = [self._exact.get(k) for k in normed]
        if all(r is not None for r in rows) and len(rows) == len(lb):
            return rows  # type: ignore[return-value]
        if self.tolerance is None or self._sorted is None:
            return None
        positions = self._sorted.get_indexer(
            pd.DatetimeIndex(normed), method="nearest", tolerance=self.tolerance
        )
        if (positions < 0).any():
            return None
        return list(self._order[positions])


class ImageWindowEngineer:
    """Sliding-window engineer that attaches a subsampled sky-image cube.

    Produces the SAME tabular blocks as ``EndogenousFeatureEngineer``
    (``X_past[n, lb, Fp]`` target-first + ``_FEATURE_COLS``, ``Y``, ``times``,
    ``X_future``, issue-time ``cos_zenith``) — windowing is row-order based and
    stride-identical, so a run at the main grid's lookback/stride yields the
    SAME windows — plus ``images[n, lb_img, 4, img_size, img_size]`` float32.

    Windows whose subsampled image lookback cannot be fully resolved (a data
    hole larger than the tolerance) are DROPPED from BOTH the returned blocks,
    keeping the model and the persistence reference on the identical subset.

    Frames are decoded once per distinct cube row (JPEG→RGB→BRR for folsom;
    uint8→float for skippd) and cached at the model's target size.
    """

    def __init__(
        self,
        target: str,
        cube: Any,
        img_ts: np.ndarray,
        *,
        img_size: int,
        image_stride: int = DEFAULT_IMAGE_STRIDE,
        tolerance: pd.Timedelta | str | None = DEFAULT_TOLERANCE,
        feature_cols: list[str] | None = None,
    ):
        self.target = target
        self.feature_cols: list[str] = list(feature_cols or _FEATURE_COLS)
        self.cube = cube
        self.img_size = img_size
        if image_stride <= 0:
            raise ValueError(f"image_stride must be positive (got {image_stride})")
        self.image_stride = image_stride
        self.tolerance = (
            pd.Timedelta(tolerance) if tolerance is not None else None
        )
        self._index = _ImageIndex(img_ts, self.tolerance)
        # row -> resized (4, img_size, img_size) float32 tile
        self._tile_cache: dict[int, np.ndarray] = {}

    def _tile(self, row: int) -> np.ndarray:
        """Decode + resize one cube row (cached)."""
        tile = self._tile_cache.get(row)
        if tile is None:
            frame = load_sky_frame(self.cube, row)  # (4, H, W) float32
            tile = resize_frames(frame[None], (self.img_size, self.img_size))[0]
            self._tile_cache[row] = tile
        return tile

    def create_features(
        self, df: pd.DataFrame, lookback: int, horizon: int, stride: int = 1
    ) -> dict:
        """Return ``{X_past, Y, times, X_future, cos_zenith_issue, images, kept}``.

        ``images`` is ``(n, lb_img, 4, img_size, img_size)`` float32; ``kept``
        is the boolean per-window coverage mask (True = image lookback fully
        resolved). Tabular blocks carry ALL windows (pre-drop) so callers can
        align; the dropped-window discipline is applied via ``images``/``kept``
        (rows where ``kept`` is False have a zero-filled image row and are
        excluded by :func:`run_crossvivit_image_benchmark`).
        """
        if lookback <= 0 or horizon <= 0 or stride <= 0:
            raise ValueError(
                f"lookback, horizon, stride must be positive "
                f"(got {lookback}, {horizon}, {stride})"
            )
        for col in ("timestamp_utc", "entity_id", self.target):
            if col not in df.columns:
                raise ValueError(f"dataframe must contain a '{col}' column")

        # Subsampled image-frame positions within a lookback, lookback-anchored
        # (ends at the issue time; the most recent frames matter most).
        img_pos = list(range(lookback - 1, -1, -self.image_stride))[::-1]

        past_cols = [self.target] + [c for c in self.feature_cols if c in df.columns]
        future_cols = ["ghi_clearsky", "cos_zenith"]
        have_future = all(c in df.columns for c in future_cols)
        have_cz = "cos_zenith" in df.columns

        df_sorted = df.sort_values(["entity_id", "timestamp_utc"], kind="mergesort")
        xp_w, y_w, t_w, xf_w, cz_w = [], [], [], [], []
        img_w: list[np.ndarray] = []
        kept: list[bool] = []
        for _, g in df_sorted.groupby("entity_id", sort=False):
            g = g.reset_index(drop=True)
            n = len(g)
            past = g[past_cols].to_numpy(dtype=float)
            tgt = g[self.target].to_numpy(dtype=float)
            ts = pd.DatetimeIndex(g["timestamp_utc"]).to_numpy()
            fut = g[future_cols].to_numpy(dtype=float) if have_future else None
            cz_col = g["cos_zenith"].to_numpy(dtype=float) if have_cz else None

            last_start = n - lookback - horizon
            for start in range(0, last_start + 1, stride):
                hs = slice(start + lookback, start + lookback + horizon)
                rows = self._index.rows_for(ts[start : start + lookback : 1][img_pos])
                if rows is None:
                    kept.append(False)
                    img_w.append(
                        np.zeros((len(img_pos), 4, self.img_size, self.img_size), np.float32)
                    )
                else:
                    kept.append(True)
                    img_w.append(np.stack([self._tile(r) for r in rows]))
                xp_w.append(past[start : start + lookback])
                y_w.append(tgt[hs])
                t_w.append(ts[start + lookback])
                if fut is not None:
                    xf_w.append(fut[hs])
                if cz_col is not None:
                    cz_w.append(cz_col[start + lookback])

        n_w = len(xp_w)
        kept_arr = np.asarray(kept, dtype=bool)
        return {
            "X_past": np.stack(xp_w, axis=0) if n_w else np.empty((0, lookback, len(past_cols))),
            "Y": np.stack(y_w, axis=0) if n_w else np.empty((0, horizon)),
            "times": np.stack(t_w, axis=0) if n_w else np.empty((0,), dtype="datetime64[ns]"),
            "X_future": (
                np.stack(xf_w, axis=0)
                if n_w and have_future
                else (np.empty((0, horizon, len(future_cols))) if have_future else None)
            ),
            "cos_zenith_issue": (
                np.stack(cz_w, axis=0).astype(float)
                if n_w and have_cz
                else (np.empty((0,)) if have_cz else None)
            ),
            "images": (
                np.stack(img_w, axis=0).astype(np.float32)
                if n_w
                else np.empty((0, len(img_pos), 4, self.img_size, self.img_size), np.float32)
            ),
            "kept": kept_arr,
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_crossvivit_image_benchmark(
    loader: _Loader,
    datasets: list[str] | None = None,
    horizons: list[int] | None = None,
    model: CrossViViTModel | None = None,
    *,
    full: bool = True,
    stride: int | None = None,
    lookback: int | None = None,
    image_stride: int = DEFAULT_IMAGE_STRIDE,
    tolerance: pd.Timedelta | str | None = DEFAULT_TOLERANCE,
    data_root: str | None = None,
) -> list[dict]:
    """Run the image-enabled CrossViViT benchmark over image datasets.

    For every ``(dataset, horizon)`` cell: build the SAME windows as the main
    E8 grid (same lookback/stride/split frames), attach the subsampled
    sky-image cube, fit a fresh deep-copy of ``model`` with images, predict
    with images, and score with the daytime-masked evaluator against
    SmartPersistence run on the SAME (image-covered) windows. Rows carry
    ``model='CrossViViT-image'``.

    Parameters
    ----------
    loader
        Object exposing the ``SolarUniBenchLoader`` tabular + sky-image APIs.
    datasets
        Datasets to evaluate. ``None`` → the image-bearing datasets in
        :data:`DATASET_CONFIG` (folsom, skippd).
    horizons
        Horizon steps. ``None`` → each dataset's ``cfg.horizons_steps``.
    model
        CrossViViT template (deep-copied fresh per cell). ``None`` → the
        FULL-config preset (epochs=50, patience=7, AdamW+Cosine, effective
        batch 64 with accum 4 → micro 16, device cuda).
    full
        Use the FULL-mode windowing stride (:func:`run_e8._full_stride`) when
        ``stride`` is not given (default True — this runner exists for the
        FULL-config image run).
    stride, lookback
        Explicit windowing stride / lookback (tests). Defaults: per-dataset
        FULL stride and ``cfg.lookback_steps``.
    image_stride
        Image-frame subsampling stride, in tabular sampling steps (see
        :class:`ImageWindowEngineer`).
    tolerance
        Sky-image nearest-neighbour match tolerance (default 120 s; ``None``
        = exact match only).
    data_root
        Standardized-data root, used to probe for folsom-style per-year sky
        shards (``sky_images_*.zarr``) when the loader reports no single cube.

    Returns
    -------
    list of dict
        One row per cell with keys ``dataset``, ``horizon``, ``model``,
        ``MAE``, ``RMSE``, ``Skill_MAE_pct``, ``Skill_RMSE_pct``,
        ``n_daytime``.
    """
    if datasets is None:
        datasets = [d for d, c in DATASET_CONFIG.items() if c.image_dataset]
    results: list[dict] = []
    for dataset in datasets:
        cfg = DATASET_CONFIG[dataset]
        cube, img_ts = _resolve_sky_cube(loader, dataset, data_root)
        if cube is None or img_ts is None:
            # No sky imagery → skip (no rows), never crash the grid.
            continue

        df_train = add_clearness_aligned(loader.load_tabular(dataset, "train"), cfg.target)
        df_test = add_clearness_aligned(loader.load_tabular(dataset, "test"), cfg.target)

        ds_stride = stride if stride is not None else (_full_stride(cfg.sampling_mins) if full else 1)
        lb = lookback if lookback is not None else cfg.lookback_steps
        hz_list = horizons if horizons is not None else list(cfg.horizons_steps)

        # Fresh template per cell (no fit-state leakage), like run_benchmark.
        template = model if model is not None else _build_full_crossvivit()
        if not isinstance(template, CrossViViTModel):
            raise TypeError(
                f"run_crossvivit_image_benchmark needs a CrossViViTModel "
                f"(got {type(template).__name__})"
            )

        # One engineer per dataset: its decode-once tile cache is shared across
        # the cells' identical train/test engineering passes and all horizons
        # (the image lookback does not depend on the horizon).
        import copy as _copy

        engineer = ImageWindowEngineer(
            cfg.target,
            cube,
            img_ts,
            img_size=template.img_size,
            image_stride=image_stride,
            tolerance=tolerance,
            feature_cols=list(_FEATURE_COLS),
        )
        for hz in hz_list:
            cell_model = _copy.deepcopy(template)
            row = _run_image_cell(
                dataset=dataset,
                cfg=cfg,
                horizon=hz,
                df_train=df_train,
                df_test=df_test,
                engineer=engineer,
                model=cell_model,
                lookback=lb,
                stride=ds_stride,
            )
            if row is not None:
                results.append(row)
    return results


def _build_full_crossvivit() -> CrossViViTModel:
    """FULL-config CrossViViT template (mirrors run_e8._build_full_models)."""
    from .run_e8 import (
        _FULL_ACCUM_CROSSVIVIT,
        _FULL_BATCH,
        _FULL_EPOCHS,
        _FULL_LR,
        _FULL_PATIENCE,
        _FULL_VAL_FRACTION,
        _FULL_WD,
    )

    return CrossViViTModel(
        batch_size=_FULL_BATCH,
        accum_steps=_FULL_ACCUM_CROSSVIVIT,
        epochs=_FULL_EPOCHS,
        lr=_FULL_LR,
        weight_decay=_FULL_WD,
        patience=_FULL_PATIENCE,
        val_fraction=_FULL_VAL_FRACTION,
        device="cuda",
        # The image engineer ALREADY subsamples the lookback to one frame per
        # image_stride steps (stride 30 → 8 frames over the 4-h lookback); the
        # model must keep ALL of them (its in-model img_frame_stride would
        # otherwise thin 8 frames down to 1 and destroy the temporal axis).
        img_frame_stride=1,
    )


def _run_image_cell(
    *,
    dataset: str,
    cfg,
    horizon: int,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    engineer: ImageWindowEngineer,
    model: CrossViViTModel,
    lookback: int,
    stride: int,
) -> dict | None:
    """Fit + predict + evaluate ONE (dataset, horizon) image cell, or None."""
    # Physical scaling: fit on train only, apply to both (identical to the
    # main grid's SolarForecastPipeline setup so numbers are comparable).
    scaler = PhysicalFeatureScaler()
    scaler.fit(df_train)
    tr = engineer.create_features(scaler.transform(df_train), lookback, horizon, stride)
    te = engineer.create_features(scaler.transform(df_test), lookback, horizon, stride)

    keep_tr = tr["kept"]
    keep_te = te["kept"]
    if not keep_te.any() or not keep_tr.any():
        return None  # no image-covered windows → skip the cell

    def _pack(blocks: dict, keep: np.ndarray) -> dict:
        return {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 else v)
                for k, v in blocks.items()}

    tr_k, te_k = _pack(tr, keep_tr), _pack(te, keep_te)

    # Reference FIRST on the SAME windows (skill is apples-to-apples).
    ref = SmartPersistenceModel()
    ref.fit(tr_k["X_past"], tr_k["Y"], tr_k["times"], tr_k["X_future"])
    persistence_pred = ref.predict(te_k["X_past"], te_k["times"], te_k["X_future"])
    y_true = te_k["Y"]
    cz_issue = te_k["cos_zenith_issue"]
    if cz_issue is None:
        raise ValueError(
            f"dataset '{dataset}' has no cos_zenith column; the E8 evaluator "
            f"requires it for the daytime mask."
        )

    # CrossViViT with images fed to fit AND predict (the load-bearing bit).
    model.fit(
        tr_k["X_past"], tr_k["Y"], tr_k["times"], tr_k["X_future"],
        images=tr_k["images"],
    )
    pred = model.predict(
        te_k["X_past"], te_k["times"], te_k["X_future"], images=te_k["images"]
    )

    metrics = evaluate_predictions(y_true, pred, cz_issue, persistence_pred)
    return {
        "dataset": dataset,
        "horizon": horizon,
        "model": MODEL_NAME,
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "Skill_MAE_pct": metrics["Skill_MAE_pct"],
        "Skill_RMSE_pct": metrics["Skill_RMSE_pct"],
        "n_daytime": metrics["n_daytime"],
    }


def save_crossvivit_image_results(rows: list[dict], out_dir: str | Path) -> Path:
    """Write ``e8_crossvivit_image.csv`` under ``out_dir``; return its path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / RESULTS_FILENAME
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
