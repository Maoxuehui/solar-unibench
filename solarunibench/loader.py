"""SolarUniBenchLoader: read standardized parquet + lazy zarr, filter by frozen split."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import zarr

_SKY = "sky_images.zarr"
_SAT = "satellite_future.zarr"
# Sentinel for the group cache (a cached None is a valid "store absent" value).
_MISSING = object()


class ConcatenatedSkyCube:
    """Lazy virtual concatenation of multiple per-year sky-image zarr arrays.

    Folsom is materialised into per-year stores (``sky_images_2014.zarr``,
    ``sky_images_2015.zarr``, ``sky_images_2016.zarr``) to avoid cross-year
    overwrite and enable independent checkpoint/resume. This wrapper presents
    them as a single logical cube so downstream code (``load_sky_frame``,
    ``_dataset_has_images``, etc.) is unaware of the partitioning.

    Supports integer indexing ``cube[idx]`` (routing to the correct sub-store
    via cumulative offsets) and exposes ``.shape`` / ``.dtype`` for compatibility
    with :func:`solarunibench.imaging.load.load_sky_frame`.
    """

    def __init__(self, arrays: list):
        assert len(arrays) > 0
        self._arrays = arrays
        self._offsets = np.cumsum([0] + [a.shape[0] for a in arrays[:-1]])
        self._n = sum(a.shape[0] for a in arrays)
        self._shape = (self._n,) + tuple(arrays[0].shape[1:])
        self.dtype = arrays[0].dtype

    @property
    def shape(self):
        return self._shape

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            # Route to the correct sub-array.
            bi = int(np.searchsorted(self._offsets, idx, side="right") - 1)
            local = idx - int(self._offsets[bi])
            return self._arrays[bi][local]
        if isinstance(idx, slice):
            return np.concatenate(
                [a[idx] for a in self._arrays if a.shape[0] > 0]
            ) if idx == slice(None) else self._slice(idx)
        # Fallback: materialise (rare path, e.g. array-like fancy indexing).
        return np.concatenate([a[:] for a in self._arrays])[idx]

    def _slice(self, s):
        start, stop, step = s.indices(self._n)
        return np.stack([self[i] for i in range(start, stop, step)])


class SolarUniBenchLoader:
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        # Memoize opened zarr groups + materialized timestamps. The sky-image
        # timestamps array is chunked (1,) -> one tiny file per frame, so a single
        # np.asarray reads 250k/363k chunk files (minutes on Windows); caching the
        # materialized array makes the common repeated calls (existence checks,
        # per-window image-attach) pay that cost ONCE per dataset per loader.
        self._group_cache: dict[tuple[str, str], object] = {}
        self._ts_cache: dict[str, np.ndarray | None] = {}

    def _path(self, dataset: str) -> Path:
        return self.data_root / dataset

    def load_tabular(self, dataset: str, split: str | None = None) -> pd.DataFrame:
        d = self._path(dataset)
        meta = pd.read_parquet(d / "metadata.parquet")
        ts = pd.read_parquet(d / "time_series.parquet")
        df = ts.merge(meta, on="entity_id", how="left", suffixes=("", "_meta"))
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)
        return df

    def _open(self, dataset: str, file: str):
        key = (dataset, file)
        cached = self._group_cache.get(key, _MISSING)
        if cached is not _MISSING:
            return cached
        p = self._path(dataset) / file
        g = zarr.open(str(p), mode="r") if p.exists() else None
        self._group_cache[key] = g
        return g

    # -- per-year sky store discovery ----------------------------------------

    def _sky_per_year_groups(self, dataset: str) -> list:
        """Return zarr groups for per-year sky stores (sorted), or empty list.

        Per-year stores follow ``sky_images_YYYY.zarr`` (e.g. Folsom 2014-2016).
        Sorted by year for deterministic row ordering.
        """
        d = self._path(dataset)
        per_year = sorted(d.glob("sky_images_20*.zarr"))
        return [zarr.open(str(p), mode="r") for p in per_year]

    def get_tensor_cube(self, dataset: str, kind: str):
        if kind == "sky_images":
            # Prefer per-year stores (Folsom); fall back to single sky_images.zarr.
            per_year = self._sky_per_year_groups(dataset)
            if per_year:
                arrays = [g["images"] for g in per_year if "images" in g]
                if arrays:
                    return arrays[0] if len(arrays) == 1 else ConcatenatedSkyCube(arrays)
            g = self._open(dataset, _SKY)
            return g["images"] if g is not None else None
        if kind == "satellite_context":
            g = self._open(dataset, _SAT)
            return g["context"] if g is not None else None
        if kind == "satellite_future":
            g = self._open(dataset, _SAT)
            return g["future"] if g is not None else None
        if kind == "satellite_uncertainty":
            g = self._open(dataset, _SAT)
            return g["uncertainty_map"] if g is not None else None
        raise ValueError(f"unknown tensor kind: {kind}")

    def get_image_timestamps(self, dataset: str):
        if dataset in self._ts_cache:
            return self._ts_cache[dataset]
        # Prefer per-year stores (Folsom); fall back to single sky_images.zarr.
        per_year = self._sky_per_year_groups(dataset)
        if per_year:
            parts = []
            for g in per_year:
                if "images_timestamps" in g:
                    parts.append(np.asarray(g["images_timestamps"]))
            ts = np.concatenate(parts) if parts else None
        else:
            g = self._open(dataset, _SKY)
            ts = (
                None
                if (g is None or "images_timestamps" not in g)
                else np.asarray(g["images_timestamps"])
            )
        self._ts_cache[dataset] = ts
        return ts
