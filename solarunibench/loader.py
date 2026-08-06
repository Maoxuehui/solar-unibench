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

    def get_tensor_cube(self, dataset: str, kind: str):
        if kind == "sky_images":
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
        g = self._open(dataset, _SKY)
        ts = (
            None
            if (g is None or "images_timestamps" not in g)
            else np.asarray(g["images_timestamps"])
        )
        self._ts_cache[dataset] = ts
        return ts
