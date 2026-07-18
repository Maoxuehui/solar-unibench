"""SolarUniBenchLoader: read standardized parquet + lazy zarr, filter by frozen split."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import zarr

_SKY = "sky_images.zarr"
_SAT = "satellite_future.zarr"


class SolarUniBenchLoader:
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)

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
        p = self._path(dataset) / file
        if not p.exists():
            return None
        return zarr.open(str(p), mode="r")

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
        g = self._open(dataset, _SKY)
        if g is None or "images_timestamps" not in g:
            return None
        return np.asarray(g["images_timestamps"])
