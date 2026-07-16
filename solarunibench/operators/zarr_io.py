"""Zarr tensor I/O for sky images and satellite cubes (zarr 2.x API)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import zarr


class SkyImageZarr:
    """Append-only (N, 4, 256, 256) float32 sky-image store, chunked (1,4,256,256)."""

    def __init__(self, path, overwrite: bool = True):
        self.path = Path(path)
        self.store = zarr.DirectoryStore(str(self.path))
        self.root = zarr.group(store=self.store, overwrite=overwrite)
        self._arr = None
        self._n = 0

    def _ensure(self):
        if self._arr is None:
            self._arr = self.root.empty("images", shape=(0, 4, 256, 256),
                                        chunks=(1, 4, 256, 256), dtype="float32")

    def append(self, frame: np.ndarray) -> None:
        """frame: (4, 256, 256) float32."""
        self._ensure()
        f = np.asarray(frame, dtype=np.float32)[None, ...]  # (1,4,256,256)
        self._arr.append(f)
        self._n += 1

    @property
    def n(self) -> int:
        return self._n


def write_array(path, name: str, data: np.ndarray, chunks) -> None:
    path = Path(path)
    store = zarr.DirectoryStore(str(path))
    root = zarr.group(store=store, overwrite=True)
    root.array(name, data=np.asarray(data), chunks=chunks, overwrite=True)
