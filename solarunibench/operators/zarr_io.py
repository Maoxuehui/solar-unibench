"""Zarr tensor I/O for sky images and satellite cubes (zarr 2.x API)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import zarr
from numcodecs import Blosc
from .image_codec import encode_image

# Light, read-friendly compressor for uint8 sky imagery. lz4 decompresses at
# multi-GB/s (faster than reading the extra uncompressed bytes from disk), so it
# helps I/O-bound training rather than hurting it. SHUFFLE (byte) over BITSHUFFLE
# for faster decode. Pass compressor=None to SkyImageZarr for fully uncompressed.
IMAGE_COMPRESSOR = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)


class SkyImageZarr:
    """Append-only (N, 4, 256, 256) uint8 sky-image store, chunked (1,4,256,256).

    ``append`` accepts the float32 frame from ``generate_brr_channel``
    ([R,G,B in 0..1, BRR in -1..1]) and stores it quantized to uint8.
    ``compressor`` defaults to IMAGE_COMPRESSOR (light lz4); pass None for
    uncompressed uint8 (max read speed, larger on disk).
    """

    def __init__(self, path, overwrite: bool = True, compressor=IMAGE_COMPRESSOR):
        self.path = Path(path)
        self.compressor = compressor
        self.store = zarr.DirectoryStore(str(self.path))
        self.root = zarr.group(store=self.store, overwrite=overwrite)
        self._arr = None
        self._n = 0

    def _ensure(self):
        if self._arr is None:
            self._arr = self.root.create_dataset(
                "images", shape=(0, 4, 256, 256), chunks=(1, 4, 256, 256),
                dtype="uint8", compressor=self.compressor, overwrite=True)

    def append(self, frame: np.ndarray) -> None:
        """frame: (4, 256, 256) float32 [RGB 0..1, BRR -1..1] — encoded to uint8."""
        self._ensure()
        u = encode_image(frame)[None, ...]  # (1,4,256,256) uint8
        self._arr.append(u)
        self._n += 1

    @property
    def n(self) -> int:
        return self._n


def write_array(path, name: str, data: np.ndarray, chunks) -> None:
    path = Path(path)
    store = zarr.DirectoryStore(str(path))
    root = zarr.open_group(store=store, mode="a")  # append/open: do NOT wipe existing arrays
    root.array(name, data=np.asarray(data), chunks=chunks, overwrite=True)  # overwrite only THIS array
