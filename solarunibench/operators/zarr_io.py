"""Zarr tensor I/O for sky images and satellite cubes (zarr 2.x API)."""
from __future__ import annotations
import io
from pathlib import Path
import numpy as np
import zarr
from numcodecs import Blosc, VLenBytes
from PIL import Image
from .image_codec import encode_image

# Light, read-friendly compressor for uint8 sky imagery. lz4 decompresses at
# multi-GB/s (faster than reading the extra uncompressed bytes from disk), so it
# helps I/O-bound training rather than hurting it. SHUFFLE (byte) over BITSHUFFLE
# for faster decode. Pass compressor=None to SkyImageZarr for fully uncompressed.
IMAGE_COMPRESSOR = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)


class SkyImageZarr:
    """Append-only (N, 4, H, W) uint8 sky-image store, chunked (1,4,H,W).

    Spatial dims (H, W) are inferred from the first ``append``ed frame — no
    hardcoded resolution. ``append`` accepts the float32 frame from
    ``generate_brr_channel`` ([R,G,B in 0..1, BRR in -1..1]) and stores it
    quantized to uint8. ``compressor`` defaults to IMAGE_COMPRESSOR (light lz4);
    pass None for uncompressed uint8 (max read speed, larger on disk).
    """

    def __init__(self, path, overwrite: bool = True, compressor=IMAGE_COMPRESSOR):
        self.path = Path(path)
        self.compressor = compressor
        self.store = zarr.DirectoryStore(str(self.path))
        self.root = zarr.group(store=self.store, overwrite=overwrite)
        self._arr = None
        self._n = 0

    def _ensure(self, frame: np.ndarray):
        if self._arr is None:
            _, h, w = frame.shape
            self._arr = self.root.create_dataset(
                "images", shape=(0, 4, h, w), chunks=(1, 4, h, w),
                dtype="uint8", compressor=self.compressor, overwrite=True)

    def append(self, frame: np.ndarray) -> None:
        """frame: (4, H, W) float32 [RGB 0..1, BRR -1..1] — encoded to uint8."""
        self._ensure(frame)
        u = encode_image(frame)[None, ...]  # (1,4,H,W) uint8
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


class JPEGSkyStore:
    """Append-only store of Pillow-JPEG-encoded RGB frames (variable-length bytes).

    For photographic sky imagery (Folsom) where lossless compresses poorly.
    Each frame stored as JPEG bytes in a zarr VLenBytes object array ``images``.
    BRR is computed on READ (load_sky_frame), not stored."""

    def __init__(self, path, quality: int = 90, overwrite: bool = True):
        self.path = Path(path); self.quality = quality
        self.store = zarr.DirectoryStore(str(self.path))
        self.root = zarr.group(store=self.store, overwrite=overwrite)
        self._arr = self.root.create_dataset(
            "images", shape=(0,), dtype=object,
            object_codec=VLenBytes(), chunks=(1,), overwrite=True)
        self._n = 0

    @staticmethod
    def _to_hwc_uint8(rgb):
        a = np.asarray(rgb)
        if a.ndim == 3 and a.shape[0] <= 4:        # (C,H,W) -> (H,W,C)
            a = np.transpose(a, (1, 2, 0))
        if a.dtype != np.uint8:
            a = np.clip(a * 255.0, 0, 255).astype(np.uint8) if a.max() <= 1.0 else a.astype(np.uint8)
        return a[..., :3]                           # RGB only
    def append(self, rgb) -> None:
        hwc = self._to_hwc_uint8(rgb)
        buf = io.BytesIO(); Image.fromarray(hwc).save(buf, format="JPEG", quality=self.quality)
        self._arr.append([buf.getvalue()]); self._n += 1
    @property
    def n(self): return self._n
