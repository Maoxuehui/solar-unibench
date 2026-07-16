"""Folsom sky-image tarball -> sky_images.zarr. Streams bz2 tar; timestamps from filename (UTC)."""
from __future__ import annotations
import io
import tarfile
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from ..operators.brr import generate_brr_channel
from ..operators.zarr_io import SkyImageZarr


def _decode(member_bytes, size=256):
    arr = np.array(Image.open(io.BytesIO(member_bytes)).convert("RGB"))
    if arr.shape[:2] != (size, size):
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    return arr


def _ts_from_name(name: str):
    stem = Path(name).stem  # YYYYMMDD_HHMMSS
    return pd.to_datetime(stem, format="%Y%m%d_%H%M%S").tz_localize("UTC")


def iter_folsom_sky_frames(tar_path, size=256):
    """Yield (timestamp_utc, frame(4,size,size)) for every .jpg in the tar.bz2."""
    with tarfile.open(tar_path, "r:bz2") as tf:
        for m in tf:
            if not m.isfile() or not m.name.lower().endswith(".jpg"):
                continue
            data = tf.extractfile(m).read()
            yield _ts_from_name(m.name), generate_brr_channel(_decode(data, size))


def build_folsom_sky_zarr(tar_paths, out_path, limit=500) -> dict:
    store = SkyImageZarr(out_path, overwrite=True)
    n = 0
    for tp in tar_paths:
        for _ts, frame in iter_folsom_sky_frames(tp):
            store.append(frame)
            n += 1
            if limit and n >= limit:
                return {"n_frames": n, "out_path": str(out_path)}
    return {"n_frames": n, "out_path": str(out_path)}
