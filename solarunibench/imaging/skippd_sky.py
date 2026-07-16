"""SKIPPD sky-image -> sky_images.zarr (RGB+BRR, 256x256). Reads default/{train,test} parquet."""
from __future__ import annotations
from pathlib import Path
import io
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from ..operators.brr import generate_brr_channel
from ..operators.zarr_io import SkyImageZarr, write_array


def _decode_resize(img_entry, size=256):
    buf = io.BytesIO(img_entry["bytes"])
    arr = np.array(Image.open(buf).convert("RGB"))
    if arr.shape[:2] != (size, size):
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    return arr


def _to_utc_iso(ts) -> str:
    """ISO-8601 UTC string. SKIPPD `time` is US/Pacific tz-aware; localize
    defensively in case a parquet round-trip dropped the tz."""
    t = pd.Timestamp(ts)
    if t.tz is None:
        t = t.tz_localize("US/Pacific")
    return t.tz_convert("UTC").isoformat()


def build_skippd_sky_zarr(parquet_dir, out_path, limit=500) -> dict:
    """Stream SKIPPD sky-image parquets into sky_images.zarr.

    Writes two arrays into the store:
      - ``images``           (N, 4, 256, 256) float32  (RGB+BRR cube, unchanged)
      - ``images_timestamps`` (N,) <U25               (ISO-8601 UTC per frame)

    The SKIPPD `time` column is US/Pacific tz-aware; it is converted to UTC and
    persisted alongside the cube so Phase 5 cross-modal alignment does not need
    to re-read the parquets.
    """
    parquet_dir = Path(parquet_dir)
    store = SkyImageZarr(out_path, overwrite=True)
    n = 0
    timestamps = []
    done = False
    for split in ("train", "test"):
        if done:
            break
        d = parquet_dir / split
        if not d.exists():
            continue
        for f in sorted(d.glob("*.parquet")):
            if done:
                break
            df = pd.read_parquet(f)
            for img, t in zip(df["image"], df["time"]):
                if img is None or not isinstance(img, dict) or not img.get("bytes"):
                    continue
                frame = generate_brr_channel(_decode_resize(img))
                store.append(frame)
                timestamps.append(_to_utc_iso(t))
                n += 1
                if limit and n >= limit:
                    done = True
                    break
    # Parallel timestamp index (ISO-8601 UTC strings). Additive write keeps the
    # `images` cube untouched; one string per chunk for cheap per-frame reads.
    write_array(out_path, "images_timestamps", np.array(timestamps), chunks=(1,))
    return {"n_frames": n, "out_path": str(out_path)}
