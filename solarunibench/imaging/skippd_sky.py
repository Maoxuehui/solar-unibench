"""SKIPPD sky-image -> sky_images.zarr (RGB+BRR, 256x256). Reads default/{train,test} parquet."""
from __future__ import annotations
from pathlib import Path
import io
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from ..operators.brr import generate_brr_channel
from ..operators.zarr_io import SkyImageZarr


def _decode_resize(img_entry, size=256):
    buf = io.BytesIO(img_entry["bytes"])
    arr = np.array(Image.open(buf).convert("RGB"))
    if arr.shape[:2] != (size, size):
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    return arr


def build_skippd_sky_zarr(parquet_dir, out_path, limit=500) -> dict:
    parquet_dir = Path(parquet_dir)
    store = SkyImageZarr(out_path, overwrite=True)
    n = 0
    for split in ("train", "test"):
        d = parquet_dir / split
        if not d.exists():
            continue
        for f in sorted(d.glob("*.parquet")):
            df = pd.read_parquet(f)
            for img in df["image"]:
                if img is None or not isinstance(img, dict) or not img.get("bytes"):
                    continue
                frame = generate_brr_channel(_decode_resize(img))
                store.append(frame)
                n += 1
                if limit and n >= limit:
                    return {"n_frames": n, "out_path": str(out_path)}
    return {"n_frames": n, "out_path": str(out_path)}
