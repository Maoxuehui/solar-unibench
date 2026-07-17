"""Folsom sky-image tarball -> sky_images.zarr. Streams bz2 tar; timestamps from filename (UTC)."""
from __future__ import annotations
import io
import tarfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from ..operators.brr import generate_brr_channel
from ..operators.zarr_io import JPEGSkyStore, write_array
from .resize import downsample_if_larger


def _decode(member_bytes, size=256):
    arr = np.array(Image.open(io.BytesIO(member_bytes)).convert("RGB"))
    return downsample_if_larger(arr, size)


def _ts_from_name(name: str):
    stem = Path(name).stem  # YYYYMMDD_HHMMSS
    return pd.to_datetime(stem, format="%Y%m%d_%H%M%S").tz_localize("UTC")


def iter_folsom_sky_frames(tar_path, size=256):
    """Yield (timestamp_utc, frame(4,size,size)) for every .jpg in the tar.bz2.

    Provided for callers that want pre-materialised 4-channel (RGB+BRR) frames.
    The storage path (:func:`build_folsom_sky_zarr`) no longer uses this — it
    stores JPEG-encoded RGB bytes and computes BRR on read — but the contract
    (4-channel float32 frames) is preserved for backwards compatibility.
    """
    with tarfile.open(tar_path, "r:bz2") as tf:
        for m in tf:
            if not m.isfile() or not m.name.lower().endswith(".jpg"):
                continue
            data = tf.extractfile(m).read()
            yield _ts_from_name(m.name), generate_brr_channel(_decode(data, size))


def _to_utc_iso(ts) -> str:
    """ISO-8601 UTC string from a tz-aware (or UTC-localizable) timestamp."""
    t = pd.Timestamp(ts)
    if t.tz is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("UTC").isoformat()


def build_folsom_sky_zarr(tar_paths, out_path, limit=500) -> dict:
    """Stream Folsom sky tarballs into sky_images.zarr (lossy JPEG RGB bytes).

    Writes two arrays into the store:
      - ``images``            (N,) object / VLenBytes  (Pillow-JPEG-encoded RGB
                            bytes; H,W per resolution policy — native if <=256
                            else downsampled to 256). BRR is NOT stored: it is
                            computed on READ via ``load_sky_frame`` reusing
                            ``generate_brr_channel``. ~12 GB at full Folsom
                            scale (q90 @256²) vs. 106–203 GB lossless uint8.
      - ``images_timestamps`` (N,) <U25               (ISO-8601 UTC per frame)

    The timestamp index is persisted now so Phase 5 cross-modal alignment does
    not need to re-stream the 49 GB tarballs.
    """
    store = JPEGSkyStore(out_path, overwrite=True)
    n = 0
    timestamps = []
    done = False
    for tp in tar_paths:
        if done:
            break
        with tarfile.open(tp, "r:bz2") as tf:
            for m in tf:
                if not m.isfile() or not m.name.lower().endswith(".jpg"):
                    continue
                data = tf.extractfile(m).read()
                rgb = _decode(data)            # (H,W,3) RGB, resolution-policy applied
                store.append(rgb)              # BRR deferred to load_sky_frame
                timestamps.append(_to_utc_iso(_ts_from_name(m.name)))
                n += 1
                if limit and n >= limit:
                    done = True
                    break
    # Parallel timestamp index (ISO-8601 UTC strings). Additive write keeps the
    # `images` array untouched; one string per chunk for cheap per-frame reads.
    write_array(out_path, "images_timestamps", np.array(timestamps), chunks=(1,))
    return {"n_frames": n, "out_path": str(out_path)}
