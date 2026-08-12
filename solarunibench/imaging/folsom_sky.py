"""Folsom sky-image tarball -> sky_images.zarr. Streams bz2 tar; timestamps from filename (UTC)."""
from __future__ import annotations
import io
import tarfile
import time
import faulthandler
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

faulthandler.enable()  # print a Python-level traceback on native segfault (bad JPEG)
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


def build_folsom_sky_zarr(tar_paths, out_path, limit=500, progress_every=10_000,
                          resume: bool = True, checkpoint_every: int = 1000) -> dict:
    """Stream Folsom sky tarballs into sky_images.zarr (lossy JPEG RGB bytes).

    Writes two arrays into the store:
      - ``images``            (N,) object / VLenBytes  (Pillow-JPEG-encoded RGB
                            bytes; H,W per resolution policy — native if <=256
                            else downsampled to 256). BRR is NOT stored: it is
                            computed on READ via ``load_sky_frame`` reusing
                            ``generate_brr_channel``. ~12 GB at full Folsom
                            scale (q90 @256²) vs. 106–203 GB lossless uint8.
      - ``images_timestamps`` (N,) <U25               (ISO-8601 UTC per frame)

    Robustness / observability for long unattended runs (Phase 8):
      - ``faulthandler`` is enabled module-wide: a native segfault (e.g. a
        malformed JPEG in libjpeg-turbo) prints a Python stack instead of a
        silent death.
      - Each tar member name is written to a ``*.LAST_MEMBER`` sidecar BEFORE
        its decode, so a crash identifies the culprit frame.
      - Per-frame ``try/except`` logs+skips soft decode failures (a hard
        segfault still kills the process — see sidecar).
      - ``progress_every`` prints frame count, elapsed, fps, skipped count.

    Checkpoint / resume (``resume=True``, the default):
      - On launch, checks if ``out_path`` already has an ``images`` array. If so,
        opens it in APPEND mode (``overwrite=False``) and skips the already-
        processed tar members (fast — tar iteration without decode is ~1000x
        faster than JPEG decode). The zarr frame count IS the checkpoint.
      - Timestamps are checkpointed to a ``.timestamps.txt`` sidecar every
        ``checkpoint_every`` frames so a mid-run kill doesn't lose them.
      - On successful completion, the timestamps are written into the zarr as
        ``images_timestamps`` and the sidecar is cleaned up.
    """
    ts_sidecar = Path(str(out_path) + ".timestamps.txt")

    # --- Resume or fresh start ---
    existing_n = 0
    if resume and (out_path / ".zgroup").exists():
        store = JPEGSkyStore(out_path, overwrite=False)
        existing_n = store.n
        if existing_n > 0:
            print(f"[folsom_sky] RESUME: {existing_n} frames already in zarr, "
                  f"skipping that many tar members...", flush=True)
        # Load existing timestamps: prefer zarr (clean exit wrote them there),
        # fall back to sidecar (killed mid-run before write_array).
        import zarr as _zg_mod
        _zg = _zg_mod.open(str(out_path), mode="r")
        if "images_timestamps" in _zg and _zg["images_timestamps"].shape[0] == existing_n:
            timestamps = list(_zg["images_timestamps"][:])
        elif ts_sidecar.exists():
            timestamps = [l for l in ts_sidecar.read_text().split("\n") if l]
        else:
            timestamps = []
    else:
        store = JPEGSkyStore(out_path, overwrite=True)
        timestamps = []

    n = existing_n
    skipped = 0
    members_seen = 0
    members_to_skip = existing_n
    done = False
    t0 = time.time()
    last_path = Path(str(out_path) + ".LAST_MEMBER")
    target = "ALL" if not limit else str(limit)
    print(f"[folsom_sky] start: {len(tar_paths)} tarball(s), limit={target}, "
          f"resume={resume}, existing={existing_n}, -> {out_path}", flush=True)
    for tp in tar_paths:
        if done:
            break
        with tarfile.open(tp, "r:bz2") as tf:
            for m in tf:
                if not m.isfile() or not m.name.lower().endswith(".jpg"):
                    continue
                members_seen += 1
                if members_seen <= members_to_skip:
                    continue                  # fast-skip already-processed
                last_path.write_text(m.name)
                try:
                    data = tf.extractfile(m).read()
                    rgb = _decode(data)
                    store.append(rgb)
                    timestamps.append(_to_utc_iso(_ts_from_name(m.name)))
                    n += 1
                except Exception as e:
                    skipped += 1
                    print(f"[folsom_sky] SKIP {m.name}: {type(e).__name__}: {e}",
                          flush=True)
                    continue
                # Checkpoint: persist timestamps to sidecar every N frames.
                if checkpoint_every and (n % checkpoint_every == 0):
                    ts_sidecar.write_text("\n".join(timestamps))
                if progress_every and (n % progress_every == 0):
                    el = time.time() - t0
                    rate = (n - existing_n) / el if el > 0 else 0.0
                    print(f"[folsom_sky] {n} frames | {tp.name} | {el:.0f}s | "
                          f"{rate:.1f} fps | skipped={skipped}", flush=True)
                if limit and n >= limit:
                    done = True
                    break
    # Write timestamps array into zarr (final, additive).
    write_array(out_path, "images_timestamps", np.array(timestamps), chunks=(1,))
    last_path.unlink(missing_ok=True)
    ts_sidecar.unlink(missing_ok=True)       # clean up checkpoint sidecar
    print(f"[folsom_sky] DONE: {n} frames, skipped={skipped} -> {out_path}",
          flush=True)
    return {"n_frames": n, "out_path": str(out_path), "skipped": skipped}
