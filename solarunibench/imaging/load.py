"""Unified sky-frame loader: any storage -> (4,H,W) float32 [RGB 0..1, BRR -1..1]."""
from __future__ import annotations
import io
import numpy as np
from PIL import Image
from ..operators.brr import generate_brr_channel
from ..operators.image_codec import decode_image


def load_sky_frame(cube, idx) -> np.ndarray:
    """Return one (4,H,W) float32 frame from a sky-image cube, regardless of storage.

    Branches on storage layout:

    - JPEG bytes (object/VLenBytes array, Folsom): Pillow-decode the RGB bytes,
      then compute BRR via :func:`generate_brr_channel` (BRR is not stored).
    - uint8 4-channel (RGB+BRR, SKIPPD): :func:`decode_image` reads the stored BRR.
    - uint8 3-channel (RGB only, defensive): :func:`generate_brr_channel` adds BRR.

    The BRR map lives in exactly one place (``operators.brr``); this loader never
    re-implements it.
    """
    if cube.dtype == object:                          # JPEG bytes (Folsom)
        rgb = np.array(Image.open(io.BytesIO(bytes(cube[idx]))).convert("RGB"))
        return generate_brr_channel(rgb)              # (4,H,W) float32 incl. BRR
    arr = np.asarray(cube[idx])
    if arr.shape[0] == 4:                              # uint8 RGB+BRR (SKIPPD)
        return decode_image(arr)
    if arr.ndim == 3 and arr.shape[0] == 3:           # uint8 RGB only (C,H,W) -> HWC -> +BRR
        return generate_brr_channel(np.transpose(arr, (1, 2, 0)))
    raise ValueError(f"unrecognized sky-image layout: dtype={arr.dtype}, shape={arr.shape}")
