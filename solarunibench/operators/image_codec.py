# solarunibench/operators/image_codec.py
"""uint8 codec for sky-image zarr storage.

Forward (encode): float32 frame [R,G,B in 0..1, BRR in -1..1] -> uint8 [0,255].
Inverse (decode): uint8 [0,255] -> float32 [R,G,B in 0..1, BRR in -1..1].

Single source of truth for the RGB/BRR quantization maps; both the zarr writer
(``SkyImageZarr``) and the reader (``SolarUniBenchDataset``) import these so the
forward/inverse pair can never drift. Round-trip error <= 1/255 per channel.
"""
from __future__ import annotations
import numpy as np


def encode_image(frame: np.ndarray) -> np.ndarray:
    """(4,H,W) float32 [R,G,B in 0..1, BRR in -1..1] -> (4,H,W) uint8 in [0,255]."""
    f = np.asarray(frame, dtype=np.float32)
    rgb = np.clip(np.rint(f[:3] * 255.0), 0, 255).astype(np.uint8)
    brr = np.clip(np.rint((f[3:4] + 1.0) * 127.5), 0, 255).astype(np.uint8)
    return np.concatenate([rgb, brr], axis=0)


def decode_image(frame: np.ndarray) -> np.ndarray:
    """(4,H,W) uint8 in [0,255] -> (4,H,W) float32 [R,G,B in 0..1, BRR in -1..1]."""
    u = np.asarray(frame)
    rgb = u[:3].astype(np.float32) / 255.0
    brr = (u[3:4].astype(np.float32) / 255.0) * 2.0 - 1.0
    return np.concatenate([rgb, brr], axis=0)
