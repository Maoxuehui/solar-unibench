"""Eq. (4)-(5) — NRBR color ratio & optical-flow reliability extraction.

Canonical physics: ``operators.brr.generate_brr_channel`` (the color-ratio
family) and ``operators.optical_flow`` (forward-splat extrapolation with
uncertainty). This module adds the two feature maps the paper names:
the NRBR map (Eq. 4) and the flow-reliability map (Eq. 5).
"""
from __future__ import annotations

import numpy as np

__all__ = ["nrbr_map", "optical_flow_reliability"]


def nrbr_map(rgb: np.ndarray) -> np.ndarray:
    """Normalized blue-to-red ratio ``(B - R) / (B + R)`` — Eq. (4).

    Args:
        rgb: ``(H, W, 3)`` or ``(T, H, W, 3)`` float/uint8 frames.

    Returns:
        ``(H, W)`` mean map (stack input) or ``(H, W)`` map (single frame);
        positive values are cloud-consistent (blue-dominated sky).
    """
    t = np.asarray(rgb, dtype=np.float32)
    if t.ndim == 3:
        t = t[None, ...]
    if t.shape[-1] < 3:
        raise ValueError(f"need RGB frames, got shape {np.shape(rgb)}")
    r, b = t[..., 0], t[..., 2]
    nrbr = (b - r) / (b + r + 1e-6)
    return nrbr[0] if np.ndim(rgb) == 3 else nrbr.mean(axis=0)


def optical_flow_reliability(frames: np.ndarray,
                             max_pairs: int = 4) -> dict:
    """Farneback flow + per-pixel reliability map — Eq. (5).

    Reliability normalizes the last pair's flow magnitude by the frame-pair
    mean, so coherent cloud motion reads ~1 and sensor noise reads ~0.

    Args:
        frames: ``(T, H, W, 3)`` RGB stack (T >= 2), any float scale.
        max_pairs: cap on consecutive pairs evaluated.

    Returns:
        ``{"flow": (u, v), "reliability": (H, W),
           "flow_mean_magnitude": [per-pair means]}``.
    """
    import cv2

    t = np.asarray(frames, dtype=np.float32)
    if t.ndim != 4 or t.shape[-1] < 3:
        raise ValueError("frames must be (T, H, W, 3) RGB, got "
                         f"{np.shape(frames)}")

    def _gray(f: np.ndarray) -> np.ndarray:
        scaled = (np.clip(f, 0, 1) * 255) if f.max() <= 1.0 else f
        return cv2.cvtColor(scaled.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    gray = [_gray(f) for f in t]
    h, w = gray[0].shape
    u = np.zeros((h, w), np.float32)
    v = np.zeros((h, w), np.float32)
    means: list[float] = []
    for a, b in zip(gray[:-1], gray[1:]):
        fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5,
                                          1.2, 0)
        u, v = fl[..., 0], fl[..., 1]
        means.append(float(np.hypot(u, v).mean()))
        if len(means) >= max_pairs:
            break
    mag = np.hypot(u, v)
    return {"flow": (u, v), "reliability": mag / (mag.mean() + 1e-9),
            "flow_mean_magnitude": means}
