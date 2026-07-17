"""Resolution policy: keep native if <= target, else downsample (never upsample)."""
from __future__ import annotations
import cv2, numpy as np

def downsample_if_larger(rgb_hwc: np.ndarray, target: int = 256) -> np.ndarray:
    h, w = rgb_hwc.shape[:2]
    if h <= target and w <= target:
        return rgb_hwc
    return cv2.resize(rgb_hwc, (target, target), interpolation=cv2.INTER_AREA)
