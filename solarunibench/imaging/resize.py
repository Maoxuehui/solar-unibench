"""Resolution policy: keep native if <= target, else downsample (never upsample)."""
from __future__ import annotations
import cv2, numpy as np

def downsample_if_larger(rgb_hwc: np.ndarray, target: int = 256) -> np.ndarray:
    h, w = rgb_hwc.shape[:2]
    if h <= target and w <= target:
        return rgb_hwc
    return cv2.resize(rgb_hwc, (target, target), interpolation=cv2.INTER_AREA)


def resize_frames(frames, target_hw):
    """Resize a batch of (4,H,W) float sky frames to ``target_hw``.

    Lets a multimodal model batch datasets whose native image sizes differ
    (Folsom 256x256, SKIPPD 64x64) by resizing each to a common target on load.
    All 4 RGB+BRR channels are resized channel-wise; resize is range-agnostic so
    RGB in [0,1] and BRR in [-1,1] both pass through untouched in value range.

    Interpolation follows OpenCV guidance: INTER_AREA for downsampling (the
    common 256->64 case; area-averaging preserves content), INTER_LINEAR
    (bilinear) for upsampling. Identity when the input is already the target.

    frames: numpy ndarray (N,4,H,W) or torch.Tensor (N,4,H,W), float.
    target_hw: (H', W') pair of ints, or a single int for a square target.
    Returns: same type as ``frames``, shape (N,4,H',W').
    """
    if isinstance(target_hw, (int, np.integer)):
        target_hw = (int(target_hw), int(target_hw))
    th, tw = int(target_hw[0]), int(target_hw[1])

    is_torch = type(frames).__name__ == "Tensor"
    arr = np.asarray(frames.detach() if hasattr(frames, "detach") else frames)
    if arr.ndim != 4 or arr.shape[1] != 4:
        raise ValueError(f"resize_frames expects (N,4,H,W); got shape {arr.shape}")
    n, _, h, w = arr.shape
    if (h, w) == (th, tw):
        return frames  # identity; nothing to do

    interp = cv2.INTER_AREA if (th <= h and tw <= w) else cv2.INTER_LINEAR
    out = np.empty((n, 4, th, tw), dtype=arr.dtype)
    for i in range(n):
        for c in range(4):
            out[i, c] = cv2.resize(arr[i, c], (tw, th), interpolation=interp)
    if is_torch:
        import torch
        return torch.from_numpy(np.ascontiguousarray(out))
    return out
