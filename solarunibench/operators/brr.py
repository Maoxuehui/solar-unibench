"""BRR (Blue-to-Red Ratio) channel operator: RGB -> 4-channel [R,G,B,BRR]."""
import numpy as np


def generate_brr_channel(rgb: np.ndarray) -> np.ndarray:
    """rgb: (H, W, 3) uint8 (or float 0-255) -> (4, H, W) float32 [R/255,G/255,B/255,BRR]."""
    img = np.asarray(rgb, dtype=np.float32) / 255.0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    eps = 1e-6
    brr = np.clip((b - r) / (b + r + eps), -1.0, 1.0)
    return np.stack([r, g, b, brr], axis=0).astype(np.float32)
