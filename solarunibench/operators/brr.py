"""BRR (Blue-to-Red Ratio) channel operator: RGB -> 4-channel [R,G,B,BRR].

Terminology / provenance
------------------------
The code variable ``brr`` computes the NORMALIZED blue-red channel ratio

    NRBR = (B - R) / (B + R + eps),  eps = 1e-6,  values in [-1, 1],

i.e. lambda_N = (b - r) / (b + r) of Li et al. (2011), J. Atmos. Oceanic
Technol., Eq. 2 — referred to uniformly as **NRBR** in the paper text.
Positive values correspond to clear-sky Rayleigh scattering (blue-dominated
sky); negative values correspond to cloud / aerosol Mie scattering (red
channel lifted). The short code name ``BRR`` is retained for backward
compatibility with the published Zarr feature tensors (the 4th channel of
the SKIPP'D sky-image store) — ``BRR`` and ``NRBR`` denote the same
physical feature.

Aliases: ``compute_nrbr = compute_brr = generate_brr_channel`` — callers
may use whichever name matches their notation without any behavioral
difference.
"""
import numpy as np

__all__ = ["generate_brr_channel", "compute_brr", "compute_nrbr"]


def generate_brr_channel(rgb: np.ndarray) -> np.ndarray:
    """rgb: (H, W, 3) uint8 (or float 0-255) -> (4, H, W) float32 [R/255,G/255,B/255,BRR].

    The 4th channel is the normalized blue-red ratio NRBR = (B - R)/(B + R + 1e-6)
    in [-1, 1] (Li et al., 2011, Eq. 2): positive ~ clear-sky Rayleigh scattering,
    negative ~ cloud Mie scattering. See the module docstring for the BRR/NRBR
    naming note and backward compatibility.
    """
    img = np.asarray(rgb, dtype=np.float32) / 255.0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    eps = 1e-6
    brr = np.clip((b - r) / (b + r + eps), -1.0, 1.0)
    return np.stack([r, g, b, brr], axis=0).astype(np.float32)


#: Canonical aliases (paper notation NRBR / neutral BRR). Same function object —
#: no behavioral difference, kept for caller ergonomics and notation alignment.
compute_brr = generate_brr_channel
compute_nrbr = generate_brr_channel
