# tests/test_image_codec.py
import numpy as np
from solarunibench.operators.image_codec import encode_image, decode_image


def _frame(rgb, brr):
    # rgb: tuple 3 floats in [0,1]; brr: float in [-1,1]; build a (4,2,2) frame
    r, g, b = rgb
    ch = lambda v: np.full((2, 2), v, np.float32)
    return np.stack([ch(r), ch(g), ch(b), ch(brr)], axis=0)


def test_encode_dtype_range_and_shape():
    u = encode_image(_frame((0.0, 0.5, 1.0), -1.0))
    assert u.dtype == np.uint8 and u.shape == (4, 2, 2)
    assert u[0, 0, 0] == 0 and u[1, 0, 0] == 128 and u[2, 0, 0] == 255  # RGB
    assert u[3, 0, 0] == 0  # brr=-1 -> 0


def test_decode_recovers_ranges():
    f = decode_image(np.full((4, 2, 2), 255, np.uint8))
    assert f.dtype == np.float32
    assert abs(f[0, 0, 0] - 1.0) < 1e-6          # rgb 255 -> 1.0
    assert abs(f[3, 0, 0] - 1.0) < 1e-6          # brr 255 -> 1.0
    f0 = decode_image(np.zeros((4, 2, 2), np.uint8))
    assert abs(f0[3, 0, 0] - (-1.0)) < 1e-6      # brr 0 -> -1.0


def test_roundtrip_within_tolerance():
    rng = np.random.default_rng(0)
    f = rng.uniform(-1.0, 1.0, (4, 8, 8)).astype(np.float32)
    f[:3] = (f[:3] + 1.0) / 2.0  # make first 3 channels valid RGB in [0,1]
    back = decode_image(encode_image(f))
    assert back.shape == f.shape and back.dtype == np.float32
    assert np.max(np.abs(back - f)) <= 1.0 / 255.0 + 1e-6
