import numpy as np
from solarunibench.operators.brr import generate_brr_channel


def test_output_shape_and_channels():
    img = np.zeros((10, 12, 3), dtype=np.uint8)
    out = generate_brr_channel(img)
    assert out.shape == (4, 10, 12)
    assert out.dtype == np.float32


def test_brr_formula_and_bounds():
    # pure-blue image: BRR = (B-R)/(B+R) = 1 ; pure-red: -1
    blue = np.zeros((4, 4, 3), dtype=np.uint8); blue[..., 2] = 200
    red = np.zeros((4, 4, 3), dtype=np.uint8); red[..., 0] = 200
    ob = generate_brr_channel(blue)[3]
    orn = generate_brr_channel(red)[3]
    assert np.allclose(ob, 1.0, atol=1e-3)
    assert np.allclose(orn, -1.0, atol=1e-3)


def test_rgb_normalized_to_unit():
    img = np.full((2, 2, 3), 255, dtype=np.uint8)
    out = generate_brr_channel(img)
    assert np.allclose(out[0], 1.0) and np.allclose(out[1], 1.0) and np.allclose(out[2], 1.0)
