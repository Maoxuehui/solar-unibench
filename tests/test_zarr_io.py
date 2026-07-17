import numpy as np
import zarr
from solarunibench.operators.zarr_io import SkyImageZarr, write_array, IMAGE_COMPRESSOR


def test_sky_zarr_append_and_read(tmp_path):
    z = SkyImageZarr(tmp_path / "sky.zarr", overwrite=True)
    frame = np.random.rand(4, 256, 256).astype(np.float32)
    z.append(frame)
    z.append(frame)
    arr = zarr.open(str(tmp_path / "sky.zarr"), mode="r")["images"]
    assert arr.shape == (2, 4, 256, 256)
    assert arr.chunks == (1, 4, 256, 256)
    assert arr.dtype == np.uint8


def test_write_array(tmp_path):
    data = np.arange(2 * 3, dtype=np.float32).reshape(2, 3)
    write_array(tmp_path / "x.zarr", "ctx", data, chunks=(1, 3))
    arr = zarr.open(str(tmp_path / "x.zarr"), mode="r")["ctx"]
    assert arr.shape == (2, 3)
    np.testing.assert_array_equal(arr[:], data)


def test_write_array_multiple_arrays_same_store(tmp_path):
    p = tmp_path / "sat.zarr"
    write_array(p, "context", np.zeros((2, 3), np.float32), chunks=(1, 3))
    write_array(p, "future", np.ones((4, 2, 3), np.float32), chunks=(1, 1, 3))
    root = zarr.open(str(p), mode="r")
    assert "context" in root and "future" in root
    assert root["context"].shape == (2, 3) and root["future"].shape == (4, 2, 3)


def test_sky_image_zarr_stores_uint8_with_blosc(tmp_path):
    store = SkyImageZarr(tmp_path / "sky.zarr", overwrite=True)
    frame = np.zeros((4, 256, 256), np.float32); frame[0] = 1.0  # R=1 -> 255
    store.append(frame); store.append(frame)
    arr = zarr.open(str(tmp_path / "sky.zarr"), mode="r")["images"]
    assert arr.shape == (2, 4, 256, 256) and arr.dtype == np.uint8
    assert arr.compressor == IMAGE_COMPRESSOR  # Blosc lz4 SHUFFLE
    assert arr[0, 0, 0, 0] == 255 and store.n == 2


def test_sky_image_zarr_uncompressed_option(tmp_path):
    store = SkyImageZarr(tmp_path / "sky2.zarr", overwrite=True, compressor=None)
    store.append(np.zeros((4, 256, 256), np.float32))
    arr = zarr.open(str(tmp_path / "sky2.zarr"), mode="r")["images"]
    assert arr.dtype == np.uint8 and arr.compressor is None  # uncompressed
