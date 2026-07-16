import numpy as np
import zarr
from solarunibench.operators.zarr_io import SkyImageZarr, write_array


def test_sky_zarr_append_and_read(tmp_path):
    z = SkyImageZarr(tmp_path / "sky.zarr", overwrite=True)
    frame = np.random.rand(4, 256, 256).astype(np.float32)
    z.append(frame)
    z.append(frame)
    arr = zarr.open(str(tmp_path / "sky.zarr"), mode="r")["images"]
    assert arr.shape == (2, 4, 256, 256)
    assert arr.chunks == (1, 4, 256, 256)
    assert arr.dtype == np.float32


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
