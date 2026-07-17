# tests/test_jpeg_store.py
import numpy as np, zarr
from numcodecs import VLenBytes
from solarunibench.operators.zarr_io import JPEGSkyStore

def test_jpeg_store_roundtrip(tmp_path):
    st = JPEGSkyStore(tmp_path/"f.zarr", quality=90, overwrite=True)
    rgb = (np.random.rand(3,256,256)*255).astype(np.uint8)
    st.append(rgb); st.append(rgb)
    arr = zarr.open(str(tmp_path/"f.zarr"), mode="r")["images"]
    assert arr.shape == (2,) and arr.dtype == object   # VLenBytes
    b0 = bytes(arr[0]); assert b0[:2] == b"\xff\xd8"    # JPEG SOI marker
    from PIL import Image; import io
    dec = np.array(Image.open(io.BytesIO(b0)))          # (H,W,3)
    assert dec.shape == (256,256,3) and dec.dtype == np.uint8
    assert st.n == 2
