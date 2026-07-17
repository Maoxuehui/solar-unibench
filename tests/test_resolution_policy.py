# tests/test_resolution_policy.py
import numpy as np, zarr
from solarunibench.imaging.resize import downsample_if_larger
from solarunibench.operators.zarr_io import SkyImageZarr

def test_downsample_only_when_larger():
    assert downsample_if_larger(np.zeros((1536,1536,3),np.uint8),256).shape[:2]==(256,256)
    assert downsample_if_larger(np.zeros((64,64,3),np.uint8),256).shape[:2]==(64,64)   # unchanged
    assert downsample_if_larger(np.zeros((256,256,3),np.uint8),256).shape[:2]==(256,256)

def test_sky_image_zarr_infers_size(tmp_path):
    s = SkyImageZarr(tmp_path/"a.zarr", overwrite=True)
    s.append(np.zeros((4,64,64),np.float32))
    assert zarr.open(str(tmp_path/"a.zarr"),mode="r")["images"].shape == (1,4,64,64)
