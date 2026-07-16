import numpy as np
from solarunibench.imaging.sunlake_satellite import extrapolate_context_series


def test_extrapolate_context_series_shape():
    # 5 frames, 2 channels, 8x8; flow is 2*(8,8)
    ctx = np.random.rand(5, 2, 8, 8).astype(np.float32)
    flow = np.random.rand(5, 4, 8, 8).astype(np.float32)  # (vx,vy) per channel pair -> use 2 here
    future, unc = extrapolate_context_series(ctx, flow, channels=[0, 1], steps_k=3)
    assert future.shape == (3, 4, 2, 8, 8)   # K, M-1, C, H, W
    assert unc.shape == (3, 4, 8, 8)          # K, M-1, H, W
    assert np.all(unc >= 0) and np.all(unc <= 1)
