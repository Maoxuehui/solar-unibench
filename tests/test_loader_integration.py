"""Real-data integration: build small standardized packages, load + iterate the Dataset. Slow."""
import os
import subprocess
import sys
import pytest
import pandas as pd
from pathlib import Path

DATA = os.environ.get("SOLAR_UNIBENCH_DATA_ROOT", "D:/monica/workspace/solar_benchmark_dataset")
slow = pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="set RUN_SLOW=1")


@slow
def test_gefcom_load_and_iterate(tmp_path):
    from solarunibench.pipeline.adapters.gefcom import GEFCom2014Pipeline
    from solarunibench.splits import build
    from solarunibench.loader import SolarUniBenchLoader
    from solarunibench.dataset import SolarUniBenchDataset
    import yaml
    GEFCom2014Pipeline(raw_data_dir=f"{DATA}/GEFCom2014_data", output_dir=str(tmp_path),
                       dataset_name="gefcom2014").run_pipeline(limit=2000)
    cfg = {"strategy": "temporal", "test_start": "2014-06-01", "val_ratio": 0.1}
    build(tmp_path / "gefcom2014", cfg, "1.0")
    ld = SolarUniBenchLoader(str(tmp_path))
    df = ld.load_tabular("gefcom2014", split="train")
    assert len(df) > 0 and (df["split"] == "train").all()
    ds = SolarUniBenchDataset(ld, ["gefcom2014"], split="train", target="pv_power_normalized",
                              lookback=4, horizon=2)
    assert len(ds) > 0
    s = ds[0]
    assert s["features"].shape[0] == 4 and s["target"].shape[0] == 2
    assert s["image"] is None  # GEFCom has no sky images


@slow
def test_skippd_dataset_with_sky_images(tmp_path):
    from solarunibench.pipeline.adapters.skippd import SKIPPDPipeline
    from solarunibench.loader import SolarUniBenchLoader
    from solarunibench.dataset import SolarUniBenchDataset
    SKIPPDPipeline(raw_data_dir=f"{DATA}/SKIPPD_all", output_dir=str(tmp_path),
                   dataset_name="skippd", config={"image_limit": 80}).run_pipeline(limit=2000)
    ld = SolarUniBenchLoader(str(tmp_path))
    # sky_images.zarr should exist (image_limit>0)
    assert ld.get_tensor_cube("skippd", "sky_images") is not None
    ds = SolarUniBenchDataset(ld, ["skippd"], split=None, target="pv_power_normalized",
                              lookback=3, horizon=2, return_images=True)
    assert len(ds) > 0
    n_with_img = sum(1 for i in range(min(len(ds), 50)) if ds[i]["image"] is not None)
    assert n_with_img > 0  # at least some samples attach sky images


@slow
def test_skippd_sky_zarr_uint8_native64_and_dataset_decode(tmp_path):
    """SKIPPD sky cube is stored lossless uint8 (N,4,64,64) at native resolution;
    SolarUniBenchDataset decodes it to float (lookback,4,64,64) in valid ranges."""
    import numpy as np
    import torch
    from solarunibench.pipeline.adapters.skippd import SKIPPDPipeline
    from solarunibench.loader import SolarUniBenchLoader
    from solarunibench.dataset import SolarUniBenchDataset
    SKIPPDPipeline(raw_data_dir=f"{DATA}/SKIPPD_all", output_dir=str(tmp_path),
                   dataset_name="skippd", config={"image_limit": 80}).run_pipeline(limit=2000)
    ld = SolarUniBenchLoader(str(tmp_path))
    cube = ld.get_tensor_cube("skippd", "sky_images")
    assert cube is not None
    # NEW compact format: uint8 RGB+BRR cube, native 64x64 (64 <= 256 -> kept, not upsampled)
    assert cube.dtype == np.uint8
    assert cube.shape[1:] == (4, 64, 64)
    ds = SolarUniBenchDataset(ld, ["skippd"], split=None, target="pv_power_normalized",
                              lookback=3, horizon=2, return_images=True)
    assert len(ds) > 0
    img = next((ds[i]["image"] for i in range(min(len(ds), 100))
                if ds[i]["image"] is not None), None)
    assert img is not None, "no SKIPPD sample attached a sky image"
    # Decoded lookback window: float32 (lookback,4,64,64); RGB in [0,1], BRR in [-1,1]
    assert img.shape == (3, 4, 64, 64) and img.dtype == torch.float32
    assert float(img[:, :3].min()) >= 0.0 - 1e-6 and float(img[:, :3].max()) <= 1.0 + 1e-6
    assert float(img[:, 3].min()) >= -1.0 - 1e-6 and float(img[:, 3].max()) <= 1.0 + 1e-6


@slow
def test_folsom_sky_zarr_jpeg_bytes_and_load_sky_frame_decode(tmp_path):
    """Folsom sky cube is stored as JPEG-q90 RGB bytes (object/VLenBytes array);
    load_sky_frame decodes -> (4,256,256) float with BRR computed on load.

    NOTE: the Dataset's timestamp-keyed image attach is exercised end-to-end on
    real JPEG bytes by tests/test_dataset.py::test_dataset_loads_jpeg_rgb_and_computes_brr
    (synthetic package). Here we validate the REAL adapter output: the on-disk
    JPEG format + load_sky_frame decoding a real Folsom frame. We do not drive
    this through SolarUniBenchDataset because the real Folsom sky-image timestamps
    (second-cadence over a few days) share ZERO exact timestamps with the hourly
    irradiance time_series -- a pre-existing data characteristic unrelated to the
    storage format, so no sample would attach via the Dataset on the real package.
    """
    import numpy as np
    from solarunibench.pipeline.adapters.folsom import FolsomPipeline
    from solarunibench.loader import SolarUniBenchLoader
    from solarunibench.imaging.load import load_sky_frame
    FolsomPipeline(raw_data_dir=f"{DATA}/Folsom_data", output_dir=str(tmp_path),
                   dataset_name="folsom", config={"image_limit": 60}).run_pipeline(limit=2000)
    ld = SolarUniBenchLoader(str(tmp_path))
    cube = ld.get_tensor_cube("folsom", "sky_images")
    assert cube is not None
    # NEW compact format: object array of Pillow-JPEG-encoded RGB bytes (VLenBytes)
    assert cube.dtype == object and cube.shape[0] >= 1
    raw = bytes(cube[0])
    assert raw[:2] == b"\xff\xd8"  # JPEG SOI marker -> genuinely JPEG bytes
    # load_sky_frame must decode JPEG -> (4,256,256) float32 and compute BRR on load.
    # Resolution policy downsamples the native 1536x1536 source to 256x256.
    frame = load_sky_frame(cube, 0)
    assert frame.shape == (4, 256, 256) and frame.dtype == np.float32
    assert float(frame[:3].min()) >= 0.0 - 1e-6 and float(frame[:3].max()) <= 1.0 + 1e-6
    assert float(frame[3].min()) >= -1.0 - 1e-6 and float(frame[3].max()) <= 1.0 + 1e-6
