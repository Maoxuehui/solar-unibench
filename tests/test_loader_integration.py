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
