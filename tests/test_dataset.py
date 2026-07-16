import numpy as np
import pandas as pd
import torch
from pathlib import Path
from solarunibench.loader import SolarUniBenchLoader
from solarunibench.dataset import SolarUniBenchDataset
from tests.test_loader import _build_pkg


def test_dataset_windows_tabular_and_counts(tmp_path):
    _build_pkg(tmp_path, dataset="toy", n=10, with_split=True)
    ld = SolarUniBenchLoader(str(tmp_path))
    ds = SolarUniBenchDataset(ld, ["toy"], split="train", target="ghi_actual",
                              feature_cols=["ghi_clearsky", "cos_zenith"], lookback=3, horizon=2, stride=1)
    # train split has 8 rows; samples = 8 - lookback - horizon + 1 = 4
    assert len(ds) == 4
    s = ds[0]
    assert isinstance(s["features"], torch.Tensor) and s["features"].shape == (3, 3)  # 2 feats + target lag
    assert s["target"].shape == (2,)
    assert s["dataset"] == "toy"
    assert s["image"] is None  # no sky_images


def test_dataset_attaches_image_when_present(tmp_path):
    _build_pkg(tmp_path, dataset="toy", n=10, with_split=True, with_sky=True)
    ld = SolarUniBenchLoader(str(tmp_path))
    ds = SolarUniBenchDataset(ld, ["toy"], split="train", target="ghi_actual",
                              feature_cols=["ghi_clearsky"], lookback=3, horizon=2, return_images=True)
    s = ds[0]
    assert s["image"] is not None and s["image"].shape == (3, 4, 256, 256)


def test_dataset_mixes_multiple_datasets(tmp_path):
    _build_pkg(tmp_path, dataset="a", n=10)
    _build_pkg(tmp_path, dataset="b", n=10)
    ld = SolarUniBenchLoader(str(tmp_path))
    ds = SolarUniBenchDataset(ld, ["a", "b"], split=None, target="ghi_actual",
                              feature_cols=["ghi_clearsky"], lookback=3, horizon=2)
    assert len(ds) == 2 * (10 - 3 - 2 + 1)
    seen = {ds[i]["dataset"] for i in range(len(ds))}
    assert seen == {"a", "b"}


def test_dataset_decodes_uint8_image_to_float_ranges(tmp_path):
    import numpy as np
    _build_pkg(tmp_path, dataset="toy", n=10, with_split=True, with_sky=True)
    ld = SolarUniBenchLoader(str(tmp_path))
    ds = SolarUniBenchDataset(ld, ["toy"], split="train", target="ghi_actual",
                              feature_cols=["ghi_clearsky"], lookback=3, horizon=2, return_images=True)
    s = ds[0]
    assert s["image"] is not None and s["image"].dtype == torch.float32
    assert s["image"].shape == (3, 4, 256, 256)
    assert float(s["image"][:, 3].min()) >= -1.0 - 1e-6 and float(s["image"][:, 3].max()) <= 1.0 + 1e-6
    assert float(s["image"][:, :3].min()) >= 0.0 - 1e-6 and float(s["image"][:, :3].max()) <= 1.0 + 1e-6


def test_dataset_raises_on_missing_target_column(tmp_path):
    _build_pkg(tmp_path, dataset="toy", n=10, with_split=True)
    ld = SolarUniBenchLoader(str(tmp_path))
    import pytest
    with pytest.raises(ValueError, match="missing required column"):
        SolarUniBenchDataset(ld, ["toy"], split="train", target="no_such_column",
                             feature_cols=["ghi_clearsky"], lookback=3, horizon=2)
