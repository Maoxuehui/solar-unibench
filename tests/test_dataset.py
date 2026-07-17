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


def test_dataset_loads_jpeg_rgb_and_computes_brr(tmp_path):
    # Folsom-style JPEG store: parquet meta/ts + JPEGSkyStore (object/VLenBytes cube).
    # BRR is NOT stored; load_sky_frame must compute it from the decoded RGB.
    import numpy as np
    import pandas as pd
    import zarr
    from solarunibench.operators.zarr_io import JPEGSkyStore
    from solarunibench.loader import SolarUniBenchLoader
    d = tmp_path / "folsom"
    d.mkdir()
    n = 10
    times = pd.date_range("2020-06-01 06:00", periods=n, freq="1h", tz="UTC")
    meta = pd.DataFrame([{"entity_id": "E1", "dataset_source": "folsom", "latitude": 38.0,
                          "longitude": -121.0, "elevation_meters": 50.0, "local_timezone": "UTC",
                          "surface_albedo": 0.2, "capacity_kw": np.nan, "slope": np.nan,
                          "azimuth": np.nan, "geo_inferred": False}])
    meta.to_parquet(d / "metadata.parquet", index=False)
    ts = pd.DataFrame({"timestamp_utc": times, "entity_id": "E1",
                       "ghi_actual": np.linspace(0, 800, n), "dni_actual": np.nan, "dhi_actual": np.nan,
                       "pv_power_normalized": np.nan, "ghi_clearsky": np.linspace(0, 900, n),
                       "dni_clearsky": np.nan, "dhi_clearsky": np.nan, "cos_zenith": np.linspace(0, 0.9, n),
                       "solar_azimuth": 90.0, "clearness_index": np.linspace(0, 0.9, n), "quality_flag": 0,
                       "split": ["train"] * (n - 2) + ["val", "test"]})
    ts.to_parquet(d / "time_series.parquet", index=False)
    st = JPEGSkyStore(d / "sky_images.zarr", quality=90, overwrite=True)
    for _ in range(n):
        st.append((np.random.rand(3, 256, 256) * 255).astype(np.uint8))
    zarr.open(str(d / "sky_images.zarr"), mode="a").array(
        "images_timestamps",
        np.array([t.isoformat() for t in times]), chunks=(1,), overwrite=True)
    ld = SolarUniBenchLoader(str(tmp_path))
    ds = SolarUniBenchDataset(ld, ["folsom"], split="train", target="ghi_actual",
                              feature_cols=["ghi_clearsky"], lookback=3, horizon=2, return_images=True)
    s = ds[0]
    assert s["image"].shape == (3, 4, 256, 256) and s["image"].dtype == torch.float32
    # BRR computed on read from JPEG RGB must land in [-1, 1]
    assert float(s["image"][:, 3].min()) >= -1.0 - 1e-6 and float(s["image"][:, 3].max()) <= 1.0 + 1e-6
    # decoded RGB must land in [0, 1]
    assert float(s["image"][:, :3].min()) >= 0.0 - 1e-6 and float(s["image"][:, :3].max()) <= 1.0 + 1e-6
