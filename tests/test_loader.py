import numpy as np
import pandas as pd
import zarr
from pathlib import Path
from solarunibench.loader import SolarUniBenchLoader


def _build_pkg(root: Path, dataset="toy", n=8, with_split=True, with_sky=False):
    d = root / dataset
    d.mkdir(parents=True)
    times = pd.date_range("2020-06-01 06:00", periods=n, freq="1h", tz="UTC")
    meta = pd.DataFrame([{"entity_id": "E1", "dataset_source": dataset, "latitude": 38.0,
                          "longitude": -121.0, "elevation_meters": 50.0, "local_timezone": "UTC",
                          "surface_albedo": 0.2, "capacity_kw": np.nan, "slope": np.nan,
                          "azimuth": np.nan, "geo_inferred": False}])
    meta.to_parquet(d / "metadata.parquet", index=False)
    ts = pd.DataFrame({"timestamp_utc": times, "entity_id": "E1",
                       "ghi_actual": np.linspace(0, 800, n), "dni_actual": np.nan, "dhi_actual": np.nan,
                       "pv_power_normalized": np.nan, "ghi_clearsky": np.linspace(0, 900, n),
                       "dni_clearsky": np.nan, "dhi_clearsky": np.nan, "cos_zenith": np.linspace(0, 0.9, n),
                       "solar_azimuth": 90.0, "clearness_index": np.linspace(0, 0.9, n), "quality_flag": 0})
    if with_split:
        ts["split"] = ["train"] * (n - 2) + ["val", "test"]
    ts.to_parquet(d / "time_series.parquet", index=False)
    if with_sky:
        store = zarr.DirectoryStore(str(d / "sky_images.zarr"))
        g = zarr.group(store=store, overwrite=True)
        g.array("images", data=np.zeros((n, 4, 256, 256), np.float32), chunks=(1, 4, 256, 256), overwrite=True)
        g.array("images_timestamps", data=np.array([t.isoformat() for t in times]), chunks=(1,), overwrite=True)


def test_load_tabular_merges_metadata_and_filters_split(tmp_path):
    _build_pkg(tmp_path, with_split=True)
    ld = SolarUniBenchLoader(str(tmp_path))
    allrows = ld.load_tabular("toy")
    assert len(allrows) == 8
    assert "latitude" in allrows.columns  # metadata merged in
    train = ld.load_tabular("toy", split="train")
    assert (train["split"] == "train").all() and len(train) == 6
    test = ld.load_tabular("toy", split="test")
    assert len(test) == 1


def test_get_tensor_cube_absent_returns_none(tmp_path):
    _build_pkg(tmp_path, with_sky=False)
    ld = SolarUniBenchLoader(str(tmp_path))
    assert ld.get_tensor_cube("toy", "sky_images") is None
    assert ld.get_image_timestamps("toy") is None


def test_get_tensor_cube_present(tmp_path):
    _build_pkg(tmp_path, with_sky=True)
    ld = SolarUniBenchLoader(str(tmp_path))
    cube = ld.get_tensor_cube("toy", "sky_images")
    assert cube is not None and cube.shape == (8, 4, 256, 256)
    ts = ld.get_image_timestamps("toy")
    assert ts is not None and len(ts) == 8
