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


def _build_pkg_offset_sky(root: Path, dataset="folsom_off", n=8, with_split=True,
                          offset_seconds=10, reverse_store=False):
    """Tabular hourly at :00:00; sky images hourly but offset by ``offset_seconds``.

    Mirrors the real Folsom mismatch (second-cadence images vs hourly tabular):
    no tabular timestamp exactly equals an image timestamp, so exact-match
    attaches ~0 while nearest-neighbor within tolerance attaches everything.
    Image ``i`` (chronological) is filled with the constant ``i*10`` so we can
    assert the nearest mapping picks the *correct* frame per lookback step.

    ``reverse_store=True`` writes the cube + timestamp array in reverse
    chronological order (cube row 0 = latest image). This is *not* how ingestion
    writes, but it stress-tests that nearest-neighbor remaps back to the right
    cube row rather than into sort-order positions.
    """
    import zarr
    d = root / dataset
    d.mkdir(parents=True)
    times = pd.date_range("2020-06-01 06:00", periods=n, freq="1h", tz="UTC")
    img_times = times + pd.Timedelta(seconds=offset_seconds)
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
    # chronological image i carries constant value i*10
    imgs = [np.full((4, 64, 64), i * 10, np.uint8) for i in range(n)]
    itimes = [t.isoformat() for t in img_times]
    if reverse_store:                      # cube row 0 = latest image
        imgs = imgs[::-1]
        itimes = itimes[::-1]
    store = zarr.DirectoryStore(str(d / "sky_images.zarr"))
    g = zarr.group(store=store, overwrite=True)
    g.array("images", data=np.stack(imgs), chunks=(1, 4, 64, 64), overwrite=True)
    g.array("images_timestamps", data=np.array(itimes), chunks=(1,), overwrite=True)


def test_dataset_exact_match_misses_when_images_offset(tmp_path):
    # Folsom-style: image timestamps offset from tabular -> exact-match attaches nothing.
    _build_pkg_offset_sky(tmp_path, offset_seconds=10)
    ld = SolarUniBenchLoader(str(tmp_path))
    ds = SolarUniBenchDataset(ld, ["folsom_off"], split="train", target="ghi_actual",
                              feature_cols=["ghi_clearsky"], lookback=3, horizon=2, return_images=True)
    assert ds[0]["image"] is None  # default match_tolerance=None -> exact only


def test_dataset_nearest_neighbor_attach_within_tolerance(tmp_path):
    # With match_tolerance, the offset images attach via nearest-neighbor; and the
    # mapping is the CORRECT frame (image i is constant i*10 -> decoded RGB i*10/255).
    _build_pkg_offset_sky(tmp_path, offset_seconds=10)
    ld = SolarUniBenchLoader(str(tmp_path))
    ds = SolarUniBenchDataset(ld, ["folsom_off"], split="train", target="ghi_actual",
                              feature_cols=["ghi_clearsky"], lookback=3, horizon=2, return_images=True,
                              match_tolerance=pd.Timedelta(minutes=30))
    s = ds[0]
    assert s["image"] is not None
    assert s["image"].shape == (3, 4, 64, 64)
    # lookback steps [t0,t1,t2] nearest-map to images [0,1,2]; RGB channel 0 = i*10/255
    for k in range(3):
        assert abs(float(s["image"][k, 0].mean()) - k * 10 / 255.0) < 1e-5


def test_dataset_nearest_neighbor_outside_tolerance_attaches_none(tmp_path):
    # Offset (10s) beyond a tight 5s tolerance -> nearest-neighbor still misses -> None.
    _build_pkg_offset_sky(tmp_path, offset_seconds=10)
    ld = SolarUniBenchLoader(str(tmp_path))
    ds = SolarUniBenchDataset(ld, ["folsom_off"], split="train", target="ghi_actual",
                              feature_cols=["ghi_clearsky"], lookback=3, horizon=2, return_images=True,
                              match_tolerance=pd.Timedelta(seconds=5))
    assert ds[0]["image"] is None


def test_dataset_nearest_neighbor_remapped_for_unsorted_store(tmp_path):
    # Cube stored REVERSE-chronologically but timestamps match each frame. NN must
    # remap to the correct cube row: lookback [06:00,07:00,08:00] -> images with
    # constants 10,20,30 (regardless of their physical cube position).
    _build_pkg_offset_sky(tmp_path, offset_seconds=10, reverse_store=True)
    ld = SolarUniBenchLoader(str(tmp_path))
    ds = SolarUniBenchDataset(ld, ["folsom_off"], split="train", target="ghi_actual",
                              feature_cols=["ghi_clearsky"], lookback=3, horizon=2, return_images=True,
                              match_tolerance=pd.Timedelta(minutes=30))
    s = ds[0]
    assert s["image"] is not None
    for k in range(3):
        assert abs(float(s["image"][k, 0].mean()) - k * 10 / 255.0) < 1e-5


def test_dataset_aligned_exact_match_unaffected_by_tolerance(tmp_path):
    # Backward-compat: aligned images (SKIPPD-style) still attach with tolerance set;
    # tolerance is a fallback on miss, never overrides a working exact match.
    _build_pkg(tmp_path, dataset="aligned", n=10, with_split=True, with_sky=True)
    ld = SolarUniBenchLoader(str(tmp_path))
    ds = SolarUniBenchDataset(ld, ["aligned"], split="train", target="ghi_actual",
                              feature_cols=["ghi_clearsky"], lookback=3, horizon=2, return_images=True,
                              match_tolerance=pd.Timedelta(minutes=30))
    s = ds[0]
    assert s["image"] is not None and s["image"].shape == (3, 4, 256, 256)


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
