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
        # uint8 NON-trivial data so the decode path is genuinely exercised
        # (zeros made the decode-range test vacuous). 4ch = RGB + BRR (SKIPPD layout).
        g.array("images", data=(np.random.rand(n, 4, 256, 256) * 255).astype(np.uint8),
                chunks=(1, 4, 256, 256), overwrite=True)
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


def _build_per_year_sky(root: Path, dataset="folsom"):
    """Create two per-year sky stores (sky_images_2014.zarr, _2015.zarr)."""
    d = root / dataset
    d.mkdir(parents=True)
    for year, n in [(2014, 5), (2015, 3)]:
        times = pd.date_range(f"{year}-06-01 06:00", periods=n, freq="1h", tz="UTC")
        store = zarr.DirectoryStore(str(d / f"sky_images_{year}.zarr"))
        g = zarr.group(store=store, overwrite=True)
        # JPEG-bytes layout (object dtype, like real Folsom stores)
        import io
        from PIL import Image
        raw = [io.BytesIO() for _ in range(n)]
        for i, buf in enumerate(raw):
            Image.fromarray(
                (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
            ).save(buf, format="JPEG")
        g.array("images", data=np.array([b.getvalue() for b in raw], dtype=object),
                chunks=(n,), object_codec=zarr.VLenBytes(), overwrite=True)
        g.array("images_timestamps", data=np.array([t.isoformat() for t in times]),
                chunks=(n,), overwrite=True)


def test_per_year_sky_concatenation(tmp_path):
    """Per-year sky stores (sky_images_YYYY.zarr) are discovered and virtually
    concatenated by the loader: cube shape = sum, timestamps = ordered concat,
    and indexing across the year boundary routes to the correct sub-store."""
    _build_per_year_sky(tmp_path)
    ld = SolarUniBenchLoader(str(tmp_path))

    cube = ld.get_tensor_cube("folsom", "sky_images")
    assert cube is not None
    assert cube.shape == (8,)  # 5 + 3
    assert cube.dtype == object  # JPEG bytes

    ts = ld.get_image_timestamps("folsom")
    assert ts is not None and len(ts) == 8
    # Ordered: all 2014 timestamps then all 2015
    assert str(ts[0]).startswith("2014-06-01")
    assert str(ts[5]).startswith("2015-06-01")

    # Indexing: frame 0 from 2014, frame 5 from 2015 (boundary)
    b0 = bytes(cube[0])
    assert b0[:2] == b"\xff\xd8"  # JPEG SOI
    b5 = bytes(cube[5])
    assert b5[:2] == b"\xff\xd8"


def test_per_year_fallback_to_single_store(tmp_path):
    """When no per-year stores exist, the loader falls back to sky_images.zarr."""
    _build_pkg(tmp_path, with_sky=True)
    ld = SolarUniBenchLoader(str(tmp_path))
    cube = ld.get_tensor_cube("toy", "sky_images")
    assert cube is not None and cube.shape[0] == 8
