"""E8 add-on — CrossViViT IMAGE benchmark (multimodal runner + CSV).

``run_crossvivit_image_benchmark`` is the image-enabled sibling of
``run_benchmark``: for the image-bearing datasets (folsom / skippd) it attaches
the per-window sky-image cube to CrossViViT's fit/predict and scores it with
the SAME daytime-masked SmartPersistence-skill evaluator. The tabular-only
``run_benchmark`` path and its recorded results are untouched.

These tests use a synthetic mock loader with tiny sky images so they run in
seconds (no real data). The load-bearing properties:

* images ARE fed to CrossViViT fit AND predict (not silently dropped),
* windows without image coverage are dropped (all-or-nothing per window),
* SmartPersistence is the reference on the SAME windows (skill comparable),
* the ``e8_crossvivit_image.csv`` row is written with the exact schema,
* a dataset without sky images is skipped (no rows) instead of crashing,
* the folsom per-year sky shards concatenate into one cube view.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solarunibench.benchmark.crossvivit_image import (
    MultiShardSkyCube,
    run_crossvivit_image_benchmark,
    save_crossvivit_image_results,
)
from solarunibench.forecast.models import CrossViViTModel

_ROW_KEYS = {
    "dataset",
    "horizon",
    "model",
    "MAE",
    "RMSE",
    "Skill_MAE_pct",
    "Skill_RMSE_pct",
    "n_daytime",
}


# -- synthetic image-bearing dataset ------------------------------------------


def _synth_frame(target_col: str, n_rows: int, seed: int) -> pd.DataFrame:
    """1-min-cadence all-daytime frame (bell-driven), 70/30 train/test."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2020-06-01 06:00", periods=n_rows, freq="1min", tz="UTC")
    bell = np.clip(np.sin(np.pi * np.arange(n_rows) / 720.0), 0.1, None)
    ghi_clear = bell * 950.0
    if target_col == "pv_power_normalized":
        target = np.clip(bell * 0.8 + rng.normal(0, 0.03, n_rows), 0, 1)
    else:
        target = bell * 900.0 + rng.normal(0, 20, n_rows)
    naive = times.tz_convert("UTC").tz_localize(None)
    n_tr = int(n_rows * 0.7)
    splits = ["train"] * n_tr + ["test"] * (n_rows - n_tr)
    return pd.DataFrame(
        {
            "timestamp_utc": naive,
            "entity_id": "E0",
            target_col: target.astype(float),
            "ghi_clearsky": ghi_clear.astype(float),
            "cos_zenith": bell.astype(float),
            "split": splits,
        }
    )


class _MockLoader:
    """Mimics SolarUniBenchLoader's tabular + sky-image APIs."""

    def __init__(self, frames: dict[str, pd.DataFrame], images: dict[str, tuple]):
        self._frames = frames
        self._images = images  # dataset -> (cube, img_ts)

    def load_tabular(self, dataset: str, split: str | None = None) -> pd.DataFrame:
        df = self._frames[dataset]
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)
        return df.copy()

    def get_tensor_cube(self, dataset: str, kind: str):
        if kind != "sky_images" or dataset not in self._images:
            return None
        return self._images[dataset][0]

    def get_image_timestamps(self, dataset: str):
        if dataset not in self._images:
            return None
        return self._images[dataset][1]


class _SpyCrossViViT(CrossViViTModel):
    """CrossViViT that records whether fit/predict received an image cube.

    The runner deep-copies the template per cell, so records go to CLASS-level
    state (shared across the copies) — ``_SpyCrossViViT.reset()`` clears it.
    """

    fit_images_shape: tuple | None = None
    pred_images_shape: tuple | None = None

    @classmethod
    def reset(cls) -> None:
        cls.fit_images_shape = None
        cls.pred_images_shape = None

    def fit(self, X_past, Y, times=None, X_future=None, *, images=None):
        if images is not None:
            _SpyCrossViViT.fit_images_shape = tuple(images.shape)
        return super().fit(X_past, Y, times, X_future, images=images)

    def predict(self, X_past, times=None, X_future=None, *, images=None):
        if images is not None:
            _SpyCrossViViT.pred_images_shape = tuple(images.shape)
        return super().predict(X_past, times, X_future, images=images)


def _tiny_model() -> _SpyCrossViViT:
    return _SpyCrossViViT(
        d_model=16,
        n_heads=4,
        depth=1,
        patch_len=4,
        img_size=8,
        img_patch=4,
        epochs=1,
        batch_size=16,
        device="cpu",
    )


def _image_bundle(frame: pd.DataFrame, seed: int, offset_s: int = 11):
    """uint8 (n,4,8,8) cube + ISO image timestamps offset by ``offset_s``."""
    rng = np.random.default_rng(seed)
    ts = pd.DatetimeIndex(frame["timestamp_utc"])
    img_ts = (ts + pd.Timedelta(seconds=offset_s)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    cube = rng.integers(0, 256, size=(len(frame), 4, 8, 8), dtype=np.uint8)
    return cube, img_ts.to_numpy()


@pytest.fixture
def image_loader() -> tuple[_MockLoader, _SpyCrossViViT]:
    frame = _synth_frame("ghi_actual", 400, seed=0)
    cube, img_ts = _image_bundle(frame, seed=1)
    loader = _MockLoader({"folsom": frame}, {"folsom": (cube, img_ts)})
    return loader, _tiny_model()


# -- the image path feeds images to fit/predict -------------------------------


def test_images_fed_to_fit_and_predict(image_loader) -> None:
    loader, model = image_loader
    _SpyCrossViViT.reset()
    rows = run_crossvivit_image_benchmark(
        loader,
        datasets=["folsom"],
        horizons=[5],
        model=model,
        lookback=12,
        stride=4,
        image_stride=3,
        tolerance="90s",
    )
    assert len(rows) == 1
    # fit AND predict received a non-None (n, lb_img, 4, H, W) cube.
    assert _SpyCrossViViT.fit_images_shape is not None, "fit never received images"
    assert _SpyCrossViViT.pred_images_shape is not None, "predict never received images"
    n, lb_img, c = _SpyCrossViViT.fit_images_shape[:3]
    assert c == 4
    assert lb_img == len(range(12 - 1, -1, -3))  # lookback-anchored subsample


def test_row_schema_and_reference_skill(image_loader) -> None:
    loader, model = image_loader
    rows = run_crossvivit_image_benchmark(
        loader,
        datasets=["folsom"],
        horizons=[5],
        model=model,
        lookback=12,
        stride=4,
        image_stride=3,
        tolerance="90s",
    )
    row = rows[0]
    assert _ROW_KEYS <= set(row.keys())
    assert row["model"] == "CrossViViT-image"
    assert row["dataset"] == "folsom" and row["horizon"] == 5
    assert np.isfinite(row["MAE"]) and np.isfinite(row["RMSE"])
    assert np.isfinite(row["Skill_MAE_pct"]) and np.isfinite(row["Skill_RMSE_pct"])
    assert row["n_daytime"] > 0


def test_windows_without_image_coverage_are_dropped(image_loader) -> None:
    """A mid-test image hole drops only the windows whose lookback hits it."""
    loader, model = image_loader
    cube, img_ts = loader._images["folsom"]
    # Remove ~40 consecutive image minutes near the end of the TEST split
    # (frame is 70/30; test = rows 280..399 → drop image rows 320..360).
    keep = np.ones(len(img_ts), dtype=bool)
    keep[320:360] = False
    loader._images["folsom"] = (cube[keep], img_ts[keep])

    rows = run_crossvivit_image_benchmark(
        loader,
        datasets=["folsom"],
        horizons=[5],
        model=model,
        lookback=12,
        stride=4,
        image_stride=3,
        tolerance="90s",
    )
    # The run still completes (windows around the hole were dropped) and the
    # reference + model share whatever windows remain (finite skill).
    assert len(rows) == 1
    assert np.isfinite(rows[0]["Skill_MAE_pct"])


def test_dataset_without_images_is_skipped(image_loader) -> None:
    loader, model = image_loader
    frame = _synth_frame("pv_power_normalized", 300, seed=5)
    loader._frames["skippd"] = frame  # tabular only, no sky cube
    rows = run_crossvivit_image_benchmark(
        loader,
        datasets=["skippd"],
        horizons=[5],
        model=model,
        lookback=12,
        stride=4,
        image_stride=3,
        tolerance="90s",
    )
    assert rows == []


# -- CSV writer ----------------------------------------------------------------


def test_csv_written_with_exact_schema(image_loader, tmp_path) -> None:
    loader, model = image_loader
    rows = run_crossvivit_image_benchmark(
        loader,
        datasets=["folsom"],
        horizons=[5],
        model=model,
        lookback=12,
        stride=4,
        image_stride=3,
        tolerance="90s",
    )
    path = save_crossvivit_image_results(rows, tmp_path)
    assert path.name == "e8_crossvivit_image.csv" and path.exists()
    df = pd.read_csv(path)
    assert set(df.columns) == _ROW_KEYS
    assert df.iloc[0]["model"] == "CrossViViT-image"


# -- folsom per-year shard concatenation ---------------------------------------


class _FakeShard:
    """Minimal zarr-array stand-in: dtype + __getitem__ + __len__."""

    def __init__(self, arr: np.ndarray):
        self._arr = arr

    @property
    def dtype(self):
        return self._arr.dtype

    def __len__(self):
        return len(self._arr)

    def __getitem__(self, i):
        return self._arr[i]


def test_multishard_cube_concatenates() -> None:
    a = np.arange(5, dtype=object)  # shard 1 (e.g. 2014)
    b = np.arange(5, 10, dtype=object)  # shard 2 (e.g. 2015)
    cube = MultiShardSkyCube([_FakeShard(a), _FakeShard(b)])
    assert len(cube) == 10
    assert cube.dtype == object
    assert [cube[i] for i in range(10)] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # negative / OOB indexing behaves like a plain array
    with pytest.raises(IndexError):
        _ = cube[10]


def test_multishard_cube_uint8_routing() -> None:
    a = np.zeros((3, 4, 8, 8), dtype=np.uint8)
    b = np.ones((2, 4, 8, 8), dtype=np.uint8)
    cube = MultiShardSkyCube([_FakeShard(a), _FakeShard(b)])
    assert cube.dtype == np.uint8
    assert cube[2].shape == (4, 8, 8) and cube[2].max() == 0
    assert cube[3].shape == (4, 8, 8) and cube[3].max() == 1
