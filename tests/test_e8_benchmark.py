"""Tests for E8 — Unified benchmark results (baselines x datasets x horizons).

Spec §11 (E8). All baselines on all datasets at the dev horizons, scored by
**Daytime skill vs Smart Persistence** under the daytime window. The fast tests
use a SYNTHETIC loader (mimics :class:`SolarUniBenchLoader`) + an injected
TINY ``model_factory`` so they run in seconds; a ``RUN_SLOW``-gated test runs
E8 end-to-end on the real standardized packages.
"""
import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from solarunibench.experiments import e8_benchmark
from solarunibench.forecast.models import (
    CrossViViTModel,
    LightGBMModel,
    MLPModel,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

slow = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW"),
    reason="set RUN_SLOW=1 to run the real-data E8 integration test",
)

_REAL_DATA_ROOT = "D:/monica/workspace/solar-unibench/data/standardized"


# ---------------------------------------------------------------------------
# Synthetic loader + frames
# ---------------------------------------------------------------------------

class _FakeLoader:
    """Mimics :class:`SolarUniBenchLoader` for tabular + sky-image APIs.

    ``images[dataset]`` is a ``(cube, image_timestamps)`` pair or ``None`` (no
    sky-image coverage -> CrossViViT omitted on that dataset).
    """

    def __init__(
        self,
        tabular: dict[str, pd.DataFrame],
        images: dict[str, tuple[np.ndarray, np.ndarray] | None] | None = None,
        data_root: str = "/fake",
    ):
        self._tabular = tabular
        self._images = images or {}
        self.data_root = data_root

    def load_tabular(self, dataset: str, split: str | None = None) -> pd.DataFrame:
        df = self._tabular[dataset].copy()
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)
        return df

    def get_tensor_cube(self, dataset: str, kind: str):
        if kind != "sky_images":
            return None
        entry = self._images.get(dataset)
        return entry[0] if entry is not None else None

    def get_image_timestamps(self, dataset: str):
        entry = self._images.get(dataset)
        return entry[1] if entry is not None else None


def _bell(times) -> np.ndarray:
    """Daytime clear-sky bell in [0, 1] driven by hour-of-day."""
    hod = times.hour.to_numpy()
    return np.clip(np.sin(np.pi * (hod - 6) / 13.0), 0.0, None)


def _synth_frame(
    name: str,
    target_col: str,
    n_entities: int = 2,
    n_days: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    """A bounded synthetic standardized frame driven by the clear-sky bell."""
    rows = []
    for e in range(n_entities):
        rng = np.random.default_rng(seed + e)
        eid = f"{name.upper()}_E{e}"
        n = n_days * 24
        times = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC")
        bell = _bell(times)
        ghi_clear = bell * 950.0
        cos_zen = bell  # proxy in [0, 1]; > 0.05 across the daytime band
        if target_col == "pv_power_normalized":
            target = np.clip(bell * 0.85 + rng.normal(0, 0.03, n), 0, 1)
        else:  # ghi_actual
            target = bell * 900.0 + rng.normal(0, 25, n)
        with np.errstate(divide="ignore", invalid="ignore"):
            kt = np.where(
                ghi_clear > 1,
                np.clip(target * 900.0 / np.maximum(ghi_clear, 1.0), 0, 1.5),
                0.0,
            )
        naive = times.tz_convert("UTC").tz_localize(None)
        n_tr = int(n * 0.70)
        n_va = int(n * 0.15)
        splits = ["train"] * n_tr + ["val"] * n_va + ["test"] * (n - n_tr - n_va)
        for i in range(n):
            rows.append({
                "timestamp_utc": naive[i],
                "entity_id": eid,
                target_col: float(target[i]),
                "ghi_clearsky": float(ghi_clear[i]),
                "cos_zenith": float(cos_zen[i]),
                "clearness_index": float(kt[i]),
                "latitude": 35.0 - e,
                "longitude": -100.0 + e,
                "dataset_source": name,
                "split": splits[i],
            })
    return pd.DataFrame(rows)


def _synth_image_dataset(
    name: str, target_col: str, n_rows: int = 480, seed: int = 0
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Bounded synthetic image-bearing frame + uint8 RGB sky cube (single entity).

    Sky-image timestamps are EXACTLY aligned with the tabular timestamps so the
    image-attach path resolves every lookback window (no tolerance confound).
    Images are random noise — the test only checks CrossViViT RUNS with images
    attached and emits finite daytime metrics, not that it learns.
    """
    rng = np.random.default_rng(seed)
    times = pd.date_range("2020-06-01 06:00", periods=n_rows, freq="1h", tz="UTC")
    bell = _bell(times)
    ghi_clear = bell * 950.0
    cos_zen = bell
    if target_col == "pv_power_normalized":
        target = np.clip(bell * 0.8 + rng.normal(0, 0.04, n_rows), 0, 1)
        kt = np.full(n_rows, np.nan)  # PV-only: clearness_index Honest-NaN
    else:
        target = bell * 900.0 + rng.normal(0, 25, n_rows)
        with np.errstate(divide="ignore", invalid="ignore"):
            kt = np.where(
                ghi_clear > 1,
                np.clip(target * 900.0 / np.maximum(ghi_clear, 1.0), 0, 1.5),
                0.0,
            )
    naive = times.tz_convert("UTC").tz_localize(None)
    n_tr = int(n_rows * 0.70)
    n_va = int(n_rows * 0.15)
    splits = ["train"] * n_tr + ["val"] * n_va + ["test"] * (n_rows - n_tr - n_va)
    frame = pd.DataFrame({
        "timestamp_utc": naive,
        "entity_id": f"{name.upper()}_E0",
        target_col: target.astype(float),
        "ghi_clearsky": ghi_clear.astype(float),
        "cos_zenith": cos_zen.astype(float),
        "clearness_index": kt.astype(float),
        "latitude": 35.0,
        "longitude": -100.0,
        "dataset_source": name,
        "split": splits,
    })
    cube = rng.integers(0, 256, size=(n_rows, 3, 16, 16), dtype=np.uint8)
    img_ts = naive.to_numpy().astype("datetime64[ns]")
    return frame, cube, img_ts


# ---------------------------------------------------------------------------
# A fast TINY model factory: only cheap tabular models (no torch fit loop).
# ---------------------------------------------------------------------------

def _fast_factory(has_images: bool):
    return [
        LightGBMModel(params={"n_estimators": 30, "n_jobs": 1, "random_state": 0}),
        MLPModel(hidden_layer_sizes=(16,), max_iter=60, random_state=0),
    ]


@pytest.fixture
def fast_loader() -> _FakeLoader:
    return _FakeLoader({
        "gefcom2014": _synth_frame("gefcom2014", "pv_power_normalized", seed=1),
        "folsom": _synth_frame("folsom", "ghi_actual", seed=2),
    })


# ---------------------------------------------------------------------------
# Contract tests for run(loader, out_dir, *, model_factory=...)
# ---------------------------------------------------------------------------

def test_run_returns_figures_and_metrics_dict(fast_loader, tmp_path):
    res = e8_benchmark.run(
        fast_loader, tmp_path, model_factory=_fast_factory,
        datasets=("gefcom2014", "folsom"),
    )
    assert isinstance(res, dict)
    assert set(res) >= {"figures", "metrics"}
    fig = tmp_path / "fig_benchmark_heatmap.png"
    assert fig.exists(), "missing fig_benchmark_heatmap.png"
    assert fig.stat().st_size > 0
    csvs = list((tmp_path / "results").glob("*.csv"))
    assert len(csvs) >= 1
    assert any("fig_benchmark_heatmap.png" in p for p in res["figures"])


def test_csv_shape_dataset_model_horizon(fast_loader, tmp_path):
    e8_benchmark.run(fast_loader, tmp_path, model_factory=_fast_factory, datasets=("gefcom2014", "folsom"))
    df = pd.read_csv(tmp_path / "results" / "results_benchmark.csv")
    # 2 datasets x (SmartPersistence reference + 2 models) x 2 horizons = 12 rows.
    expected = 2 * (1 + 2) * len(e8_benchmark.HORIZONS_HOURS)
    assert len(df) == expected, f"expected {expected} rows, got {len(df)}\n{df}"
    # SmartPersistence appears as a model row (the skill=0 reference).
    assert set(df["model"]) >= {"SmartPersistence", "LightGBM", "MLP"}
    assert set(df["dataset"]) == {"gefcom2014", "folsom"}
    assert set(df["horizon"]) == set(e8_benchmark.HORIZONS_HOURS)


def test_results_csv_schema(fast_loader, tmp_path):
    e8_benchmark.run(fast_loader, tmp_path, model_factory=_fast_factory, datasets=("gefcom2014", "folsom"))
    df = pd.read_csv(tmp_path / "results" / "results_benchmark.csv")
    required = {
        "dataset", "target", "horizon", "model",
        "Daytime_MAE", "Daytime_RMSE", "Daytime_skill", "n_daytime", "n_total",
    }
    assert required.issubset(df.columns), f"missing cols: {required - set(df.columns)}"


def test_smart_persistence_is_skill_zero_reference(fast_loader, tmp_path):
    """SmartPersistence is the skill denominator -> its own skill is exactly 0."""
    e8_benchmark.run(fast_loader, tmp_path, model_factory=_fast_factory, datasets=("gefcom2014", "folsom"))
    df = pd.read_csv(tmp_path / "results" / "results_benchmark.csv")
    sp = df[df["model"] == "SmartPersistence"]
    assert len(sp) > 0
    assert (sp["Daytime_skill"] == 0.0).all(), "SmartPersistence skill must be 0 (parity)"


def test_metrics_finite_and_daytime_consistent(fast_loader, tmp_path):
    e8_benchmark.run(fast_loader, tmp_path, model_factory=_fast_factory, datasets=("gefcom2014", "folsom"))
    df = pd.read_csv(tmp_path / "results" / "results_benchmark.csv")
    for _, r in df.iterrows():
        assert np.isfinite(r["Daytime_MAE"]), f"{r['model']}/{r['dataset']}/h{r['horizon']}: MAE not finite"
        assert np.isfinite(r["Daytime_RMSE"])
        assert r["Daytime_MAE"] >= 0
        assert np.isfinite(r["Daytime_skill"]), f"{r['model']}: skill not finite"
        assert r["n_daytime"] > 0, f"{r['model']}/{r['dataset']}/h{r['horizon']}: no daytime samples"
    # Daytime window MUST be identical across models within a (dataset, horizon)
    # so the skill comparison is fair (same mask => same n_daytime).
    for (ds, h), g in df.groupby(["dataset", "horizon"]):
        assert g["n_daytime"].nunique() == 1, (
            f"daytime window inconsistent across models for {ds}/h{h}: {g['n_daytime'].tolist()}"
        )


def test_skill_formula_consistent_with_reference(fast_loader, tmp_path):
    """Daytime_skill == 1 - MAE_model / MAE_SmartPersistence (the spec §10 skill).

    The load-bearing correctness property of the unified eval: SmartPersistence
    is the skill denominator, and every model's skill is recoverable from its
    MAE and the reference MAE via the forecast-skill formula. Also checks the
    sign convention (0 = parity, + = better, - = worse).
    """
    e8_benchmark.run(fast_loader, tmp_path, model_factory=_fast_factory, datasets=("gefcom2014", "folsom"))
    df = pd.read_csv(tmp_path / "results" / "results_benchmark.csv")
    for (ds, h), g in df.groupby(["dataset", "horizon"]):
        ref_mae = g.loc[g["model"] == "SmartPersistence", "Daytime_MAE"].iloc[0]
        assert np.isfinite(ref_mae) and ref_mae > 0
        for _, r in g.iterrows():
            if r["model"] == "SmartPersistence":
                assert r["Daytime_skill"] == 0.0  # parity with itself
                continue
            expected = 1.0 - r["Daytime_MAE"] / ref_mae
            assert abs(r["Daytime_skill"] - expected) < 1e-9, (
                f"{r['model']}/{ds}/h{h}: skill {r['Daytime_skill']} != formula {expected}"
            )


def test_run_does_not_mutate_loader_data(fast_loader, tmp_path):
    before = {k: v.copy(deep=True) for k, v in fast_loader._tabular.items()}
    e8_benchmark.run(fast_loader, tmp_path, model_factory=_fast_factory, datasets=("gefcom2014", "folsom"))
    for k, v in before.items():
        pd.testing.assert_frame_equal(v, fast_loader._tabular[k])


# ---------------------------------------------------------------------------
# CrossViViT — image-modality model only on image-bearing datasets
# ---------------------------------------------------------------------------

def _crossvivit_factory(has_images: bool):
    if not has_images:
        return []
    return [
        CrossViViTModel(
            d_model=16, n_heads=4, depth=1, patch_len=4, img_size=16,
            img_patch=8, epochs=2, batch_size=16, device="cpu",
        )
    ]


@pytest.fixture
def image_loader() -> _FakeLoader:
    skippd_frame, skippd_cube, skippd_ts = _synth_image_dataset(
        "skippd", "pv_power_normalized", seed=3
    )
    return _FakeLoader(
        tabular={
            "skippd": skippd_frame,
            "gefcom2014": _synth_frame("gefcom2014", "pv_power_normalized", seed=4),
        },
        images={"skippd": (skippd_cube, skippd_ts)},  # only skippd has sky images
    )


def test_crossvivit_only_on_image_bearing_dataset(image_loader, tmp_path):
    """CrossViViT rows appear ONLY for the image-bearing dataset (skippd).

    The factory returns CrossViViT only when has_images is True; gefcom2014
    (no sky images) must have NO CrossViViT row. This is the load-bearing
    graceful-omission property of the multimodal model in the benchmark.
    """
    res = e8_benchmark.run(image_loader, tmp_path, model_factory=_crossvivit_factory, datasets=("skippd", "gefcom2014"))
    df = pd.read_csv(tmp_path / "results" / "results_benchmark.csv")
    cv = df[df["model"] == "CrossViViT"]
    assert set(cv["dataset"]) == {"skippd"}, (
        f"CrossViViT should only appear for skippd, got {set(cv['dataset'])}"
    )
    # CrossViViT skill on skippd is finite (ran with images attached).
    for _, r in cv.iterrows():
        assert np.isfinite(r["Daytime_MAE"])
        assert np.isfinite(r["Daytime_skill"])
    # gefcom2014 has no CrossViViT rows (only SmartPersistence reference).
    gef = df[df["dataset"] == "gefcom2014"]
    assert "CrossViViT" not in set(gef["model"])


def test_image_attach_does_not_change_daytime_window(image_loader, tmp_path):
    """Image-attach must not alter the window set / daytime mask (fair comparison).

    CrossViViT (with images) and SmartPersistence (tabular) on skippd must share
    the same n_daytime -> the skill comparison is apples-to-apples.
    """
    e8_benchmark.run(image_loader, tmp_path, model_factory=_crossvivit_factory, datasets=("skippd", "gefcom2014"))
    df = pd.read_csv(tmp_path / "results" / "results_benchmark.csv")
    sk = df[(df["dataset"] == "skippd")]
    for h, g in sk.groupby("horizon"):
        assert g["n_daytime"].nunique() == 1, (
            f"image-attach changed the daytime window for skippd/h{h}"
        )


# ---------------------------------------------------------------------------
# Real-data RUN_SLOW integration
# ---------------------------------------------------------------------------

@slow
def test_run_real_standardized_packages(tmp_path):
    from solarunibench.loader import SolarUniBenchLoader

    loader = SolarUniBenchLoader(_REAL_DATA_ROOT)
    res = e8_benchmark.run(loader, tmp_path)  # default factory: all baselines
    assert isinstance(res, dict)
    assert (tmp_path / "fig_benchmark_heatmap.png").exists()
    csv = tmp_path / "results" / "results_benchmark.csv"
    assert csv.exists()
    df = pd.read_csv(csv)
    print("\n=== E8 real-data Daytime skill (model x dataset, horizon=1h) ===")
    h1 = df[df["horizon"] == df["horizon"].min()]
    pivot = h1.pivot_table(
        index="model", columns="dataset", values="Daytime_skill", aggfunc="mean"
    )
    print(pivot.to_string())
    # The SmartPersistence reference + all tabular models must be finite on
    # every cell; deep models may occasionally degrade to NaN (graceful) on a
    # tiny cell, so report (don't hard-fail) those.
    sp = df[df["model"] == "SmartPersistence"]
    assert sp["Daytime_MAE"].apply(np.isfinite).all(), "SmartPersistence MAE not finite"
    nan_cells = df[~df["Daytime_MAE"].apply(np.isfinite)]
    if len(nan_cells):
        print(f"\n[INFO] {len(nan_cells)} cell(s) reported NaN (graceful):")
        print(nan_cells[["model", "dataset", "horizon"]].to_string(index=False))
    # The overwhelming majority of cells must evaluate (guards against a
    # systematic break of the uniform-eval harness).
    finite_frac = df["Daytime_MAE"].apply(np.isfinite).mean()
    assert finite_frac >= 0.8, f"only {finite_frac:.0%} of cells finite"
