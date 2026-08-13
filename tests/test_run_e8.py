"""E8 unified benchmark — Task 4 tests (the 9-model runner).

``run_benchmark`` wires per-dataset config + the F=4 clearness feature + the
forecast pipeline + the daytime-masked evaluator into one cell loop over
``(dataset, horizon_step, model)``. SmartPersistence is always run FIRST as the
physical skill reference (its own Skill% is therefore 0 by construction); the
other models are scored against it. CrossViViT is run only on image datasets.

These tests use a MOCK loader + synthetic frames + 2 fast models so they stay
sub-second and dependency-free (the real-data run is the Task 6 RUN_SLOW suite).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solarunibench.benchmark.run_e8 import build_models, run_benchmark
from solarunibench.forecast.models import (
    CrossViViTModel,
    LightGBMModel,
    MLPModel,
)

# The spec'd per-row keys (Task 4 brief).
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


# -- synthetic data + mock loader --------------------------------------------


def _synthetic_split(n: int, seed: int) -> pd.DataFrame:
    """Deterministic sunlake-shaped frame (ghi_actual target, all-daytime)."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2021-06-01", periods=n, freq="1h", tz="UTC")
    phase = np.arange(n)
    # cos_zenith always > 0.05 → every window is daytime (finite metrics).
    cos_zenith = np.clip(0.5 + 0.3 * np.sin(phase / 4.0), 0.1, 0.95)
    ghi_clearsky = 1000.0 * cos_zenith
    # Target tracks clearsky with mild noise → nonzero, finite MAE.
    ghi_actual = np.clip(
        ghi_clearsky * (0.8 + 0.05 * rng.standard_normal(n)), 0.0, None
    )
    return pd.DataFrame(
        {
            "timestamp_utc": t,
            "entity_id": "E1",
            "ghi_actual": ghi_actual,
            "ghi_clearsky": ghi_clearsky,
            "cos_zenith": cos_zenith,
        }
    )


class _MockLoader:
    """Stand-in for ``SolarUniBenchLoader``: serves pre-built per-split frames."""

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        self._frames = {"train": train, "test": test}

    def load_tabular(self, dataset: str, split: str | None = None) -> pd.DataFrame:
        assert split in self._frames, f"unexpected split={split!r}"
        return self._frames[split].copy()


@pytest.fixture
def sunlake_loader() -> _MockLoader:
    # sunlake lookback_steps=24, horizon=1 → need >= 25 rows per split.
    return _MockLoader(_synthetic_split(50, seed=0), _synthetic_split(35, seed=1))


# -- run_benchmark: shape + reference ----------------------------------------


def test_run_benchmark_row_shape_and_keys(sunlake_loader: _MockLoader) -> None:
    """Each result row is a dict carrying exactly the spec'd benchmark columns."""
    models = [LightGBMModel(), MLPModel(max_iter=10)]
    results = run_benchmark(
        sunlake_loader, datasets=["sunlake"], horizons=[1], models=models
    )

    assert isinstance(results, list)
    assert len(results) == 3  # SmartPersistence (reference) + 2 models
    for row in results:
        assert _ROW_KEYS <= set(row.keys()), f"missing keys in {row!r}"
        assert row["dataset"] == "sunlake"
        assert row["horizon"] == 1
        assert np.isfinite(row["MAE"]) and np.isfinite(row["RMSE"])
        assert row["n_daytime"] > 0


def test_smartpersistence_is_auto_reference(sunlake_loader: _MockLoader) -> None:
    """SmartPersistence is always run first as the reference (not passed in)."""
    results = run_benchmark(
        sunlake_loader,
        datasets=["sunlake"],
        horizons=[1],
        models=[LightGBMModel()],
    )
    names = [r["model"] for r in results]
    assert "SmartPersistence" in names
    assert names[0] == "SmartPersistence"  # reference emitted first


def test_smartpersistence_skill_is_zero(sunlake_loader: _MockLoader) -> None:
    """The reference scores 0% skill against itself (parity)."""
    results = run_benchmark(
        sunlake_loader,
        datasets=["sunlake"],
        horizons=[1],
        models=[LightGBMModel()],
    )
    sp = next(r for r in results if r["model"] == "SmartPersistence")
    assert abs(sp["Skill_MAE_pct"]) < 1e-9
    assert abs(sp["Skill_RMSE_pct"]) < 1e-9


def test_other_models_scored_against_reference(sunlake_loader: _MockLoader) -> None:
    """Non-reference models share the reference's persistence_pred + daytime mask."""
    results = run_benchmark(
        sunlake_loader,
        datasets=["sunlake"],
        horizons=[1],
        models=[LightGBMModel()],
    )
    lgb = next(r for r in results if r["model"] == "LightGBM")
    # Skill is a finite percentage (sign depends on fit quality; just finite).
    assert np.isfinite(lgb["Skill_MAE_pct"])
    assert np.isfinite(lgb["Skill_RMSE_pct"])
    # Same daytime count as the reference (shared mask + windows).
    sp = next(r for r in results if r["model"] == "SmartPersistence")
    assert lgb["n_daytime"] == sp["n_daytime"]


# -- CrossViViT gating --------------------------------------------------------


def test_crossvivit_skipped_on_non_image_dataset(sunlake_loader: _MockLoader) -> None:
    """sunlake has image_dataset=False → CrossViViT is dropped, not run."""
    results = run_benchmark(
        sunlake_loader,
        datasets=["sunlake"],
        horizons=[1],
        models=[CrossViViTModel(epochs=1)],
    )
    names = [r["model"] for r in results]
    assert "CrossViViT" not in names
    # SmartPersistence reference still emitted.
    assert names == ["SmartPersistence"]


# -- horizons default + multiple cells ---------------------------------------


def test_horizons_default_to_dataset_config(sunlake_loader: _MockLoader) -> None:
    """horizons=None → uses the dataset's cfg.horizons_steps ([1,2,3] for sunlake)."""
    results = run_benchmark(
        sunlake_loader,
        datasets=["sunlake"],
        horizons=None,
        models=[LightGBMModel()],
    )
    # 3 horizons × (SmartPersistence + LightGBM).
    assert len(results) == 6
    assert sorted({r["horizon"] for r in results}) == [1, 2, 3]


# -- build_models factory ----------------------------------------------------


def test_build_models_returns_eight_non_reference_models() -> None:
    """Default model battery = 8 (excludes the SmartPersistence reference)."""
    models = build_models(tiny=True)
    assert len(models) == 8
    # Exactly one CrossViViT in the battery (gated per-dataset by the runner).
    assert sum(1 for m in models if isinstance(m, CrossViViTModel)) == 1


def test_build_models_tiny_reduces_epochs() -> None:
    """tiny=True forces the smallest tractable epoch budget on deep models."""
    tiny = build_models(tiny=True)
    full = build_models(tiny=False)
    tiny_lstm = next(m for m in tiny if m.name == "LSTM")
    full_lstm = next(m for m in full if m.name == "LSTM")
    assert tiny_lstm.epochs == 1
    assert full_lstm.epochs > 1


def test_build_models_patchtst_has_ci_and_cd_variants() -> None:
    """The battery carries both PatchTST channel-independent and -dependent."""
    models = build_models(tiny=True)
    variants = [getattr(m, "variant", None) for m in models]
    assert "patchtst" in variants and "patchtst_cd" in variants
