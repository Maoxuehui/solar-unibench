"""End-to-end integration: 3 forecast baselines on the FIXED GEFCom.

RUN_SLOW-gated. Loads the real GEFCom2014 standardized package through
:class:`solarunibench.loader.SolarUniBenchLoader` (all three standardized
splits — ``train``/``val``/``test`` — with the ``split`` column preserved),
runs :class:`SmartPersistenceModel`, :class:`LightGBMModel`, and
:class:`MLPModel` through :class:`SolarForecastPipeline` (which now scales
features by default via :class:`PhysicalFeatureScaler` and emits a
``daytime`` metric block via the ``cos_zenith > 0.05`` anti-fraud filter),
and asserts:

* all 3 models produce finite ``Daytime_MAE``;
* **MLP converges** — ``Daytime_MAE < 0.5`` (the OLD integration saw MAE ~3.8
  on unscaled features; the physical scaler fixes the divergence);
* LightGBM daytime skill > 0 vs SmartPersistence.

A 3-row daytime skill table is printed to stdout.

What changed vs the previous (Phase 6a T6) integration
------------------------------------------------------
The previous version worked around TWO data-quality regressions that the
benchmark-integrity work has now fixed at the source:

* **GEFCom test target was identically zero.** BI-T1 rebuilt the GEFCom
  package so the 2014-06 Solution period POWER is present in the ``test``
  split (test ``pv_power_normalized`` max 0.74). We therefore run on the REAL
  standardized ``test`` split — no per-entity 80/20 train re-split override.
* **Unscaled features next to a [0, 1] target.** BI-T2 made
  :class:`PhysicalFeatureScaler` default-on in the pipeline (irradiance ÷ 1500,
  NWP train-fit z-score). MLP no longer diverges.

The previous ``_FiniteTargetWindowFilter`` workaround is also dropped: the
fixed GEFCom ``pv_power_normalized`` has zero NaNs in every split (night is
encoded as ``0.0``, not NaN), so the daytime-window evaluator — not a
NaN-dropping filter — is what excludes nighttime rows from the headline
metric.

Real-data accommodations (all operating on an in-memory copy — the on-disk
parquet is never mutated):
* **``clearness_index`` is entirely NaN** in the current GEFCom package. We
  keep it in ``feature_cols`` per the task brief and ``fillna(0)`` so the
  column is harmless (LightGBM ignores it; MLP sees a constant). This is a
  documented data finding, not a framework behavior.
"""
from __future__ import annotations

import functools
import os
import warnings

import numpy as np
import pytest

DATA_ROOT = "D:/monica/workspace/solar-unibench/data/standardized"
DATASET = "gefcom2014"
TARGET = "pv_power_normalized"
FEATURE_COLS = ["ghi_clearsky", "cos_zenith", "clearness_index"]

slow = pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="set RUN_SLOW=1")


@functools.lru_cache(maxsize=1)
def _gefcom_standardized_frame() -> "pd.DataFrame":
    """Load all 3 standardized GEFCom splits with the ``split`` column intact.

    Returns an in-memory copy; the on-disk parquet is never mutated. The
    frame carries the standardized ``train``/``val``/``test`` partitioning
    unchanged — :class:`SolarForecastPipeline._load_split` filters on the
    ``split`` column. The 2014-06 Solution period (the BI-T1 fix) lives in
    the ``test`` split.
    """
    import pandas as pd

    from solarunibench.loader import SolarUniBenchLoader

    full = SolarUniBenchLoader(DATA_ROOT).load_tabular(DATASET).copy()
    # ``clearness_index`` is entirely NaN in this package → fill with 0 so the
    # tabular MLP (which cannot ingest NaN features) can fit. The column is a
    # documented data-quality casualty, not a framework behavior.
    if "clearness_index" in full.columns:
        full["clearness_index"] = full["clearness_index"].fillna(0.0)
    return full


def _provider(_config) -> "pd.DataFrame":
    """df_provider hook: returns the full GEFCom frame (train+val+test)."""
    return _gefcom_standardized_frame()


@slow
def test_three_baselines_on_fixed_gefcom_daytime_skill(capsys):
    """Run SP + LightGBM + MLP on the FIXED GEFCom; assert + print daytime table.

    On the rebuilt GEFCom (real 2014-06 test POWER) with the pipeline's
    default :class:`PhysicalFeatureScaler` and the daytime-window evaluator,
    MLP must converge (``Daytime_MAE < 0.5``, not the ~3.8 divergence seen on
    unscaled features) and LightGBM must beat SmartPersistence on daytime
    skill.
    """
    from solarunibench.forecast.core import (
        EndogenousFeatureEngineer,
        PipelineConfig,
        SolarForecastPipeline,
    )
    from solarunibench.forecast.models.lightgbm_model import LightGBMModel
    from solarunibench.forecast.models.mlp_model import MLPModel
    from solarunibench.forecast.models.smart_persistence import SmartPersistenceModel

    # lookback=6, horizon=3 per the task brief ("modest, e.g. 6/3"). Stride=3
    # (half-overlap) rather than =lookback: the GEFCom ``test`` split is a
    # single month (2014-06 Solution period, 2163 rows / 721 per zone), so
    # non-overlapping windows yield only ~180 daytime samples — borderline
    # statistical power for a stable skill ratio. Half-overlap roughly doubles
    # the daytime count (~260) for a more robust skill estimate, which is the
    # standard reason forecasters use overlapping eval windows on short test
    # periods.
    cfg = PipelineConfig(
        dataset=DATASET,
        target=TARGET,
        lookback=6,
        horizon=3,
        stride=3,
        feature_cols=FEATURE_COLS,
        data_root=DATA_ROOT,
    )
    # No NaN-target filter: the fixed GEFCom encodes night as 0.0 (zero NaNs
    # across splits), and the daytime-window evaluator — not a NaN-dropping
    # filter — is what excludes nighttime from the headline metric.
    fe = EndogenousFeatureEngineer(target=TARGET, feature_cols=FEATURE_COLS)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        res_sp = SolarForecastPipeline(cfg, fe, SmartPersistenceModel()).run(
            df_provider=_provider
        )
        res_lgb = SolarForecastPipeline(cfg, fe, LightGBMModel()).run(df_provider=_provider)
        res_mlp = SolarForecastPipeline(cfg, fe, MLPModel()).run(df_provider=_provider)

    # Windows are deterministic across the 3 runs (same split, same sort) →
    # identical ground truth and daytime mask.
    y = res_sp["y_true"]
    assert res_lgb["y_true"].shape == y.shape
    assert res_mlp["y_true"].shape == y.shape
    np.testing.assert_array_equal(res_lgb["y_true"], y)
    np.testing.assert_array_equal(res_mlp["y_true"], y)

    results = {
        "SmartPersistence": res_sp,
        "LightGBM": res_lgb,
        "MLP": res_mlp,
    }

    # --- All 3 models: finite Daytime_MAE / Daytime_RMSE, n_daytime > 0. ---
    for name, res in results.items():
        assert "daytime" in res, (
            f"{name} pipeline did not emit a daytime metric block "
            "(cos_zenith missing from frame?)"
        )
        dm = res["daytime"]
        assert dm["n_daytime"] > 0, f"{name} has zero daytime test samples"
        assert np.isfinite(dm["Daytime_MAE"]), (
            f"{name} Daytime_MAE not finite: {dm['Daytime_MAE']}"
        )
        assert np.isfinite(dm["Daytime_RMSE"]), (
            f"{name} Daytime_RMSE not finite: {dm['Daytime_RMSE']}"
        )

    # --- Daytime skill vs SmartPersistence for the two learned models. ---
    # skill = 1 - mae_model / mae_ref (0 = parity, positive = better); this is
    # exactly `skill_vs_reference` over the daytime-masked arrays, which share
    # the same y_true and cos_zenith mask across the 3 deterministic runs.
    sp_daytime_mae = res_sp["daytime"]["Daytime_MAE"]
    assert sp_daytime_mae > 0, "SmartPersistence Daytime_MAE == 0; skill undefined"
    skill_lgb = 1.0 - res_lgb["daytime"]["Daytime_MAE"] / sp_daytime_mae
    skill_mlp = 1.0 - res_mlp["daytime"]["Daytime_MAE"] / sp_daytime_mae

    # --- Print the 3-row daytime skill table BEFORE the gating assertions so ---
    # the numbers are always captured for diagnosis even on a failed assert.
    table = _format_daytime_skill_table(
        ("SmartPersistence", res_sp, float("nan")),
        ("LightGBM", res_lgb, skill_lgb),
        ("MLP", res_mlp, skill_mlp),
    )
    with capsys.disabled():
        print(table)

    # --- MLP converges: Daytime_MAE < 0.5 (was ~3.8 on unscaled features). ---
    mlp_daytime_mae = res_mlp["daytime"]["Daytime_MAE"]
    assert mlp_daytime_mae < 0.5, (
        f"MLP failed to converge: Daytime_MAE={mlp_daytime_mae:.4f} >= 0.5 — "
        "the PhysicalFeatureScaler should have fixed the unscaled-feature divergence"
    )

    # --- LightGBM daytime skill > 0 vs SmartPersistence. ---
    assert skill_lgb > 0, (
        f"LightGBM daytime skill vs SmartPersistence not > 0: {skill_lgb:+.4f}"
    )


def _format_daytime_skill_table(*rows) -> str:
    """Format the 3-row daytime skill table (model | MAE | RMSE | skill | n)."""
    lines = [
        "",
        "=== GEFCom daytime forecast skill "
        "(reference: SmartPersistence, cos_zenith>0.05) ===",
        f"{'model':<20}{'Daytime_MAE':>14}{'Daytime_RMSE':>14}"
        f"{'Daytime_skill-vs-SP':>22}{'n_daytime':>12}",
    ]
    for name, res, skill in rows:
        dm = res["daytime"]
        skill_str = (
            "(reference)" if name == "SmartPersistence" else f"{skill:+.4f}"
        )
        lines.append(
            f"{name:<20}{dm['Daytime_MAE']:>14.4f}{dm['Daytime_RMSE']:>14.4f}"
            f"{skill_str:>22}{dm['n_daytime']:>12}"
        )
    return "\n".join(lines)
