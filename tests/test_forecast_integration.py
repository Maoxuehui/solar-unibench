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

Phase 6b Task 6 extension — deep baselines + consolidated skill table
--------------------------------------------------------------------
A second RUN_SLOW test (``test_deep_baselines_consolidated_daytime_skill``)
adds the three Phase-6b deep baselines and prints ONE consolidated daytime
skill table covering every model:

* **LSTM + PatchTST on GEFCom** (tabular sequence models; same GEFCom frame,
  same ``feature_cols`` as the trio above, lookback=12 / horizon=3 so the
  sequence/patch tokenizers have a real lookback — ``patch_len=4`` needs
  ``lookback >= 4``).
* **CrossViViT on SKIPPD** (multimodal: tabular + sky images) via
  ``EndogenousFeatureEngineer(..., return_images=True, image_loader=...,
  image_dataset="skippd")`` so the image branch activates end-to-end.

SKIPPD multimodal bounding (documented): the standardized SKIPPD sky cube
holds only 2000 frames (2017-03-09 → 2017-05-13) and EVERY one of those
timestamps sits in the original ``train`` split — the chronological
``test``/``val`` splits have ZERO sky-image coverage, so CrossViViT's image
branch cannot activate on them. We therefore carve a small, fully
image-covered, fully DAYTIME slice (the first 400 image-matched rows — a
2017-03-09 block, ``cos_zenith ∈ [0.05, 0.75]``) and re-stamp its ``split``
70/30 so BOTH the fit and predict frames resolve sky images. The on-disk
parquet is untouched (in-memory copy only).

Deep models use TINY configs (``epochs=4``, small hidden) so the RUN_SLOW
suite stays tractable; their daytime skill is REPORTED, not gated — a 4-epoch
net is not expected to beat SmartPersistence, the point is that all three deep
baselines run end-to-end on real standardized data through the full pipeline
(default :class:`PhysicalFeatureScaler` + daytime evaluator).
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


# --- Phase 6b Task 6: deep baselines + consolidated skill table -------------

# SKIPPD multimodal run. CrossViViT fuses tabular + sky images; the bounded
# slice + re-split is documented in the module docstring.
_SK_DATASET = "skippd"
_SK_TARGET = "pv_power_normalized"
_SK_FEATURE_COLS = ["ghi_clearsky", "cos_zenith", "clearness_index"]
# First N image-matched rows (sorted by time): a fully image-covered, fully
# DAYTIME 2017-03-09 block (cos_zenith ∈ [0.05, 0.75]).
_SK_IMAGE_LIMIT = 400
_SK_TRAIN_FRACTION = 0.70

# Lookback/horizon/stride shared by EVERY model in the consolidated table so
# each dataset's models are scored against SmartPersistence on IDENTICAL
# windows / daytime masks (skill is directly comparable). 12/3 gives the
# sequence/patch tokenizers a real lookback (patch_len=4 needs lookback >= 4)
# while staying TINY.
_DEEP_LOOKBACK = 12
_DEEP_HORIZON = 3
_DEEP_STRIDE = 3


@functools.lru_cache(maxsize=1)
def _skippd_image_covered_frame() -> "pd.DataFrame":
    """Return a small, fully image-covered SKIPPD slice, re-split 70/30.

    The standardized SKIPPD sky cube holds only 2000 frames (2017-03-09 →
    2017-05-13) at ~1-min cadence, and EVERY one of those timestamps lives in
    the original ``train`` split — the chronological ``test``/``val`` splits
    have ZERO sky-image coverage. CrossViViT's image branch can only activate
    end-to-end when BOTH the fit and predict frames resolve images, so we:

    1. load the full SKIPPD tabular frame (in-memory copy; parquet untouched);
    2. keep only rows whose ``timestamp_utc`` EXACTLY matches a sky-image
       timestamp (the 2000 image-covered rows);
    3. take the first ``_SK_IMAGE_LIMIT`` (sorted by time) — a small, fully
       image-covered, fully DAYTIME 2017-03-09 slice;
    4. re-stamp ``split`` so the first 70% are ``train`` and the last 30% are
       ``test`` — BOTH halves image-covered and daytime.

    ``clearness_index`` is entirely NaN in this package too, so it is
    ``fillna(0)``-ed (harmless constant column) for parity with the GEFCom
    treatment above.
    """
    import pandas as pd

    from solarunibench.loader import SolarUniBenchLoader

    loader = SolarUniBenchLoader(DATA_ROOT)
    full = loader.load_tabular(_SK_DATASET).copy()
    if "clearness_index" in full.columns:
        full["clearness_index"] = full["clearness_index"].fillna(0.0)

    img_ts = loader.get_image_timestamps(_SK_DATASET)
    img_naive = (
        pd.to_datetime([str(t) for t in np.asarray(img_ts)])
        .tz_convert("UTC")
        .tz_localize(None)
        .floor("s")
    )
    img_set = pd.Index(img_naive)
    raw = pd.to_datetime(full["timestamp_utc"])
    tab_naive = raw.dt.tz_convert("UTC").dt.tz_localize(None).dt.floor("s")

    covered = (
        full[tab_naive.isin(img_set)]
        .sort_values("timestamp_utc")
        .head(_SK_IMAGE_LIMIT)
        .reset_index(drop=True)
        .copy()
    )
    cut = int(len(covered) * _SK_TRAIN_FRACTION)
    covered["split"] = np.where(covered.index < cut, "train", "test")
    return covered


def _skippd_provider(_config) -> "pd.DataFrame":
    """df_provider hook: returns the bounded image-covered SKIPPD slice."""
    return _skippd_image_covered_frame()


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


@slow
def test_deep_baselines_consolidated_daytime_skill(capsys):
    """Run LSTM + PatchTST (GEFCom) and CrossViViT (SKIPPD multimodal) through
    the pipeline + default :class:`PhysicalFeatureScaler` + daytime evaluator;
    print ONE consolidated daytime skill table covering every model (SP,
    LightGBM, MLP, LSTM, PatchTST on GEFCom; SP, CrossViViT on SKIPPD).

    Asserts each deep model emits a finite ``Daytime_MAE`` with ``n_daytime > 0``
    and that CrossViViT's image branch ACTIVATES (every lookback resolves a sky
    frame on the bounded SKIPPD slice). Deep models use TINY configs (``epochs=4``,
    small hidden) so the RUN_SLOW suite stays tractable; skill is REPORTED, not
    gated — a 4-epoch net is not expected to beat SmartPersistence. The point is
    that all three Phase-6b deep baselines run end-to-end on REAL standardized
    data through the full pipeline.
    """
    from solarunibench.forecast.core import (
        EndogenousFeatureEngineer,
        PipelineConfig,
        SolarForecastPipeline,
    )
    from solarunibench.forecast.models.crossvivit_model import CrossViViTModel
    from solarunibench.forecast.models.lightgbm_model import LightGBMModel
    from solarunibench.forecast.models.lstm_model import LSTMModel
    from solarunibench.forecast.models.mlp_model import MLPModel
    from solarunibench.forecast.models.patchtst_model import PatchTSTModel
    from solarunibench.forecast.models.smart_persistence import SmartPersistenceModel
    from solarunibench.loader import SolarUniBenchLoader

    # ---------------- GEFCom: SP + LightGBM + MLP + LSTM + PatchTST ----------
    # All five share one PipelineConfig + one FeatureEngineer → identical windows
    # and daytime mask, so skill-vs-SP is directly comparable across the set.
    cfg_gef = PipelineConfig(
        dataset=DATASET,
        target=TARGET,
        lookback=_DEEP_LOOKBACK,
        horizon=_DEEP_HORIZON,
        stride=_DEEP_STRIDE,
        feature_cols=FEATURE_COLS,
        data_root=DATA_ROOT,
    )
    fe_gef = EndogenousFeatureEngineer(target=TARGET, feature_cols=FEATURE_COLS)

    # TINY deep configs: few epochs, small hidden — keep RUN_SLOW tractable.
    # ``seed=0`` makes fits reproducible (the torch baselines seed RNG + cuDNN).
    lstm = LSTMModel(
        hidden_size=32, num_layers=1, dropout=0.0, epochs=4, batch_size=64, seed=0
    )
    patch = PatchTSTModel(
        d_model=32, n_heads=4, patch_len=4, stride=2, depth=1,
        epochs=4, batch_size=64, seed=0,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        res_sp_gef = SolarForecastPipeline(
            cfg_gef, fe_gef, SmartPersistenceModel()
        ).run(df_provider=_provider)
        res_lgb = SolarForecastPipeline(
            cfg_gef, fe_gef, LightGBMModel()
        ).run(df_provider=_provider)
        res_mlp = SolarForecastPipeline(cfg_gef, fe_gef, MLPModel()).run(
            df_provider=_provider
        )
        res_lstm = SolarForecastPipeline(cfg_gef, fe_gef, lstm).run(
            df_provider=_provider
        )
        res_patch = SolarForecastPipeline(cfg_gef, fe_gef, patch).run(
            df_provider=_provider
        )

    # ---------------- SKIPPD: SP + CrossViViT (multimodal, images ON) --------
    cfg_sk = PipelineConfig(
        dataset=_SK_DATASET,
        target=_SK_TARGET,
        lookback=_DEEP_LOOKBACK,
        horizon=_DEEP_HORIZON,
        stride=_DEEP_STRIDE,
        feature_cols=_SK_FEATURE_COLS,
        data_root=DATA_ROOT,
    )
    loader = SolarUniBenchLoader(DATA_ROOT)
    # Image-aware engineer for CrossViViT; tabular-only engineer for the SP
    # reference. Both produce IDENTICAL windows (image-attach is post-hoc and
    # does not affect windowing) → comparable daytime skill.
    fe_sk_img = EndogenousFeatureEngineer(
        target=_SK_TARGET,
        feature_cols=_SK_FEATURE_COLS,
        return_images=True,
        image_loader=loader,
        image_dataset=_SK_DATASET,
    )
    fe_sk_tab = EndogenousFeatureEngineer(
        target=_SK_TARGET, feature_cols=_SK_FEATURE_COLS
    )

    # Activation guard: confirm the image branch resolves on the TRAIN frame.
    # Image-attach is keyed on timestamps (scale-invariant), so this pre-check
    # on the raw frame exactly mirrors what the pipeline sees post-scaling; it
    # also warms the FE's image-timestamp index cache for the run.
    df_sk = _skippd_image_covered_frame()
    df_sk_tr = df_sk[df_sk["split"] == "train"]
    _, _, _, _, _, img_tr_check = fe_sk_img.create_features(
        df_sk_tr, _DEEP_LOOKBACK, _DEEP_HORIZON, _DEEP_STRIDE
    )
    assert img_tr_check is not None, (
        "CrossViViT image branch did not activate: every lookback must resolve a "
        "sky frame on the bounded SKIPPD slice (got images=None on the train frame)"
    )

    # TINY CrossViViT (img_patch=8 → (32/8)^2 = 16 image tokens per frame).
    cv = CrossViViTModel(
        d_model=32, n_heads=4, depth=1, patch_len=4,
        img_size=32, img_patch=8, in_channels=4,
        epochs=4, batch_size=32, seed=0,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        res_sp_sk = SolarForecastPipeline(
            cfg_sk, fe_sk_tab, SmartPersistenceModel()
        ).run(df_provider=_skippd_provider)
        res_cv = SolarForecastPipeline(cfg_sk, fe_sk_img, cv).run(
            df_provider=_skippd_provider
        )

    # ---------------- Assert finite Daytime_MAE for the deep baselines -------
    deep_results = {
        "LSTM": res_lstm,
        "PatchTST": res_patch,
        "CrossViViT": res_cv,
    }
    for name, res in deep_results.items():
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

    # ---------------- Skill vs each dataset's SmartPersistence --------------
    # Windows + daytime mask are deterministic per (cfg, frame) → identical
    # across a dataset's models, so skill = 1 - mae_model / mae_SP is exact.
    sp_gef_mae = res_sp_gef["daytime"]["Daytime_MAE"]
    sp_sk_mae = res_sp_sk["daytime"]["Daytime_MAE"]
    assert sp_gef_mae > 0, "GEFCom SmartPersistence Daytime_MAE == 0; skill undefined"
    assert sp_sk_mae > 0, "SKIPPD SmartPersistence Daytime_MAE == 0; skill undefined"

    gef_rows = [
        ("SmartPersistence", res_sp_gef, float("nan")),
        ("LightGBM", res_lgb, 1.0 - res_lgb["daytime"]["Daytime_MAE"] / sp_gef_mae),
        ("MLP", res_mlp, 1.0 - res_mlp["daytime"]["Daytime_MAE"] / sp_gef_mae),
        ("LSTM", res_lstm, 1.0 - res_lstm["daytime"]["Daytime_MAE"] / sp_gef_mae),
        ("PatchTST", res_patch, 1.0 - res_patch["daytime"]["Daytime_MAE"] / sp_gef_mae),
    ]
    sk_rows = [
        ("SmartPersistence", res_sp_sk, float("nan")),
        ("CrossViViT", res_cv, 1.0 - res_cv["daytime"]["Daytime_MAE"] / sp_sk_mae),
    ]

    # Print BEFORE any further assertions so the table is always captured.
    table = _format_consolidated_daytime_skill_table(
        ("GEFCom2014", gef_rows), ("SKIPPD", sk_rows)
    )
    with capsys.disabled():
        print(table)


def _format_consolidated_daytime_skill_table(*dataset_blocks) -> str:
    """Format the consolidated daytime skill table across datasets.

    Each ``dataset_blocks`` item is ``(dataset_name, [(model, res, skill), ...])``.
    Skill is ``1 - mae_model / mae_SP`` over the shared daytime mask
    (0 = parity, positive = better than SmartPersistence); ``nan`` for the SP
    reference row (printed as ``(reference)``).
    """
    lines = [
        "",
        "=== Consolidated daytime forecast skill "
        "(reference: SmartPersistence, cos_zenith>0.05) ===",
        f"{'model':<18}{'dataset':<14}{'Daytime_MAE':>13}{'Daytime_RMSE':>14}"
        f"{'Daytime_skill-vs-SP':>22}{'n_daytime':>11}",
    ]
    for dataset, rows in dataset_blocks:
        for name, res, skill in rows:
            dm = res["daytime"]
            skill_str = (
                "(reference)" if name == "SmartPersistence" else f"{skill:+.4f}"
            )
            lines.append(
                f"{name:<18}{dataset:<14}{dm['Daytime_MAE']:>13.4f}"
                f"{dm['Daytime_RMSE']:>14.4f}{skill_str:>22}{dm['n_daytime']:>11}"
            )
    return "\n".join(lines)
