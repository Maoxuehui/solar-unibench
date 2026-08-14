"""E8 unified benchmark runner — the 9-model cell loop.

Wires the per-dataset physics-time config
(:data:`solarunibench.benchmark.config.DATASET_CONFIG`), the F=4 clearness
feature (:func:`solarunibench.benchmark.features.add_clearness_aligned`), the
forecast pipeline (:class:`solarunibench.forecast.core.SolarForecastPipeline`),
and the daytime-masked evaluator
(:func:`solarunibench.benchmark.evaluator.evaluate_predictions`) into one loop
over every ``(dataset, horizon_step, model)`` cell.

Discipline (see ``docs/superpowers/plans/2026-08-13-e8-unified-benchmark.md``):

* **SmartPersistence is the reference.** It is always run FIRST per cell; every
  other model's Forecast Skill% is measured against its predictions on the SAME
  daytime subset. The reference's own Skill% is therefore 0 by construction
  (``evaluate_predictions`` with identical model and reference predictions).
* **Fresh model per cell.** Each cell fits a pristine model instance (the
  ``models`` list carries templates; the runner deep-copies one per cell so fit
  state never leaks across cells/horizons).
* **CrossViViT only on image datasets.** The multimodal model is skipped
  (``cfg.image_dataset is False``) rather than run without its image modality.
* **TINY configs by default** for tractability (deep models at 1 epoch, MLP at
  10 iterations); ``tiny=False`` restores the framework defaults for the later
  full-config run.
* **Pure:** loads via the injected loader (a ``SolarUniBenchLoader`` in
  production, a mock in tests) and never writes to disk.

Per-sample ``cos_zenith`` for the daytime mask
----------------------------------------------
The pipeline returns ``{pred, y_true, times, model_name, daytime?}`` but does
NOT expose the per-sample issue-time ``cos_zenith`` the evaluator needs. We
derive it here by calling the SAME
:class:`~solarunibench.forecast.core.EndogenousFeatureEngineer` on the test
split (its 5th return value). This is safe and exact because:

* ``cos_zenith`` is **passthrough** under
  :class:`~solarunibench.forecast.scaling.PhysicalFeatureScaler` (it is neither
  an irradiance nor an NWP column), so its values are identical whether
  engineered from the scaled or unscaled frame; and
* windowing is row-order based and therefore identical regardless of scaling.

So the ``cos_zenith_issue`` array extracted here is element-wise identical to
the one the pipeline computes internally on its scaled test frame.
"""
from __future__ import annotations

import copy
from typing import Callable, Protocol

import pandas as pd

from .config import DATASET_CONFIG, DatasetConfig
from .evaluator import evaluate_predictions
from .features import add_clearness_aligned
from ..forecast.core import (
    EndogenousFeatureEngineer,
    PipelineConfig,
    SolarForecastPipeline,
)
from ..forecast.models import (
    CrossViViTModel,
    DLinearModel,
    iTransformerModel,
    LightGBMModel,
    LSTMModel,
    MLPModel,
    PatchTSTModel,
    SmartPersistenceModel,
)
from ..forecast.scaling import PhysicalFeatureScaler

# The fixed F=4 past-channel spec: [target, cos_zenith, ghi_clearsky,
# clearness_index_aligned]. The target is always the FIRST past channel
# (EndogenousFeatureEngineer prepends it); the other three are feature_cols.
_FEATURE_COLS: list[str] = ["cos_zenith", "ghi_clearsky", "clearness_index_aligned"]

# TINY vs full-config epoch budgets for the deep (torch) baselines. TINY is the
# default so the framework + first table are tractable; full-config is the later
# large-epoch run (see plan §"Notes / out of scope").
_DEEP_EPOCHS_TINY = 1
_DEEP_EPOCHS_FULL = 8
_MLP_MAX_ITER_TINY = 10


class _Loader(Protocol):
    """Structural type for the loader dependency (injectable for tests)."""

    def load_tabular(self, dataset: str, split: str | None = None) -> pd.DataFrame: ...


# ---------------------------------------------------------------------------
# Model battery
# ---------------------------------------------------------------------------


def build_models(
    tiny: bool = True, device: str | None = None
) -> list:
    """Return the 8 non-reference model templates (SmartPersistence is the ref).

    The battery is the E8 model suite minus SmartPersistence (which the runner
    always runs as the skill reference): LightGBM, MLP, DLinear, LSTM,
    iTransformer, PatchTST channel-independent + channel-dependent, and
    CrossViViT. Each is returned UNFITTED; the runner deep-copies a fresh
    instance per cell.

    ``tiny=True`` (default) shrinks the deep-model epoch budget to 1 and the
    MLP iteration cap to 10 so the framework + first table are tractable.
    ``tiny=False`` restores the framework defaults for the later full-config run.

    ``device`` is forwarded to every torch deep model's constructor (``"cpu"``
    gives deterministic tiny runs — the deep-model docstrings recommend it for
    tiny tests; auto-selecting CUDA on a broken GPU makes tiny mode fail).
    ``None`` (default) preserves each model's native auto-select behavior so
    existing Task 4 callers are unaffected.
    """
    epochs = _DEEP_EPOCHS_TINY if tiny else _DEEP_EPOCHS_FULL
    mlp_kw = {"max_iter": _MLP_MAX_ITER_TINY} if tiny else {}
    deep_kw = {"device": device} if device is not None else {}
    return [
        LightGBMModel(),
        MLPModel(**mlp_kw),
        DLinearModel(epochs=epochs, **deep_kw),
        LSTMModel(epochs=epochs, **deep_kw),
        iTransformerModel(epochs=epochs, **deep_kw),
        PatchTSTModel(variant="patchtst", name="PatchTST-CI", epochs=epochs, **deep_kw),
        PatchTSTModel(variant="patchtst_cd", name="PatchTST-CD", epochs=epochs, **deep_kw),
        CrossViViTModel(epochs=epochs, **deep_kw),
    ]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_benchmark(
    loader: _Loader,
    datasets: list[str] | None = None,
    horizons: list[int] | None = None,
    models: list | None = None,
    tiny: bool = True,
) -> list[dict]:
    """Run the E8 benchmark over every ``(dataset, horizon, model)`` cell.

    Parameters
    ----------
    loader
        Object exposing ``load_tabular(dataset, split)`` (a
        :class:`~solarunibench.loader.SolarUniBenchLoader` in production; a mock
        in tests). Pure: the runner never touches disk.
    datasets
        Datasets to evaluate. ``None`` → every dataset in
        :data:`DATASET_CONFIG` (folsom, skippd, sunlake, gefcom2014).
    horizons
        Horizon step counts applied to EVERY dataset. ``None`` → each dataset's
        ``cfg.horizons_steps`` (physics-time-aligned short/medium/long).
    models
        Non-reference model templates (each cell fits a fresh deep-copy).
        ``None`` → :func:`build_models` with the same ``tiny`` flag.
    tiny
        Forwarded to :func:`build_models` when ``models is None``. Ignored
        when ``models`` is supplied explicitly.

    Returns
    -------
    list of dict
        One row per ``(dataset, horizon, model)`` with keys ``dataset``,
        ``horizon``, ``model``, ``MAE``, ``RMSE``, ``Skill_MAE_pct``,
        ``Skill_RMSE_pct``, ``n_daytime``. SmartPersistence is emitted first
        per cell with Skill% = 0 (the reference).
    """
    if datasets is None:
        datasets = list(DATASET_CONFIG)
    if models is None:
        models = build_models(tiny=tiny)

    results: list[dict] = []
    for dataset in datasets:
        cfg = DATASET_CONFIG[dataset]
        df_train = add_clearness_aligned(loader.load_tabular(dataset, "train"), cfg.target)
        df_test = add_clearness_aligned(loader.load_tabular(dataset, "test"), cfg.target)
        # One frame with a `split` column so the pipeline can filter both splits
        # from a single df_provider(config) call (its internal contract).
        full = pd.concat(
            [df_train.assign(split="train"), df_test.assign(split="test")],
            ignore_index=True,
        )
        provider = _provider_for(full)

        hz_list = horizons if horizons is not None else list(cfg.horizons_steps)
        for hz in hz_list:
            results.extend(_run_cell(dataset, cfg, hz, df_test, full, provider, models))
    return results


# -- per-cell orchestration --------------------------------------------------


def _run_cell(
    dataset: str,
    cfg: DatasetConfig,
    horizon: int,
    df_test: pd.DataFrame,
    full: pd.DataFrame,
    provider: Callable,
    models: list,
) -> list[dict]:
    """Run SmartPersistence (reference) + every eligible model for one cell."""
    fe = EndogenousFeatureEngineer(cfg.target, feature_cols=list(_FEATURE_COLS))
    pconfig = PipelineConfig(
        dataset=dataset,
        target=cfg.target,
        lookback=cfg.lookback_steps,
        horizon=horizon,
        feature_cols=list(_FEATURE_COLS),
    )

    # Per-sample issue-time cos_zenith aligned with this cell's test windows.
    # cos_zenith is passthrough under PhysicalFeatureScaler (see module docstring)
    # so this matches the pipeline's internal scaled engineering exactly.
    cos_zenith_issue = fe.create_features(
        df_test, cfg.lookback_steps, horizon, 1
    )[4]
    if cos_zenith_issue is None:
        raise ValueError(
            f"dataset '{dataset}' has no cos_zenith column; the E8 evaluator "
            f"requires it for the daytime mask."
        )

    # Reference: SmartPersistence. Skill% is 0 by construction (model == ref).
    ref = _run_pipeline(pconfig, fe, SmartPersistenceModel(), provider)
    y_true = ref["y_true"]
    persistence_pred = ref["pred"]
    ref_metrics = evaluate_predictions(
        y_true, persistence_pred, cos_zenith_issue, persistence_pred
    )
    rows = [_row(dataset, horizon, "SmartPersistence", ref_metrics)]

    for template in models:
        # The reference is always SmartPersistence; skip a duplicate.
        if isinstance(template, SmartPersistenceModel):
            continue
        # CrossViViT is multimodal — run only where sky images exist.
        if isinstance(template, CrossViViTModel) and not cfg.image_dataset:
            continue
        # Fresh, unfitted instance per cell (no fit-state leakage).
        model = copy.deepcopy(template)
        res = _run_pipeline(pconfig, fe, model, provider)
        metrics = evaluate_predictions(
            y_true, res["pred"], cos_zenith_issue, persistence_pred
        )
        rows.append(_row(dataset, horizon, model.name, metrics))
    return rows


def _run_pipeline(
    config: PipelineConfig,
    feature_engineer: EndogenousFeatureEngineer,
    model,
    provider: Callable,
) -> dict:
    """Run one fit/predict pass and return the pipeline result dict."""
    return SolarForecastPipeline(
        config, feature_engineer, model, scaler=PhysicalFeatureScaler()
    ).run(df_provider=provider)


def _provider_for(full: pd.DataFrame) -> Callable[[PipelineConfig], pd.DataFrame]:
    """Build the df_provider the pipeline expects (returns the full frame)."""

    def _provider(_config: PipelineConfig) -> pd.DataFrame:
        return full

    return _provider


def _row(dataset: str, horizon: int, name: str, metrics: dict) -> dict:
    """Assemble one benchmark result row from the evaluator output."""
    return {
        "dataset": dataset,
        "horizon": horizon,
        "model": name,
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "Skill_MAE_pct": metrics["Skill_MAE_pct"],
        "Skill_RMSE_pct": metrics["Skill_RMSE_pct"],
        "n_daytime": metrics["n_daytime"],
    }
