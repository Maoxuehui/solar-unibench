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
import warnings
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd
import torch

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

# ---------------------------------------------------------------------------
# FULL-config presets (E8 upgrade: TINY → FULL). These encode the user's
# checklist: epochs=50 + patience=7 val-based early stopping, batch 64
# (CrossViViT 64 + accum 4 → micro-batch 16 for 24GB at 256x256), AdamW +
# CosineAnnealingLR, and a full-data stride for tractable window counts.
# ---------------------------------------------------------------------------

_FULL_EPOCHS = 50          # give PatchTST/CrossViViT room to converge
_FULL_PATIENCE = 7         # val-based early stopping
_FULL_BATCH = 64           # effective batch size
_FULL_ACCUM_CROSSVIVIT = 4  # CrossViViT: micro-batch 64//4 = 16 (24GB-friendly)
_FULL_LR = 1e-3
_FULL_WD = 1e-4            # AdamW weight decay
_FULL_VAL_FRACTION = 0.15  # chronological val holdout (last 15% of train windows)

# Full-data windowing stride by native sampling rate. Goal: full SEASONAL
# coverage (all 3 years) at a tractable window count, NOT every stride-1 window
# (stride-1 over 1.55M rows x lookback-240 ~ OOMs). 1-min data → stride 30
# (~50k windows over 3 years); 30-min / 1-h data is small enough for stride 1.
_FULL_STRIDE_BY_SAMPLING_MINS: dict[int, int] = {1: 30, 30: 1, 60: 1}


def _full_stride(sampling_mins: int) -> int:
    """FULL-mode windowing stride for a dataset's native sampling rate."""
    if sampling_mins not in _FULL_STRIDE_BY_SAMPLING_MINS:
        raise KeyError(
            f"no FULL stride configured for sampling_mins={sampling_mins} "
            f"(known: {sorted(_FULL_STRIDE_BY_SAMPLING_MINS)})"
        )
    return _FULL_STRIDE_BY_SAMPLING_MINS[sampling_mins]


# ---------------------------------------------------------------------------
# CUDA auto-fallback helpers
# ---------------------------------------------------------------------------


def probe_cuda(device: str = "cuda") -> bool:
    """Quick CUDA viability probe: run a tiny matmul; return whether it worked.

    Used to REPORT GPU status before a FULL run; a per-cell retry
    (:func:`run_with_cuda_fallback`) is the real safety net for transient errors.
    """
    if not torch.cuda.is_available() and device.startswith("cuda"):
        return False
    try:
        dev = torch.device(device)
        a = torch.randn(4, 4, device=dev)
        (a @ a).sum().item()
        if device.startswith("cuda"):
            torch.cuda.synchronize(dev)
        return True
    except RuntimeError:
        return False


def _is_cuda_runtime_error(exc: BaseException) -> bool:
    """Heuristic: did a torch op raise because of CUDA (vs a logic error)?"""
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc).lower()
    return any(
        tok in msg
        for tok in ("cuda", "out of memory", "cublas", "cudnn", "device-side")
    )


def _set_model_device_cpu(model: Any) -> None:
    """Best-effort move a torch ForecastModel's target device to CPU.

    Torch models read ``self.device`` at fit time (they rebuild the net there),
    so flipping ``model.device`` to CPU makes the next fit a CPU fit. CPU-only /
    non-torch models (LightGBM, MLP) are unaffected (they have no ``device``).
    """
    if hasattr(model, "device"):
        try:
            model.device = torch.device("cpu")
        except (AttributeError, TypeError):
            pass


def run_with_cuda_fallback(
    fn: Callable[[Any], Any],
    model: Any,
    *,
    on_retry: Callable[[Any], None] | None = None,
) -> Any:
    """Run ``fn(model)``; on a CUDA :class:`RuntimeError`, retry once on CPU.

    A transient CUDA error mid-fit (the classic OOM on a big image batch) must
    not kill a multi-hour FULL grid run: this wrapper catches CUDA runtime
    errors, warns, flips the model to CPU, drops any half-fit GPU net, and
    retries the whole fit/predict pass on CPU. Non-CUDA errors re-raise.
    """
    try:
        return fn(model)
    except RuntimeError as exc:
        if not _is_cuda_runtime_error(exc):
            raise
        warnings.warn(
            f"CUDA RuntimeError during {getattr(model, 'name', model)!s} fit/predict "
            f"({exc!r}); retrying this cell on CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        _set_model_device_cpu(model)
        if on_retry is not None:
            on_retry(model)
        return fn(model)


class _Loader(Protocol):
    """Structural type for the loader dependency (injectable for tests)."""

    def load_tabular(self, dataset: str, split: str | None = None) -> pd.DataFrame: ...


# ---------------------------------------------------------------------------
# Model battery
# ---------------------------------------------------------------------------


def build_models(
    tiny: bool = True, device: str | None = None, full: bool = False
) -> list:
    """Return the 8 non-reference model templates (SmartPersistence is the ref).

    The battery is the E8 model suite minus SmartPersistence (which the runner
    always runs as the skill reference): LightGBM, MLP, DLinear, LSTM,
    iTransformer, PatchTST channel-independent + channel-dependent, and
    CrossViViT. Each is returned UNFITTED; the runner deep-copies a fresh
    instance per cell.

    Parameters
    ----------
    tiny
        ``True`` (default) shrinks the deep-model epoch budget to 1 and the MLP
        iteration cap to 10 so the framework + first table are tractable.
        ``False`` restores the framework defaults (8-epoch deep models).
    full
        FULL-config presets (the E8 upgrade): overrides ``tiny`` and sets
        epochs=50, patience=7 val-based early stopping, AdamW(lr=1e-3,
        weight_decay=1e-4) + CosineAnnealingLR, chronological 15% val split,
        effective batch 64 (CrossViViT gets accum_steps=4 → micro-batch 16 for
        24GB at 256x256). ``device`` defaults to ``"cuda"`` (with per-cell
        CPU fallback in the runner) so the FULL grid trains on GPU.
    device
        Forwarded to every torch deep model's constructor. ``"cpu"`` gives
        deterministic tiny runs; ``"cuda"`` trains on GPU (FULL default). For
        FULL, ``None`` auto-selects ``"cuda"``.
    """
    if full:
        return _build_full_models(device)
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


def _build_full_models(device: str | None = None) -> list:
    """Build the 8-model battery with FULL-config presets (see :func:`build_models`)."""
    dev = device or "cuda"
    # Shared deep-model FULL kwargs. batch_size / accum_steps differ per model
    # (CrossViViT accumulates), so they are passed individually below.
    common = dict(
        epochs=_FULL_EPOCHS,
        lr=_FULL_LR,
        weight_decay=_FULL_WD,
        patience=_FULL_PATIENCE,
        val_fraction=_FULL_VAL_FRACTION,
        device=dev,
    )
    return [
        LightGBMModel(),
        MLPModel(),  # framework-default MLP iters (no tiny cap in FULL)
        DLinearModel(batch_size=_FULL_BATCH, **common),
        LSTMModel(batch_size=_FULL_BATCH, **common),
        iTransformerModel(batch_size=_FULL_BATCH, **common),
        PatchTSTModel(
            variant="patchtst", name="PatchTST-CI", batch_size=_FULL_BATCH, **common
        ),
        PatchTSTModel(
            variant="patchtst_cd", name="PatchTST-CD", batch_size=_FULL_BATCH, **common
        ),
        # CrossViViT image path (256x256, lookback 240): micro-batch 16 fits
        # 24GB; 4 accumulated micro-batches give an effective batch of 64.
        CrossViViTModel(
            batch_size=_FULL_BATCH, accum_steps=_FULL_ACCUM_CROSSVIVIT, **common
        ),
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
    full: bool = False,
    stride: int | None = None,
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
        ``None`` → :func:`build_models` with the same ``tiny``/``full`` flags.
    tiny
        Forwarded to :func:`build_models` when ``models is None``. Ignored
        when ``models`` is supplied explicitly.
    full
        FULL-config mode: builds models via :func:`build_models` with
        ``full=True`` (epochs=50, patience=7, AdamW+Cosine, effective batch 64,
        CrossViViT accum) and — unless ``stride`` is given explicitly — sets the
        windowing stride per dataset sampling rate (:func:`_full_stride`) so the
        full multi-year range windows tractably (~50k windows on 1-min data
        instead of the stride-1 OOM). Each cell is wrapped in
        :func:`run_with_cuda_fallback` so a transient CUDA error degrades to CPU
        instead of killing the grid.
    stride
        Explicit windowing stride applied to EVERY dataset (overrides the
        FULL-mode per-dataset default). ``None`` → ``1`` (tiny/default) or
        :func:`_full_stride` (full).

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
        # FULL → train on CUDA (per-cell CPU fallback handles transient errors).
        dev = "cuda" if full else None
        models = build_models(tiny=tiny, full=full, device=dev)

    results: list[dict] = []
    for dataset in datasets:
        cfg = DATASET_CONFIG[dataset]
        df_train = add_clearness_aligned(loader.load_tabular(dataset, "train"), cfg.target)
        df_test = add_clearness_aligned(loader.load_tabular(dataset, "test"), cfg.target)
        # One frame with a `split` column so the pipeline can filter both splits
        # from a single df_provider(config) call (its internal contract).
        full_df = pd.concat(
            [df_train.assign(split="train"), df_test.assign(split="test")],
            ignore_index=True,
        )
        provider = _provider_for(full_df)

        # FULL-mode per-dataset stride (full seasonal coverage, tractable
        # window count); explicit ``stride`` always wins.
        ds_stride = stride
        if ds_stride is None:
            ds_stride = _full_stride(cfg.sampling_mins) if full else 1

        hz_list = horizons if horizons is not None else list(cfg.horizons_steps)
        for hz in hz_list:
            results.extend(
                _run_cell(
                    dataset, cfg, hz, df_test, full_df, provider, models, ds_stride, full
                )
            )
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
    stride: int = 1,
    cuda_fallback: bool = False,
) -> list[dict]:
    """Run SmartPersistence (reference) + every eligible model for one cell.

    ``stride`` is the windowing stride applied to BOTH the pipeline (train+test
    windows) and the per-sample ``cos_zenith_issue`` mask so they stay aligned
    (a stride-1 mask over strided test predictions would misalign the evaluator).
    ``cuda_fallback`` wraps each model's fit/predict in
    :func:`run_with_cuda_fallback` so a transient CUDA error degrades to CPU.
    """
    fe = EndogenousFeatureEngineer(cfg.target, feature_cols=list(_FEATURE_COLS))
    pconfig = PipelineConfig(
        dataset=dataset,
        target=cfg.target,
        lookback=cfg.lookback_steps,
        horizon=horizon,
        stride=stride,
        feature_cols=list(_FEATURE_COLS),
    )

    # Per-sample issue-time cos_zenith aligned with this cell's test windows.
    # cos_zenith is passthrough under PhysicalFeatureScaler (see module docstring)
    # so this matches the pipeline's internal scaled engineering exactly. MUST
    # use the SAME stride as the pipeline so the mask length == n test windows.
    cos_zenith_issue = fe.create_features(
        df_test, cfg.lookback_steps, horizon, stride
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
        if cuda_fallback:
            res = _run_pipeline_with_fallback(pconfig, fe, model, provider)
        else:
            res = _run_pipeline(pconfig, fe, model, provider)
        metrics = evaluate_predictions(
            y_true, res["pred"], cos_zenith_issue, persistence_pred
        )
        rows.append(_row(dataset, horizon, model.name, metrics))
        # Release this torch model's GPU footprint before the next model so
        # the CUDA caching allocator does not accumulate ~24 GB across the
        # successive deep models in a cell and trip spurious kernel-launch
        # failures ("invalid configuration argument") under memory pressure.
        # Numerics are unaffected: the result row is already captured.
        del model, res
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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


def _run_pipeline_with_fallback(
    config: PipelineConfig,
    feature_engineer: EndogenousFeatureEngineer,
    model,
    provider: Callable,
) -> dict:
    """Run a cell's pipeline with CUDA → CPU fallback (see :func:`run_with_cuda_fallback`).

    On a CUDA RuntimeError the model is flipped to CPU and its half-fit GPU net
    is dropped before the retry, freeing GPU memory and ensuring a clean refit.
    """
    def _drop_net(m: Any) -> None:
        if hasattr(m, "_net"):
            m._net = None  # free the half-fit GPU net before the CPU refit

    return run_with_cuda_fallback(
        lambda m: _run_pipeline(config, feature_engineer, m, provider),
        model,
        on_retry=_drop_net,
    )


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
