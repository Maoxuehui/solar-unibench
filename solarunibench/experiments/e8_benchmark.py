"""E8 — Unified benchmark results (baselines x datasets x horizons).

Spec §11 (E8). *One standardized eval proving the benchmark is uniform.* Every
baseline is run on every dataset at the dev horizons and scored by **Daytime
skill vs Smart Persistence** (clear-sky-scaled persistence, the spec §10 skill
denominator) under the daytime window (``cos_zenith > 0.05``) — the unified
evaluation line across the whole grid.

Outputs (both routed through the shared :mod:`solarunibench.viz` design system):

* ``fig_benchmark_heatmap.png`` — Daytime-skill heatmap (rows = model,
  cols = dataset x horizon; diverging colormap centered at 0 = SmartPersistence
  parity; missing cells marked ``n/a``).
* ``results/results_benchmark.csv`` — one row per ``dataset x model x horizon``
  with ``Daytime_MAE``, ``Daytime_RMSE``, ``Daytime_skill``, ``n_daytime``.

Dev-tractable config (full-config numbers come at Phase 8)
---------------------------------------------------------
TINY model configs (MLP / LSTM / PatchTST / CrossViViT ``epochs=3``, small
width; small LightGBM) + a per-entity row cap + a coarse stride bound the run
on Folsom (~0.9M rows). **Deep models WILL be undertrained at these configs** —
PatchTST / CrossViViT may trail Smart Persistence. That is expected and
reported honestly: the E8 deliverable is the uniform-eval *infrastructure* (one
pipeline + scaler + daytime evaluator across every model x dataset x horizon);
the publishable numbers come at Phase 8 with full configs and the full
3-horizon grid (intra-day + day-ahead 24h).

CrossViViT runs ONLY where sky images resolve (Folsom / SKIPPD in the
standardized packages); it is omitted on GEFCom / SunLake where the image
modality has no signal (the model class otherwise degrades to a tabular-only
self-attention path, which would be a misleading benchmark row).

Each ``(model, dataset, horizon)`` cell uses a FRESH model instance + the shared
:class:`SolarForecastPipeline`, so :class:`PhysicalFeatureScaler` scaling
(default-on, fit on TRAIN only) and the daytime evaluator are uniform across the
whole grid — the load-bearing correctness property of the benchmark. Smart
Persistence is run once per ``(dataset, horizon)`` as the skill reference and
also appears as a model row with ``Daytime_skill = 0`` (parity with itself).

Pure function contract: ``run(loader, out_dir, *, model_factory=None,
datasets=None) -> dict``. Reads only standardized packages via ``loader``;
writes to ``out_dir``; never mutates loader state or standardized data.
``model_factory`` and ``datasets`` are optional test/dev hooks (scope the model
list / dataset grid); the orchestrator calls ``run(loader, out_dir)`` with the
defaults (all baselines, all standardized datasets).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..forecast.core import (
    EndogenousFeatureEngineer,
    PipelineConfig,
    SolarForecastPipeline,
)
from ..forecast.models import (
    CrossViViTModel,
    LightGBMModel,
    LSTMModel,
    MLPModel,
    PatchTSTModel,
    SmartPersistenceModel,
)
from ..viz.io import save_fig, save_metrics
from ..viz.style import set_style
from ._config import (  # Phase 8 Stream B: TINY dev by default, FULL when SUB_FULL_CONFIG=1
    CROSSVIVIT_PARAMS as _CROSSVIVIT_PARAMS,
    HORIZONS_HOURS,
    LGBM_PARAMS as _LGBM_PARAMS,
    LSTM_PARAMS as _LSTM_PARAMS,
    MAX_ROWS_PER_ENTITY as _MAX_ROWS_PER_ENTITY,
    MLP_PARAMS as _MLP_PARAMS,
    PATCHTST_PARAMS as _PATCHTST_PARAMS,
    FULL as _FULL,
)

__all__ = ["run", "DATASET_TARGETS", "HORIZONS_HOURS"]


# ---------------------------------------------------------------------------
# Bounded dev configuration (full-config numbers come at Phase 8).
# ---------------------------------------------------------------------------

#: Per-dataset target (spec §10): PV for GEFCom/SKIPPD, GHI for Folsom/SunLake.
DATASET_TARGETS: dict[str, str] = {
    "gefcom2014": "pv_power_normalized",
    "skippd": "pv_power_normalized",
    "folsom": "ghi_actual",
    "sunlake": "ghi_actual",
}

#: Forecast-horizon grid imported from :mod:`._config` (dev=(1,6); FULL adds 24h).

#: Modest lookback + coarse stride that bound the window count on Folsom while
#: keeping enough daytime samples for a finite daytime MAE on small frames.
_LOOKBACK: int = 12
_STRIDE: int = 12

#: Row cap imported from :mod:`._config` (dev=2000; FULL=0 / no cap).

#: Nearest-neighbor match tolerance (image<->tabular timestamp) for sky-image
#: attach on Folsom (second-cadence images vs hourly tabular) and SKIPPD.
_IMAGE_MATCH_TOLERANCE: str = "30min"

#: Common feature block (physics anchors) every tabular model sees, so the
#: design matrix is uniform across the grid. ``clearness_index`` is Honest-NaN
#: on PV-only datasets and is filtered out per dataset by :func:`_usable_features`.
_FEATURES: tuple[str, ...] = ("ghi_clearsky", "cos_zenith", "clearness_index")
_FINITE_FRAC_THRESHOLD: float = 0.5

# -- Model param presets imported from :mod:`._config` (TINY dev / FULL Phase 8).

# Active-config provenance stamp for the results JSON.
_CONFIG_TAG: str = "FULL (Phase 8)" if _FULL else "TINY dev"

#: Display-name order (low -> high complexity) for the heatmap row ordering.
_MODEL_ORDER: tuple[str, ...] = (
    "SmartPersistence", "LightGBM", "MLP", "LSTM", "PatchTST", "CrossViViT",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _finite_fraction(series: pd.Series) -> float:
    """Fraction of finite values in a column (0 for all-NaN / object-None)."""
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return float(np.isfinite(arr).mean()) if arr.size else 0.0


def _usable_features(df: pd.DataFrame) -> list[str]:
    """Feature columns present AND mostly finite (Honest-NaN filter).

    ``clearness_index`` is dropped on PV-only datasets where it is entirely NaN.
    """
    return [
        c for c in _FEATURES
        if c in df.columns and _finite_fraction(df[c]) > _FINITE_FRAC_THRESHOLD
    ]


def _dataset_has_images(loader, dataset: str) -> bool:
    """True iff ``loader`` exposes a non-empty sky-image cube for ``dataset``."""
    get_ts = getattr(loader, "get_image_timestamps", None)
    get_cube = getattr(loader, "get_tensor_cube", None)
    if get_ts is None or get_cube is None:
        return False
    try:
        ts = get_ts(dataset)
        cube = get_cube(dataset, "sky_images")
    except Exception:  # defensive: a minimal/loader stub raises on unknown kind
        return False
    return ts is not None and cube is not None and len(ts) > 0


def _cap_rows_per_entity(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Keep the first ``cap`` rows per (entity, split) (temporal dev slice).

    Bounds window counts on Folsom (~0.9M rows) so the deep models fit in
    seconds. Capping is done PER SPLIT (not on the whole frame) so each split
    retains coverage — standardized splits are temporal (train early, test
    late), so capping the whole frame would keep only train rows and empty the
    test windows. Applied identically to TRAIN and TEST so the daytime mask is
    shared across all models (fair comparison). No-op when ``cap <= 0``.
    """
    if cap <= 0:
        return df
    group_cols = ["entity_id"] + (["split"] if "split" in df.columns else [])
    return (
        df.sort_values([*group_cols, "timestamp_utc"], kind="mergesort")
        .groupby(group_cols, sort=False, group_keys=False)
        .head(cap)
        .reset_index(drop=True)
    )


def _skill(arm_mae: float, ref_mae: float) -> float:
    """Forecast skill vs the Smart Persistence reference.

    ``1 - mae_arm / mae_ref`` (0 = parity, positive = better). NaN when either
    MAE is non-finite or the reference MAE is exactly 0.
    """
    if not (np.isfinite(arm_mae) and np.isfinite(ref_mae)) or ref_mae == 0.0:
        return float("nan")
    return float(1.0 - arm_mae / ref_mae)


def _uses_images(model) -> bool:
    """True for image-aware models that consume the sky-image cube (CrossViViT,
    non-tabular variant)."""
    return isinstance(model, CrossViViTModel) and getattr(
        model, "variant", "crossvivit"
    ) != "tabular"


def _build_models(has_images: bool) -> list:
    """Fresh TINY baseline instances for one ``(dataset, horizon)`` cell.

    Returns the 5 non-reference baselines (SmartPersistence is the reference,
    handled separately). CrossViViT is included only when the dataset has sky
    images — its image modality has no signal on purely tabular packages.
    """
    models = [
        LightGBMModel(params=_LGBM_PARAMS),
        MLPModel(**_MLP_PARAMS),
        LSTMModel(**_LSTM_PARAMS),
        PatchTSTModel(**_PATCHTST_PARAMS),
    ]
    if has_images:
        models.append(CrossViViTModel(**_CROSSVIVIT_PARAMS))
    return models


def _run_one(
    loader,
    dataset: str,
    target: str,
    feature_cols: list[str],
    horizon: int,
    model,
    with_images: bool,
) -> dict:
    """Run one model arm via :class:`SolarForecastPipeline` -> daytime metrics.

    The pipeline applies :class:`PhysicalFeatureScaler` (fit on TRAIN only) and
    the daytime evaluator uniformly, so every cell in the grid is scored on the
    same footing. The frame is read-only (loaded + NaN-target rows dropped + a
    dev row cap); standardized data is never mutated.
    """
    data_root = str(getattr(loader, "data_root", "") or "")
    config = PipelineConfig(
        dataset=dataset,
        target=target,
        lookback=_LOOKBACK,
        horizon=horizon,
        stride=_STRIDE,
        feature_cols=list(feature_cols),
        data_root=data_root or None,
    )
    if with_images:
        engineer = EndogenousFeatureEngineer(
            target,
            feature_cols=list(feature_cols),
            return_images=True,
            image_loader=loader,
            image_dataset=dataset,
            match_tolerance=_IMAGE_MATCH_TOLERANCE,
        )
    else:
        engineer = EndogenousFeatureEngineer(target, feature_cols=list(feature_cols))
    pipeline = SolarForecastPipeline(config, engineer, model)

    def _df_provider(cfg):
        d = loader.load_tabular(cfg.dataset)
        d = d.dropna(subset=[target]).reset_index(drop=True)
        d = _cap_rows_per_entity(d, _MAX_ROWS_PER_ENTITY)
        return d

    try:
        res = pipeline.run(df_provider=_df_provider)
    except (ValueError, RuntimeError) as exc:
        # Honest NaN: a cell whose engineered test set is empty (e.g. a split
        # too short for any lookback+horizon window) or a model that cannot
        # shape an empty batch is reported as NaN rather than aborting the whole
        # benchmark grid. Visible via the warning; the cell shows ``n/a``.
        import warnings

        warnings.warn(
            f"E8 {model.name}/{dataset}/h{horizon}: evaluation failed ({exc!r}); "
            f"reporting NaN",
            stacklevel=2,
        )
        return {
            "Daytime_MAE": float("nan"),
            "Daytime_RMSE": float("nan"),
            "n_daytime": 0,
            "n_total": 0,
        }
    day = res.get("daytime", {}) or {}
    return {
        "Daytime_MAE": float(day.get("Daytime_MAE", float("nan"))),
        "Daytime_RMSE": float(day.get("Daytime_RMSE", float("nan"))),
        "n_daytime": int(day.get("n_daytime", 0)),
        "n_total": int(day.get("n_total", 0)),
    }


def _row(
    dataset: str, target: str, horizon: int, model_name: str, m: dict, skill: float
) -> dict:
    return {
        "dataset": dataset,
        "target": target,
        "horizon": horizon,
        "model": model_name,
        "Daytime_MAE": m["Daytime_MAE"],
        "Daytime_RMSE": m["Daytime_RMSE"],
        "Daytime_skill": skill,
        "n_daytime": m["n_daytime"],
        "n_total": m["n_total"],
    }


# ---------------------------------------------------------------------------
# Figure: Daytime-skill heatmap (model x dataset x horizon)
# ---------------------------------------------------------------------------

def _plot_heatmap(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Daytime-skill heatmap: rows = model, cols = dataset x horizon.

    Diverging colormap centered at 0 (SmartPersistence parity). Missing cells
    (CrossViViT on tabular-only datasets) are masked grey and labelled ``n/a``.
    """
    from matplotlib.colors import TwoSlopeNorm

    work = df.copy()
    work["col"] = work["dataset"] + "\n" + work["horizon"].astype(str) + "h"
    pivot = work.pivot_table(
        index="model", columns="col", values="Daytime_skill", aggfunc="mean"
    )
    # Stable row order: complexity rank (missing models just drop out).
    row_order = [m for m in _MODEL_ORDER if m in pivot.index]
    pivot = pivot.reindex(row_order)
    # Stable column order: dataset (declaration order) then horizon.
    col_order: list[str] = []
    for ds in DATASET_TARGETS:
        for h in HORIZONS_HOURS:
            label = f"{ds}\n{h}h"
            if label in pivot.columns:
                col_order.append(label)
    # Append any unexpected columns (e.g. a custom horizon) at the end, sorted.
    extras = sorted(c for c in pivot.columns if c not in col_order)
    pivot = pivot[col_order + extras]

    vals = pivot.to_numpy(dtype=float)
    n_rows, n_cols = pivot.shape
    fig, ax = plt.subplots(
        figsize=(max(8.0, 1.15 * n_cols + 2.5), max(3.8, 0.65 * n_rows + 1.8))
    )

    finite = vals[np.isfinite(vals)]
    vmax = float(np.percentile(np.abs(finite), 95)) if finite.size else 0.5
    vmax = max(vmax, 0.05)  # avoid a degenerate norm when skill is near-zero
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#dddddd")  # masked (missing) cells
    masked = np.ma.masked_invalid(vals)
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            v = vals[i, j]
            if np.isfinite(v):
                ax.text(
                    j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                    color="black",
                )
            else:
                ax.text(
                    j, i, "n/a", ha="center", va="center", fontsize=8,
                    color="#888888",
                )

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(pivot.columns, rotation=0)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("dataset x horizon")
    ax.set_ylabel("model")
    ax.set_title(
        "Unified benchmark — Daytime skill vs Smart Persistence (E8)\n"
        f"{_CONFIG_TAG} configs"
    )
    fig.colorbar(im, ax=ax, shrink=0.85,
                 label="Daytime skill (0 = SmartPersistence parity, + = better)")
    return save_fig(fig, "fig_benchmark_heatmap", out_dir)


# ---------------------------------------------------------------------------
# Results CSV
# ---------------------------------------------------------------------------

def _save_results_csv(rows: list[dict], out_dir: Path) -> Path:
    return save_metrics(pd.DataFrame(rows), "results_benchmark", out_dir)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(loader, out_dir, *, model_factory=None, datasets=None) -> dict:
    """Run experiment E8 and return ``{figures: [paths], metrics: {...}}``.

    Reads only standardized packages via ``loader``; writes the heatmap figure +
    a results CSV under ``out_dir``. Pure: does not mutate ``loader`` state or
    the standardized data.

    Parameters
    ----------
    loader:
        Standardized-package loader (:class:`SolarUniBenchLoader` or test stub).
    out_dir:
        Directory the figure + ``results/`` CSV are written under.
    model_factory:
        Optional callable ``has_images -> list[ForecastModel]`` (test/dev hook).
        Defaults to :func:`_build_models`, which builds all baselines with TINY
        configs (CrossViViT only where the dataset has sky images).
    datasets:
        Optional iterable of dataset names to scope the grid (test/dev hook).
        Defaults to every entry of :data:`DATASET_TARGETS`. Per-dataset targets
        are resolved from :data:`DATASET_TARGETS`.
    """
    set_style("light")
    out_dir = Path(out_dir)
    build = model_factory or _build_models
    grid = list(datasets) if datasets is not None else list(DATASET_TARGETS)

    # Pre-load each frame once to compute the shared feature block per dataset.
    feature_blocks: dict[str, list[str]] = {}
    image_flags: dict[str, bool] = {}
    for dataset in grid:
        df = loader.load_tabular(dataset)
        feature_blocks[dataset] = _usable_features(df)
        image_flags[dataset] = _dataset_has_images(loader, dataset)

    rows: list[dict] = []
    for dataset in grid:
        target = DATASET_TARGETS[dataset]
        feats = feature_blocks[dataset]
        has_images = image_flags[dataset]
        for horizon in HORIZONS_HOURS:
            # Reference: SmartPersistence (defines the daytime subset + the
            # skill denominator). Run once per (dataset, horizon).
            ref_model = SmartPersistenceModel()
            ref = _run_one(
                loader, dataset, target, feats, horizon, ref_model,
                with_images=False,
            )
            ref_mae = ref["Daytime_MAE"]
            # SmartPersistence is also a model row: skill = 0 (parity w/ self).
            rows.append(_row(dataset, target, horizon, "SmartPersistence", ref, 0.0))

            # Every other baseline, fresh instance, uniform pipeline + daytime eval.
            for model in build(has_images):
                m = _run_one(
                    loader, dataset, target, feats, horizon, model,
                    with_images=_uses_images(model),
                )
                rows.append(
                    _row(dataset, target, horizon, model.name, m, _skill(m["Daytime_MAE"], ref_mae))
                )

    df_out = pd.DataFrame(rows)
    _save_results_csv(rows, out_dir)
    figures: list[Path] = []
    if len(df_out):
        figures += _plot_heatmap(df_out, out_dir)

    metrics = {
        "rows": rows,
        "config": {
            "lookback": _LOOKBACK,
            "horizons": list(HORIZONS_HOURS),
            "stride": _STRIDE,
            "max_rows_per_entity": _MAX_ROWS_PER_ENTITY,
            "reference": "SmartPersistence",
            "config_tag": _CONFIG_TAG,
        },
        "image_datasets": [d for d, f in image_flags.items() if f],
    }
    return {"figures": [str(p) for p in figures], "metrics": metrics}
