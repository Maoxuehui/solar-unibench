"""E9 — Data governance vs model complexity (spec §11, E9; the 治理重于模型 centerpiece).

*Hypothesis:* cleaning + alignment buys more metric gain than scaling model
complexity — **a simple model on well-governed data can match or beat a complex
model on raw data.** Governance dominates complexification.

**2-axis ablation (the spec's design):**

* **Axis 1 — data state:** {raw / unaligned / uncleaned} vs {pipeline-cleaned +
  aligned}. The RAW arm is the pre-governance frame (a ``raw_provider(dataset)
  -> df`` hook, exactly E5's pattern: ``adapter.run_pipeline(raw=True, limit=...)``
  into a tmp dir). The CLEANED arm is derived by running
  :func:`apply_quality_audit` on the raw frame — i.e. precisely what
  ``run_pipeline(raw=False)`` does to the anchor-merged frame — so the two arms
  are ROW-ALIGNED and differ ONLY in governance (quality-audit + night-zeroing +
  physical clamp). Anchors (``ghi_clearsky`` / ``cos_zenith``) are computed
  BEFORE the audit and are IDENTICAL in both arms; the only difference is the
  governed target. (This is equivalent to the standardized package, which is
  ``raw + audit``.)
* **Axis 2 — model complexity:** a clear ladder
  ``SmartPersistence -> LightGBM -> MLP -> LSTM -> PatchTST`` (CrossViViT is
  omitted to keep the grid tabular/sequence-only + tractable, per the plan).

The full grid (2 data states x N models) is run on a representative dataset
(GEFCom — PV, fixed data, fast). Every cell is scored by **Daytime skill vs
SmartPersistence** under the daytime window (``cos_zenith > 0.05``).

**Fair comparison (E5 discipline):** both arms evaluate on the SAME cleaned test
windows (clean ground truth) and share an identical per-entity temporal split.
So ``n_daytime`` is identical across the whole grid — the ONLY thing that varies
between the raw and cleaned arms is whether the TRAINING labels were governed.
The shared reference is SmartPersistence on the cleaned frame (skill = 0 by
construction); every cell's skill is ``1 - MAE_cell / MAE_SP_cleaned``, so skill
is directly comparable across the whole 2-axis grid.

**Key contrasts (the headline numbers):**

1. **Δ-hygiene[model]** = ``skill(model, cleaned) - skill(model, raw)`` — the
   metric gain from data governance, same model (raw -> cleaned).
2. **Δ-complexity[model, data]** = ``skill(model, data) - skill(SP, data)`` —
   the metric gain from scaling complexity, same data state.
3. **The crossover** — report where **"simple + clean >= complex + raw"**: does
   the simplest non-trivial model on CLEANED match/beat the MOST complex model
   on RAW? This is the empirical proof that governance dominates.

**Honesty is paramount (this is the centerpiece):** the REAL grid is reported
even when the crossover is ambiguous at TINY dev configs (deep models are
undertrained at ``epochs=3``). The synthetic test constructs a case where
governance clearly dominates (injected label noise the audit removes + a model
ladder) and asserts the crossover there — proving the MEASUREMENT works. The
publishable crossover comes at Phase 8 with full configs.

Outputs (routed through the shared :mod:`solarunibench.viz` design system):

* ``fig_governance_vs_complexity.png`` — grouped bars of Daytime skill over the
  {data state x model} grid (raw hatched grey vs cleaned colored per model),
  with Δ-hygiene annotated per model — the 2-axis picture.
* ``fig_cleaning_delta.png`` — per-model Δ-hygiene (cleaning gain) vs
  Δ-complexity on raw (complexity gain), side-by-side, with the crossover
  annotated — the "hygiene > complexity" visual.
* ``results/results_e9_governance.csv`` — the full grid + the two Δ columns.

Pure function contract: ``run(loader, out_dir, *, raw_provider=None,
model_ladder=None, datasets=None) -> dict``. ``raw_provider`` is REQUIRED to
build the grid (RAW data is not in the standardized packages); without it E9
honestly skips. The ``loader`` is the contract's first arg (uniform with E1-E8);
E9 sources both arms from ``raw_provider`` for rigorous row-alignment, so the
loader is reserved for the orchestrator's uniform dispatch. Reads only the raw
frames + writes figures + a results CSV under ``out_dir``; never mutates
standardized data.
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
    LightGBMModel,
    LSTMModel,
    MLPModel,
    PatchTSTModel,
    SmartPersistenceModel,
)
from ..operators.quality import apply_quality_audit
from ..viz.io import save_fig, save_metrics
from ..viz.style import MODEL_COLORS, PALETTE, set_style
from ._config import (  # Phase 8 Stream B: TINY dev by default, FULL when SUB_FULL_CONFIG=1
    FULL as _FULL,
    LGBM_PARAMS as _BASE_LGBM,
    MLP_PARAMS as _BASE_MLP,
    LSTM_PARAMS as _BASE_LSTM,
    PATCHTST_PARAMS as _BASE_PATCHTST,
)

__all__ = ["run", "DATASET_TARGETS", "COMPLEXITY_LADDER", "DEFAULT_DATASETS"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Per-dataset forecasting target (spec §10) — PV for GEFCom/SKIPPD, GHI for
#: Folsom/SunLake.
DATASET_TARGETS: dict[str, str] = {
    "gefcom2014": "pv_power_normalized",
    "skippd": "pv_power_normalized",
    "folsom": "ghi_actual",
    "sunlake": "ghi_actual",
}

#: Representative fast dataset for the 2-axis grid (spec E9: GEFCom — PV, fixed
#: data, tabular-only). Phase 8 extends the grid to more datasets.
DEFAULT_DATASETS: tuple[str, ...] = ("gefcom2014",)

#: Display-name order (low -> high complexity) — the complexity axis. Drives
#: ``complexity_rank`` and the figure axis order.
COMPLEXITY_LADDER: tuple[str, ...] = (
    "SmartPersistence",
    "LightGBM",
    "MLP",
    "LSTM",
    "PatchTST",
)

#: Shared feature block (the clear-sky anchors). Computed BEFORE the quality
#: audit so they are IDENTICAL in raw and cleaned arms — the only arm difference
#: is the governed target.
_SHARED_FEATURES: tuple[str, ...] = ("ghi_clearsky", "cos_zenith")

#: Bounded dev configuration (full-config numbers come at Phase 8).
_LOOKBACK: int = 12
_HORIZON: int = 1
_STRIDE: int = 6
_TRAIN_FRAC: float = 0.70
_DAYTIME_COS_ZENITH: float = 0.05
_DEFAULT_CAPACITY: float = 1.0

# -- Model param presets (Phase 8 Stream B). TINY dev (E9's exact ladder,
# LightGBM n_estimators=40) by default; FULL capacity (shared experiment-wide
# presets, GPU, to-convergence) when SUB_FULL_CONFIG=1.
if _FULL:
    _LGBM_PARAMS: dict = _BASE_LGBM
    _MLP_PARAMS: dict = _BASE_MLP
    _LSTM_PARAMS: dict = _BASE_LSTM
    _PATCHTST_PARAMS: dict = _BASE_PATCHTST
else:
    _LGBM_PARAMS = {"n_estimators": 40, "n_jobs": 1, "random_state": 0}
    _MLP_PARAMS = {"hidden_layer_sizes": (32,), "max_iter": 80, "random_state": 0}
    _LSTM_PARAMS = {
        "hidden_size": 16,
        "num_layers": 1,
        "epochs": 3,
        "batch_size": 64,
        "device": "cpu",
    }
    _PATCHTST_PARAMS = {
        "d_model": 32,
        "n_heads": 4,
        "patch_len": 4,
        "stride": 2,
        "depth": 1,
        "epochs": 3,
        "batch_size": 64,
        "device": "cpu",
    }
_CONFIG_TAG: str = "FULL (Phase 8)" if _FULL else "TINY dev"

#: SmartPersistence is the physical skill reference AND the ladder's first rung.
#: The "non-trivial" rungs (everything after SP) are the complexity ladder whose
#: gains Δ-complexity measures; the crossover uses the FIRST non-trivial rung as
#: "simple" and the LAST rung as "complex".
_REFERENCE_MODEL: str = "SmartPersistence"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skill(arm_mae: float, ref_mae: float) -> float:
    """Forecast skill vs the Smart Persistence reference.

    ``1 - mae_arm / mae_ref`` (0 = parity, positive = better than reference).
    NaN when either MAE is non-finite or the reference MAE is exactly 0.
    """
    if not (np.isfinite(arm_mae) and np.isfinite(ref_mae)) or ref_mae == 0.0:
        return float("nan")
    return float(1.0 - arm_mae / ref_mae)


def _derive_cleaned(raw_df: pd.DataFrame, capacity: float = _DEFAULT_CAPACITY) -> pd.DataFrame:
    """Run the quality audit on a raw frame -> the cleaned (governed) frame.

    Mirrors ``run_pipeline(raw=False)``: the audit runs on the anchor-merged
    frame, so the NA ``quality_flag`` / ``clearness_index`` the raw snapshot
    carries are dropped first and the audit owns them. Produces a frame
    row-aligned with ``raw_df`` and differing ONLY in governance.
    """
    base = raw_df.drop(
        columns=[c for c in ("quality_flag", "clearness_index") if c in raw_df.columns]
    )
    return apply_quality_audit(base, capacity=capacity)


def _assign_temporal_split(df: pd.DataFrame, train_frac: float = _TRAIN_FRAC) -> pd.Series:
    """Per-entity temporal split label (``train``/``test``), row-aligned with ``df``.

    Within each entity, rows are ordered by ``timestamp_utc`` and the first
    ``train_frac`` are labeled ``train`` (the rest ``test``). Row-aligned so the
    SAME split label transfers from cleaned to raw (row-aligned arms share
    identical train/test boundaries).
    """
    labels = pd.Series("test", index=df.index, dtype=object)
    for _, g in df.groupby("entity_id", sort=False):
        g_sorted = g.sort_values("timestamp_utc", kind="mergesort")
        n = len(g_sorted)
        n_tr = int(round(n * train_frac))
        labels.loc[g_sorted.index[:n_tr]] = "train"
    return labels


# ---------------------------------------------------------------------------
# Default model ladder (TINY configs; full configs at Phase 8)
# ---------------------------------------------------------------------------

def _build_ladder() -> list:
    """Fresh instances of the full complexity ladder (low -> high).

    Returns fresh instances on every call so each (dataset, arm) cell gets a
    clean model — no weight leakage between the raw and cleaned arms.
    """
    return [
        SmartPersistenceModel(),
        LightGBMModel(params=_LGBM_PARAMS),
        MLPModel(**_MLP_PARAMS),
        LSTMModel(**_LSTM_PARAMS),
        PatchTSTModel(**_PATCHTST_PARAMS),
    ]


# ---------------------------------------------------------------------------
# Run one (model, arm) cell via the shared SolarForecastPipeline
# ---------------------------------------------------------------------------

def _run_arm(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    target: str,
    model,
) -> dict:
    """Run one model arm -> daytime MAE + counts.

    ``df_train`` / ``df_test`` carry a ``split`` column (``train`` / ``test``).
    The pipeline applies :class:`PhysicalFeatureScaler` (fit on TRAIN only) and
    the daytime evaluator uniformly, so every cell is scored on the same footing.
    """
    config = PipelineConfig(
        dataset="e9",
        target=target,
        lookback=_LOOKBACK,
        horizon=_HORIZON,
        stride=_STRIDE,
        feature_cols=list(_SHARED_FEATURES),
        data_root=None,  # df_provider supplies the frame; data_root unused.
    )
    engineer = EndogenousFeatureEngineer(target, feature_cols=list(_SHARED_FEATURES))
    pipeline = SolarForecastPipeline(config, engineer, model)

    combined = pd.concat([df_train, df_test], ignore_index=True)

    def _df_provider(cfg):
        return combined

    try:
        res = pipeline.run(df_provider=_df_provider)
    except (ValueError, RuntimeError) as exc:
        # Honest NaN: a cell whose engineered test set is empty (e.g. a split too
        # short for any lookback+horizon window) or a model that cannot shape an
        # empty/tiny batch (deep models at TINY configs) is reported as NaN
        # rather than aborting the whole grid — the centerpiece must publish the
        # REAL grid even when a deep-model cell degrades. Visible via the warning.
        import warnings

        warnings.warn(
            f"E9 {model.name} arm: evaluation failed ({exc!r}); reporting NaN",
            stacklevel=2,
        )
        return {
            "daytime_mae": float("nan"),
            "daytime_rmse": float("nan"),
            "n_daytime": 0,
            "n_total": 0,
        }
    day = res.get("daytime", {}) or {}
    return {
        "daytime_mae": float(day.get("Daytime_MAE", float("nan"))),
        "daytime_rmse": float(day.get("Daytime_RMSE", float("nan"))),
        "n_daytime": int(day.get("n_daytime", 0)),
        "n_total": int(day.get("n_total", 0)),
    }


def _evaluate_dataset(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    target: str,
    build_ladder,
) -> dict:
    """Run the 2-axis grid (raw/cleaned x model ladder) for one dataset.

    Returns ``{model_name: {complexity_rank, raw: {...}, cleaned: {...},
    delta_hygiene, delta_complexity_raw, delta_complexity_cleaned,
    crossover_simple_clean_beats_complex_raw}}``. Both arms evaluate on the SAME
    cleaned test windows (E5 discipline); the shared reference is
    SmartPersistence on cleaned (skill = 0).
    """
    # Shared temporal split — assign once on cleaned, transfer to raw (row-aligned).
    split = _assign_temporal_split(cleaned_df)
    cleaned_df = cleaned_df.assign(split=split)
    raw_df = raw_df.assign(split=split)

    # Drop governance-metadata columns from arm inputs: they are NOT model
    # features (the engineer uses only [target, ghi_clearsky, cos_zenith]) and
    # leaving them would (a) leak cleaning into the raw arm's test frame and
    # (b) trigger a pandas concat dtype warning on all-NA vs populated
    # quality_flag columns. The audit's effect is measured via the TARGET only.
    meta_cols = [c for c in ("quality_flag", "clearness_index") if c in cleaned_df.columns]
    cleaned_train = (
        cleaned_df[cleaned_df["split"] == "train"]
        .drop(columns=meta_cols)
        .reset_index(drop=True)
    )
    cleaned_test = (
        cleaned_df[cleaned_df["split"] == "test"]
        .drop(columns=meta_cols)
        .reset_index(drop=True)
    )
    raw_train = (
        raw_df[raw_df["split"] == "train"]
        .drop(columns=meta_cols)
        .reset_index(drop=True)
    )

    # Two FRESH ladder instances — one per arm — so no model weights leak
    # between the raw and cleaned arms. Both ladders have identical names/order.
    raw_models = build_ladder()
    cl_models = build_ladder()
    names = [m.name for m in cl_models]

    cells: dict[str, dict] = {}
    for raw_m, cl_m, rank in zip(raw_models, cl_models, range(len(names))):
        assert raw_m.name == cl_m.name, "ladder order unstable across calls"
        name = cl_m.name
        # Raw arm: trained on RAW train, scored on CLEANED test (clean ground truth).
        raw_arm = _run_arm(raw_train, cleaned_test, target=target, model=raw_m)
        # Cleaned arm: trained on CLEANED train, scored on CLEANED test.
        cl_arm = _run_arm(cleaned_train, cleaned_test, target=target, model=cl_m)
        cells[name] = {
            "complexity_rank": int(rank),
            "raw": {
                "daytime_mae": raw_arm["daytime_mae"],
                "n_daytime": raw_arm["n_daytime"],
                "n_total": raw_arm["n_total"],
            },
            "cleaned": {
                "daytime_mae": cl_arm["daytime_mae"],
                "n_daytime": cl_arm["n_daytime"],
                "n_total": cl_arm["n_total"],
            },
        }
        # Free the model pair + clear the CUDA caching allocator between fits.
        # E9 does 2 torch fits per ladder rung in one process; without this the
        # allocator fragments on Windows + cuDNN and stalls (same fix as E8).
        try:
            import torch as _torch
            del raw_m, cl_m
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass

    # Shared reference = SmartPersistence on cleaned (skill = 0 by construction).
    # The spec's ladder starts with SmartPersistence; a custom ladder without it
    # cannot define the skill denominator → fail loud rather than silently.
    if _REFERENCE_MODEL not in cells:
        raise ValueError(
            f"model_ladder must include '{_REFERENCE_MODEL}' (the SmartPersistence "
            f"skill reference); got {names}"
        )
    ref_mae = cells[_REFERENCE_MODEL]["cleaned"]["daytime_mae"]

    # Skill + Δ-hygiene + Δ-complexity per model.
    sp_raw_skill = _skill(cells[_REFERENCE_MODEL]["raw"]["daytime_mae"], ref_mae)
    sp_cl_skill = 0.0  # SP-on-cleaned IS the reference.
    for name, cell in cells.items():
        cell["raw"]["skill"] = _skill(cell["raw"]["daytime_mae"], ref_mae)
        cell["cleaned"]["skill"] = _skill(cell["cleaned"]["daytime_mae"], ref_mae)
        cell["delta_hygiene"] = (
            float(cell["cleaned"]["skill"] - cell["raw"]["skill"])
            if (np.isfinite(cell["cleaned"]["skill"]) and np.isfinite(cell["raw"]["skill"]))
            else float("nan")
        )
        cell["delta_complexity_raw"] = (
            float(cell["raw"]["skill"] - sp_raw_skill)
            if np.isfinite(cell["raw"]["skill"]) and np.isfinite(sp_raw_skill)
            else float("nan")
        )
        cell["delta_complexity_cleaned"] = (
            float(cell["cleaned"]["skill"] - sp_cl_skill)
            if np.isfinite(cell["cleaned"]["skill"])
            else float("nan")
        )

    # Crossover per model: does THIS model on cleaned beat the MOST complex on raw?
    most_complex = names[-1]
    most_complex_raw_skill = cells[most_complex]["raw"]["skill"]
    for name, cell in cells.items():
        sc = cell["cleaned"]["skill"]
        if np.isfinite(sc) and np.isfinite(most_complex_raw_skill):
            cell["crossover_simple_clean_beats_complex_raw"] = bool(sc >= most_complex_raw_skill)
        else:
            cell["crossover_simple_clean_beats_complex_raw"] = False

    return cells


def _crossover_summary(cells: dict[str, dict]) -> dict:
    """Headline crossover for one dataset: simplest non-trivial model on
    CLEANED vs MOST complex model on RAW.

    ``margin = skill(simplest_nontrivial, cleaned) - skill(most_complex, raw)``;
    ``holds`` iff ``margin >= 0``. The simplest NON-trivial model = the first
    rung after SmartPersistence (the physical reference); the most complex = the
    last rung. Returns NaN margin if either skill is non-finite (reported, not
    forced).
    """
    names = sorted(cells, key=lambda n: cells[n]["complexity_rank"])
    non_trivial = [n for n in names if n != _REFERENCE_MODEL]
    if len(non_trivial) < 1:
        return {
            "simple_model": "",
            "complex_model": "",
            "margin": float("nan"),
            "holds": False,
        }
    simple = non_trivial[0]
    complex_m = names[-1]
    sc = cells[simple]["cleaned"]["skill"]
    cr = cells[complex_m]["raw"]["skill"]
    if not (np.isfinite(sc) and np.isfinite(cr)):
        margin = float("nan")
        holds = False
    else:
        margin = float(sc - cr)
        holds = bool(margin >= 0.0)
    return {
        "simple_model": simple,
        "complex_model": complex_m,
        "margin": margin,
        "holds": holds,
    }


# ---------------------------------------------------------------------------
# Figure 1: governance-vs-complexity grouped bars (the 2-axis picture)
# ---------------------------------------------------------------------------

def _plot_governance_vs_complexity(
    grid: dict[str, dict], out_dir: Path
) -> list[Path]:
    """Grouped bars of Daytime skill over the {data state x model} grid.

    One dataset panel (GEFCom default); raw arm hatched grey, cleaned colored
    per model. Δ-hygiene annotated above each model pair; horizontal line at 0
    (SmartPersistence-on-cleaned parity). The 2-axis picture in one panel.
    """
    dataset = next(iter(grid), None)
    if dataset is None:
        return []
    cells = grid[dataset]
    names = sorted(cells, key=lambda n: cells[n]["complexity_rank"])
    skills_raw = [cells[n]["raw"]["skill"] for n in names]
    skills_cl = [cells[n]["cleaned"]["skill"] for n in names]

    fig, ax = plt.subplots(figsize=(max(7.0, 1.5 * len(names) + 2.5), 5.6))
    x = np.arange(len(names))
    width = 0.38
    ax.bar(
        x - width / 2,
        [v if np.isfinite(v) else 0.0 for v in skills_raw],
        width,
        color="#9a9a9a",
        edgecolor="black",
        linewidth=0.5,
        hatch="//",
        label="raw (pre-governance)",
    )
    ax.bar(
        x + width / 2,
        [v if np.isfinite(v) else 0.0 for v in skills_cl],
        width,
        color=[MODEL_COLORS.get(n, PALETTE[0]) for n in names],
        edgecolor="black",
        linewidth=0.5,
        label="cleaned (governed)",
    )
    # Per-bar skill value labels (makes the SmartPersistence reference explicit:
    # both its arms sit at 0.0 because SP is the physical skill denominator and
    # is identical across arms — it does not learn from training labels).
    for i in range(len(names)):
        for off, val in ((-width / 2, skills_raw[i]), (width / 2, skills_cl[i])):
            if np.isfinite(val):
                ax.text(
                    x[i] + off, val, f"{val:+.2f}", ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=8, color="#444444",
                )
    for i, n in enumerate(names):
        dh = cells[n]["delta_hygiene"]
        top = max(
            skills_raw[i] if np.isfinite(skills_raw[i]) else 0.0,
            skills_cl[i] if np.isfinite(skills_cl[i]) else 0.0,
        )
        if np.isfinite(dh):
            ax.text(
                x[i], top, f"Δ{dh:+.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=PALETTE[4],
            )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Daytime skill vs Smart Persistence")
    ax.set_title(
        f"Data governance vs model complexity (E9) — {dataset}\n"
        "2-axis grid: {raw vs cleaned} x model ladder  "
        "(TINY dev configs; Phase 8 full)"
    )
    ax.legend(loc="best")
    return save_fig(fig, "fig_governance_vs_complexity", out_dir)


# ---------------------------------------------------------------------------
# Figure 2: cleaning Δ vs complexity Δ (the "hygiene > complexity" visual)
# ---------------------------------------------------------------------------

def _plot_cleaning_delta(grid: dict[str, dict], out_dir: Path) -> list[Path]:
    """Per-model Δ-hygiene (cleaning gain) vs Δ-complexity on raw (complexity
    gain), side-by-side. The crossover margin is annotated — the visual makes
    the "hygiene > complexity" point (cleaning bars dominate complexity bars when
    governance dominates).
    """
    dataset = next(iter(grid), None)
    if dataset is None:
        return []
    cells = grid[dataset]
    names = sorted(cells, key=lambda n: cells[n]["complexity_rank"])
    dh = [cells[n]["delta_hygiene"] for n in names]
    dc_raw = [cells[n]["delta_complexity_raw"] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6), sharey=True)
    x = np.arange(len(names))
    colors = [MODEL_COLORS.get(n, PALETTE[0]) for n in names]

    ax1.bar(
        x, [v if np.isfinite(v) else 0.0 for v in dh],
        color=colors, edgecolor="black", linewidth=0.5,
    )
    ax1.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha="right")
    ax1.set_title("Δ from data hygiene\nskill(model, cleaned) − skill(model, raw)")
    ax1.set_ylabel("Daytime skill gain")

    ax2.bar(
        x, [v if np.isfinite(v) else 0.0 for v in dc_raw],
        color=colors, edgecolor="black", linewidth=0.5, hatch="..",
    )
    ax2.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha="right")
    ax2.set_title("Δ from complexity (on raw data)\nskill(model, raw) − skill(SP, raw)")

    cross = _crossover_summary(cells)
    margin = cross["margin"]
    holds_txt = (
        f"crossover HOLDS (margin {margin:+.3f})"
        if (np.isfinite(margin) and cross["holds"])
        else (f"crossover ambiguous (margin {margin:+.3f})" if np.isfinite(margin)
              else "crossover n/a (non-finite cell)")
    )
    fig.suptitle(
        f"Cleaning gain vs complexity gain (E9) — {dataset}\n"
        f"{cross['simple_model']}/cleaned vs {cross['complex_model']}/raw → "
        f"{holds_txt}",
        y=1.03,
    )
    return save_fig(fig, "fig_cleaning_delta", out_dir)


# ---------------------------------------------------------------------------
# Results CSV (one row per dataset x model x data_state + Δ columns)
# ---------------------------------------------------------------------------

def _save_results_csv(grid: dict[str, dict], out_dir: Path) -> Path:
    rows: list[dict] = []
    for dataset, cells in grid.items():
        for name, cell in cells.items():
            common = {
                "dataset": dataset,
                "model": name,
                "complexity_rank": cell["complexity_rank"],
                "delta_hygiene": cell["delta_hygiene"],
                "delta_complexity_raw": cell["delta_complexity_raw"],
                "delta_complexity_cleaned": cell["delta_complexity_cleaned"],
                "crossover_simple_clean_beats_complex_raw": cell[
                    "crossover_simple_clean_beats_complex_raw"
                ],
            }
            for arm in ("raw", "cleaned"):
                rows.append({
                    **common,
                    "data_state": arm,
                    "daytime_mae": cell[arm]["daytime_mae"],
                    "daytime_skill": cell[arm]["skill"],
                    "n_daytime": cell[arm]["n_daytime"],
                })
    return save_metrics(pd.DataFrame(rows), "results_e9_governance", out_dir)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    loader,
    out_dir,
    *,
    raw_provider=None,
    model_ladder=None,
    datasets=None,
) -> dict:
    """Run experiment E9 and return ``{figures: [paths], metrics: {...}}``.

    Builds the 2-axis grid {raw, cleaned} x model-complexity-ladder per dataset
    and scores every cell by Daytime skill vs SmartPersistence (shared reference
    = SP-on-cleaned; both arms evaluate on the SAME cleaned test windows).

    Parameters
    ----------
    loader:
        Standardized-package loader (uniform contract with E1-E8). E9 sources
        both arms from ``raw_provider`` for rigorous row-alignment, so the
        loader is reserved for orchestrator dispatch.
    out_dir:
        Directory the figures + ``results/`` CSV are written under.
    raw_provider:
        REQUIRED callable ``raw_provider(dataset) -> raw_df | None`` returning
        the pre-governance frame (E5 pattern: ``adapter.run_pipeline(raw=True,
        limit=...)`` into a tmp dir). Without it E9 honestly skips — RAW data is
        not in the standardized packages.
    model_ladder:
        Optional callable ``() -> list[ForecastModel]`` returning FRESH model
        instances in complexity order (test/dev hook; called once per arm so
        weights don't leak between raw and cleaned). Defaults to the full ladder
        :func:`_build_ladder` (TINY configs).
    datasets:
        Optional iterable of dataset names to scope the grid. Defaults to
        :data:`DEFAULT_DATASETS` (GEFCom, the representative fast PV dataset).
    """
    set_style("light")
    out_dir = Path(out_dir)

    if raw_provider is None:
        return {
            "figures": [],
            "metrics": {
                "status": "skipped",
                "grid": {},
                "crossover": {},
                "config": _config_dict(model_ladder is not None, datasets),
                "note": (
                    "no raw_provider supplied; E9 needs the pre-governance (RAW) "
                    "frame to build the 2-axis grid. Pass raw_provider(dataset) "
                    "-> raw_df, or run scripts/build_dataset.py --dataset <d> "
                    "--raw to emit a RAW snapshot."
                ),
            },
        }

    build_ladder = model_ladder or _build_ladder
    grid_scope = list(datasets) if datasets is not None else list(DEFAULT_DATASETS)

    grid: dict[str, dict] = {}
    for dataset in grid_scope:
        target = DATASET_TARGETS.get(dataset)
        if target is None:
            continue
        try:
            raw_df = raw_provider(dataset)
        except Exception:  # pragma: no cover - defensive provider contract
            raw_df = None
        if raw_df is None or len(raw_df) == 0:
            continue
        raw_df = raw_df.copy()
        if target not in raw_df.columns:
            continue

        # Capacity: prefer the raw frame's capacity_kw, else the PV default.
        cap = _DEFAULT_CAPACITY
        if "capacity_kw" in raw_df.columns and raw_df["capacity_kw"].notna().any():
            cap = float(raw_df["capacity_kw"].dropna().iloc[0])
        cleaned_df = _derive_cleaned(raw_df, capacity=cap)

        grid[dataset] = _evaluate_dataset(raw_df, cleaned_df, target, build_ladder)

    if not grid:
        return {
            "figures": [],
            "metrics": {
                "status": "skipped",
                "grid": {},
                "crossover": {},
                "config": _config_dict(model_ladder is not None, datasets),
                "note": (
                    "raw_provider returned no usable raw frame for the selected "
                    "datasets; grid not built."
                ),
            },
        }

    crossover = {ds: _crossover_summary(cells) for ds, cells in grid.items()}
    figures: list[Path] = []
    figures += _plot_governance_vs_complexity(grid, out_dir)
    figures += _plot_cleaning_delta(grid, out_dir)
    _save_results_csv(grid, out_dir)

    metrics = {
        "status": "ran",
        "grid": grid,
        "crossover": crossover,
        "config": _config_dict(model_ladder is not None, datasets),
    }
    return {"figures": [str(p) for p in figures], "metrics": metrics}


def _config_dict(custom_ladder: bool, datasets) -> dict:
    return {
        "lookback": _LOOKBACK,
        "horizon": _HORIZON,
        "stride": _STRIDE,
        "train_frac": _TRAIN_FRAC,
        "features": list(_SHARED_FEATURES),
        "reference": _REFERENCE_MODEL,
        "daytime_cos_zenith": _DAYTIME_COS_ZENITH,
        "ladder": list(COMPLEXITY_LADDER),
        "ladder_source": "custom" if custom_ladder else "default_tiny",
        "datasets": list(datasets) if datasets is not None else list(DEFAULT_DATASETS),
        "dev_configs": "TINY (epochs=3 deep models) — Phase 8 full",
    }
