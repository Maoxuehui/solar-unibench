"""E2 — Clear-sky / clearness-index feature value (all 4 datasets).

Spec §11 (E2). *Hypothesis: physics anchors (clear-sky / clearness-index) beat
raw endogenous features.* A per-dataset ablation scored against the physical
Smart Persistence reference (clear-sky-scaled persistence, the spec §10 skill
denominator for every dataset):

* **baseline** — endogenous only: the model sees ONLY target lags.
* **+clear-sky anchors** — endogenous + ``ghi_clearsky`` / ``cos_zenith``
  (+ ``clearness_index`` where the column is populated).

One figure (``fig_clearsky_skill.png`` — grouped bars of Daytime skill per
dataset, baseline vs +clear-sky) + one results CSV, both routed through the
shared :mod:`solarunibench.viz` design system.

Honest-NaN + clean-ablation notes
---------------------------------
* ``clearness_index`` is entirely NaN on PV-only datasets (GEFCom, SKIPPD carry
  no measured ``ghi_actual`` to form ``ghi_actual / ghi_clearsky``). Per the
  spec's Honest-NaN principle it is NOT fabricated — it is dropped per dataset
  and the omission is reported in ``clearsky_features_used`` vs ``...requested``.
* :class:`EndogenousFeatureEngineer` auto-builds a **future** clear-sky context
  block ``X_future`` whenever both ``ghi_clearsky`` and ``cos_zenith`` are
  present. Left untouched this would secretly feed clear-sky features to the
  "endogenous-only" baseline and confound the ablation. E2 therefore drops
  ``ghi_clearsky`` for the baseline arm (keeping ``cos_zenith`` so the daytime
  mask still resolves) — so baseline sees target lags only. The Smart
  Persistence reference runs on the full frame (clear-sky-scaled), as required.
  Each arm reports ``past_channels`` + ``future_context`` so the design matrix
  is auditable.

Pure function contract: ``run(loader, out_dir) -> {figures, metrics}`` reads the
loader and writes to ``out_dir``; it never mutates standardized data. TINY
config for fast dev iteration; full-config numbers come at Phase 8.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..forecast.core import EndogenousFeatureEngineer, PipelineConfig, SolarForecastPipeline
from ..forecast.models.lightgbm_model import LightGBMModel
from ..forecast.models.smart_persistence import SmartPersistenceModel
from ..viz.io import save_fig, save_metrics
from ..viz.style import DATASET_COLORS, PALETTE, set_style

__all__ = ["run"]


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

#: Physics anchors (the treatment arm). clearness_index is Honest-NaN on PV-only
#: datasets and is filtered out per dataset by :func:`_usable_clearsky_features`.
_CLEARSKY_FEATURES: tuple[str, ...] = ("ghi_clearsky", "cos_zenith", "clearness_index")

#: Modest lookback / single-step horizon / stride that bounds the window count
#: on the largest datasets (Folsom ~0.9M rows) while keeping enough daytime
#: samples for a finite daytime MAE on the small synthetic frames.
_LOOKBACK: int = 12
_HORIZON: int = 1
_STRIDE: int = 6

#: TINY LightGBM (fast + robust tabular baseline). n_jobs=1 keeps runs
#: deterministic; the model already forces verbose=-1.
_LGBM_PARAMS: dict = {"n_estimators": 40, "n_jobs": 1, "random_state": 0}

#: Finite-fraction threshold above which a clear-sky feature column is "present".
_FINITE_FRAC_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _finite_fraction(series: pd.Series) -> float:
    """Fraction of finite values in a column (0 for all-NaN / object-None)."""
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return float(np.isfinite(arr).mean()) if arr.size else 0.0


def _usable_clearsky_features(df: pd.DataFrame) -> list[str]:
    """Clear-sky features that are columns AND mostly finite (Honest-NaN filter).

    ``clearness_index`` is dropped on PV-only datasets where it is entirely NaN.
    """
    return [
        c for c in _CLEARSKY_FEATURES
        if c in df.columns and _finite_fraction(df[c]) > _FINITE_FRAC_THRESHOLD
    ]


def _skill(arm_mae: float, ref_mae: float) -> float:
    """Forecast skill vs the Smart Persistence reference.

    ``1 - mae_arm / mae_ref`` (0 = parity, positive = better than reference).
    NaN when either MAE is non-finite or the reference MAE is exactly 0.
    """
    if not (np.isfinite(arm_mae) and np.isfinite(ref_mae)) or ref_mae == 0.0:
        return float("nan")
    return float(1.0 - arm_mae / ref_mae)


def _run_arm(
    loader,
    dataset: str,
    target: str,
    feature_cols: list[str],
    drop_cols: tuple[str, ...],
    model,
) -> dict:
    """Run one model arm and return its daytime MAE/RMSE + design-matrix descriptor.

    ``drop_cols`` is removed from the frame BEFORE feature engineering (used to
    suppress the auto future clear-sky context for the endogenous baseline).
    The frame is otherwise read-only.
    """
    data_root = str(getattr(loader, "data_root", "") or "")
    config = PipelineConfig(
        dataset=dataset,
        target=target,
        lookback=_LOOKBACK,
        horizon=_HORIZON,
        stride=_STRIDE,
        feature_cols=list(feature_cols),
        data_root=data_root or None,
    )
    engineer = EndogenousFeatureEngineer(target, feature_cols=list(feature_cols))
    pipeline = SolarForecastPipeline(config, engineer, model)

    # Reuse the injected loader (no hidden SolarUniBenchLoader re-instantiation).
    # Drop NaN-target rows — the model cannot fit on missing labels (Honest NaN).
    def _df_provider(cfg):
        d = loader.load_tabular(cfg.dataset)
        d = d.dropna(subset=[target]).reset_index(drop=True)
        for c in drop_cols:
            if c in d.columns:
                d = d.drop(columns=[c])
        return d

    res = pipeline.run(df_provider=_df_provider)
    day = res.get("daytime", {}) or {}
    return {
        "daytime_mae": float(day.get("Daytime_MAE", float("nan"))),
        "daytime_rmse": float(day.get("Daytime_RMSE", float("nan"))),
        "n_daytime": int(day.get("n_daytime", 0)),
        "n_total": int(day.get("n_total", 0)),
        # Auditable design-matrix descriptor (no internal-state introspection).
        "past_channels": 1 + len([c for c in feature_cols if c not in drop_cols]),
        "future_context": "ghi_clearsky" not in drop_cols,
    }


def _evaluate_dataset(loader, dataset: str, target: str) -> dict | None:
    """Run the 3-arm ablation (reference / baseline / +clear-sky) for one dataset.

    Returns ``None`` when the dataset lacks the target or the core clear-sky
    anchors (Honest skip — the dataset cannot test the hypothesis).
    """
    df = loader.load_tabular(dataset)
    if target not in df.columns or "ghi_clearsky" not in df.columns or "cos_zenith" not in df.columns:
        return None
    used = _usable_clearsky_features(df)

    # Reference: Smart Persistence on the FULL frame (clear-sky-scaled, the
    # spec §10 skill denominator). Defines the daytime subset all arms share.
    ref = _run_arm(
        loader, dataset, target, feature_cols=[], drop_cols=(),
        model=SmartPersistenceModel(),
    )
    # Baseline: endogenous ONLY — drop ghi_clearsky so the engineer's auto
    # future clear-sky block (X_future) is NOT built; cos_zenith kept for mask.
    base = _run_arm(
        loader, dataset, target, feature_cols=[], drop_cols=("ghi_clearsky",),
        model=LightGBMModel(params=_LGBM_PARAMS, name="LGBM_baseline"),
    )
    # Treatment: endogenous + clear-sky anchors (past channels + future context).
    clr = _run_arm(
        loader, dataset, target, feature_cols=used, drop_cols=(),
        model=LightGBMModel(params=_LGBM_PARAMS, name="LGBM_clearsky"),
    )

    ref_mae = ref["daytime_mae"]
    base_skill = _skill(base["daytime_mae"], ref_mae)
    clr_skill = _skill(clr["daytime_mae"], ref_mae)
    gain = float(clr_skill - base_skill) if (np.isfinite(base_skill) and np.isfinite(clr_skill)) else float("nan")

    return {
        "target": target,
        "clearsky_features_requested": list(_CLEARSKY_FEATURES),
        "clearsky_features_used": used,
        "reference": {k: ref[k] for k in ("daytime_mae", "n_daytime")},
        "baseline": {**base, "skill": base_skill},
        "clearsky": {**clr, "skill": clr_skill},
        "skill_gain": gain,
    }


# ---------------------------------------------------------------------------
# Figure: grouped bars of Daytime skill per dataset (baseline vs +clear-sky)
# ---------------------------------------------------------------------------

def _plot_skill(per_dataset: dict[str, dict], out_dir: Path) -> list[Path]:
    datasets = list(per_dataset.keys())
    n = len(datasets)
    fig, ax = plt.subplots(figsize=(max(6.0, 1.7 * n + 2.0), 5.2))
    width = 0.38
    x = np.arange(n)
    for i, ds in enumerate(datasets):
        arm = per_dataset[ds]
        b_skill = arm["baseline"]["skill"]
        c_skill = arm["clearsky"]["skill"]
        ds_color = DATASET_COLORS.get(ds, PALETTE[0])
        # baseline: muted grey + hatch (the control). clearsky: dataset color.
        ax.bar(
            x[i] - width / 2, b_skill if np.isfinite(b_skill) else 0.0, width,
            color="#9a9a9a", edgecolor="black", linewidth=0.5, hatch="//",
        )
        ax.bar(
            x[i] + width / 2, c_skill if np.isfinite(c_skill) else 0.0, width,
            color=ds_color, edgecolor="black", linewidth=0.5,
        )
        # Per-group Δskill annotation (the headline E2 number).
        gain = arm["skill_gain"]
        if np.isfinite(gain):
            top = max(
                b_skill if np.isfinite(b_skill) else 0.0,
                c_skill if np.isfinite(c_skill) else 0.0,
            )
            ax.text(
                x[i], top, f"Δ{gain:+.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=PALETTE[4],
            )

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Daytime forecast skill vs Smart Persistence")
    ax.set_title("Clear-sky anchor value (E2)\nDaytime skill: baseline (endogenous) vs +clear-sky anchors")
    # Two proxy legend entries (per-dataset tinting is a visual cue; x-axis
    # carries dataset identity).
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor="#9a9a9a", edgecolor="black", hatch="//", label="baseline (endogenous only)"),
            Patch(facecolor=PALETTE[2], edgecolor="black", label="+clear-sky anchors"),
        ],
        loc="best",
    )
    return save_fig(fig, "fig_clearsky_skill", out_dir)


# ---------------------------------------------------------------------------
# Results CSV
# ---------------------------------------------------------------------------

def _save_results_csv(metrics: dict, out_dir: Path) -> Path:
    rows: list[dict] = []
    for ds, arm in metrics["datasets"].items():
        common = {
            "dataset": ds,
            "target": arm["target"],
            "clearsky_features_used": ",".join(arm["clearsky_features_used"]),
            "skill_gain": arm["skill_gain"],
        }
        for arm_name in ("baseline", "clearsky"):
            a = arm[arm_name]
            rows.append({
                **common,
                "arm": arm_name,
                "daytime_mae": a["daytime_mae"],
                "daytime_rmse": a.get("daytime_rmse", float("nan")),
                "n_daytime": a["n_daytime"],
                "skill": a["skill"],
                "past_channels": a["past_channels"],
                "future_context": a["future_context"],
            })
        rows.append({
            **common, "arm": "reference",
            "daytime_mae": arm["reference"]["daytime_mae"],
            "n_daytime": arm["reference"]["n_daytime"],
        })
    return save_metrics(pd.DataFrame(rows), "results_e2_clearsky", out_dir)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(loader, out_dir) -> dict:
    """Run experiment E2 and return ``{figures: [paths], metrics: {...}}``.

    Reads only standardized packages via ``loader``; writes a figure + a results
    CSV under ``out_dir``. Pure: does not mutate ``loader`` state or the
    standardized data.
    """
    set_style("light")
    out_dir = Path(out_dir)

    per_dataset: dict[str, dict] = {}
    for dataset, target in DATASET_TARGETS.items():
        res = _evaluate_dataset(loader, dataset, target)
        if res is not None:
            per_dataset[dataset] = res

    figures: list[Path] = []
    if per_dataset:
        figures += _plot_skill(per_dataset, out_dir)

    metrics = {
        "datasets": per_dataset,
        "config": {
            "lookback": _LOOKBACK,
            "horizon": _HORIZON,
            "stride": _STRIDE,
            "model": "LightGBM",
            "n_estimators": _LGBM_PARAMS["n_estimators"],
            "reference": "SmartPersistence",
        },
    }
    _save_results_csv(metrics, out_dir)

    return {"figures": [str(p) for p in figures], "metrics": metrics}
