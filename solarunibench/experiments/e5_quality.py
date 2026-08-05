"""E5 — Quality-audit cleaning value (spec §11, E5).

*Hypothesis:* the ``quality_flag`` governance (night-zeroing, de-clipping,
fault removal) cuts label noise and lifts forecast skill.

**Ablation.** For each dataset with an available RAW pre-governance snapshot,
train a LightGBM model on (a) the RAW frame and (b) the CLEANED
(quality-audited) frame, then score both on the SAME daytime test windows
derived from the cleaned frame (clean ground truth). Skill is measured against
the physical Smart Persistence reference. The two arms share identical model,
lookback, horizon, stride and feature columns — the ONLY difference is whether
the training labels were governed, so any Δ skill is attributable to the audit.

**Raw-data sourcing.** The standardized packages are all ``raw=False``
(post-governance). E5 therefore takes the RAW frame via a ``raw_provider``
hook (``raw_provider(dataset) -> df | None``). When no provider is supplied
(the default orchestrator path) the cleaning-effect ablation is honestly
SKIPPED — only ``fig_quality_flags`` (which reads cleaned packages via the
loader) is produced. The matching CLEANED frame is derived in-run by applying
:func:`apply_quality_audit` to the raw frame (exactly what
``run_pipeline(raw=False)`` does), so the two arms are row-aligned and differ
ONLY in governance. A bounded RAW snapshot for one dataset can be built via
``scripts/build_dataset.py --dataset <d> --raw`` (see the RUN_SLOW test, which
builds a bounded GEFCom snapshot in-run).

**Target-distribution shift.** Per dataset E5 reports the fraction of rows the
audit flagged (``quality_flag != 0``) and the target mean/std on the daytime
finite subset before (raw) vs after (cleaned) governance — the label-noise cut
the hypothesis predicts. If the audit is a no-op on a dataset (nothing flagged),
E5 reports that honestly rather than fabricating a gain.

Two figures (``fig_quality_flags.png`` — composition per dataset;
``fig_quality_cleaning_effect.png`` — skill raw vs cleaned + target shift) + a
results CSV, all routed through the shared :mod:`solarunibench.viz` design
system.

Pure function contract: ``run(loader, out_dir, *, raw_provider=None) ->
{figures, metrics}`` reads the loader + the optional raw provider and writes to
``out_dir``; it never mutates standardized data. TINY LightGBM config for fast
dev iteration; full-config numbers come at Phase 8.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..forecast.core import EndogenousFeatureEngineer, PipelineConfig, SolarForecastPipeline
from ..forecast.models.lightgbm_model import LightGBMModel
from ..forecast.models.smart_persistence import SmartPersistenceModel
from ..operators.quality import apply_quality_audit
from ..viz.io import save_fig, save_metrics
from ..viz.style import DATASET_COLORS, PALETTE, set_style

__all__ = ["run"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: All four standardized packages (for fig_quality_flags composition).
DATASETS: tuple[str, ...] = ("gefcom2014", "folsom", "skippd", "sunlake")

#: Per-dataset forecasting target (spec §10) — PV for GEFCom/SKIPPD, GHI for
#: Folsom/SunLake.
DATASET_TARGETS: dict[str, str] = {
    "gefcom2014": "pv_power_normalized",
    "skippd": "pv_power_normalized",
    "folsom": "ghi_actual",
    "sunlake": "ghi_actual",
}

#: Feature columns shared by both arms (the clear-sky anchors). These are
#: computed BEFORE the quality audit, so they are IDENTICAL in the raw and
#: cleaned frames — the only difference between arms is the training target
#: (the noisy raw target vs the governed cleaned target).
_SHARED_FEATURES: tuple[str, ...] = ("ghi_clearsky", "cos_zenith")

#: Bounded dev configuration. A modest lookback + single-step horizon + stride
#: keeps the window count tractable on the largest frames while retaining
#: enough daytime samples for a finite daytime MAE.
_LOOKBACK: int = 12
_HORIZON: int = 1
_STRIDE: int = 6

#: TINY LightGBM (fast + robust tabular baseline). n_jobs=1 keeps runs
#: deterministic; the model already forces verbose=-1.
_LGBM_PARAMS: dict = {"n_estimators": 40, "n_jobs": 1, "random_state": 0}

#: Daytime threshold (spec global constraint cos_zenith > 0.05).
_DAYTIME_COS_ZENITH: float = 0.05

#: Default PV capacity passed to the quality audit (pv_power_normalized ∈ [0,1]
#: for every PV dataset; capacity_kw is NaN on GEFCom → the base pipeline uses
#: 1.0). GHI datasets carry no PV column, so capacity is unused for them.
_DEFAULT_CAPACITY: float = 1.0

#: Train fraction of the per-entity temporal split E5 owns (the experiment does
#: not depend on the canonical benchmark splits — it assigns its own so raw and
#: cleaned share identical train/test boundaries).
_TRAIN_FRAC: float = 0.70

#: quality_flag semantics (spec §4.2) — for human-readable figure legends.
_QUALITY_FLAG_NAMES: dict[int, str] = {
    0: "0 good",
    1: "1 clipping",
    2: "2 fault",
    3: "3 night",
}


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
    """Run the quality audit on a raw frame → the cleaned (governed) frame.

    Mirrors ``run_pipeline(raw=False)``: the audit runs on the anchor-merged
    frame, so the NA ``quality_flag`` / ``clearness_index`` the raw snapshot
    carries are dropped first and the audit owns them. Produces a frame that is
    row-aligned with ``raw_df`` and differs ONLY in governance.
    """
    base = raw_df.drop(
        columns=[c for c in ("quality_flag", "clearness_index") if c in raw_df.columns]
    )
    return apply_quality_audit(base, capacity=capacity)


def _assign_temporal_split(df: pd.DataFrame, train_frac: float = _TRAIN_FRAC) -> pd.Series:
    """Per-entity temporal split label (``train``/``test``), row-aligned with ``df``.

    Within each entity, rows are ordered by ``timestamp_utc`` and the first
    ``train_frac`` are labeled ``train`` (the rest ``test``). The returned
    Series is indexed by ``df.index`` so it can be assigned onto a row-aligned
    frame (raw or cleaned) by index.
    """
    labels = pd.Series("test", index=df.index, dtype=object)
    for _, g in df.groupby("entity_id", sort=False):
        g_sorted = g.sort_values("timestamp_utc", kind="mergesort")
        n = len(g_sorted)
        n_tr = int(round(n * train_frac))
        labels.loc[g_sorted.index[:n_tr]] = "train"
    return labels


def _quality_flag_composition(df: pd.DataFrame) -> dict:
    """``{flag: pct}`` composition + total count. Absent column → empty dict."""
    out: dict[str, float] = {}
    n = len(df)
    if "quality_flag" not in df.columns or n == 0:
        return {"composition_pct": out, "counts": {}, "n_total": 0}
    vc = df["quality_flag"].value_counts(dropna=False)
    counts: dict[str, int] = {}
    for k, v in vc.items():
        key = "nan" if pd.isna(k) else str(int(k))
        counts[key] = int(v)
        out[key] = float(v / n * 100.0) if n else 0.0
    return {"composition_pct": out, "counts": counts, "n_total": int(n)}


def _target_stats(df: pd.DataFrame, target: str) -> dict:
    """Mean/std of the target on the daytime finite subset (cos_zenith > 0.05)."""
    arr = pd.to_numeric(df.get(target, pd.Series(dtype=float)), errors="coerce")
    cz = pd.to_numeric(df.get("cos_zenith", pd.Series(dtype=float)), errors="coerce")
    daytime = cz > _DAYTIME_COS_ZENITH
    arr = arr[daytime]
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(arr.size)}


#: GHI is flagged as a physical violation when it exceeds this multiple of the
#: clear-sky irradiance at the same zenith (measured GHI physically cannot
#: exceed ~clear-sky; 1.2 is a generous sensor-glare margin).
_GHI_OVERCLEAR_FACTOR: float = 1.2


def physics_violation_rate(df: pd.DataFrame, capacity: float = _DEFAULT_CAPACITY,
                           ghi_over_factor: float = _GHI_OVERCLEAR_FACTOR) -> dict:
    """Fraction of DAYTIME rows that violate physical limits (raw vs cleaned).

    * **GHI over clear-sky**: ``ghi_actual > ghi_over_factor · ghi_clearsky`` —
      physically impossible (measured irradiance cannot exceed ~clear-sky at the
      same zenith); a generous sensor-glare margin is used.
    * **PV over installed capacity**: ``pv_power_normalized > capacity``
      (``capacity_kw`` when present, else 1.0 for normalized PV).

    Returns per-violation fractions + the daytime sample count. Honest NaN when
    the relevant column is absent (Honest NaN principle — never fabricates).
    """
    cz = pd.to_numeric(df.get("cos_zenith", pd.Series(dtype=float)), errors="coerce")
    sub = df.loc[cz > _DAYTIME_COS_ZENITH]
    n = int(len(sub))
    out = {"n_daytime": n, "ghi_overclear_frac": float("nan"),
           "pv_overcap_frac": float("nan"), "any_violation_frac": float("nan")}
    if n == 0:
        return out
    ghi = pd.to_numeric(sub.get("ghi_actual"), errors="coerce")
    gcs = pd.to_numeric(sub.get("ghi_clearsky"), errors="coerce")
    ghi_viol = pd.Series(False, index=sub.index)
    has_ghi = ghi.notna().any() and gcs.notna().any()
    if has_ghi:
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = ghi / np.maximum(gcs, 1e-6)
        ghi_viol = (ratio > ghi_over_factor).fillna(False)
        out["ghi_overclear_frac"] = float(ghi_viol.mean())  # 0.0 when none
    pv = pd.to_numeric(sub.get("pv_power_normalized"), errors="coerce")
    pv_viol = pd.Series(False, index=sub.index)
    has_pv = pv.notna().any()
    if has_pv:
        pv_viol = (pv > capacity).fillna(False)
        out["pv_overcap_frac"] = float(pv_viol.mean())  # 0.0 when none
    # any_violation is finite (0.0) whenever at least one channel was scored;
    # NaN only when neither GHI nor PV is present.
    if has_ghi or has_pv:
        out["any_violation_frac"] = float((ghi_viol | pv_viol).mean())
    return out


def three_way_closure(df: pd.DataFrame) -> dict:
    """Three-way irradiance closure ``GHI ≈ DNI·cos(z) + DHI`` (Folsom/SunLake).

    The decomposed irradiance must reconstruct the global: residual =
    ``GHI − (DNI·cos(z) + DHI)``. Reports absolute-residual MAE/median/RMSE +
    the regression R² of the reconstruction vs GHI (1.0 = perfect closure), on
    the daytime subset. Honest skip when ``ghi/dni/dhi/cos_zenith`` are absent.
    """
    needed = ("ghi_actual", "dni_actual", "dhi_actual", "cos_zenith")
    if any(c not in df.columns for c in needed):
        return {"status": "skipped", "reason": "missing DNI/DHI/GHI/cos_zenith"}
    sub = df[list(needed)].apply(pd.to_numeric, errors="coerce")
    cz = sub["cos_zenith"]
    sub = sub[(cz > _DAYTIME_COS_ZENITH).fillna(False)]  # daytime only
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(sub))
    if n < 8:
        return {"status": "skipped", "reason": f"too few daytime closure rows ({n})"}
    dni_cos = (sub["dni_actual"].clip(lower=0) * sub["cos_zenith"].clip(lower=0)).to_numpy()
    recon = dni_cos + sub["dhi_actual"].clip(lower=0).to_numpy()
    ghi = sub["ghi_actual"].to_numpy()
    resid = ghi - recon
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((ghi - ghi.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "status": "ran", "n_daytime": n,
        "residual_mae": float(np.mean(np.abs(resid))),
        "residual_median": float(np.median(np.abs(resid))),
        "residual_rmse": float(np.sqrt(ss_res / n)),
        "r2": r2,
        "ghi_mean": float(ghi.mean()), "recon_mean": float(recon.mean()),
    }


# ---------------------------------------------------------------------------
# Figure 1: quality_flag composition per dataset (from cleaned packages)
# ---------------------------------------------------------------------------

def _plot_quality_flags(qf: dict[str, dict], out_dir: Path) -> list[Path]:
    datasets = list(qf.keys())
    fig, ax = plt.subplots(figsize=(max(6.5, 1.6 * len(datasets) + 2.5), 5.2))
    x = np.arange(len(datasets))
    all_flags = sorted(
        {f for d in datasets for f in qf[d]["composition_pct"]},
        key=lambda s: (s == "nan", s),
    )
    bottoms = np.zeros(len(datasets))
    for fi, flag in enumerate(all_flags):
        vals = np.array(
            [qf[d]["composition_pct"].get(flag, 0.0) for d in datasets], dtype=float
        )
        label = _QUALITY_FLAG_NAMES.get(int(flag), f"flag={flag}") if flag != "nan" else "NaN"
        ax.bar(
            x, vals, bottom=bottoms, color=PALETTE[fi % len(PALETTE)],
            edgecolor="black", linewidth=0.4, label=label,
        )
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel("% of rows")
    ax.set_ylim(0, 100)
    ax.set_title("quality_flag composition per dataset (E5)\nwhat the audit flagged")
    ax.legend(loc="best", fontsize=9)
    return save_fig(fig, "fig_quality_flags", out_dir)


# ---------------------------------------------------------------------------
# Ablation: train LightGBM on raw vs cleaned, score on cleaned test windows
# ---------------------------------------------------------------------------

def _run_arm(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    dataset: str,
    target: str,
    data_root: str,
    model,
) -> dict:
    """Run one model arm and return its daytime MAE/RMSE + design descriptor.

    ``df_train`` / ``df_test`` are already split-specific frames carrying a
    ``split`` column (``"train"`` / ``"test"`` respectively). The pipeline
    filters by split internally, so the provider concatenates them.
    """
    config = PipelineConfig(
        dataset=dataset,
        target=target,
        lookback=_LOOKBACK,
        horizon=_HORIZON,
        stride=_STRIDE,
        feature_cols=list(_SHARED_FEATURES),
        data_root=data_root or None,
    )
    engineer = EndogenousFeatureEngineer(target, feature_cols=list(_SHARED_FEATURES))
    pipeline = SolarForecastPipeline(config, engineer, model)

    combined = pd.concat([df_train, df_test], ignore_index=True)

    def _df_provider(cfg):
        return combined

    res = pipeline.run(df_provider=_df_provider)
    day = res.get("daytime", {}) or {}
    return {
        "daytime_mae": float(day.get("Daytime_MAE", float("nan"))),
        "daytime_rmse": float(day.get("Daytime_RMSE", float("nan"))),
        "n_daytime": int(day.get("n_daytime", 0)),
        "n_total": int(day.get("n_total", 0)),
        "past_channels": 1 + len(_SHARED_FEATURES),
    }


def _evaluate_dataset(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    target: str,
) -> dict:
    """Run the 3-arm ablation (reference / raw / cleaned) for one dataset.

    Both arms are evaluated on the SAME cleaned test windows (clean ground
    truth). The reference is Smart Persistence on the cleaned frame.
    """
    # Shared temporal split — assign once on cleaned, transfer to raw (row-aligned).
    split = _assign_temporal_split(cleaned_df)
    cleaned_df = cleaned_df.assign(split=split)
    raw_df = raw_df.assign(split=split)

    data_root = ""  # df_provider supplies frames; data_root unused
    # Drop governance-metadata columns from the arm inputs: they are NOT model
    # features (the engineer uses only [target, ghi_clearsky, cos_zenith]) and
    # leaving them would (a) leak the cleaning into the raw arm's test frame
    # and (b) trigger a pandas concat dtype warning on all-NA vs populated
    # quality_flag columns. The audit's effect is measured via the TARGET only.
    meta_cols = [c for c in ("quality_flag", "clearness_index") if c in cleaned_df.columns]
    cleaned_train = cleaned_df[cleaned_df["split"] == "train"].drop(columns=meta_cols).reset_index(drop=True)
    cleaned_test = cleaned_df[cleaned_df["split"] == "test"].drop(columns=meta_cols).reset_index(drop=True)
    raw_train = raw_df[raw_df["split"] == "train"].drop(columns=meta_cols).reset_index(drop=True)

    # Reference: Smart Persistence on the cleaned frame (defines the daytime set).
    ref = _run_arm(
        cleaned_train, cleaned_test, dataset="e5", target=target,
        data_root=data_root, model=SmartPersistenceModel(),
    )
    # Raw arm: LightGBM trained on RAW train, scored on CLEANED test.
    raw_arm = _run_arm(
        raw_train, cleaned_test, dataset="e5", target=target,
        data_root=data_root,
        model=LightGBMModel(params=_LGBM_PARAMS, name="LGBM_raw"),
    )
    # Cleaned arm: LightGBM trained on CLEANED train, scored on CLEANED test.
    cl_arm = _run_arm(
        cleaned_train, cleaned_test, dataset="e5", target=target,
        data_root=data_root,
        model=LightGBMModel(params=_LGBM_PARAMS, name="LGBM_cleaned"),
    )

    ref_mae = ref["daytime_mae"]
    raw_skill = _skill(raw_arm["daytime_mae"], ref_mae)
    cl_skill = _skill(cl_arm["daytime_mae"], ref_mae)
    gain = (
        float(cl_skill - raw_skill)
        if (np.isfinite(raw_skill) and np.isfinite(cl_skill))
        else float("nan")
    )

    # Target-distribution shift (label-noise cut).
    n_total = len(cleaned_df)
    qf = _quality_flag_composition(cleaned_df)
    flagged = {k: v for k, v in qf["counts"].items() if k != "0" and k != "nan"}
    n_flagged = int(sum(flagged.values()))
    frac_flagged = float(n_flagged / n_total) if n_total else 0.0
    # fraction_target_changed: rows where the audit actually MODIFIED the target
    # value (the honest no-op measure — governance can flag night rows without
    # changing a target that was already 0).
    raw_t = pd.to_numeric(raw_df.get(target, pd.Series(dtype=float)), errors="coerce")
    cl_t = pd.to_numeric(cleaned_df.get(target, pd.Series(dtype=float)), errors="coerce")
    both_finite = raw_t.notna() & cl_t.notna()
    n_changed = int((both_finite & (raw_t != cl_t)).sum())
    frac_changed = float(n_changed / n_total) if n_total else 0.0
    raw_stats = _target_stats(raw_df, target)
    cl_stats = _target_stats(cleaned_df, target)

    return {
        "target": target,
        "reference": {k: ref[k] for k in ("daytime_mae", "n_daytime")},
        "raw": {**raw_arm, "skill": raw_skill, "train_source": "raw"},
        "cleaned": {**cl_arm, "skill": cl_skill, "train_source": "cleaned"},
        "skill_gain": gain,
        "fraction_flagged": frac_flagged,
        "n_flagged": n_flagged,
        "fraction_target_changed": frac_changed,
        "n_target_changed": n_changed,
        "n_total": n_total,
        "target_mean_raw": raw_stats["mean"],
        "target_std_raw": raw_stats["std"],
        "target_mean_cleaned": cl_stats["mean"],
        "target_std_cleaned": cl_stats["std"],
        "target_n_daytime": raw_stats["n"],
    }


# ---------------------------------------------------------------------------
# Figure 2: skill raw vs cleaned + target-distribution shift
# ---------------------------------------------------------------------------

def _plot_cleaning_effect(
    per_dataset: dict[str, dict], raw_samples: dict[str, np.ndarray],
    cleaned_samples: dict[str, np.ndarray], out_dir: Path,
) -> list[Path]:
    datasets = list(per_dataset.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    width = 0.38
    x = np.arange(len(datasets))

    # Left: grouped bars of daytime skill (raw vs cleaned) per dataset.
    for i, ds in enumerate(datasets):
        arm = per_dataset[ds]
        r_skill = arm["raw"]["skill"]
        c_skill = arm["cleaned"]["skill"]
        ax1.bar(x[i] - width / 2, r_skill if np.isfinite(r_skill) else 0.0, width,
                color="#9a9a9a", edgecolor="black", linewidth=0.5, hatch="//")
        ax1.bar(x[i] + width / 2, c_skill if np.isfinite(c_skill) else 0.0, width,
                color=DATASET_COLORS.get(ds, PALETTE[0]), edgecolor="black", linewidth=0.5)
        gain = arm["skill_gain"]
        if np.isfinite(gain):
            top = max(
                r_skill if np.isfinite(r_skill) else 0.0,
                c_skill if np.isfinite(c_skill) else 0.0,
            )
            ax1.text(x[i], top, f"Δ{gain:+.3f}", ha="center", va="bottom",
                     fontsize=9, fontweight="bold", color=PALETTE[4])
    ax1.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=30, ha="right")
    ax1.set_ylabel("Daytime forecast skill vs Smart Persistence")
    ax1.set_title("Cleaning value (E5)\nLightGBM trained on RAW vs CLEANED labels")
    from matplotlib.patches import Patch
    ax1.legend(
        handles=[
            Patch(facecolor="#9a9a9a", edgecolor="black", hatch="//", label="raw (pre-governance)"),
            Patch(facecolor=PALETTE[2], edgecolor="black", label="cleaned (quality-audited)"),
        ],
        loc="best",
    )

    # Right: target ECDF before (raw) vs after (cleaned) for the first dataset —
    # the visual label-noise cut. Falls back to empty if no samples available.
    if datasets and datasets[0] in raw_samples:
        ds0 = datasets[0]
        rs = raw_samples[ds0]
        cs = cleaned_samples[ds0]
        if rs.size and cs.size:
            r_sorted = np.sort(rs)
            c_sorted = np.sort(cs)
            r_ecdf = np.arange(1, len(r_sorted) + 1) / len(r_sorted)
            c_ecdf = np.arange(1, len(c_sorted) + 1) / len(c_sorted)
            ax2.step(r_sorted, r_ecdf, color="#9a9a9a", linewidth=1.8, label="raw")
            ax2.step(c_sorted, c_ecdf, color=DATASET_COLORS.get(ds0, PALETTE[0]),
                     linewidth=1.8, label="cleaned")
            arm0 = per_dataset[ds0]
            ax2.set_title(
                f"Target distribution shift ({ds0})\n"
                f"fraction flagged={arm0['fraction_flagged']:.3f}"
            )
            ax2.set_xlabel(f"{per_dataset[ds0]['target']} (daytime finite subset)")
            ax2.set_ylabel("ECDF")
            ax2.legend(loc="lower right")
        else:
            ax2.text(0.5, 0.5, "no daytime target samples", ha="center", va="center",
                     transform=ax2.transAxes)
            ax2.set_axis_off()
    else:
        ax2.text(0.5, 0.5, "no ablation dataset available", ha="center", va="center",
                 transform=ax2.transAxes)
        ax2.set_axis_off()

    fig.suptitle("Quality-audit cleaning effect (E5)", y=1.02)
    return save_fig(fig, "fig_quality_cleaning_effect", out_dir)


# ---------------------------------------------------------------------------
# Results CSV (long format: one row per dataset / arm + shift metrics)
# ---------------------------------------------------------------------------

def _save_results_csv(metrics: dict, out_dir: Path) -> Path:
    rows: list[dict] = []
    # Quality-flag composition rows (all datasets).
    for ds, q in metrics["quality_flag"].items():
        for flag, pct in q["composition_pct"].items():
            rows.append({
                "dataset": ds, "section": "quality_flag", "flag": flag,
                "pct": pct, "count": q["counts"].get(flag, 0), "n_total": q["n_total"],
            })
    # Ablation rows (datasets with a raw snapshot).
    abl = metrics["ablation"]
    for ds, arm in abl["datasets"].items():
        common = {
            "dataset": ds, "target": arm["target"],
            "fraction_flagged": arm["fraction_flagged"],
            "fraction_target_changed": arm["fraction_target_changed"],
            "skill_gain": arm["skill_gain"],
            "target_std_raw": arm["target_std_raw"],
            "target_std_cleaned": arm["target_std_cleaned"],
        }
        for arm_name in ("raw", "cleaned"):
            a = arm[arm_name]
            rows.append({
                **common, "section": "ablation", "arm": arm_name,
                "daytime_mae": a["daytime_mae"],
                "daytime_rmse": a.get("daytime_rmse", float("nan")),
                "n_daytime": a["n_daytime"], "skill": a["skill"],
                "past_channels": a["past_channels"], "train_source": a["train_source"],
            })
        rows.append({
            **common, "section": "ablation", "arm": "reference",
            "daytime_mae": arm["reference"]["daytime_mae"],
            "n_daytime": arm["reference"]["n_daytime"],
        })
    return save_metrics(pd.DataFrame(rows), "results_e5_quality", out_dir)


def _save_physics_csv(physics: dict, out_dir: Path) -> Path | None:
    """Violation-rate (raw vs cleaned) + three-way irradiance closure → CSV.

    Long format: one row per (dataset, state) for the violation rate and one row
    per dataset for the closure. Empty → no file written (Honest skip)."""
    rows: list[dict] = []
    for ds, arm in physics.get("violation", {}).items():
        for state, m in arm.items():
            rows.append({"section": "violation_rate", "dataset": ds, "state": state, **m})
    for ds, cl in physics.get("closure", {}).items():
        rows.append({"section": "three_way_closure", "dataset": ds, **cl})
    if not rows:
        return None
    return save_metrics(pd.DataFrame(rows), "results_physics_closure", out_dir)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(loader, out_dir, *, raw_provider=None) -> dict:
    """Run experiment E5 and return ``{figures: [paths], metrics: {...}}``.

    Always produces ``fig_quality_flags.png`` (composition per dataset from the
    cleaned packages via ``loader``). When ``raw_provider`` is given and returns
    a raw frame for at least one dataset, also produces
    ``fig_quality_cleaning_effect.png`` (skill raw vs cleaned + target shift).
    Without a raw provider the cleaning-effect ablation is honestly skipped.

    Reads only standardized packages via ``loader`` + the optional raw frames
    from ``raw_provider``; writes figures + a results CSV under ``out_dir``.
    Pure: does not mutate ``loader`` state or the standardized data.
    """
    set_style("light")
    out_dir = Path(out_dir)

    # --- Sub-analysis 1: quality_flag composition (always; from cleaned pkgs) ---
    quality_flag: dict[str, dict] = {}
    for dataset in DATASETS:
        try:
            df = loader.load_tabular(dataset)
        except (FileNotFoundError, KeyError):
            quality_flag[dataset] = {"composition_pct": {}, "counts": {}, "n_total": 0}
            continue
        quality_flag[dataset] = _quality_flag_composition(df)

    figures: list[Path] = []
    figures += _plot_quality_flags(quality_flag, out_dir)

    # --- Sub-analysis 2: raw-vs-cleaned ablation (only if raw frames given) ---
    ablation_datasets: dict[str, dict] = {}
    raw_samples: dict[str, np.ndarray] = {}
    cleaned_samples: dict[str, np.ndarray] = {}
    violation: dict[str, dict] = {}  # physics violation rate raw vs cleaned

    if raw_provider is not None:
        for dataset, target in DATASET_TARGETS.items():
            try:
                raw_df = raw_provider(dataset)
            except Exception:  # pragma: no cover - defensive provider contract
                raw_df = None
            if raw_df is None:
                continue
            raw_df = raw_df.copy()
            if target not in raw_df.columns:
                continue
            # Capacity: prefer the raw frame's capacity_kw, else the PV default.
            cap = _DEFAULT_CAPACITY
            if "capacity_kw" in raw_df.columns and raw_df["capacity_kw"].notna().any():
                cap = float(raw_df["capacity_kw"].dropna().iloc[0])
            cleaned_df = _derive_cleaned(raw_df, capacity=cap)

            arm = _evaluate_dataset(raw_df, cleaned_df, target=target)
            ablation_datasets[dataset] = arm
            # Physics violation rate raw vs cleaned (governance should cut it).
            violation[dataset] = {
                "raw": physics_violation_rate(raw_df, capacity=cap),
                "cleaned": physics_violation_rate(cleaned_df, capacity=cap),
            }

            # Cache daytime target samples for the ECDF panel (first dataset only
            # is plotted; cheap to collect for all, used for one).
            cz = pd.to_numeric(raw_df.get("cos_zenith", pd.Series(dtype=float)),
                               errors="coerce")
            day = cz > _DAYTIME_COS_ZENITH
            r = pd.to_numeric(raw_df[target], errors="coerce")[day]
            c = pd.to_numeric(cleaned_df[target], errors="coerce")[day]
            raw_samples[dataset] = r[np.isfinite(r)].to_numpy(dtype=float)
            cleaned_samples[dataset] = c[np.isfinite(c)].to_numpy(dtype=float)

    if ablation_datasets:
        figures += _plot_cleaning_effect(
            ablation_datasets, raw_samples, cleaned_samples, out_dir
        )
        abl_status = "ran"
    else:
        abl_status = "skipped"

    # --- Sub-analysis 3: three-way irradiance closure (Folsom/SunLake) + the
    #     physics CSV (violation rate raw vs cleaned + closure R²). Violation
    #     rates are collected above (raw frames needed); closure uses the
    #     loader's standardized packages, which always carry DNI/DHI. ---
    closure: dict[str, dict] = {}
    for dataset in DATASETS:
        try:
            dfc = loader.load_tabular(dataset)
        except (FileNotFoundError, KeyError):
            continue
        cl = three_way_closure(dfc)
        if cl.get("status") == "ran":
            closure[dataset] = cl
    physics = {"violation": violation, "closure": closure}
    _save_physics_csv(physics, out_dir)

    metrics = {
        "quality_flag": quality_flag,
        "physics": physics,
        "ablation": {
            "status": abl_status,
            "datasets": ablation_datasets,
            "note": (
                "no raw_provider supplied; pass raw_provider(dataset)->raw_df "
                "or run scripts/build_dataset.py --dataset <d> --raw to enable "
                "the raw-vs-cleaned ablation"
            ) if abl_status == "skipped" else "",
        },
        "config": {
            "lookback": _LOOKBACK,
            "horizon": _HORIZON,
            "stride": _STRIDE,
            "model": "LightGBM",
            "n_estimators": _LGBM_PARAMS["n_estimators"],
            "features": list(_SHARED_FEATURES),
            "reference": "SmartPersistence",
            "train_frac": _TRAIN_FRAC,
            "daytime_cos_zenith": _DAYTIME_COS_ZENITH,
        },
    }
    _save_results_csv(metrics, out_dir)

    return {"figures": [str(p) for p in figures], "metrics": metrics}
