"""E7 — Data-defect diagnostics (descriptive, all datasets, NO model).

Spec §11 (E7) — the methodology's "缺陷热力图" (defect heatmap) diagnostic
reports. E7 is purely DESCRIPTIVE: it characterizes the standardized packages'
data defects WITHOUT training any model. Three diagnostic axes:

1. **Missingness** — per-canonical-column NaN fraction across the union schema.
   Honest-NaN modalities (GEFCom no ``ghi_actual``/``clearness_index``; Folsom /
   SunLake no ``pv_power_normalized``; SKIPPD irradiance absent at the tabular
   level) are reported as such — they document each dataset's modality contract,
   NOT a defect to "fix". A canonical column entirely absent from a dataset's
   frame is reported as 100% NaN (the loader probes existence to switch
   modality).

2. **Irregularity** — per-entity timestamp-gap distribution: modal cadence vs
   observed; count of gaps (>1.5x modal) and duplicates (Δt ≤ 0); percentiles
   of the positive inter-step delta. Validates the unique
   ``(timestamp_utc, entity_id)`` contract + expected cadence.

3. **Outliers / quality_flag composition** — distribution of ``quality_flag``
   values (0 good / 1 curtailment-clipping / 2 instrument-fault / 3 night) per
   dataset; IQR outlier fraction on the target + ghi columns over the daytime
   finite subset (``cos_zenith > 0.05``) — the physically-meaningful signal
   range for solar targets.

Three figures (``fig_defect_missingness`` / ``_irregularity`` / ``_outliers``)
+ a ``defect_summary`` CSV, all routed through the shared :mod:`solarunibench.viz`
design system.

Pure function contract: ``run(loader, out_dir) -> {figures, metrics}`` reads the
loader and writes to ``out_dir``; it never mutates standardized data.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..viz.io import save_fig, save_metrics
from ..viz.style import DATASET_COLORS, PALETTE, set_style

__all__ = ["run"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: All four standardized packages.
DATASETS: tuple[str, ...] = ("gefcom2014", "folsom", "skippd", "sunlake")

#: Per-dataset forecasting target (spec §10) — PV for GEFCom/SKIPPD, GHI for
#: Folsom/SunLake. Drives the target-column outlier diagnostic.
DATASET_TARGETS: dict[str, str] = {
    "gefcom2014": "pv_power_normalized",
    "skippd": "pv_power_normalized",
    "folsom": "ghi_actual",
    "sunlake": "ghi_actual",
}

#: Canonical column set (contract Tables 2-3 anchors + targets + derived) tracked
#: in the missingness matrix. The Honest-NaN modalities live here. Per-dataset
#: NWP extensions (skippd has ~190) are intentionally excluded — they are not
#: comparable across datasets and would bury the Honest-NaN signal in noise.
CANONICAL_COLUMNS: tuple[str, ...] = (
    "pv_power_normalized",
    "ghi_actual", "dni_actual", "dhi_actual",
    "cos_zenith", "solar_azimuth",
    "ghi_clearsky", "dni_clearsky", "dhi_clearsky",
    "clearness_index",
    "quality_flag",
    "split",
)

#: Daytime threshold for the outlier diagnostic (spec global constraint
#: cos_zenith > 0.05). E7 has no model, but solar outliers are physically
#: meaningful only on the daytime finite subset (night = 0 is correct, not an
#: outlier).
_DAYTIME_COS_ZENITH: float = 0.05

#: Inter-step delta multiple above the modal cadence counted as a gap.
_GAP_THRESHOLD_X: float = 1.5

#: quality_flag semantics (spec §4.2) — for human-readable figure legends.
_QUALITY_FLAG_NAMES: dict[int, str] = {
    0: "0 good",
    1: "1 clipping",
    2: "2 fault",
    3: "3 night",
}


# ---------------------------------------------------------------------------
# Sub-analysis 1: missingness (Honest-NaN matrix)
# ---------------------------------------------------------------------------

def _missingness(df: pd.DataFrame) -> dict[str, float]:
    """Per-canonical-column NaN fraction (0-100). Absent column -> 100.0.

    An absent column is an Honest-NaN modality (the loader probes file existence
    to switch modality), so it is reported as 100% NaN — NOT silently dropped.
    """
    out: dict[str, float] = {}
    n = len(df)
    for col in CANONICAL_COLUMNS:
        if col not in df.columns or n == 0:
            out[col] = 100.0
        else:
            # ``isna`` correctly detects None/NaN for both numeric and object
            # dtypes (real-data all-None object columns like folsom's
            # pv_power_normalized report 100% here).
            out[col] = float(df[col].isna().mean() * 100.0)
    return out


def _plot_missingness(miss: dict[str, dict[str, float]], out_dir: Path) -> list[Path]:
    datasets = list(miss.keys())
    cols = list(CANONICAL_COLUMNS)
    mat = np.full((len(datasets), len(cols)), np.nan)
    for i, ds in enumerate(datasets):
        for j, c in enumerate(cols):
            mat[i, j] = miss[ds].get(c, 100.0)
    fig, ax = plt.subplots(
        figsize=(max(8.5, 0.72 * len(cols) + 2.0), max(3.5, 0.55 * len(datasets) + 2.0))
    )
    # cividis: perceptually-uniform + colorblind-safe sequential colormap.
    im = ax.imshow(mat, aspect="auto", cmap="cividis", vmin=0, vmax=100)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets)
    for i in range(len(datasets)):
        for j in range(len(cols)):
            v = mat[i, j]
            if np.isfinite(v):
                # light annotation on dark cells (high NaN), dark on light (low NaN)
                color = "white" if v > 55 else "black"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color=color)
    ax.set_title("Data missingness — % NaN per canonical column\n(Honest-NaN modalities, not defects)")
    fig.colorbar(im, ax=ax, label="% NaN")
    return save_fig(fig, "fig_defect_missingness", out_dir)


# ---------------------------------------------------------------------------
# Sub-analysis 2: irregularity (timestamp gap distribution)
# ---------------------------------------------------------------------------

def _irregularity(df: pd.DataFrame) -> tuple[dict, np.ndarray]:
    """Return (summary, positive_diffs_minutes).

    summary: modal_cadence_seconds, n_total_steps, n_gaps, n_duplicates,
    n_entities, gap_minutes_{p50,p95,max}.
    positive_diffs_minutes: per-step positive deltas (for the boxplot).
    """
    empty_summary = {
        "modal_cadence_seconds": float("nan"),
        "n_total_steps": 0,
        "n_gaps": 0,
        "n_duplicates": 0,
        "n_entities": 0,
        "gap_minutes_p50": float("nan"),
        "gap_minutes_p95": float("nan"),
        "gap_minutes_max": float("nan"),
    }
    if (
        "timestamp_utc" not in df.columns
        or "entity_id" not in df.columns
        or len(df) == 0
    ):
        return empty_summary, np.array([], dtype=float)

    n_entities = int(df["entity_id"].nunique())
    pos_diffs_s: list[float] = []
    n_duplicates = 0
    n_total = 0
    for _, g in df.groupby("entity_id", sort=False):
        ts = pd.to_datetime(g["timestamp_utc"]).sort_values()
        d = ts.diff().dropna()
        seconds = d.dt.total_seconds().to_numpy(dtype=float)
        if seconds.size == 0:
            continue
        n_total += int(seconds.size)
        n_duplicates += int((seconds <= 0).sum())
        pos = seconds[seconds > 0]
        if pos.size:
            pos_diffs_s.extend(pos.tolist())

    arr = np.asarray(pos_diffs_s, dtype=float)
    if arr.size == 0:
        empty_summary["n_entities"] = n_entities
        empty_summary["n_total_steps"] = n_total
        return empty_summary, np.array([], dtype=float)

    # Modal cadence: mode of positive diffs (fall back to median if no unique mode).
    mode_s = pd.Series(arr).mode()
    modal = float(mode_s.iloc[0]) if len(mode_s) else float(np.median(arr))
    n_gaps = int((arr > _GAP_THRESHOLD_X * modal).sum())

    summary = {
        "modal_cadence_seconds": modal,
        "n_total_steps": n_total,
        "n_gaps": n_gaps,
        "n_duplicates": n_duplicates,
        "n_entities": n_entities,
        "gap_minutes_p50": float(np.percentile(arr, 50) / 60.0),
        "gap_minutes_p95": float(np.percentile(arr, 95) / 60.0),
        "gap_minutes_max": float(arr.max() / 60.0),
    }
    return summary, arr / 60.0  # minutes for the figure


def _plot_irregularity(
    diffs_min: dict[str, np.ndarray], summary: dict[str, dict], out_dir: Path
) -> list[Path]:
    datasets = [ds for ds in diffs_min if diffs_min[ds].size > 0]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    if datasets:
        data = [diffs_min[ds] for ds in datasets]
        bp = ax.boxplot(
            data, tick_labels=datasets, showfliers=True, patch_artist=True,
            medianprops=dict(color="black", linewidth=1.2),
        )
        for patch, ds in zip(bp["boxes"], datasets):
            patch.set_facecolor(DATASET_COLORS.get(ds, PALETTE[0]))
            patch.set_alpha(0.75)
        ax.set_yscale("log")
    # Multi-line x labels: cadence | gaps | dups (the headline irregularity stats).
    labels = []
    for ds in diffs_min:
        s = summary.get(ds, {})
        cad = s.get("modal_cadence_seconds", float("nan"))
        cad_str = f"{cad / 60.0:.0f}min" if np.isfinite(cad) else "?"
        labels.append(
            f"{ds}\ncad={cad_str} | gaps={s.get('n_gaps', 0)} | dups={s.get('n_duplicates', 0)}"
        )
    ax.set_xticks(range(1, len(diffs_min) + 1))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Inter-step gap (minutes, log scale)")
    ax.set_title("Timestamp irregularity — inter-step gap distribution per dataset")
    return save_fig(fig, "fig_defect_irregularity", out_dir)


# ---------------------------------------------------------------------------
# Sub-analysis 3a: quality_flag composition
# ---------------------------------------------------------------------------

def _quality_flag_composition(df: pd.DataFrame) -> dict:
    if "quality_flag" not in df.columns or len(df) == 0:
        return {"counts": {}, "composition_pct": {}, "n_total": 0}
    vc = df["quality_flag"].value_counts(dropna=False)
    n = int(len(df))
    counts: dict[str, int] = {}
    pct: dict[str, float] = {}
    for k, v in vc.items():
        key = "nan" if pd.isna(k) else str(int(k))
        counts[key] = int(v)
        pct[key] = float(v / n * 100.0) if n else 0.0
    return {"counts": counts, "composition_pct": pct, "n_total": n}


# ---------------------------------------------------------------------------
# Sub-analysis 3b: IQR outlier fraction (target + ghi, daytime finite subset)
# ---------------------------------------------------------------------------

def _iqr_outlier_fraction(series: pd.Series) -> tuple[float, int]:
    """Return (outlier_fraction, n_finite). NaN fraction when no finite values."""
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), 0
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    if iqr == 0:  # degenerate (constant) -> no outliers definable
        return 0.0, int(arr.size)
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (arr < lo) | (arr > hi)
    return float(mask.mean()), int(arr.size)


def _outliers(df: pd.DataFrame, target_col: str) -> dict:
    daytime = pd.Series(True, index=df.index)
    if "cos_zenith" in df.columns:
        cz = pd.to_numeric(df["cos_zenith"], errors="coerce")
        daytime = cz > _DAYTIME_COS_ZENITH
    out = {
        "target_col": target_col,
        "ghi_col": "ghi_actual",
        "n_daytime": int(daytime.sum()),
    }
    # Target outliers (daytime finite subset).
    if target_col and target_col in df.columns:
        frac, n = _iqr_outlier_fraction(df.loc[daytime, target_col])
    else:
        frac, n = float("nan"), 0
    out["target_outlier_fraction"] = frac
    out["target_outlier_n"] = n
    # GHI outliers (daytime finite subset); NaN when the column is absent/all-NaN.
    if "ghi_actual" in df.columns:
        frac, n = _iqr_outlier_fraction(df.loc[daytime, "ghi_actual"])
    else:
        frac, n = float("nan"), 0
    out["ghi_outlier_fraction"] = frac
    out["ghi_outlier_n"] = n
    return out


def _plot_outliers(
    qf: dict[str, dict], outl: dict[str, dict], out_dir: Path
) -> list[Path]:
    datasets = list(qf.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    x = np.arange(len(datasets))

    # Left: stacked bar — quality_flag composition (%) per dataset.
    all_flags = sorted({f for ds in datasets for f in qf[ds]["composition_pct"]},
                       key=lambda s: (s == "nan", s))
    bottoms = np.zeros(len(datasets))
    for fi, flag in enumerate(all_flags):
        vals = np.array(
            [qf[ds]["composition_pct"].get(flag, 0.0) for ds in datasets], dtype=float
        )
        label = _QUALITY_FLAG_NAMES.get(int(flag), f"flag={flag}") if flag != "nan" else "NaN"
        ax1.bar(x, vals, bottom=bottoms, color=PALETTE[fi % len(PALETTE)],
                edgecolor="black", linewidth=0.4, label=label)
        bottoms += vals
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=30, ha="right")
    ax1.set_ylabel("% of rows")
    ax1.set_title("quality_flag composition")
    ax1.set_ylim(0, 100)
    ax1.legend(loc="best", fontsize=9)

    # Right: grouped bar — IQR outlier fractions (target + ghi), daytime subset.
    target_fracs = [outl[ds].get("target_outlier_fraction", float("nan")) for ds in datasets]
    ghi_fracs = [outl[ds].get("ghi_outlier_fraction", float("nan")) for ds in datasets]
    w = 0.38
    ax2.bar(
        x - w / 2, [f if np.isfinite(f) else 0.0 for f in target_fracs], w,
        color=PALETTE[0], edgecolor="black", linewidth=0.4, label="target IQR outlier",
    )
    ax2.bar(
        x + w / 2, [f if np.isfinite(f) else 0.0 for f in ghi_fracs], w,
        color=PALETTE[1], edgecolor="black", linewidth=0.4, label="ghi IQR outlier",
    )
    # Annotate NaN (Honest-NaN modality) so the zero-height bar is not misread.
    for i, (tf, gf) in enumerate(zip(target_fracs, ghi_fracs)):
        if not np.isfinite(tf):
            ax2.text(x[i] - w / 2, 0.0, "Honest\nNaN", ha="center", va="bottom",
                     fontsize=7, color=PALETTE[5])
        if not np.isfinite(gf):
            ax2.text(x[i] + w / 2, 0.0, "Honest\nNaN", ha="center", va="bottom",
                     fontsize=7, color=PALETTE[5])
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, rotation=30, ha="right")
    ax2.set_ylabel("daytime IQR outlier fraction")
    ax2.set_title("Outlier fraction (IQR, daytime subset)")
    ax2.legend(loc="best", fontsize=9)

    fig.suptitle("Data outliers + quality_flag composition (E7)", y=1.02)
    return save_fig(fig, "fig_defect_outliers", out_dir)


# ---------------------------------------------------------------------------
# Results CSV (long format: one row per dataset / section / metric)
# ---------------------------------------------------------------------------

def _save_results_csv(metrics: dict, out_dir: Path) -> Path:
    rows: list[dict] = []
    for ds, cols in metrics["missingness"].items():
        for col, pct in cols.items():
            rows.append({"dataset": ds, "section": "missingness",
                         "column": col, "pct_nan": pct})
    for ds, s in metrics["irregularity"].items():
        rows.append({"dataset": ds, "section": "irregularity", **s})
    for ds, q in metrics["quality_flag"].items():
        for flag, cnt in q["counts"].items():
            rows.append({"dataset": ds, "section": "quality_flag",
                         "flag": flag, "count": cnt,
                         "pct": q["composition_pct"].get(flag, 0.0),
                         "n_total": q["n_total"]})
    for ds, o in metrics["outliers"].items():
        rows.append({"dataset": ds, "section": "outliers", **o})
    return save_metrics(pd.DataFrame(rows), "defect_summary", out_dir)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(loader, out_dir) -> dict:
    """Run experiment E7 and return ``{figures: [paths], metrics: {...}}``.

    Reads only standardized packages via ``loader``; writes three figures + a
    defect summary CSV under ``out_dir``. Pure: does not mutate ``loader`` state
    or the standardized data. E7 is descriptive — no forecasting model.
    """
    set_style("light")
    out_dir = Path(out_dir)

    missingness: dict[str, dict[str, float]] = {}
    irregularity: dict[str, dict] = {}
    diffs_min: dict[str, np.ndarray] = {}
    quality_flag: dict[str, dict] = {}
    outliers: dict[str, dict] = {}

    for dataset in DATASETS:
        try:
            df = loader.load_tabular(dataset)
        except (FileNotFoundError, KeyError):
            # A dataset whose standardized package is not yet built is reported
            # honestly as fully missing rather than crashing the diagnostic.
            missingness[dataset] = {c: 100.0 for c in CANONICAL_COLUMNS}
            irregularity[dataset] = {
                "modal_cadence_seconds": float("nan"), "n_total_steps": 0,
                "n_gaps": 0, "n_duplicates": 0, "n_entities": 0,
                "gap_minutes_p50": float("nan"), "gap_minutes_p95": float("nan"),
                "gap_minutes_max": float("nan"),
            }
            diffs_min[dataset] = np.array([], dtype=float)
            quality_flag[dataset] = {"counts": {}, "composition_pct": {}, "n_total": 0}
            outliers[dataset] = {
                "target_col": DATASET_TARGETS.get(dataset, ""),
                "ghi_col": "ghi_actual", "n_daytime": 0,
                "target_outlier_fraction": float("nan"), "target_outlier_n": 0,
                "ghi_outlier_fraction": float("nan"), "ghi_outlier_n": 0,
            }
            continue

        missingness[dataset] = _missingness(df)
        ir, dmin = _irregularity(df)
        irregularity[dataset] = ir
        diffs_min[dataset] = dmin
        quality_flag[dataset] = _quality_flag_composition(df)
        outliers[dataset] = _outliers(df, DATASET_TARGETS.get(dataset, ""))

    figures: list[Path] = []
    figures += _plot_missingness(missingness, out_dir)
    figures += _plot_irregularity(diffs_min, irregularity, out_dir)
    figures += _plot_outliers(quality_flag, outliers, out_dir)

    metrics = {
        "missingness": missingness,
        "irregularity": irregularity,
        "quality_flag": quality_flag,
        "outliers": outliers,
        "config": {
            "canonical_columns": list(CANONICAL_COLUMNS),
            "daytime_cos_zenith": _DAYTIME_COS_ZENITH,
            "gap_threshold_x": _GAP_THRESHOLD_X,
            "outlier_method": "IQR (1.5x) on daytime finite subset",
        },
    }
    _save_results_csv(metrics, out_dir)

    return {"figures": [str(p) for p in figures], "metrics": metrics}
