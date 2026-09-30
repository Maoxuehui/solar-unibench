"""Outliers & noise across the four datasets — statistics first, then figure.

Stage 1 (statistics, REAL standardized tables, full records):
  * PHYSICAL outliers  GHI datasets (folsom, sunlake): kt = ghi_actual /
      ghi_clearsky > 1.5 recomputed from the RAW columns (the stored
      clearness_index is governance-clipped at 1.5 and would hide them);
      PV datasets (gefcom2014, skippd): pv_power_normalized > 1.0 capacity.
  * STATISTICAL outliers  daytime first-difference robust z-score:
      |d - median(d)| / (1.4826 * MAD) > 3.5 (spike/jump fraction).
  * NOISE level  high-frequency residual r = target - rolling_median(target,
      15 rows); block std over 30-row windows divided by the dataset's median
      DAYTIME target -> unitless noise CV (comparable across W/m^2 and
      normalized-PV unit families).
  Writes figures_full/results/results_outliers_noise.csv + stdout table.

Stage 2 (figure, 1x2, 180 mm, 300 dpi, house style):
  (a) outlier fractions per dataset (physical vs statistical, log y, labels)
  (b) real example trace: the Folsom day with the MOST super-physical kt
      samples — GHI + clear-sky, flagged outliers in vermillion.
  (The noise-CV box panel was dropped: its native-row blocking confounds
  cross-dataset comparison; the full noise statistics stay in the CSV.)

Run (conda env solar_uni):
    python scripts/plot_fig_outliers_noise.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({
    "font.family": "Arial", "font.size": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": "#E5E5E5", "grid.linewidth": 0.3,
    "axes.axisbelow": True, "xtick.minor.visible": False,
    "ytick.minor.visible": False, "axes.facecolor": "white",
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})

STD = _REPO_ROOT / "data" / "standardized"
OUT_CSV = _REPO_ROOT / "figures_full" / "results" / "results_outliers_noise.csv"
OUT_FIG = _REPO_ROOT / "Figures" / "fig_outliers_noise"
DS_ORDER = ["gefcom2014", "folsom", "skippd", "sunlake"]
DS_LABELS = {"gefcom2014": "GEFCom2014", "folsom": "Folsom",
             "skippd": "SKIPPD'", "sunlake": "SunLake"}
DS_COLORS = {"gefcom2014": "#56B4E9", "folsom": "#E69F00",
             "skippd": "#009E73", "sunlake": "#CC79A7"}
TARGET = {"gefcom2014": "pv_power_normalized", "folsom": "ghi_actual",
          "skippd": "pv_power_normalized", "sunlake": "ghi_actual"}
IS_GHI = {"folsom": True, "sunlake": True, "gefcom2014": False, "skippd": False}
DAYTIME_COSZ = 0.05
KT_SUPER = 1.5
ROBUST_Z = 3.5
ROLL = 15
BLOCK = 30
MAX_BLOCKS = 4000          # uniform cap for the box plots
VERM = "#D55E00"


def load(ds: str) -> pd.DataFrame:
    cols = ["timestamp_utc", "entity_id", "cos_zenith", "ghi_clearsky",
            "ghi_actual", "pv_power_normalized"]
    df = pd.read_parquet(STD / ds / "time_series.parquet", columns=cols)
    df = df.set_index("timestamp_utc").sort_index()
    return df


def stage1() -> dict:
    """Per-dataset outlier/noise statistics on the FULL records."""
    stats, cvs = {}, {}
    for ds in DS_ORDER:
        df = load(ds)
        day = df["cos_zenith"] > DAYTIME_COSZ
        tgt = df[TARGET[ds]]
        # -- physical outliers --------------------------------------------
        if IS_GHI[ds]:
            kt = tgt / (df["ghi_clearsky"] + 1e-6)
            phys = (kt > KT_SUPER) & day & (df["ghi_clearsky"] > 20)
            phys_extreme = float(kt[phys].max()) if phys.any() else float("nan")
        else:
            phys = (tgt > 1.0) & day
            phys_extreme = float(tgt[phys].max()) if phys.any() else float("nan")
        # -- statistical outliers (robust z on daytime first difference) ---
        d = tgt[day].diff().dropna()
        med = d.median()
        mad = (d - med).abs().median()
        z = (d - med).abs() / (1.4826 * mad if mad > 0 else np.nan)
        stat = z > ROBUST_Z
        # -- noise CV (high-frequency residual, 30-row blocks) -------------
        r = (tgt - tgt.rolling(ROLL, center=True, min_periods=1).median())[day]
        n_blocks = len(r) // BLOCK
        blk = r.to_numpy()[:n_blocks * BLOCK].reshape(n_blocks, BLOCK)
        blk_std = blk.std(axis=1)
        med_day = float(tgt[day].median())
        cv = blk_std / med_day if med_day > 0 else blk_std * np.nan
        keep = np.flatnonzero(np.isfinite(cv))
        if len(keep) > MAX_BLOCKS:                      # uniform subsample
            keep = keep[:: len(keep) // MAX_BLOCKS][:MAX_BLOCKS]
        cvs[ds] = cv[keep]
        stats[ds] = {
            "n_daytime": int(day.sum()),
            "phys_outlier_n": int(phys.sum()),
            "phys_outlier_pct": 100.0 * float(phys.sum()) / max(1, int(day.sum())),
            "phys_extreme": phys_extreme,
            "stat_outlier_n": int(stat.sum()),
            "stat_outlier_pct": 100.0 * float(stat.sum()) / max(1, len(d)),
            "noise_cv_median": float(np.nanmedian(cvs[ds])),
            "noise_cv_p95": float(np.nanpercentile(cvs[ds], 95)),
        }
        s = stats[ds]
        print(f"[{ds:10s}] n_day={s['n_daytime']:8d} "
              f"phys={s['phys_outlier_pct']:.4f}% (n={s['phys_outlier_n']}, "
              f"extreme={s['phys_extreme']:.2f}) "
              f"stat={s['stat_outlier_pct']:.4f}% "
              f"noiseCV med={s['noise_cv_median']:.4f} "
              f"p95={s['noise_cv_p95']:.4f}", flush=True)
    rows = [{"dataset": ds, **stats[ds]} for ds in DS_ORDER]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")
    return {"stats": stats, "cvs": cvs}


def pick_folsom_day() -> pd.Timestamp:
    """The Folsom day with the MOST super-physical kt samples (deterministic)."""
    df = load("folsom")
    day = df["cos_zenith"] > DAYTIME_COSZ
    kt = df["ghi_actual"] / (df["ghi_clearsky"] + 1e-6)
    sup = (kt > KT_SUPER) & day & (df["ghi_clearsky"] > 20)
    per_day = sup.groupby(df.index.date).sum()
    best = per_day.idxmax()
    print(f"[example] Folsom day with most kt>{KT_SUPER} samples: "
          f"{best} (n={int(per_day.max())})")
    return pd.Timestamp(best)


def stage2(res: dict) -> None:
    stats, cvs = res["stats"], res["cvs"]
    fig = plt.figure(figsize=(180 / 25.4, 60 / 25.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.30,
                          left=0.055, right=0.99, top=0.88, bottom=0.19)

    # (a) outlier fractions (physical vs statistical), log scale
    ax_a = fig.add_subplot(gs[0])
    x = np.arange(len(DS_ORDER))
    phys = [stats[ds]["phys_outlier_pct"] for ds in DS_ORDER]
    stat = [stats[ds]["stat_outlier_pct"] for ds in DS_ORDER]
    ax_a.bar(x - 0.19, np.maximum(phys, 1e-5), width=0.35, color="#0072B2",
             label="Physical (kt > 1.5 / PV > cap)", zorder=3)
    ax_a.bar(x + 0.19, np.maximum(stat, 1e-5), width=0.35, color="#E69F00",
             label="Statistical (robust z > 3.5)", zorder=3)
    for xi, v in zip(x - 0.19, phys):
        ax_a.text(xi, v + 2.0, f"{v:.2f}" if v >= 0.01 else "0",
                  ha="center", va="bottom", fontsize=6.5)
    for xi, v in zip(x + 0.19, stat):
        ax_a.text(xi, v + 2.0, f"{v:.2f}" if v >= 0.01 else "0",
                  ha="center", va="bottom", fontsize=6.5)
    ax_a.set_ylim(0, 100)                      # linear; headroom for legend
    ax_a.set_yticks([0, 25, 50, 75, 100])
    ax_a.set_xticks(x, [DS_LABELS[d] for d in DS_ORDER],
                    fontsize=7.5, rotation=20, ha="right")
    ax_a.set_ylabel("Outlier fraction (%)")
    ax_a.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax_a.set_title("(a) Outlier fractions", fontsize=9, loc="left", pad=4)

    # (c) real example trace: worst Folsom day, residual + flagged outliers
    ax_c = fig.add_subplot(gs[1])
    day0 = pick_folsom_day()
    df = load("folsom").loc[str(day0.date())]
    t_h = (df.index - df.index[0]).total_seconds() / 3600.0
    ax_c.plot(t_h, df["ghi_actual"], color="#E69F00", lw=0.7,
              label="Observed GHI")
    ax_c.plot(t_h, df["ghi_clearsky"], color="#52514E", lw=0.6, ls=(0, (4, 2)),
              label="Clear-sky GHI")
    kt = df["ghi_actual"] / (df["ghi_clearsky"] + 1e-6)
    sup = (kt > KT_SUPER) & (df["cos_zenith"] > DAYTIME_COSZ) \
        & (df["ghi_clearsky"] > 20)
    ax_c.plot(t_h[sup], df["ghi_actual"][sup], linestyle="none", marker="o",
              ms=3.0, color=VERM, zorder=5,
              label=f"kt > {KT_SUPER} ({int(sup.sum())})")
    ax_c.set_xlabel(f"Hour (UTC), {day0:%d %b %Y}")
    ax_c.set_ylabel(r"GHI (W m$^{-2}$)")
    ax_c.set_xlim(0, 24)
    ax_c.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax_c.set_title("(b) Example: Folsom flagged outliers", fontsize=9,
                   loc="left", pad=4)

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_FIG}.{ext}", dpi=300, bbox_inches="tight")
        print(f"  {OUT_FIG.name}.{ext}")
    plt.close(fig)


def main() -> int:
    print("Stage 1 — statistics ...")
    res = stage1()
    print("Stage 2 — figure ...")
    stage2(res)
    print(f"done -> {OUT_FIG}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
