"""Merged 2x2 defects figure — panels (a)+(b) ported VERBATIM from the
working fig4() in scripts/plot_essd_figures.py, panels (c)+(d) from the
working scripts/plot_fig_outliers_noise.py.

  (a) Missing values        fig4's presence matrix (merged runs, red crosses)
  (b) Irregular intervals   fig4's per-dataset gap histograms
  (c) Outlier fractions     physical vs statistical, linear 0-100 %
  (d) Example               real Folsom day, super-physical kt flagged

Run (conda env solar_uni):
    python scripts/plot_fig04_defects_outliers.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_fig_outliers_noise import (
    DAYTIME_COSZ, KT_SUPER, VERM,
    load as load_ds, pick_folsom_day, stage1,
)
DS_ORDER = ["gefcom2014", "folsom", "skippd", "sunlake"]
#: labels carry each dataset's NATIVE cadence (from defect_summary)
DS_LABELS = {"gefcom2014": "GEFCom2014 (1 h)", "folsom": "Folsom (1 min)",
             "skippd": "SKIPP'D (1 min)", "sunlake": "SunLake (30 min)"}

mpl.rcParams.update({
    "font.family": "Arial", "font.size": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": "#E5E5E5", "grid.linewidth": 0.3,
    "axes.axisbelow": True, "xtick.minor.visible": False,
    "ytick.minor.visible": False, "axes.facecolor": "white",
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})

#: ---- ported constants from plot_essd_figures.py (fig4's working set) ----
ACADEMIC_LABELS = {
    "pv_power_normalized": r"$\mathrm{PV}_{\mathrm{norm}}$",
    "ghi_actual": r"$\mathrm{GHI}_{\mathrm{act}}$",
    "dni_actual": r"$\mathrm{DNI}_{\mathrm{act}}$",
    "dhi_actual": r"$\mathrm{DHI}_{\mathrm{act}}$",
    "cos_zenith": r"$\cos\theta_z$",
    "solar_azimuth": r"$\gamma_s$",
    "ghi_clearsky": r"$\mathrm{GHI}_{\mathrm{cs}}$",
    "dni_clearsky": r"$\mathrm{DNI}_{\mathrm{cs}}$",
    "dhi_clearsky": r"$\mathrm{DHI}_{\mathrm{cs}}$",
    "clearness_index": r"$k_t$",
    "quality_flag": "Quality Flag",
    "split": "Data Split",
}
TEXT_SIZE_PT = 8.5
GREEN, RED, EDGE = "#009E73", "#D55E00", "#D5D5D1"
GLYPH = dict(fontfamily="DejaVu Sans", ha="center", va="center",
             fontsize=TEXT_SIZE_PT + 2, fontweight="bold")
RESULTS = _REPO_ROOT / "figures_full" / "results"
OUT_FIG = _REPO_ROOT / "Figures" / "fig04_defects_outliers"


def e7_inputs():
    """Exactly fig4()'s data path: CANONICAL_COLUMNS + missingness + diffs."""
    from solarunibench.experiments.e7_defects import CANONICAL_COLUMNS, _irregularity
    from solarunibench.loader import SolarUniBenchLoader
    summary = pd.read_csv(RESULTS / "defect_summary.csv")
    miss = summary[summary.section == "missingness"]
    loader = SolarUniBenchLoader(str(_REPO_ROOT / "data" / "standardized"))
    diffs = {}
    for ds in DS_ORDER:
        _, diffs[ds] = _irregularity(loader.load_tabular(ds))
    return list(CANONICAL_COLUMNS), miss, diffs


def draw_matrix(ax_m, cols, miss):
    """PORTED VERBATIM from plot_essd_figures.fig4() — working version."""
    ax_m.imshow(np.zeros((len(DS_ORDER), len(cols))), cmap="Greys", vmin=0,
                vmax=1, aspect="auto")
    present_rows = []
    for i, ds in enumerate(DS_ORDER):
        row = miss[miss.dataset == ds].set_index("column")["pct_nan"]
        present = [float(row.get(c, 100.0)) < 99.5 for c in cols]
        present_rows.append(present)
        for j, ok in enumerate(present):          # one glyph PER CELL
            ax_m.text(j, i, "✓" if ok else "✗",
                      color=GREEN if ok else RED, **GLYPH)
    for yb in np.arange(-0.5, len(DS_ORDER) + 0.5, 1):
        ax_m.axhline(yb, color=EDGE, linewidth=0.5, zorder=1)
    for xb in np.arange(-0.5, len(cols) + 0.5, 1):   # full cell grid
        ax_m.axvline(xb, color=EDGE, linewidth=0.5, zorder=1)
    ax_m.set_xticks(range(len(cols)),
                    [ACADEMIC_LABELS.get(c, c) for c in cols],
                    rotation=30, ha="right", fontsize=TEXT_SIZE_PT - 2,
                    rotation_mode="anchor")
    ax_m.set_yticks(range(len(DS_ORDER)),
                    [DS_LABELS[d].split(" (")[0] for d in DS_ORDER],
                    fontsize=TEXT_SIZE_PT - 2)
    ax_m.grid(False)
    ax_m.set_xlim(-0.5, len(cols) - 0.5)
    ax_m.set_ylim(len(DS_ORDER) - 0.5, -0.5)
    ax_m.set_title("(a) Missing values", fontsize=9, loc="left", pad=3)


def draw_minis(sub, diffs):
    """PORTED from fig4()'s working histogram block (shared 10^n ladder)."""
    edges = [0, 2, 5, 15, 60, 360, 1440, np.inf]
    labels = ["≤2", "2–5", "5–15", "15–60", "1–6 h", "6–24 h", ">24 h"]
    x = np.arange(len(labels))
    axs = [fig.add_subplot(s) for s in sub]
    all_cnt = {}
    for ds in DS_ORDER:
        d = pd.Series(diffs[ds])
        all_cnt[ds] = np.histogram(d[d > d.value_counts().idxmax()],
                                   bins=edges)[0]
    g_top = int(np.ceil(np.log10(max(int(c.max()) for c in all_cnt.values()))))
    n_ds = len(DS_ORDER)
    for k, (ax_h, ds) in enumerate(zip(axs, DS_ORDER)):
        cnt = all_cnt[ds]
        ax_h.bar(x, np.maximum(cnt, 0), width=0.55, color="#0072B2", zorder=3)
        for xb, c in zip(x, cnt):
            if c > 0:
                ax_h.text(xb, c * 0.72, f"{int(c)}", ha="center", va="top",
                          fontsize=TEXT_SIZE_PT - 2.5, color="white")
        ax_h.set_yscale("log")
        ax_h.set_ylim(0.5, 10 ** g_top)
        ticks = [10 ** e for e in range(0, g_top + 1)]
        ax_h.set_yticks(ticks,
                        [f"$10^{{{e}}}$" for e in range(0, g_top + 1)],
                        fontsize=TEXT_SIZE_PT - 2)
        ax_h.text(0.015, 0.93, DS_LABELS[ds], transform=ax_h.transAxes,
                  fontsize=TEXT_SIZE_PT - 2, ha="left", va="top",
                  color="#0B0B0B")
        if k == 1:
            ax_h.set_ylabel("Count", rotation=90, va="center",
                            fontsize=TEXT_SIZE_PT - 1, labelpad=2)
        if k < n_ds - 1:
            ax_h.set_xticks(x)
            ax_h.tick_params(labelbottom=False)
        else:
            ax_h.set_xticks(x, labels, fontsize=TEXT_SIZE_PT - 2)
            ax_h.set_xlabel("Gap (min)")
        ax_h.grid(which="minor", visible=False)
        ax_h.grid(which="major", color="#E5E5E5", linewidth=0.3)
    axs[0].set_title("(b) Irregular intervals", fontsize=9, loc="left", pad=3)


def draw_outlier_bars(ax, stats):
    x = np.arange(len(DS_ORDER))
    phys = [stats[ds]["phys_outlier_pct"] for ds in DS_ORDER]
    stat = [stats[ds]["stat_outlier_pct"] for ds in DS_ORDER]
    ax.bar(x - 0.19, phys, width=0.35, color="#0072B2",
           label="Physical (kt > 1.5 / PV > cap)", zorder=3)
    ax.bar(x + 0.19, stat, width=0.35, color="#E69F00",
           label="Statistical (robust z > 3.5)", zorder=3)
    for xi, v in zip(x - 0.19, phys):
        ax.text(xi, v + 2.0, f"{v:.2f}" if v >= 0.01 else "0",
                ha="center", va="bottom", fontsize=6.5)
    for xi, v in zip(x + 0.19, stat):
        ax.text(xi, v + 2.0, f"{v:.2f}" if v >= 0.01 else "0",
                ha="center", va="bottom", fontsize=6.5)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(x, [DS_LABELS[d].split(" (")[0] for d in DS_ORDER],
                  fontsize=7.5)
    ax.set_ylabel("Outlier fraction (%)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.set_title("(c) Outlier fractions", fontsize=9, loc="left", pad=3)


def draw_example(ax):
    day0 = pick_folsom_day()
    df = load_ds("folsom").loc[str(day0.date())]
    t_h = (df.index - df.index[0]).total_seconds() / 3600.0
    ax.plot(t_h, df["ghi_actual"], color="#E69F00", lw=0.7,
            label="Observed GHI")
    ax.plot(t_h, df["ghi_clearsky"], color="#52514E", lw=0.6, ls=(0, (4, 2)),
            label="Clear-sky GHI")
    kt = df["ghi_actual"] / (df["ghi_clearsky"] + 1e-6)
    sup = (kt > KT_SUPER) & (df["cos_zenith"] > DAYTIME_COSZ) \
        & (df["ghi_clearsky"] > 20)
    ax.plot(t_h[sup], df["ghi_actual"][sup], linestyle="none", marker="o",
            ms=3.0, color=VERM, zorder=5,
            label=f"kt > {KT_SUPER} ({int(sup.sum())})")
    ax.set_xlabel(f"Hour (UTC), {day0:%d %b %Y}")
    ax.set_ylabel(r"GHI (W m$^{-2}$)")
    ax.set_xlim(0, 24)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.set_title("(d) Example: Folsom flagged outliers", fontsize=9,
                 loc="left", pad=3)


def main() -> int:
    global fig
    print("[stage1] outlier/noise statistics ...", flush=True)
    res = stage1()
    print("[stage2] E7 defects inputs ...", flush=True)
    cols, miss, diffs = e7_inputs()

    fig = plt.figure(figsize=(180 / 25.4, 128 / 25.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.85], hspace=0.42,
                          wspace=0.25, left=0.055, right=0.99, top=0.955,
                          bottom=0.075)
    draw_matrix(fig.add_subplot(gs[0, 0]), cols, miss)
    draw_minis(gs[0, 1].subgridspec(len(DS_ORDER), 1, hspace=0.30), diffs)
    draw_outlier_bars(fig.add_subplot(gs[1, 0]), res["stats"])
    draw_example(fig.add_subplot(gs[1, 1]))

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_FIG}.{ext}", dpi=300, bbox_inches="tight")
        print(f"  {OUT_FIG.name}.{ext}")
    plt.close(fig)
    print(f"done -> {OUT_FIG}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
