"""E9 combined figure — (a) hierarchical crossover frontier + (b) governance waterfall.

One publication panel merging the two core dimensions of E9:
* **(a) The Hierarchical CrossOver Frontier** — grouped bars: margin of the
  cleaned micro-model ceiling (max of LightGBM/MLP) over the RAW deep-model
  camp, both vs. the camp MEAN (LSTM/iTransformer/PatchTST average) and vs.
  the camp CHAMPION (best of the three). 4/4 positive vs mean; 3/4 vs best
  (Folsom's LSTM+raw edges out LightGBM+cleaned — honestly shown as −0.013).
* **(b) The Causal Governance Waterfall** — Folsom, LightGBM probe: label
  governance steps are no-ops; the clear-sky anchor features deliver the
  breakthrough (−13.1%).

Colorblind-safe palette per the user spec: blue #005ab5 / orange #dc7d28
(with '//' hatch double-encoding), serif fonts, no titles (LaTeX captions
own the explanation), LaTeX-math axis labels, y-only faint dashed grid.

Run: python scripts/plots/plot_fig08.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -- global aesthetics: match the shared viz design system (DejaVu Sans),
# -- 10-12pt, no titles, faint y-grid ----------------------------------------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
})

BLUE, ORANGE = "#005ab5", "#dc7d28"
OUT = _REPO_ROOT / "figures" / "paper" / "fig08"
RES = _REPO_ROOT / "results"


def _compute_frontier() -> tuple[list[str], list[float], list[float]]:
    """Margins from the contiguous E9 rerun chi table (skill % -> fractions).

    Branch exp/e9-contiguous-rerun: reads results/results_e9_contiguous_chi.csv
    produced by scripts/experiments/e9_contiguous.py (full-data, E8-aligned
    splits, strict window contiguity). chi_mu / chi_best use the full deep
    pool ({LSTM, iTransformer, PatchTST-CD} + CrossViViT on image datasets).
    """
    chi = pd.read_csv(RES / "results_e9_contiguous_chi.csv")
    chi = chi.set_index("dataset")
    datasets = ["folsom", "skippd", "sunlake", "gefcom2014"]
    disp = ["Folsom", "SKIPP'D", "SunLake", "GEFCom2014"]
    vs_mean = [float(chi.loc[ds, "chi_mu"]) / 100.0 for ds in datasets]
    vs_best = [float(chi.loc[ds, "chi_best"]) / 100.0 for ds in datasets]
    return disp, vs_mean, vs_best


def _panel_a(ax: plt.Axes) -> None:
    disp, vs_mean, vs_best = _compute_frontier()
    xpos = np.arange(len(disp))
    b1 = ax.bar(xpos - 0.19, vs_mean, width=0.35, facecolor=BLUE, edgecolor="none",
                label=r"vs. deep-model $\mu$")
    b2 = ax.bar(xpos + 0.19, vs_best, width=0.35, facecolor=ORANGE,
                edgecolor=ORANGE, linewidth=0.5, hatch="//",
                label=r"vs. deep-model best")
    ax.axhline(0.0, color="black", linewidth=0.8)
    for bars, vals, col in ((b1, vs_mean, BLUE), (b2, vs_best, ORANGE)):
        dx = 0.0 if col == BLUE else 0.04   # nudge orange labels rightward
        for b, v in zip(bars, vals):
            va = "bottom" if v >= 0 else "top"
            off = 0.006 * (1 if v >= 0 else -1)
            ax.text(b.get_x() + b.get_width() / 2 + dx, v + off,
                    f"{v * 100:+.1f}%",
                    ha="center", va=va, fontsize=8.5, color=col,
                    fontweight="bold")
    # Percent axis: adaptive frame with quantized (5% or 10%) tick steps so
    # the chi range always fits (the contiguous rerun reached chi_mu +46%).
    all_vals = vs_mean + vs_best
    lo, hi = min(all_vals), max(all_vals)
    span = hi - lo
    lo_f, hi_f = lo - 0.18 * span - 0.02, hi + 0.24 * span + 0.02
    step = 0.05 if span <= 0.45 else 0.10
    lo_q = np.floor(lo_f / step) * step
    hi_q = np.ceil(hi_f / step) * step
    ax.set_ylim(lo_q, hi_q)
    ax.set_yticks(np.arange(lo_q, hi_q + step / 2, step))
    ax.yaxis.set_major_formatter(
        mpl.ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xticks(xpos, disp)
    ax.set_ylabel(r"Crossover Margin $\chi$ (Skill Difference, %)")
    ax.grid(axis="y", linestyle="--", alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.text(0.01, 0.02,
            r"$\chi=\mathrm{Skill}_{\mathrm{lightweight,\ cleaned}}^{\mathrm{best}}"
            r"-\mathrm{Skill}_{\mathrm{deep,\ raw}}$",
            transform=ax.transAxes, fontsize=9.5, color="0.25")


def _panel_b(ax: plt.Axes) -> None:
    """(b) Governance waterfall — contiguous full-data rerun (branch).

    Values are read LIVE from results/results_e10b_waterfall_contiguous.csv
    (scripts/experiments/e10b_waterfall_contiguous.py: full Folsom record,
    E8 splits, L=720/H=60/stride=30, strict contiguity — the same regime as
    panel (a)). Styling is unchanged vs the legacy panel: Base/+Hygiene keep
    the blue ramp, the +Clear-sky bar + connectors + % labels carry the
    academic orange.
    """
    import pandas as pd

    res = pd.read_csv(RES / "results_e10b_waterfall_contiguous.csv")
    m = res.set_index("stage")
    mae = [float(m.loc["raw", "mae"]),
           float(m.loc["+hygiene", "mae"]),
           float(m.loc["+clear_sky", "mae"])]
    pct_lbl = [None,
               float(m.loc["+hygiene", "delta_mae_pct"]),
               float(m.loc["+clear_sky", "delta_mae_pct"])]
    steps = ["Raw (target-lag)", "+Hygiene", "+Clear-sky"]

    colors = ["#08306b", "#2171b5", ORANGE]   # blue ramp -> academic orange
    bars = ax.bar(np.arange(3), mae, width=0.55, facecolor=colors,
                  edgecolor="none")
    ax.axhline(0.0, color="black", linewidth=0.8)
    span = max(mae) - min(mae)
    for i, (b, v) in enumerate(zip(bars, mae)):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.22 * span, f"{v:.5f}",
                ha="center", va="bottom", fontsize=8.5, color="black")
    for i in (1, 2):
        pct = pct_lbl[i]
        ax.annotate("", xy=(i, mae[i]), xytext=(i - 1, mae[i - 1]),
                    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.0,
                                    connectionstyle="arc3,rad=-0.15"))
        # -7.51% sits slightly BELOW -0.15%'s band (its column pair
        # spans lower; visually anchors the deeper drop).
        band = 0.0 if i == 2 else 0.14
        ax.text(i - 0.5, max(mae[i], mae[i - 1]) + band * span,
                f"{pct:+.2f}%", ha="center", va="bottom", fontsize=8.5,
                color=ORANGE, fontweight="bold")
    ax.set_xticks(np.arange(3), steps, fontsize=9.5)
    ax.set_ylabel(r"Daytime Forecast Error (MAE)")
    # Normal upright axis zoomed onto the tight contiguous-rerun range
    # (0.02555..0.02821): bars, value labels, and both % annotations all
    # fit INSIDE the frame. (The legacy hard ylim 0.04.. flew the bars out.)
    ax.set_ylim(0.0235, 0.0295)
    ax.set_yticks(np.arange(0.024, 0.0296, 0.001))
    ax.grid(axis="y", linestyle="--", alpha=0.15)
    ax.set_axisbelow(True)


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6),
                             gridspec_kw={"wspace": 0.28})
    _panel_a(axes[0])
    _panel_b(axes[1])
    # Subfigure tags in AXES coordinates: above the outer edge of each
    # panel's y-label column (-0.22 aligns with the label text).
    axes[0].text(-0.22, 1.05, "(a)", transform=axes[0].transAxes,
                 fontsize=14, fontweight="bold", va="bottom", ha="left")
    axes[1].text(-0.22, 1.05, "(b)", transform=axes[1].transAxes,
                 fontsize=14, fontweight="bold", va="bottom", ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    for ext in ("pdf", "png"):
        fig.savefig(OUT.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
        # LaTeX-convention twin at figures/ (manuscript references
        # Figures/fig08.pdf; the repo maps Figures/ -> figures/).
        fig.savefig(_REPO_ROOT / "figures" / f"fig08.{ext}", dpi=300,
                    bbox_inches="tight")
    print(f"wrote {OUT}.pdf (+ .png, + figures/ twins)")
    return 0


if __name__ == "__main__":
    sys.exit(main())


run = main
