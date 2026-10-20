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
RES = _REPO_ROOT / "results" / "plots"


def _compute_frontier() -> tuple[list[str], list[float], list[float]]:
    """Margins from the cross-dataset grid CSV (micro-cleaned max vs deep camp)."""
    x = pd.read_csv(RES / "results_e9_cross_dataset.csv")
    datasets = ["folsom", "skippd", "sunlake", "gefcom2014"]
    disp = ["Folsom", "SKIPP'D", "SunLake", "GEFCom2014"]
    vs_mean, vs_best = [], []
    for ds in datasets:
        d = x[x.dataset == ds]
        micro = max(d[d.model == m].skill_cleaned.iloc[0] for m in ("LightGBM", "MLP"))
        deep = [d[d.model == m].skill_raw.iloc[0] for m in ("LSTM", "iTransformer", "PatchTST")]
        vs_mean.append(micro - float(np.mean(deep)))
        vs_best.append(micro - max(deep))
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
        for b, v in zip(bars, vals):
            va = "bottom" if v >= 0 else "top"
            off = 0.006 * (1 if v >= 0 else -1)
            ax.text(b.get_x() + b.get_width() / 2, v + off, f"{v:+.3f}",
                    ha="center", va=va, fontsize=8.5, color=col,
                    fontweight="bold")
    # Y head/foot room so the +/- labels never collide with the zero line.
    all_vals = vs_mean + vs_best
    lo, hi = min(all_vals), max(all_vals)
    span = hi - lo
    ax.set_ylim(lo - 0.22 * span, hi + 0.22 * span)
    ax.set_xticks(xpos, disp)
    ax.set_ylabel(r"CrossOver Margin $\chi$ (Skill Difference)")
    ax.grid(axis="y", linestyle="--", alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.text(0.01, 0.02,
            r"$\chi=\max(\mathrm{LGBM},\mathrm{MLP})_{\mathrm{cleaned}}-\mathrm{deep}_{\mathrm{raw}}$",
            transform=ax.transAxes, fontsize=9.5, color="0.25")


def _panel_b(ax: plt.Axes) -> None:
    """(b) Governance waterfall — the MET-informed 3-stage variant (E10b-MET).

    Values are read LIVE from results_e10b_waterfall_met.csv (no hardcoding);
    Base/+Hygiene keep the blue ramp of panel (a), the +Clear-sky
    breakthrough bar + its connector + the -7.51% label carry panel (a)'s
    academic orange.
    """
    import pandas as pd

    res = pd.read_csv(RES / "results_e10b_waterfall_met.csv")
    m = res[res.waterfall == "main_3stage"].set_index("stage")["mae"]
    mae = [float(m["base"]), float(m["+hygiene"]), float(m["+clear_sky"])]
    steps = ["Base (Target-lag + 5 MET)", "+Hygiene", "+Clear-sky"]

    colors = ["#08306b", "#2171b5", ORANGE]   # blue ramp -> academic orange
    bars = ax.bar(np.arange(3), mae, width=0.55, facecolor=colors,
                  edgecolor="none")
    ax.axhline(0.0, color="black", linewidth=0.8)
    span = max(mae) - min(mae)
    for i, (b, v) in enumerate(zip(bars, mae)):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.22 * span, f"{v:.5f}",
                ha="center", va="bottom", fontsize=8.5, color="black")
    for i in (1, 2):
        pct = (mae[i] / mae[i - 1] - 1) * 100
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
    ax.set_ylim(0.04, max(mae) + 0.42 * span)   # headroom for both bands
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
    fig.savefig(OUT.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}.pdf (+ .png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())


run = main
