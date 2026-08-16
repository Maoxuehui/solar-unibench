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

Run: python scripts/plot_e9_gemini.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
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
OUT = _REPO_ROOT / "figures_full" / "fig_e9_crossover_waterfall_combined"
RES = _REPO_ROOT / "figures_full" / "results"


def _compute_frontier() -> tuple[list[str], list[float], list[float]]:
    """Margins from the cross-dataset grid CSV (micro-cleaned max vs deep camp)."""
    x = pd.read_csv(RES / "results_e9_cross_dataset.csv")
    datasets = ["gefcom2014", "folsom", "skippd", "sunlake"]
    disp = ["GEFCom2014", "Folsom", "SKIPPD", "SunLake"]
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
                label=r"vs.\ deep-model $\mu$")
    b2 = ax.bar(xpos + 0.19, vs_best, width=0.35, facecolor=ORANGE,
                edgecolor=ORANGE, linewidth=0.5, hatch="//",
                label=r"vs.\ deep-model best")
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
    steps = ["Raw\nBaseline", "+Night", "+Range\nClamp", "+UTC\nAlign", "+ClearSky\nFeatures"]
    mae = [0.0532, 0.0542, 0.0542, 0.0542, 0.0462]
    # Monotone blue ramp (dark -> light) for governance steps; the final
    # breakthrough step flips to the high-contrast orange emphasis.
    ramp = ["#08306b", "#2171b5", "#4292c6", "#6baed6"]
    colors = ramp + [ORANGE]
    bars = ax.bar(np.arange(len(mae)), mae, width=0.62, facecolor=colors,
                  edgecolor="none")
    ax.axhline(0.0, color="black", linewidth=0.8)
    span = max(mae) - min(mae)
    for i, (b, v) in enumerate(zip(bars, mae)):
        # Absolute MAE value just above the bar top.
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012 * span, f"{v:.4f}",
                ha="center", va="bottom", fontsize=9, color="black")
        if i < 4:
            note = "no-op"
            ncol, nfw = "0.25", "normal"
        else:
            note = r"$\downarrow$ $-13.1\%$"
            ncol, nfw = ORANGE, "bold"
        # Note floats well above the value (no overlap with it).
        ax.text(b.get_x() + b.get_width() / 2, v + 0.085 * span, note,
                ha="center", va="bottom", fontsize=10, color=ncol,
                fontweight=nfw)
    ax.set_xticks(np.arange(len(mae)), steps, fontsize=9.5)
    ax.set_ylabel(r"Daytime Forecast Error (MAE)")
    ax.set_ylim(0.038, max(mae) + 0.50 * span)   # headroom for note + value
    ax.grid(axis="y", linestyle="--", alpha=0.15)
    ax.set_axisbelow(True)


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6),
                             gridspec_kw={"wspace": 0.28})
    _panel_a(axes[0])
    _panel_b(axes[1])
    fig.text(0.012, 0.955, "(a)", fontsize=13, fontweight="bold")
    fig.text(0.545, 0.955, "(b)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(OUT.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}.pdf (+ .png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
