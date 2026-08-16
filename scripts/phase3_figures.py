"""Phase 4 — ESSD figure export for the Phase-3 results.

Reads the Phase-3 CSVs and renders three publication figures via the shared
viz design system:
  1. fig_e9_cross_dataset_crossover.png — crossover margin per dataset (4-bar).
  2. fig_e9_waterfall.png              — governance waterfall (MAE vs step).
  3. fig_e9_granularity_sp_collapse.png — SP physical MAE vs granularity.

Run: python scripts/phase3_figures.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from solarunibench.viz.io import save_fig
from solarunibench.viz.style import PALETTE, set_style

RES = Path("figures_full/results")
OUT = Path("figures_full")


def _crossover():
    """Per-dataset crossover margin (simplest+cleaned vs most-complex+raw).
    Uses the per-dataset margins if recorded, else derives from the grid."""
    df = pd.read_csv(RES / "results_e9_cross_dataset.csv")
    datasets = ["gefcom2014", "folsom", "skippd", "sunlake"]
    margins, holds = [], []
    for ds in datasets:
        sub = df[df.dataset == ds]
        if sub.empty:
            margins.append(np.nan); holds.append(False); continue
        # simplest non-trivial = lowest-rank non-SP; most-complex = highest-rank.
        non_sp = sub[sub.model != "SmartPersistence"]
        if non_sp.empty:
            margins.append(np.nan); holds.append(False); continue
        simple = non_sp.loc[non_sp.complexity_rank.idxmin()]
        complex_m = sub.loc[sub.complexity_rank.idxmax()]
        margin = simple.skill_cleaned - complex_m.skill_raw
        margins.append(float(margin))
        holds.append(bool(margin >= 0))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    colors = [PALETTE[0] if h else PALETTE[3] for h in holds]
    bars = ax.bar(datasets, margins, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0.0, color="black", linewidth=0.9)
    # Display names consistent with the other figures (E7 combined panel etc.).
    disp = {"gefcom2014": "GEFCom2014", "folsom": "Folsom",
            "skippd": "SKIPPD", "sunlake": "SunLake"}
    ax.set_xticks(range(len(datasets)), [disp[d] for d in datasets], fontsize=10)
    for b, m in zip(bars, margins):
        if np.isfinite(m):
            # 3-decimal labels: gefcom (+0.012) and skippd (+0.013) must be
            # distinguishable (2-decimal "+0.01" read ambiguously at print size).
            # Offset is 1.5% of the y-span (tight to the bar top; the SunLake
            # label must stay inside the axes).
            span = max(margins) - min(min(margins), 0.0)
            ax.text(b.get_x() + b.get_width() / 2, m + 0.015 * span,
                    f"{m:+.3f}", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold")
    # Keep ~12% headroom above the tallest bar so its label fits inside.
    ymax = max(max(margins), 0.0)
    ymin = min(min(margins), 0.0)
    pad = 0.12 * (ymax - ymin if ymax > ymin else ymax)
    ax.set_ylim(ymin - 0.08 * (ymax - ymin + pad), ymax + pad)
    ax.set_ylabel("Crossover margin (skill_simple+cleaned - skill_complex+raw)")
    ax.set_title("E9 cross-dataset crossover (physics-aligned 1h-ahead)\n"
                 "bar >= 0: governance > complexity holds")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=PALETTE[0], edgecolor="black", label="holds (>=0)"),
                       Patch(facecolor=PALETTE[3], edgecolor="black", label="fails (<0)")],
              loc="best", fontsize=9)
    return save_fig(fig, "fig_e9_cross_dataset_crossover", OUT)


def _waterfall():
    df = pd.read_csv(RES / "results_e9_waterfall.csv")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    steps = df["step"].tolist()
    vals = df["daytime_mae"].tolist()
    colors = [PALETTE[4]] + ["#bbbbbb"] * (len(steps) - 2) + [PALETTE[0]]
    bars = ax.bar(range(len(steps)), vals, color=colors[:len(steps)],
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(steps)), steps, rotation=20, ha="right", fontsize=9)
    span = max(vals) - min(vals)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02 * span, f"{v:.4f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold")
    # arrow from first to last (the cumulative governance gain)
    ax.annotate("", xy=(len(steps) - 1, vals[-1]), xytext=(0, vals[0]),
                arrowprops=dict(arrowstyle="->", color=PALETTE[4], lw=1.3))
    ymax = max(vals)
    ax.set_ylim(0, ymax * 1.14)
    ax.set_ylabel("Daytime MAE (LightGBM, scaled)")
    ax.set_title("E9 governance waterfall (Folsom)\n"
                 "clearsky anchor features dominate the governance gain")
    return save_fig(fig, "fig_e9_waterfall", OUT)


def _granularity():
    df = pd.read_csv(RES / "results_e9_granularity.csv")
    df = df[df.status == "ran"].reset_index(drop=True)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(len(df))
    ax.plot(x, df["mae_sp_wm2"], color=PALETTE[3], marker="o", markersize=8,
            linewidth=2.2, label="SmartPersistence MAE (W/m^2)")
    ax.plot(x, df["mae_dlinear_wm2"], color=PALETTE[0], marker="s", markersize=7,
            linewidth=2.0, linestyle="--", label="DLinear MAE (W/m^2)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["granularity"])
    ax.set_xlabel("Temporal granularity (Folsom resampled)")
    ax.set_ylabel("Daytime MAE (W/m^2, unscaled physical)")
    ax.set_title("E9 SP-skill collapse with coarser granularity (Folsom)\n"
                 "SP weakens (MAE rises) as cadence coarsens; honest DLinear caveat shown")
    ax.legend(loc="best", fontsize=9)
    return save_fig(fig, "fig_e9_granularity_sp_collapse", OUT)


def main():
    set_style("light")
    figs = []
    for fn in (_crossover, _waterfall, _granularity):
        try:
            figs += fn()
        except Exception as e:  # pragma: no cover
            print(f"  {fn.__name__} skipped: {e!r}", flush=True)
    print(f"wrote {len(figs)} figures: {[str(p) for p in figs]}", flush=True)


if __name__ == "__main__":
    main()
