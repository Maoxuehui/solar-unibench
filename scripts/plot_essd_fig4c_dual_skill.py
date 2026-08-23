"""ESSD fig4c — E8 dual-metric skill trajectories (2x4 facet grid).

Extends fig4b (skill vs native horizon per dataset) to BOTH metrics:
row 1 = Daytime Skill_MAE (%), row 2 = Daytime Skill_RMSE (%); columns =
Folsom (1-min), SKIPP'D (1-min), SunLake (30-min), GEFCom2014 (1-h).
Free y per column; zero baseline dashed in every panel; (a)-(h) panel tags;
single-row 7-model legend on top (Wong palette, frameless).

Run: python scripts/plot_essd_fig4c_dual_skill.py
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

# ===========================================================================
# MANDATORY PARAMETER BLOCK
# ===========================================================================
TEXT_SIZE_PT = 8.5
FONT_FAMILY = "Arial"
LINE_WIDTH_DATA = 0.6
MARKER_SIZE = 1.5
WIDTH_MM = 180
HEIGHT_MM = 110
DPI_RASTER = 300
WONG_PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7",
                "#D55E00", "#F0E442", "#0072B2", "#000000"]
GRID_COLOR = "#E5E5E5"
GRID_WIDTH = 0.3
BASELINE_COLOR = "#A0A0A0"
PANEL_TAG_PT = 8.5

MODELS = ["LightGBM", "MLP", "LSTM", "iTransformer",
          "PatchTST-CI", "PatchTST-CD", "CrossViViT"]
MODEL_COLORS = dict(zip(MODELS, WONG_PALETTE[:7]))

# Column spec: (dataset key, strip label, csv step horizons, physical labels, unit)
COLS = [
    ("folsom", "Folsom (1-min)", [5, 10, 15], ["5", "10", "15"], "min"),
    ("skippd", "SKIPP'D (1-min)", [5, 10, 15], ["5", "10", "15"], "min"),
    ("sunlake", "SunLake (30-min)", [1, 2, 3], ["30", "60", "90"], "min"),
    ("gefcom2014", "GEFCom2014 (1-h)", [1, 2, 3], ["1", "2", "3"], "h"),
]
ROWS = [("Skill_MAE_pct", "MAE Skill (%)"),
        ("Skill_RMSE_pct", "RMSE Skill (%)")]

mpl.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": TEXT_SIZE_PT,
    "axes.labelsize": TEXT_SIZE_PT,
    "xtick.labelsize": TEXT_SIZE_PT,
    "ytick.labelsize": TEXT_SIZE_PT,
    "axes.linewidth": LINE_WIDTH_DATA,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.linewidth": GRID_WIDTH,
    "axes.axisbelow": True,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT_FAMILY,
    "mathtext.it": f"{FONT_FAMILY}:italic",
    "mathtext.bf": f"{FONT_FAMILY}:bold",
    "mathtext.default": "rm",
})
mm = 1 / 25.4

E8_SUMMARY = _REPO_ROOT / "figures" / "results" / "e8_summary.csv"
OUT = _REPO_ROOT / "figures_essd" / "fig4c_e8_dual_skill_grid"


def main() -> int:
    s = pd.read_csv(E8_SUMMARY)
    fig, axes = plt.subplots(
        2, 4, figsize=(WIDTH_MM * mm, HEIGHT_MM * mm),
        gridspec_kw={"wspace": 0.28, "hspace": 0.42},
    )
    tag = "a"
    for r, (metric, rowlab) in enumerate(ROWS):
        for c, (ds, strip, csv_hz, phys, unit) in enumerate(COLS):
            ax = axes[r, c]
            d = s[(s.Dataset == ds) & (s.Metric == metric)].sort_values("Horizon")
            x = np.arange(len(csv_hz))
            for m in MODELS:
                ys = [d[d.Horizon == h][m].iloc[0] for h in csv_hz]
                if all(pd.isna(v) for v in ys):
                    continue
                ax.plot(x, ys, marker="o", markersize=MARKER_SIZE + 2,
                        linewidth=LINE_WIDTH_DATA + 0.4,
                        color=MODEL_COLORS[m], label=m)
            ax.axhline(0, color=BASELINE_COLOR, linestyle="--",
                       linewidth=GRID_WIDTH + 0.2)
            ax.set_xticks(x, [f"{p} {unit}" for p in phys])
            ax.grid(which="minor", visible=False)
            if r == 0:
                ax.set_title(strip, fontsize=TEXT_SIZE_PT)
            if c == 0:
                ax.set_ylabel(rowlab)
            # (a)-(h) tag, top-left inside each panel
            ax.text(0.02, 0.97, f"({tag})", transform=ax.transAxes,
                    fontsize=PANEL_TAG_PT, fontweight="bold",
                    ha="left", va="top")
            tag = chr(ord(tag) + 1)

    # Single-row unified legend at the very top, frameless.
    handles = [plt.Line2D([], [], marker="o", markersize=MARKER_SIZE + 2,
                          linewidth=LINE_WIDTH_DATA + 0.4,
                          color=MODEL_COLORS[m], label=m) for m in MODELS]
    fig.legend(handles=handles, ncol=7, loc="lower center",
               bbox_to_anchor=(0.5, 0.985), frameon=False,
               handletextpad=0.35, columnspacing=1.1)

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=DPI_RASTER)
    plt.close(fig)
    print(f"wrote {OUT}.pdf (+ .png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
