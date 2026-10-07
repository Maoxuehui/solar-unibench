"""ESSD fig8 — Folsom clock phase-shift response (E10), Copernicus-formatted.

Two-panel line figure from results_folsom_phase_sweep.csv:
(a) Validation RMSE (W m^-2) vs phase shift Delta (-120..+120 min),
(b) Validation MAE (W m^-2) — three modality curves (TSI / MET / NWP).
Wong palette binding matches the rest of the ESSD set; Arial 8.5 pt; zero
reference line at Delta=0; PDF (fonttype 42) + 300 dpi PNG.

Run: python scripts/plot_essd_fig8_phase.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# ===========================================================================
# MANDATORY PARAMETER BLOCK
# ===========================================================================
TEXT_SIZE_PT = 8.5
FONT_FAMILY = "Arial"
LINE_WIDTH_DATA = 0.6
MARKER_SIZE = 1.5
WIDTH_MM = 180
HEIGHT_MM = 62
DPI_RASTER = 300
WONG_PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7",
                "#D55E00", "#F0E442", "#0072B2", "#000000"]
GRID_COLOR = "#E5E5E5"
GRID_WIDTH = 0.3
TITLE_PT = 10

#: Line styling matched to fig3_benchmark_horizon_lines: uniform thin lines
#: (LINE_WIDTH_DATA + 0.4 = 1.0) + small markers (MARKER_SIZE + 2 = 3.5),
#: Wong palette order as in the rest of the set.
MOD_COLORS = {"MET": "#E69F00", "TSI": "#56B4E9", "NWP": "#009E73"}
MOD_LABELS = {"TSI": "TSI", "MET": "Surface MET", "NWP": "NAM NWP (Control)"}
MOD_STYLE = {g: dict(linewidth=LINE_WIDTH_DATA + 0.4,
                     markersize=MARKER_SIZE + 2) for g in MOD_COLORS}
BEST_MET_DELTA = 30          # the discovered systematic clock delay (min)

mpl.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": TEXT_SIZE_PT,
    "axes.labelsize": TEXT_SIZE_PT,
    "xtick.labelsize": TEXT_SIZE_PT,
    "ytick.labelsize": TEXT_SIZE_PT,
    "legend.fontsize": TEXT_SIZE_PT,
    "axes.linewidth": LINE_WIDTH_DATA,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.linewidth": GRID_WIDTH,
    "axes.axisbelow": True,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "axes.unicode_minus": False,
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

RESULTS = _REPO_ROOT / "figures_full" / "results" / "results_folsom_phase_sweep.csv"
OUT = _REPO_ROOT / "figures" / "fig8_phase_response"


def main() -> int:
    res = pd.read_csv(RESULTS)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_MM * mm, HEIGHT_MM * mm))
    plt.subplots_adjust(wspace=0.32, left=0.09, right=0.97, top=0.86, bottom=0.17)

    for ax, metric, title in ((axes[0], "rmse", r"$\mathbf{(a)}$"),
                              (axes[1], "mae", r"$\mathbf{(b)}$")):
        for g, d in res.groupby("modality"):
            d = d.sort_values("delta_min")
            st = MOD_STYLE[g]
            ax.plot(d.delta_min, d[metric], marker="o",
                    markersize=st["markersize"],
                    linewidth=st["linewidth"],
                    color=MOD_COLORS[g], label=MOD_LABELS[g],
                    zorder=5 if g == "MET" else 3)
        # baseline clock line (grey) + the discovered MET delay (light red)
        ax.axvline(0, color="#A0A0A0", linestyle="--", linewidth=GRID_WIDTH + 0.2)
        ax.axvline(BEST_MET_DELTA, color="#E66A6A", linestyle="--",
                   linewidth=1.0, alpha=0.7, zorder=2)
        ax.set_xlabel(r"Clock phase shift $\Delta$ [min]")
        ax.set_ylabel((r"Validation RMSE [W$\cdot$m$^{-2}$]" if metric == "rmse"
                       else r"Validation MAE [W$\cdot$m$^{-2}$]"))
        ax.set_title(title, loc="left", fontsize=TITLE_PT, pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Annotate the discovered systematic delay in the MID-UPPER band of the
    # left panel, offset RIGHT of the +30 min line so the dashed vline does
    # not strike through the text (that was the 'ghosting').
    ax0 = axes[0]
    ymin, ymax = ax0.get_ylim()
    ax0.text(BEST_MET_DELTA + 6, ymin + 0.55 * (ymax - ymin),
             r"$\hat{\Delta}_{\mathrm{MET}} = +30$ min",
             ha="left", va="center", fontsize=TEXT_SIZE_PT - 0.5,
             color="#C04040",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                       edgecolor="none", alpha=0.9))

    # Single-row horizontal legend floating ABOVE both panels (no occlusion).
    handles, labels = axes[0].get_legend_handles_labels()
    order = ["MET", "TSI", "NWP"]
    label_to_handle = dict(zip(labels, handles))
    ordered = [label_to_handle[MOD_LABELS[g]] for g in order]
    fig.legend(ordered, [MOD_LABELS[g] for g in order],
               ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.985),
               frameon=False, handletextpad=0.4, columnspacing=1.4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=DPI_RASTER, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}.pdf (+ .png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
