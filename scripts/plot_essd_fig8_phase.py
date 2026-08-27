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

MOD_COLORS = {"TSI": "#56B4E9", "MET": "#E69F00", "NWP": "#009E73"}
MOD_LABELS = {"TSI": "TSI Features", "MET": "Surface Met", "NWP": "NAM NWP"}

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
OUT = _REPO_ROOT / "figures_essd" / "fig8_phase_response"


def main() -> int:
    res = pd.read_csv(RESULTS)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_MM * mm, HEIGHT_MM * mm))
    plt.subplots_adjust(wspace=0.32, left=0.09, right=0.97, top=0.82, bottom=0.17)

    for ax, metric, title in ((axes[0], "rmse", r"$\mathbf{(a)}$ RMSE"),
                              (axes[1], "mae", r"$\mathbf{(b)}$ MAE")):
        for g, d in res.groupby("modality"):
            d = d.sort_values("delta_min")
            ax.plot(d.delta_min, d[metric], marker="o",
                    markersize=MARKER_SIZE + 2,
                    linewidth=LINE_WIDTH_DATA + 0.4,
                    color=MOD_COLORS[g], label=MOD_LABELS[g])
        ax.axvline(0, color="#A0A0A0", linestyle="--", linewidth=GRID_WIDTH + 0.2)
        ax.set_xlabel(r"Clock phase shift $\Delta$ (min)")
        ax.set_ylabel(f"Validation {metric.upper()} (W m$^{{-2}}$)"
                      if metric == "rmse" else f"Validation {metric.upper()} (W m$^{{-2}}$)")
        ax.set_title(title, loc="left", fontsize=TITLE_PT, pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(frameon=False, loc="upper left")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=DPI_RASTER, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}.pdf (+ .png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
