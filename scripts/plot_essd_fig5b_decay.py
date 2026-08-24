"""ESSD fig5b — PSNR / SSIM decay (2-panel line figure).

(a) PSNR (dB) and (b) SSIM vs forecast horizon K. Clean bold left-aligned
titles (no stacked centered headers); SSIM y-range 0.5-1.0 (dead space below
the curve eliminated); Wong Sky Blue / Wong Orange lines matching fig4c's
palette binding; wspace 0.32; Arial throughout; PDF + 300 dpi PNG.

Run: python scripts/plot_essd_fig5b_decay.py
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
HEIGHT_MM = 62
DPI_RASTER = 300
WONG_PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7",
                "#D55E00", "#F0E442", "#0072B2", "#000000"]
GRID_COLOR = "#E5E5E5"
GRID_WIDTH = 0.3
TITLE_PT = 10
COL_PSNR = "#56B4E9"   # Wong sky blue  (same binding as fig4c)
COL_SSIM = "#E69F00"   # Wong orange
SSIM_YMIN, SSIM_YMAX = 0.5, 1.0

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

RESULTS = _REPO_ROOT / "figures_full" / "results" / "results_e4_optflow.csv"
OUT = _REPO_ROOT / "figures_essd" / "fig5b_optflow_decay"


def main() -> int:
    df = pd.read_csv(RESULTS)
    psnr = df["psnr_db"].to_numpy(dtype=float)
    ssim = df["ssim"].to_numpy(dtype=float)
    kk = np.arange(1, psnr.size + 1)

    fig, axs = plt.subplots(1, 2, figsize=(WIDTH_MM * mm, HEIGHT_MM * mm))
    plt.subplots_adjust(wspace=0.32, left=0.09, right=0.97, top=0.82, bottom=0.17)

    axs[0].plot(kk, psnr, color=COL_PSNR, marker="o",
                markersize=MARKER_SIZE + 2, linewidth=LINE_WIDTH_DATA + 0.4)
    axs[0].set_title(r"$\mathbf{(a)}$ PSNR (dB)", loc="left",
                     fontsize=TITLE_PT, pad=8)
    axs[0].set_xlabel(r"Forecast Horizon $K$ (steps)")
    axs[0].set_ylabel(r"PSNR (dB)")

    axs[1].plot(kk, ssim, color=COL_SSIM, marker="s",
                markersize=MARKER_SIZE + 2, linewidth=LINE_WIDTH_DATA + 0.4)
    axs[1].set_title(r"$\mathbf{(b)}$ SSIM", loc="left",
                     fontsize=TITLE_PT, pad=8)
    axs[1].set_xlabel(r"Forecast Horizon $K$ (steps)")
    axs[1].set_ylabel(r"SSIM")
    axs[1].set_ylim(SSIM_YMIN, SSIM_YMAX)   # kill the dead space below 0.5

    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=DPI_RASTER, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}.pdf (+ .png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
