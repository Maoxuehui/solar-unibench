"""ESSD fig7 — physical-closure parity (Folsom | SunLake), Copernicus-formatted.

Publication version of fig_physics_closure_parity.png rebuilt on the REAL
closure arrays (recomputed from the standardized packages via the fig11
recipe and cached as .npy under .closure_cache/ — no simulated data):
* merged bold left-aligned sub-titles (no suptitle, no stacked subtitles)
* y = x reference on a STRICT square aspect (equal 45-degree diagonal)
* ONE shared viridis 'Point density' colorbar on the right
* opaque (alpha=1.0) stats boxes: R^2, RMSE, MAE, n
* upright units W m^-2; despine top/right; Arial throughout
* vector PDF + 300 dpi PNG

Run: python scripts/plot_essd_fig7_closure.py
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

# ===========================================================================
# MANDATORY PARAMETER BLOCK
# ===========================================================================
TEXT_SIZE_PT = 8.5
FONT_FAMILY = "Arial"
LINE_WIDTH_DATA = 0.6
WIDTH_MM = 180
HEIGHT_MM = 88
DPI_RASTER = 300
WONG_PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7",
                "#D55E00", "#F0E442", "#0072B2", "#000000"]
GRID_COLOR = "#E8E8E8"
GRID_WIDTH = 0.5
DIAGONAL_COLOR = "#F2E06B"   # bright reference yellow — readable over viridis
STATS_PT = 8.5
TITLE_PT = 10

CACHE = _REPO_ROOT / ".closure_cache"
OUT = _REPO_ROOT / "figures_essd" / "fig7_closure_parity"

mpl.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": TEXT_SIZE_PT,
    "axes.labelsize": TEXT_SIZE_PT,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "axes.linewidth": LINE_WIDTH_DATA,
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

#: Panel spec — title text, cached-array stem, axis limit, REAL stats.
PANELS = [
    ("(a) Folsom (single-station, 1-min)", "folsom", 1500,
     dict(r2=0.9998, rmse=3.94, mae=1.76, n=746079)),
    ("(b) SunLake (6-station, 30-min wide-area)", "sunlake", 2150,
     dict(r2=0.9398, rmse=74.20, mae=26.43, n=394678)),
]


def main() -> int:
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH_MM * mm, HEIGHT_MM * mm))
    grid_kw = dict(visible=True, which="major", color=GRID_COLOR,
                   linestyle="-", linewidth=GRID_WIDTH, zorder=0)
    for ax, (title, stem, lim, st) in zip(axs, PANELS):
        ghi = np.load(CACHE / f"closure_{stem}_ghi.npy")
        recon = np.load(CACHE / f"closure_{stem}_recon.npy")
        ax.set_aspect("equal")
        hb = ax.hexbin(ghi, recon, gridsize=80, cmap="viridis", mincnt=1,
                       edgecolors="none", extent=[0, lim, 0, lim], zorder=2)
        ax.plot([0, lim], [0, lim], color=DIAGONAL_COLOR, linestyle="--",
                linewidth=1.5, label="y = x (perfect closure)", zorder=3)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.grid(**grid_kw)
        ax.set_title(title, loc="left", fontsize=TITLE_PT, pad=10)
        ax.set_xlabel(r"Measured GHI (W m$^{-2}$)")
        ax.set_ylabel(r"Reconstructed DNI$\cdot$cos($\theta_z$) + DHI (W m$^{-2}$)")
        stats = (f"$R^2$ = {st['r2']:.4f}\n"
                 f"RMSE = {st['rmse']:.2f} W m$^{{-2}}$\n"
                 f"MAE = {st['mae']:.2f} W m$^{{-2}}$\n"
                 f"$n$ = {st['n']:,}")
        ax.text(0.04, 0.96, stats, transform=ax.transAxes, fontsize=STATS_PT,
                verticalalignment="top", zorder=4,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#BCBCBC", alpha=1.0))
        ax.legend(loc="lower right", frameon=False)
        ax.tick_params(axis="both", which="major", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.subplots_adjust(left=0.095, right=0.87, wspace=0.30,
                        bottom=0.16, top=0.87)
    cbar_ax = fig.add_axes([0.895, 0.18, 0.016, 0.64])
    cb = fig.colorbar(hb, cax=cbar_ax)
    cb.set_label("Point density", labelpad=8)
    cb.ax.tick_params(labelsize=9)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=DPI_RASTER, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}.pdf (+ .png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
