"""Paper fig10 — E8 daytime-skill heatmaps (MAE | RMSE).

Extracted from dev/plot_essd_figures.py fig2() verbatim; reads
results/plots/e8_summary.csv; writes figures/paper/fig10.{png,pdf}.

Run:  python scripts/plots/plot_fig10.py
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

# ===========================================================================
# MANDATORY PARAMETER BLOCK — the ONLY place sizes/colors/dimensions live
# ===========================================================================
TEXT_SIZE_PT = 8.5          # single text size for ALL elements
FONT_FAMILY = "Arial"       # Copernicus/ESSD preferred sans-serif
LINE_WIDTH_DATA = 0.6       # stroke unit for primary data lines
MARKER_SIZE = 1.5           # marker size for scatter points
WIDTH_MM = 180              # 85 = single column, 180 = double column
HEIGHT_MM = 90              # flat canvas aspect ratio (per-panel rows adjust)
DPI_RASTER = 300            # ESSD raster resolution requirement
WONG_PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7",
                "#D55E00", "#F0E442", "#0072B2", "#000000"]
GRID_COLOR = "#E5E5E5"
GRID_WIDTH = 0.3
PANEL_LABEL_PT = 9.5        # (a)/(b) corner tags (slightly larger, still >=6)

# Derived canvas
WIDTH_IN = WIDTH_MM / 25.4
HEIGHT_IN = HEIGHT_MM / 25.4

#: Dataset -> Wong color binding (semantic, shared by every figure).
DS_COLORS = {
    "gefcom2014": "#56B4E9",   # sky blue
    "folsom": "#E69F00",       # orange
    "skippd": "#009E73",       # bluish green
    "sunlake": "#CC79A7",      # reddish purple
}
DS_LABELS = {"gefcom2014": "GEFCom2014", "folsom": "Folsom",
             "skippd": "SKIPP'D", "sunlake": "SunLake"}
DS_ORDER = ["gefcom2014", "folsom", "skippd", "sunlake"]

#: Academic LaTeX column notation (no snake_case on axes).
ACADEMIC_LABELS = {
    "pv_power_normalized": r"$\mathrm{PV}_{\mathrm{norm}}$",
    "ghi_actual": r"$\mathrm{GHI}_{\mathrm{act}}$",
    "dni_actual": r"$\mathrm{DNI}_{\mathrm{act}}$",
    "dhi_actual": r"$\mathrm{DHI}_{\mathrm{act}}$",
    "cos_zenith": r"$\cos	heta_z$",
    "solar_azimuth": r"$\gamma_s$",
    "ghi_clearsky": r"$\mathrm{GHI}_{\mathrm{cs}}$",
    "dni_clearsky": r"$\mathrm{DNI}_{\mathrm{cs}}$",
    "dhi_clearsky": r"$\mathrm{DHI}_{\mathrm{cs}}$",
    "clearness_index": r"$k_t$",
    "quality_flag": "Quality Flag",
    "split": "Data Split",
}

FIGURES_DIR = _REPO_ROOT / "figures" / "paper"
E8_SUMMARY = _REPO_ROOT / "results" / "plots" / "e8_summary.csv"

mpl.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": TEXT_SIZE_PT,
    "axes.labelsize": TEXT_SIZE_PT,
    "xtick.labelsize": TEXT_SIZE_PT - 1,
    "ytick.labelsize": TEXT_SIZE_PT - 1,
    "axes.linewidth": LINE_WIDTH_DATA,
    "axes.grid": True,
    "axes.edgecolor": "#4D4D4D",
    "axes.labelcolor": "#000000",
    "axes.axisbelow": True,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "grid.color": GRID_COLOR,
    "grid.linewidth": GRID_WIDTH,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT_FAMILY,
    "mathtext.it": FONT_FAMILY + ":italic",
    "mathtext.bf": FONT_FAMILY + ":bold",
    "mathtext.default": "rm",
})


def _save(fig: plt.Figure, stem: str) -> None:
    """Dual export: vector PDF + 300 dpi PNG at identical physical size."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=DPI_RASTER,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}.pdf + .png")


def fig2() -> None:
    s = pd.read_csv(E8_SUMMARY)
    models = ["LightGBM", "MLP", "LSTM", "iTransformer",
              "PatchTST-CI", "PatchTST-CD", "CrossViViT"]
    cells = [("folsom", 5, "5", "min"), ("folsom", 10, "10", "min"),
             ("folsom", 15, "15", "min"), ("skippd", 5, "5", "min"),
             ("skippd", 10, "10", "min"), ("skippd", 15, "15", "min"),
             ("sunlake", 1, "30", "min"), ("sunlake", 2, "60", "min"),
             ("sunlake", 3, "90", "min"), ("gefcom2014", 1, "1", "h"),
             ("gefcom2014", 2, "2", "h"), ("gefcom2014", 3, "3", "h")]
    # Horizon-only ticks (6.5 pt — back above the 6 pt floor); dataset names
    # are drawn ONCE per 3-column group below the ticks + a '// ' group
    # separator on the tick row (ESSD-style grouped axis).
    labels = [f"{ph} {unit}" for _, _, ph, unit in cells]

    fig, axes = plt.subplots(2, 1, figsize=(WIDTH_MM * mm, HEIGHT_MM * mm * 1.08),
                             sharex=True, gridspec_kw={"hspace": 0.50})
    vmax = 0.0
    mats = []
    for metric in ("Skill_MAE_pct", "Skill_RMSE_pct"):
        mat = np.full((len(models), len(cells)), np.nan)
        for j, (ds, hz, _, _unit) in enumerate(cells):
            d = s[(s.Dataset == ds) & (s.Metric == metric) & (s.Horizon == hz)]
            for i, m in enumerate(models):
                if len(d) and not pd.isna(d[m].iloc[0]):
                    mat[i, j] = d[m].iloc[0]
        mats.append(mat)
        vmax = max(vmax, float(np.nanmax(np.abs(mat))))
    shared_im = None
    for ax, mat, ylab, subtitle in zip(
            axes, mats,
            (r"Skill, MAE (%)", r"Skill, RMSE (%)"),
            (r"(a) Daytime Skill (MAE) %",
             r"(b) Daytime Skill (RMSE) %")):
        shared_im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                              aspect="auto")
        ax.set_ylabel(ylab)
        ax.set_title(subtitle, fontsize=TEXT_SIZE_PT, loc="left", pad=3)
        ax.grid(False)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    ax.text(j, i, "n/a", ha="center", va="center",
                            fontsize=TEXT_SIZE_PT - 2, color="0.45")
                else:
                    ax.text(j, i, f"{v:+.0f}", ha="center", va="center",
                            fontsize=TEXT_SIZE_PT - 1.5,
                            color="black" if abs(v) < vmax * 0.55 else "white")
        ax.set_yticks(range(len(models)), models)
    axes[1].set_xticks(range(len(labels)), labels,
                       fontsize=TEXT_SIZE_PT - 2)     # 6.5 pt horizon-only ticks
    axes[1].set_xlabel("")   # bottom caption row removed per user request
    # Grouped dataset axis: one name per 3-column block, BELOW the horizon
    # ticks (axes-fraction y < 0 so it can never climb onto panel (a));
    # order follows the cell layout: Folsom, SKIPP'D, SunLake, GEFCom2014.
    cell_ds_order = []
    for ds, _, _, _ in cells:
        if ds not in cell_ds_order:
            cell_ds_order.append(ds)
    for g, ds in enumerate(cell_ds_order):
        c0, c1 = 3 * g, 3 * g + 2
        axes[1].text((c0 + c1) / 2, -0.16, DS_LABELS[ds],
                     transform=axes[1].get_xaxis_transform(),
                     ha="center", va="top", fontsize=TEXT_SIZE_PT - 0.5,
                     color="black", fontweight="bold")
        if g > 0:  # separator between groups (on both heatmap panels)
            for ax in axes:
                ax.axvline(c0 - 0.5, color="white", linewidth=2.0)
    # Shared diverging colorbar on the right edge (both panels share the scale).
    cbar = fig.colorbar(shared_im, ax=list(axes), fraction=0.025, pad=0.015)
    cbar.set_label(r"Daytime Skill (%)", fontsize=TEXT_SIZE_PT - 1)
    cbar.ax.tick_params(labelsize=TEXT_SIZE_PT - 1.5)
    # panel letters are already inside the sub-titles above; keep the corner
    # tags off fig2 to avoid duplication
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    _save(fig, "fig10")


# ===========================================================================
# fig3 — E8: skill vs native horizon per dataset
# ===========================================================================

mm = 1 / 25.4


def main() -> int:
    fig2()
    return 0


if __name__ == "__main__":
    sys.exit(main())


run = main
