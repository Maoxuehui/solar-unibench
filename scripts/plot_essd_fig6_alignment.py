"""ESSD fig6 — E6 cross-modal alignment lag (Folsom | SKIPP'D joint panel).

Two-panel histogram of the image<->tabular timestamp lag:
* **(a) Folsom** — NN matching dominates (second-level camera vs hourly
  tabular); median NN lag ~1 s inside the 30 s tolerance.
* **(b) SKIPP'D** — exact timestamp matches dominate (native alignment);
  the ΔSkill subtitle honestly shows the negative control.

Design contract: one parameter block (Arial 8.5 pt, 180x75 mm, Wong palette,
vector PDF + 300 dpi PNG), no debug strings in headers, clean academic titles
+ ΔSkill subtitles, x range forced to [-5, 45] s.

Run: python scripts/plot_essd_fig6_alignment.py
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
HEIGHT_MM = 75
DPI_RASTER = 300
WONG_PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7",
                "#D55E00", "#F0E442", "#0072B2", "#000000"]
GRID_COLOR = "#E5E5E5"
GRID_WIDTH = 0.3
TITLE_PT = 10          # panel titles (dataset names)
SUBTITLE_PT = 8.5      # ΔSkill subtitles
XMIN, XMAX = -5.0, 45.0

COL_EXACT = "#CC79A7"   # Wong reddish purple — exact matches
COL_NN = "#56B4E9"      # Wong sky blue — NN matching
COL_MEDIAN = "#D55E00"  # Wong vermillion — median dashed line

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

OUT = _REPO_ROOT / "figures_essd" / "fig6_alignment_lag"

#: Panel spec: dataset -> (title, ΔSkill subtitle text).
_PANELS = [
    ("folsom", "Folsom", r"$\Delta\mathrm{Skill} = +0.186$"),
    ("skippd", "SKIPP'D", r"$\Delta\mathrm{Skill} = -0.247$"),
]

_TOLERANCE_S = 30.0


def _lag_arrays(dataset: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Re-derive exact/NN lag arrays (seconds) for one dataset via E6's path."""
    from solarunibench.benchmark.crossvivit_image import _resolve_sky_cube
    from solarunibench.experiments import e6_alignment as e6
    from solarunibench.loader import SolarUniBenchLoader

    loader = SolarUniBenchLoader(str(_REPO_ROOT / "data" / "standardized"))
    cube, img_ts = _resolve_sky_cube(loader, dataset,
                                     str(_REPO_ROOT / "data" / "standardized"))
    tab_ts = loader.load_tabular(dataset)["timestamp_utc"].to_numpy()
    diag = e6._lag_diagnostic(img_ts, tab_ts, tolerance=f"{int(_TOLERANCE_S)}s")
    if diag is None:
        raise RuntimeError(f"no lag diagnostic for {dataset}")
    # Recompute the arrays from the same recipe (diag carries only summaries):
    img_times = pd.DatetimeIndex(e6._naive_utc(np.asarray(img_ts)))
    tab_times = pd.DatetimeIndex(sorted(set(
        t for t in e6._naive_utc(np.asarray(tab_ts)) if not pd.isna(t))))
    pos = tab_times.get_indexer(img_times, method="nearest")
    lags = np.abs((img_times.values - tab_times.values[pos])
                  .astype("timedelta64[s]").astype(float))
    exact = lags[lags == 0.0]
    nn = lags[lags <= _TOLERANCE_S]
    med = float(np.median(nn)) if nn.size else 0.0
    return exact, nn, med


def main() -> int:
    fig, axes = plt.subplots(
        1, 2, figsize=(WIDTH_MM * mm, HEIGHT_MM * mm),
        gridspec_kw={"wspace": 0.22},
    )
    for ax, (ds, title, sub) in zip(axes, _PANELS):
        exact, nn, med = _lag_arrays(ds)
        bins = np.linspace(0, _TOLERANCE_S, 31)
        if nn.size:
            ax.hist(nn, bins=bins, color=COL_NN, alpha=0.8,
                    edgecolor="black", linewidth=0.4,
                    label=r"NN Matching ($\leq 30\,\mathrm{s}$)")
        if exact.size:
            ax.hist(exact, bins=bins, color=COL_EXACT, alpha=0.8,
                    edgecolor="black", linewidth=0.4, label="Exact Match")
        ax.axvline(med if np.isfinite(med) else 0.0, color=COL_MEDIAN,
                   linestyle="--", linewidth=1.3, label="Median NN Lag")
        ax.set_xlim(XMIN, XMAX)
        ax.set_xlabel(r"Time Alignment Lag $|t_{\mathrm{img}} - t_{\mathrm{tab}}|$ (s)")
        # Clean header: bold dataset name + ΔSkill subtitle, axes-fraction top.
        ax.set_title(title, fontsize=TITLE_PT, fontweight="bold", pad=14)
        ax.text(0.5, 1.02, sub, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=SUBTITLE_PT)
        leg = ax.legend(frameon=False, loc="upper right")
        leg.set_zorder(5)
    axes[0].set_ylabel("Frame Count")

    fig.text(0.005, 0.985, "(a)", fontsize=9.5, fontweight="bold",
             ha="left", va="top")
    fig.text(0.515, 0.985, "(b)", fontsize=9.5, fontweight="bold",
             ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=DPI_RASTER)
    plt.close(fig)
    print(f"wrote {OUT}.pdf (+ .png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
