"""ESSD submission figure set — Copernicus-formatted, standalone.

Regenerates the core benchmark figures at ESSD/Copernicus specifications into
``figures/``:

  fig1_crossover_frontier_waterfall  E9 — (a) hierarchical crossover margin,
                                     (b) governance waterfall
  fig2_benchmark_skill_heatmap       E8 — daytime Skill (MAE | RMSE) heatmaps
  fig3_benchmark_horizon_lines       E8 — skill vs native horizon per dataset
  fig4_data_defects                  E7 — (a) timestamp gaps, (b) missingness
  fig5_optflow_quality               E4 — (a) extrapolation sequence,
                                     (b) PSNR/SSIM decay

Design contract (user spec, 2026-08):
* ONE parameter block owns every tunable value — no magic numbers downstream.
* Single text size (8.5 pt) + Arial everywhere; >=6 pt at print scale.
* 180 mm double-column canvas (85 mm would be single-column).
* Wong/Okabe-Ito discrete palette; no red/green diverging.
* NO plot titles — panel letters + concise academic sub-labels only; the LaTeX
  caption owns the explanation.
* No raw snake_case code variables on axes — academic LaTeX notation.
* White canvas, light major grid only (#E5E5E5, 0.3), no minor grid.
* Dual export: vector PDF + 300 dpi PNG at identical physical size.

Run:  python scripts/plot_essd_figures.py
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

#: Dataset → Wong color binding (semantic, shared by every figure).
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
    "cos_zenith": r"$\cos\theta_z$",
    "solar_azimuth": r"$\gamma_s$",
    "ghi_clearsky": r"$\mathrm{GHI}_{\mathrm{cs}}$",
    "dni_clearsky": r"$\mathrm{DNI}_{\mathrm{cs}}$",
    "dhi_clearsky": r"$\mathrm{DHI}_{\mathrm{cs}}$",
    "clearness_index": r"$k_t$",
    "quality_flag": "Quality Flag",
    "split": "Data Split",
}

FIGURES_DIR = _REPO_ROOT / "figures"
RESULTS = _REPO_ROOT / "figures_full" / "results"
E8_SUMMARY = _REPO_ROOT / "figures" / "results" / "e8_summary.csv"

# -- apply the design system globally ---------------------------------------
mpl.rcParams.update({
    "font.family": FONT_FAMILY,
    # Mathtext rendered with the SAME Arial family so formulas
    # (\chi=\max(...)_{cleaned} etc.) match the axis text in weight and style.
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT_FAMILY,
    "mathtext.it": f"{FONT_FAMILY}:italic",
    "mathtext.bf": f"{FONT_FAMILY}:bold",
    "mathtext.default": "rm",
    "font.size": TEXT_SIZE_PT,
    "axes.labelsize": TEXT_SIZE_PT,
    "axes.titlesize": TEXT_SIZE_PT,
    "xtick.labelsize": TEXT_SIZE_PT,
    "ytick.labelsize": TEXT_SIZE_PT,
    "legend.fontsize": TEXT_SIZE_PT,
    "axes.linewidth": LINE_WIDTH_DATA,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.linewidth": GRID_WIDTH,
    "grid.linestyle": "-",
    "axes.axisbelow": True,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,          # editable text in the vector PDF
})
mm = 1 / 25.4


def _save(fig: plt.Figure, stem: str) -> None:
    """Dual export: vector PDF + 300 dpi PNG at identical physical size.

    bbox_inches='tight' trims surplus canvas margins (e.g. the space rotated
    x-tick labels reserve below the axes) so the exported artwork is compact.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=DPI_RASTER,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}.pdf + .png")


def _panel_tag(fig: plt.Figure, x: float, y: float, letter: str) -> None:
    fig.text(x, y, f"({letter})", fontsize=PANEL_LABEL_PT, fontweight="bold",
             ha="left", va="top")


def _no_minor(ax: plt.Axes) -> None:
    ax.grid(which="minor", visible=False)
    ax.grid(which="major", color=GRID_COLOR, linewidth=GRID_WIDTH)


# ===========================================================================
# fig1 — E9: (a) crossover frontier + (b) governance waterfall
# ===========================================================================
def fig1() -> None:
    x = pd.read_csv(RESULTS / "results_e9_cross_dataset.csv")
    margins_mean, margins_best = [], []
    for ds in DS_ORDER:
        d = x[x.dataset == ds]
        micro = max(d[d.model == m].skill_cleaned.iloc[0] for m in ("LightGBM", "MLP"))
        deep = [d[d.model == m].skill_raw.iloc[0]
                for m in ("LSTM", "iTransformer", "PatchTST")]
        margins_mean.append(micro - float(np.mean(deep)))
        margins_best.append(micro - max(deep))

    wf = pd.read_csv(RESULTS / "results_e9_waterfall.csv")
    mae = wf["daytime_mae"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(WIDTH_MM * mm, HEIGHT_MM * mm * 0.78),
        gridspec_kw={"wspace": 0.42},   # breathing room between (a) and (b)
    )
    xpos = np.arange(len(DS_ORDER))
    b1 = ax1.bar(xpos - 0.19, margins_mean, width=0.35, facecolor="#0072B2",
                 edgecolor="none", label=r"vs. deep-model mean")
    b2 = ax1.bar(xpos + 0.19, margins_best, width=0.35, facecolor="#E69F00",
                 edgecolor="#E69F00", linewidth=0.4, hatch="//",
                 label=r"vs. deep-model best")
    ax1.axhline(0, color="black", linewidth=0.8)
    for bars, vals, col in ((b1, margins_mean, "#0072B2"), (b2, margins_best, "#E69F00")):
        for b, v in zip(bars, vals):
            va = "bottom" if v >= 0 else "top"
            off = 0.006 * (1 if v >= 0 else -1)
            ax1.text(b.get_x() + b.get_width() / 2, v + off, f"{v:+.3f}",
                     ha="center", va=va, fontsize=TEXT_SIZE_PT - 2.5, color=col)
    lo = min(margins_mean + margins_best); hi = max(margins_mean + margins_best)
    ax1.set_ylim(lo - 0.25 * (hi - lo), hi + 0.25 * (hi - lo))
    ax1.set_xticks(xpos, [DS_LABELS[d] for d in DS_ORDER],
                   fontsize=TEXT_SIZE_PT - 2)
    ax1.set_ylabel(r"Crossover margin $\chi$ (skill difference)")
    ax1.legend(frameon=False, loc="upper left")
    # The chi definition formula, lower-left (kept from the gemini figure).
    ax1.text(0.02, 0.03,
             r"$\chi=\max(\mathrm{LGBM},\mathrm{MLP})_{\mathrm{cleaned}}-\mathrm{deep}_{\mathrm{raw}}$",
             transform=ax1.transAxes, fontsize=TEXT_SIZE_PT - 1.5, color="0.25")
    _no_minor(ax1)

    steps = ["Raw", "+Night", "+Clamp", "+UTC", "+Clear-sky"]
    ramp = ["#08306b", "#2171b5", "#4292c6", "#6baed6"]
    ax2.bar(np.arange(len(mae)), mae, width=0.62,
            facecolor=ramp + ["#D55E00"], edgecolor="none")
    ax2.axhline(0, color="black", linewidth=0.8)
    span = mae.max() - mae.min()
    for i, v in enumerate(mae):
        ax2.text(i, v + 0.012 * span, f"{v:.4f}", ha="center", va="bottom",
                 fontsize=TEXT_SIZE_PT - 1.5, color="black")
    # Notes float at 22%-span — fully clear of the value labels below them.
    for i in range(4):
        ax2.text(i, mae[i] + 0.22 * span, "no-op", ha="center",
                 va="bottom", fontsize=TEXT_SIZE_PT - 1, color="0.30")
    ax2.text(4, mae[4] + 0.22 * span, r"$\downarrow$ $-13.1\%$",
             ha="center", va="bottom", fontsize=TEXT_SIZE_PT - 1,
             color="#D55E00", fontweight="bold")
    ax2.set_xticks(np.arange(len(mae)), steps, fontsize=TEXT_SIZE_PT - 2)
    ax2.set_ylabel(r"Daytime forecast error (MAE)")
    ax2.set_ylim(0.038, mae.max() + 0.70 * span)
    _no_minor(ax2)

    _panel_tag(fig, 0.005, 0.985, "a")
    _panel_tag(fig, 0.545, 0.985, "b")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "fig1_crossover_frontier_waterfall")


# ===========================================================================
# fig2 — E8: skill heatmaps (MAE | RMSE)
# ===========================================================================
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
    _save(fig, "fig2_benchmark_skill_heatmap")


# ===========================================================================
# fig3 — E8: skill vs native horizon per dataset
# ===========================================================================
def fig3() -> None:
    s = pd.read_csv(E8_SUMMARY)
    models = ["LightGBM", "MLP", "LSTM", "iTransformer",
              "PatchTST-CI", "PatchTST-CD", "CrossViViT"]
    unit = {"folsom": "min", "skippd": "min", "sunlake": "min", "gefcom2014": "h"}
    phys = {"folsom": [5, 10, 15], "skippd": [5, 10, 15],
            "sunlake": [30, 60, 90], "gefcom2014": [1, 2, 3]}
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH_MM * mm, HEIGHT_MM * mm * 0.62),
                             sharey=False, gridspec_kw={"wspace": 0.30})
    for ax, ds in zip(axes, DS_ORDER):
        d = s[(s.Dataset == ds) & (s.Metric == "Skill_MAE_pct")].sort_values("Horizon")
        x = np.arange(3)
        for k, m in enumerate(models):
            ys = [d[d.Horizon == h][m].iloc[0] for h in sorted(d.Horizon.unique())]
            if all(pd.isna(v) for v in ys):
                continue
            ax.plot(x, ys, marker="o", markersize=MARKER_SIZE + 1.5,
                    linewidth=LINE_WIDTH_DATA + 0.4,
                    color=WONG_PALETTE[k % len(WONG_PALETTE)], label=m)
        ax.axhline(0, color="0.3", linewidth=0.6, linestyle="--")
        ax.set_xticks(x, [f"{p} {unit[ds]}" for p in phys[ds]])
        ax.set_title(DS_LABELS[ds], fontsize=TEXT_SIZE_PT)
        _no_minor(ax)
    axes[0].set_ylabel(r"Skill, MAE (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    for ax_ in axes[1:]:
        h_, l_ = ax_.get_legend_handles_labels()
        handles += h_; labels += l_
    uniq = dict(zip(labels, handles))
    fig.legend(uniq.values(), uniq.keys(), loc="lower center", ncol=7,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, "fig3_benchmark_horizon_lines")


# ===========================================================================
# fig4 — E7: (a) timestamp gaps + (b) column-wise missingness
# ===========================================================================
def fig4() -> None:
    from solarunibench.experiments.e7_defects import CANONICAL_COLUMNS, _irregularity
    from solarunibench.loader import SolarUniBenchLoader

    summary = pd.read_csv(RESULTS / "defect_summary.csv")
    stats = summary[summary.section == "irregularity"].set_index("dataset")
    miss = summary[summary.section == "missingness"]

    loader = SolarUniBenchLoader(str(_REPO_ROOT / "data" / "standardized"))
    diffs = {}
    for ds in DS_ORDER:
        _, diffs[ds] = _irregularity(loader.load_tabular(ds))

    fig = plt.figure(figsize=(WIDTH_MM * mm, HEIGHT_MM * mm * 0.70))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.85])
    ax_m = fig.add_subplot(gs[0, 0])                      # (a) matrix, LEFT
    sub_h = gs[0, 1].subgridspec(len(DS_ORDER), 1, hspace=0.28)
    axs_h = [fig.add_subplot(s) for s in sub_h]           # (b) per-dataset

    # ---- (a) Missing values: presence matrix with merged check runs -------
    cols = list(CANONICAL_COLUMNS)
    ax_m.imshow(np.zeros((len(DS_ORDER), len(cols))), cmap="Greys", vmin=0,
                vmax=1, aspect="auto")
    GREEN, RED = "#009E73", "#D55E00"      # house Wong green / vermillion
    GLYPH = dict(fontfamily="DejaVu Sans", ha="center", va="center",
                 fontsize=TEXT_SIZE_PT + 2, fontweight="bold")
    EDGE = "#D5D5D1"
    # Fixed check segments for the two long irradiance rows (user layout):
    # cols 2/3/4 each their own check, 5-9 merged, 10 alone, 11-12 merged.
    SEGMENTS_2_12 = [(1, 1), (2, 2), (3, 3), (4, 8), (9, 9), (10, 11)]
    present_rows = []
    for i, ds in enumerate(DS_ORDER):
        row = miss[miss.dataset == ds].set_index("column")["pct_nan"]
        present = [float(row.get(c, 100.0)) < 99.5 for c in cols]
        present_rows.append(present)
        if ds in ("folsom", "sunlake") and all(present[1:12]):
            if not present[0]:                  # keep the col-1 red cross
                ax_m.text(0, i, "✗", color=RED, **GLYPH)
            for a_i, b_i in SEGMENTS_2_12:      # one centered check per segment
                ax_m.text((a_i + b_i) / 2.0, i, "✓", color=GREEN, **GLYPH)
            continue
        j = 0
        while j < len(cols):
            if present[j]:                 # merge horizontal check runs
                k = j
                while k + 1 < len(cols) and present[k + 1]:
                    k += 1
                ax_m.text((j + k) / 2.0, i, "✓", color=GREEN, **GLYPH)
                j = k + 1
            else:                          # absent -> individual red cross
                ax_m.text(j, i, "✗", color=RED, **GLYPH)
                j += 1
    # cell grid: horizontal lines everywhere; vertical lines ONLY where the
    # presence state changes (no separators inside merged check runs)
    for yb in np.arange(-0.5, len(DS_ORDER) + 0.5, 1):
        ax_m.axhline(yb, color=EDGE, linewidth=0.5, zorder=1)
    for xb in np.arange(-0.5, len(cols) + 0.5, 1):
        jb = int(xb + 0.5)
        draw = xb in (-0.5, len(cols) - 0.5)
        if not draw and 0 < jb < len(cols):
            # boundary between jb-1 and jb: omit inside a present-present run
            any_row_splits = any(pr[jb - 1] != pr[jb] or not pr[jb - 1]
                                 or not pr[jb] for pr in present_rows)
            draw = any_row_splits
        if draw:
            ax_m.axvline(xb, color=EDGE, linewidth=0.5, zorder=1)
    ax_m.set_xticks(range(len(cols)), [ACADEMIC_LABELS.get(c, c) for c in cols],
                    rotation=30, ha="right", fontsize=TEXT_SIZE_PT - 2,
                    rotation_mode="anchor")
    ax_m.set_yticks(range(len(DS_ORDER)), [DS_LABELS[d] for d in DS_ORDER],
                    fontsize=TEXT_SIZE_PT - 2)
    ax_m.grid(False)
    ax_m.set_xlim(-0.5, len(cols) - 0.5)
    ax_m.set_ylim(len(DS_ORDER) - 0.5, -0.5)
    ax_m.set_title("(a) Missing values", fontsize=TEXT_SIZE_PT, loc="left",
                   pad=3)

    # ---- (b) per-dataset gap histograms -----------------------------------
    edges = [0, 2, 5, 15, 60, 360, 1440, np.inf]
    labels = ["≤2", "2–5", "5–15", "15–60", "1–6 h", "6–24 h", ">24 h"]
    x = np.arange(len(labels))
    n_ds = len(DS_ORDER)
    all_cnt = {}
    for ds in DS_ORDER:
        d = pd.Series(diffs[ds])
        modal = d.value_counts().idxmax()
        all_cnt[ds] = np.histogram(d[d > modal], bins=edges)[0]
    g_top = int(np.ceil(np.log10(max(int(c.max()) for c in all_cnt.values()
                                     if c.size))))
    for k, (ax_h, ds) in enumerate(zip(axs_h, DS_ORDER)):
        cnt = all_cnt[ds]
        ax_h.bar(x, np.maximum(cnt, 0), width=0.55, color="#0072B2",
                 zorder=3)
        for xb, c in zip(x, cnt):            # count INSIDE the bar, near top
            if c > 0:
                ax_h.text(xb, c * 0.72, f"{int(c)}", ha="center",
                          va="top", fontsize=TEXT_SIZE_PT - 2.5,
                          color="white")
        ax_h.set_yscale("log")
        ax_h.set_ylim(0.5, 10 ** g_top)          # shared full 10^n ladder
        ticks = [10 ** e for e in range(0, g_top + 1)]
        ax_h.set_yticks(ticks, [f"$10^{{{e}}}$" for e in range(0, g_top + 1)],
                        fontsize=TEXT_SIZE_PT - 2)
        # dataset name at the frame's top-left corner
        ax_h.text(0.015, 0.93, DS_LABELS[ds], transform=ax_h.transAxes,
                  fontsize=TEXT_SIZE_PT - 2, ha="left", va="top",
                  color="#0B0B0B")
        if k == 1:                             # one shared y-axis label
            ax_h.set_ylabel("Count", rotation=90, va="center",
                            fontsize=TEXT_SIZE_PT - 1, labelpad=2)
        if k < n_ds - 1:
            ax_h.set_xticks(x)
            ax_h.tick_params(labelbottom=False)
        else:
            ax_h.set_xticks(x, labels, fontsize=TEXT_SIZE_PT - 2)
            ax_h.set_xlabel("Gap (min)")
        _no_minor(ax_h)
    axs_h[0].set_title("(b) Irregular intervals", fontsize=TEXT_SIZE_PT,
                       loc="left", pad=3)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.subplots_adjust(left=0.075, wspace=0.25)   # (a)-(b) breathing gap
    _save(fig, "fig4_data_defects")


# ===========================================================================
# fig5 — E4: (a) extrapolation sequence + (b) PSNR/SSIM decay
# ===========================================================================
def fig5() -> None:
    from solarunibench.experiments import e4_optflow as e4
    from solarunibench.loader import SolarUniBenchLoader

    loader = SolarUniBenchLoader(str(_REPO_ROOT / "data" / "standardized"))
    ctx = loader.get_tensor_cube(e4._DATASET, "satellite_context")
    fut = loader.get_tensor_cube(e4._DATASET, "satellite_future")
    unc = loader.get_tensor_cube(e4._DATASET, "satellite_uncertainty")
    m = min(ctx.shape[0], e4._MAX_PAIRS_DEV + 1)
    metrics = e4._evaluate(
        np.asarray(ctx[:m], dtype=np.float32),
        np.asarray(fut[:, : m - 1], dtype=np.float32),
        np.asarray(unc[:, : m - 1], dtype=np.float32),
        max_pairs=e4._MAX_PAIRS_DEV, nominal_levels=e4._NOMINAL_LEVELS,
    )
    per_lead, decay = metrics["per_lead"], metrics["decay"]

    # 180 x 130 mm: top row (a-d) image sequence, bottom row (e-f) decay.
    fig = plt.figure(figsize=(WIDTH_MM * mm, 130 * mm))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.72], hspace=0.50,
                          wspace=0.26)
    sub_img = gs[0].subgridspec(1, 4, wspace=0.06)
    sub_line = gs[1].subgridspec(1, 2, wspace=0.30)

    k = 1 if 1 in per_lead else list(per_lead)[0]
    ch = e4._GALLERY_CHANNEL
    seed = per_lead[k]["seed"][0, ch]
    pred = per_lead[k]["pred"][0, ch]
    truth = per_lead[k]["truth"][0, ch]
    unc = per_lead[k]["unc"][0]

    # (a-c) share ONE inferno scale + a single colorbar; (d) keeps viridis.
    img_axes = []
    for s_, im, t in zip(
            sub_img, (seed, pred, truth),
            ("(a) Context Seed", f"(b) Extrapolated ($K$={k})", "(c) Ground Truth")):
        ax = fig.add_subplot(s_)
        me = ax.imshow(im, cmap="inferno", vmin=0.0, vmax=0.9)
        ax.set_title(t, fontsize=TEXT_SIZE_PT)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        img_axes.append(ax)
    cbar = fig.colorbar(me, ax=img_axes, fraction=0.025, pad=0.015)
    cbar.ax.tick_params(labelsize=TEXT_SIZE_PT - 1)
    ax = fig.add_subplot(sub_img[3])
    me_u = ax.imshow(unc, cmap="viridis")
    ax.set_title("(d) Alignment Uncertainty", fontsize=TEXT_SIZE_PT)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    cbar_u = fig.colorbar(me_u, ax=[ax], fraction=0.08, pad=0.04)
    cbar_u.ax.tick_params(labelsize=TEXT_SIZE_PT - 1.5)

    psnr = np.asarray(decay["psnr_per_lead"], dtype=float)
    ssim = np.asarray(decay["ssim_per_lead"], dtype=float)
    kk = np.arange(1, psnr.size + 1)
    ax_l = fig.add_subplot(sub_line[0])
    ax_l.plot(kk, psnr, color=WONG_PALETTE[6], marker="o",
              markersize=MARKER_SIZE + 2, linewidth=LINE_WIDTH_DATA + 0.4)
    ax_l.set_xlabel(r"Forecast horizon $K$ (steps)")
    ax_l.set_ylabel(r"PSNR (dB)")
    ax_l.set_title("(e) PSNR Decay", fontsize=TEXT_SIZE_PT, loc="left")
    ax_l.spines["top"].set_visible(False)
    ax_l.spines["right"].set_visible(False)
    _no_minor(ax_l)
    ax_r = fig.add_subplot(sub_line[1])
    ax_r.plot(kk, ssim, color=WONG_PALETTE[0], marker="s",
              markersize=MARKER_SIZE + 2, linewidth=LINE_WIDTH_DATA + 0.4)
    ax_r.set_xlabel(r"Forecast horizon $K$ (steps)")
    ax_r.set_ylabel(r"SSIM")
    ax_r.set_title("(f) SSIM Decay", fontsize=TEXT_SIZE_PT, loc="left")
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(False)
    _no_minor(ax_r)

    fig.tight_layout(rect=(0, 0, 1, 0.99))
    _save(fig, "fig5_optflow_quality")


def main() -> int:
    print(f"ESSD figure set -> {FIGURES_DIR}  ({WIDTH_MM}x~{HEIGHT_MM} mm, "
          f"{FONT_FAMILY} {TEXT_SIZE_PT} pt, {DPI_RASTER} dpi)")
    for fn in (fig1, fig2, fig3, fig4, fig5):
        fn()
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
