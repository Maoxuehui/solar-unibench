"""ESSD fig5a — optical-flow extrapolation sequence (4-panel strip).

(a) Context (seed) | (b) Extrapolated (K=1) | (c) Real future share ONE
inferno scale (vmin 0.0, vmax 0.85) with a SINGLE shared colorbar (right of
(c), physical unit = normalized radiance); (d) Reliability (uncertainty) uses
a distinct 'Blues' ramp so it cannot be confused with the irradiance scale.
Bold left-aligned panel titles, no ticks, equal aspect, wspace 0.25.

Run: python scripts/plot_essd_fig5a_sequence.py   (uses .closure_cache-like
.npy? No — the E4 gallery pair is tiny; re-evaluation takes seconds.)
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
HEIGHT_MM = 48
DPI_RASTER = 300
WONG_PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7",
                "#D55E00", "#F0E442", "#0072B2", "#000000"]
TITLE_PT = 9.5
VMIN, VMAX = 0.0, 0.85          # shared (a-c) inferno range

mpl.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": TEXT_SIZE_PT,
    "axes.labelsize": TEXT_SIZE_PT,
    "xtick.labelsize": TEXT_SIZE_PT,
    "ytick.labelsize": TEXT_SIZE_PT,
    "axes.linewidth": LINE_WIDTH_DATA,
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

OUT = _REPO_ROOT / "figures_essd" / "fig5a_optflow_sequence"


def _gallery() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(seed, pred, truth, uncertainty) of the K=1 gallery pair (IR_016)."""
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
    per_lead = metrics["per_lead"]
    k = 1 if 1 in per_lead else list(per_lead)[0]
    ch = e4._GALLERY_CHANNEL
    return (per_lead[k]["seed"][0, ch], per_lead[k]["pred"][0, ch],
            per_lead[k]["truth"][0, ch], per_lead[k]["unc"][0])


def main() -> int:
    seed, pred, truth, unc = _gallery()
    fig, axs = plt.subplots(1, 4, figsize=(WIDTH_MM * mm, HEIGHT_MM * mm))
    # Equal image boxes: fixed margins + equal per-panel widths; the shared
    # colorbar returns BETWEEN (c) and (d) — those two get a wide gap by
    # shifting (d) right via position edit after layout.
    plt.subplots_adjust(wspace=0.12, left=0.015, right=0.845,
                        top=0.82, bottom=0.02)
    # Widen the (c)-(d) gap: move (d) to the freed right zone.
    pos_d0 = axs[3].get_position()
    axs[3].set_position([pos_d0.x0 + 0.145, pos_d0.y0,
                         pos_d0.width, pos_d0.height])

    titles = (r"$\mathbf{(a)}$ Context (seed)",
              r"$\mathbf{(b)}$ Extrapolated ($K = 1$)",
              r"$\mathbf{(c)}$ Real future",
              r"$\mathbf{(d)}$ Reliability")
    imgs = (seed, pred, truth)
    for ax, im, t in zip(axs[:3], imgs, titles[:3]):
        h = ax.imshow(im, cmap="inferno", vmin=VMIN, vmax=VMAX,
                      aspect="equal")          # one shared scale object
        ax.set_title(t, loc="left", fontsize=TITLE_PT, pad=8)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
    # SINGLE shared colorbar back BETWEEN (c) and (d): the wide reserved gap
    # (~0.13 fig-width) holds bar + rotated label + ticks comfortably.
    pos_a = axs[0].get_position()
    pos_c = axs[2].get_position()
    pos_d = axs[3].get_position()
    gap_mid = (pos_c.x1 + pos_d.x0) / 2
    cb_ax = fig.add_axes([gap_mid - 0.006, pos_c.y0,
                          0.011, pos_c.height])
    cbar = fig.colorbar(h, cax=cb_ax)
    cbar.set_label("Normalized radiance", fontsize=TEXT_SIZE_PT,
                   rotation=90, labelpad=10)
    cbar.ax.tick_params(labelsize=TEXT_SIZE_PT - 1)

    ax_d = axs[3]
    hu = ax_d.imshow(unc, cmap="Blues", aspect="equal")   # distinct ramp
    ax_d.set_title(titles[3], loc="left", fontsize=TITLE_PT, pad=8)
    ax_d.set_xticks([]); ax_d.set_yticks([])
    for s in ax_d.spines.values():
        s.set_visible(True)
    # (d)'s own small bar, far right, same fixed-width rule; label shortened
    # to avoid repeating 'uncertainty' from the panel title.
    cbu_ax = fig.add_axes([pos_d.x1 + 0.006, pos_d.y0,
                           0.008, pos_d.height])
    cbar_u = fig.colorbar(hu, cax=cbu_ax)
    cbar_u.set_label("Reliability", fontsize=TEXT_SIZE_PT - 1,
                     rotation=90, labelpad=2)
    cbar_u.ax.tick_params(labelsize=TEXT_SIZE_PT - 1.5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=DPI_RASTER, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}.pdf (+ .png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
