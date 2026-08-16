"""E7 combined data-defect figure — (a) timestamp irregularity + (b) missingness matrix.

One publication figure consolidating the two E7 diagnostics side by side (the
ESSD "data-defect evidence chain" panel):

* **(a) Timestamp irregularity** — per-dataset inter-step gap distribution
  (log-scale strip plot; the raw per-step diffs are NOT in defect_summary.csv,
  so the distributions are re-derived from the standardized packages) plus the
  headline stats (cadence / gaps / dups) as an inset table.
* **(b) Missingness matrix** — % NaN per canonical column × dataset, straight
  from ``defect_summary.csv``.

Color system — dataset-semantic binding, colorblind-safe (user spec 2026-08):

| dataset    | Okabe-Ito    | hex       | marker |
|------------|--------------|-----------|--------|
| gefcom2014 | sky blue     | #56B4E9   | o      |
| folsom     | orange       | #E69F00   | s      |
| skippd     | bluish green | #009E73   | ^      |
| sunlake    | reddish purple | #CC79A7 | d      |

Panel (b) renders 0% cells as near-white grey and highlights 100%-missing
cells in the ROW's dataset color (semantic cross-reference with panel (a));
only 100 is annotated (zeros are suppressed).

Usage:
    python scripts/plot_e7_defects_combined.py
        [--csv figures_full/results/defect_summary.csv]
        [--out figures_full/fig_defects_combined.png]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solarunibench.experiments.e7_defects import (  # noqa: E402
    CANONICAL_COLUMNS,
    _irregularity,
)
from solarunibench.viz.style import set_style  # noqa: E402

#: Dataset-semantic Okabe-Ito binding (shared by both panels).
_DS_STYLE: dict[str, dict] = {
    "gefcom2014": {"color": "#56B4E9", "marker": "o", "label": "GEFCom2014"},
    "folsom":     {"color": "#E69F00", "marker": "s", "label": "Folsom"},
    "skippd":     {"color": "#009E73", "marker": "^", "label": "SKIPPD"},
    "sunlake":    {"color": "#CC79A7", "marker": "D", "label": "SunLake"},
}
_DS_ORDER = ["gefcom2014", "folsom", "skippd", "sunlake"]

_DATA_ROOT = "D:/monica/workspace/solar-unibench/data/standardized"


def _gap_distributions() -> dict[str, np.ndarray]:
    """Re-derive per-dataset positive inter-step gap arrays (minutes)."""
    from solarunibench.loader import SolarUniBenchLoader

    loader = SolarUniBenchLoader(_DATA_ROOT)
    out: dict[str, np.ndarray] = {}
    for ds in _DS_ORDER:
        df = loader.load_tabular(ds)
        _, diffs_min = _irregularity(df)
        out[ds] = diffs_min
    return out


def _panel_a(ax: plt.Axes, diffs: dict[str, np.ndarray], stats: pd.DataFrame) -> None:
    """Strip plot of gap distributions + stats annotation block."""
    rng = np.random.default_rng(0)
    for i, ds in enumerate(_DS_ORDER):
        st = _DS_STYLE[ds]
        arr = diffs.get(ds, np.array([]))
        arr = arr[arr > 0]
        # Subsample to <=4000 points per dataset (rendering speed; the bulk
        # sits at the modal cadence anyway).
        if arr.size > 4000:
            idx = rng.choice(arr.size, 4000, replace=False)
            arr = arr[idx]
        y = np.full(arr.size, i, dtype=float) + rng.uniform(-0.16, 0.16, arr.size)
        ax.scatter(arr, y, s=9, marker=st["marker"], c=st["color"],
                   alpha=0.4, edgecolors=st["color"], linewidths=0.4)
        # Modal-cadence reference line per dataset.
        cad = stats.loc[ds, "modal_cadence_seconds"] / 60.0
        if np.isfinite(cad):
            ax.axvline(cad, color=st["color"], lw=1.0, ls="--", alpha=0.8)
    ax.set_yticks(range(len(_DS_ORDER)),
                  [_DS_STYLE[ds]["label"] for ds in _DS_ORDER])
    for tick, ds in zip(ax.get_yticklabels(), _DS_ORDER):
        tick.set_color(_DS_STYLE[ds]["color"])
        tick.set_fontweight("bold")
    ax.set_xscale("log")
    ax.set_xlabel("Inter-step gap (minutes, log scale)")
    ax.set_ylim(-0.6, len(_DS_ORDER) - 0.4)
    ax.grid(True, axis="x", color="#E0E0E0", linestyle="--", linewidth=0.6)
    ax.set_title("(a) Timestamp irregularity", fontsize=12, loc="left")
    # Stats block: top-right inside the axes (away from the dense low-end).
    lines = ["cad / gaps / dups"]
    for ds in _DS_ORDER:
        r = stats.loc[ds]
        cad = r["modal_cadence_seconds"] / 60.0
        cad_s = f"{cad:g}m" if cad >= 1 else f"{cad*60:g}s"
        lines.append(
            f"{_DS_STYLE[ds]['label']}: {cad_s} / {int(r['n_gaps'])} / {int(r['n_duplicates'])}"
        )
    ax.text(0.985, 0.03, "\n".join(lines), transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#BBBBBB", alpha=0.9))


def _panel_b(ax: plt.Axes, miss: pd.DataFrame) -> None:
    """Missingness matrix: 0% -> near-white; 100% -> row's dataset color."""
    cols = list(CANONICAL_COLUMNS)
    mat = np.zeros((len(_DS_ORDER), len(cols)))
    for i, ds in enumerate(_DS_ORDER):
        row = miss[miss.dataset == ds].set_index("column")["pct_nan"]
        for j, c in enumerate(cols):
            mat[i, j] = float(row.get(c, 100.0))
    # Background: all-present cells in near-white; draw a base grey matrix.
    ax.imshow(np.zeros_like(mat), cmap="Greys", vmin=0, vmax=1, aspect="auto")
    for i, ds in enumerate(_DS_ORDER):
        color = _DS_STYLE[ds]["color"]
        for j, c in enumerate(cols):
            v = mat[i, j]
            if v >= 99.5:  # 100%-missing cell -> dataset color + bold "100"
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           facecolor=color, edgecolor="white", lw=1.2))
                ax.text(j, i, "100", ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color="white")
            elif v > 0.5:  # partial NaN (rare): light hatch of the same color
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           facecolor=color, alpha=0.35,
                                           edgecolor="white", lw=1.2))
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=7, color="black")
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(_DS_ORDER)),
                  [_DS_STYLE[ds]["label"] for ds in _DS_ORDER], fontsize=9)
    ax.set_title("(b) Missingness (Honest-NaN modalities)", fontsize=12, loc="left")
    # X limits/y limits fixed to the matrix extent.
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(len(_DS_ORDER) - 0.5, -0.5)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="figures_full/results/defect_summary.csv")
    p.add_argument("--out", default="figures_full/fig_defects_combined.png")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    set_style("light")
    summary = pd.read_csv(args.csv)
    stats = summary[summary.section == "irregularity"].set_index("dataset")
    miss = summary[summary.section == "missingness"]

    diffs = _gap_distributions()

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5.5),
        gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.22},
    )
    _panel_a(axes[0], diffs, stats)
    _panel_b(axes[1], miss)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    # PDF companion for ESSD submission.
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} (+ .pdf)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
