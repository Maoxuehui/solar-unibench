"""E8 unified-benchmark visualizations (physics-time-aligned grid).

Reads the canonical results file ``figures/results/e8_summary.csv`` (written by
``scripts/run_e8_unified.py --full``; CrossViViT rows = v2 architecture) and
renders publication figures through the shared viz design system:

1. ``fig_e8_skill_heatmap.png`` — Daytime Skill% heatmap, rows = model, cols =
   dataset × physical horizon, diverging colormap centered at 0 (Smart
   Persistence parity). One panel per metric (MAE skill, RMSE skill).
2. ``fig_e8_horizon_lines.png`` — Skill_MAE% vs horizon per model, one panel
   per dataset (the "when do models beat persistence" crossover view).

Values are read from the CSV only — no recomputation, so the figures can never
drift from the published table.

Usage:
    python scripts/plot_e8_summary.py [--csv figures/results/e8_summary.csv]
                                      [--out-dir figures/results]
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

from solarunibench.viz.style import MODEL_COLORS, PALETTE, set_style  # noqa: E402

#: Column order for the displayed non-reference models. DLinear is EXCLUDED
#: from figures + doc tables (user decision 2026-08: its −590%..−381% MAE
#: skill on Folsom compresses the shared color scale and the crossover lines);
#: its results remain in e8_summary.csv for completeness.
_MODELS = [
    "LightGBM", "MLP", "LSTM",
    "iTransformer", "PatchTST-CI", "PatchTST-CD", "CrossViViT",
]

#: Per-model color: prefer the shared design-system mapping; the newer
#: variants (DLinear/iTransformer/PatchTST-CD) fall back to a fixed assignment
#: so all 8 lines are distinguishable and stable across runs. The base palette
#: has 6 Wong/Okabe-Ito colors, so the fallbacks add linestyle separation.
_FALLBACK = {
    "iTransformer": "#555555",
    "PatchTST-CI": MODEL_COLORS["PatchTST"],
    "PatchTST-CD": "#66CCEE",
}


def _color(model: str) -> str:
    return MODEL_COLORS.get(model) or _FALLBACK.get(model) or "0.4"

#: Dataset display order. Each entry: (dataset key, display, CSV step horizons,
#: physical-horizon x-labels, unit). The CSV stores STEP counts (sunlake 1/2/3
#: = 30/60/90 min; gefcom 1/2/3 = 1/2/3 h) — never mix the two.
_DATASETS = [
    ("folsom", "Folsom (1-min)", [5, 10, 15], ["5", "10", "15"], "min"),
    ("skippd", "SKIPPD (1-min)", [5, 10, 15], ["5", "10", "15"], "min"),
    ("sunlake", "SunLake (30-min)", [1, 2, 3], ["30", "60", "90"], "min"),
    ("gefcom2014", "GEFCom2014 (1-h)", [1, 2, 3], ["1", "2", "3"], "h"),
]

_SKILL_METRICS = [("Skill_MAE_pct", "Daytime Skill (MAE) %"),
                  ("Skill_RMSE_pct", "Daytime Skill (RMSE) %")]


def _load(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df["Horizon"] = df["Horizon"].astype(int)
    return df


def plot_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    """Two-panel diverging heatmap (MAE skill | RMSE skill), model × cell."""
    set_style()
    fig, axes = plt.subplots(
        2, 1, figsize=(11.5, 7.6), sharex=True,
        gridspec_kw={"hspace": 0.30},
    )
    cols, labels = [], []
    for ds, disp, csv_hz, phys_hz, unit in _DATASETS:
        for hz, phz in zip(csv_hz, phys_hz):
            cols.append((ds, int(hz)))
            labels.append(f"{disp.split(' (')[0]}\n{phz} {unit}")
    vmax_abs = 0.0
    mats = []
    for metric, _ in _SKILL_METRICS:
        mat = np.full((len(_MODELS), len(cols)), np.nan)
        for j, (ds, hz) in enumerate(cols):
            d = df[(df.Dataset == ds) & (df.Metric == metric) & (df.Horizon == hz)]
            for i, m in enumerate(_MODELS):
                if len(d) and not pd.isna(d[m].iloc[0]):
                    mat[i, j] = d[m].iloc[0]
        mats.append(mat)
        vmax_abs = max(vmax_abs, np.nanmax(np.abs(mat)))
    for ax, (metric, title), mat in zip(axes, _SKILL_METRICS, mats):
        im = ax.imshow(mat, cmap="RdYlGn", vmin=-vmax_abs, vmax=vmax_abs,
                       aspect="auto")
        ax.set_xticks(range(len(labels)), labels, fontsize=8)
        ax.set_yticks(range(len(_MODELS)), _MODELS, fontsize=9)
        ax.set_title(f"{title}  —  0 = Smart Persistence parity (+ = better)",
                     fontsize=11)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    ax.text(j, i, "n/a", ha="center", va="center",
                            fontsize=7, color="0.45")
                else:
                    ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                            fontsize=7.5,
                            color="black" if abs(v) < vmax_abs * 0.55 else "white")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    out = out_dir / "fig_e8_skill_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_horizon_lines(df: pd.DataFrame, out_dir: Path) -> Path:
    """Skill_MAE% vs horizon per model; one panel per dataset."""
    set_style()
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6), sharey=False)
    for ax, (ds, disp, csv_hz, phys_hz, unit) in zip(axes, _DATASETS):
        d = df[(df.Dataset == ds) & (df.Metric == "Skill_MAE_pct")].sort_values("Horizon")
        x = np.arange(len(csv_hz))
        for m in _MODELS:
            ys = [d[d.Horizon == int(h)][m].iloc[0] for h in csv_hz]
            if all(pd.isna(v) for v in ys):
                continue
            ax.plot(x, ys, marker="o", ms=4, lw=1.6,
                    color=_color(m), label=m)
        ax.axhline(0, color="0.3", lw=0.8, ls="--")
        ax.set_xticks(x, [f"{h} {unit}" for h in phys_hz], fontsize=9)
        ax.set_title(disp, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Daytime Skill (MAE) %")
    handles, labs = axes[0].get_legend_handles_labels()
    for ax_ in axes[1:]:
        h_, l_ = ax_.get_legend_handles_labels()
        handles += h_
        labs += l_
    uniq = dict(zip(labs, handles))
    fig.legend(uniq.values(), uniq.keys(), loc="lower center",
               ncol=len(_MODELS), fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    out = out_dir / "fig_e8_horizon_lines.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="figures/results/e8_summary.csv")
    p.add_argument("--out-dir", default="figures/results")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _load(Path(args.csv))
    paths = [plot_heatmap(df, out_dir), plot_horizon_lines(df, out_dir)]
    for path in paths:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
