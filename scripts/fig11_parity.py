"""Figure 11 — Physical-closure parity plot (Folsom + SunLake).

ESSD data-validation figure: every dataset with measured GHI/DNI/DHI is plotted.
Two panels:
  (a) Folsom  — single-station, 1-min, R^2 = 0.9998 (near-perfect closure).
  (b) SunLake — 6-station wide-area, R^2 ~ 0.94 (closure holds, more noise).
Together they show the governance pipeline achieves physical closure of the
irradiance decomposition (GHI ~= DNI*cos(theta_z) + DHI) on both single-station
high-frequency AND multi-station wide-area settings.

Density hexbin (the daytime point count is too large for a scatter), y=x
reference line, per-panel annotation box (R^2, RMSE, MAE, n).
Output: figures_full/fig_physics_closure_parity.png (source for select_figures).
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from solarunibench.loader import SolarUniBenchLoader
from solarunibench.viz.io import save_fig
from solarunibench.viz.style import PALETTE, set_style

DATASETS = ["folsom", "sunlake"]
DAYTIME = 0.05


def closure_arrays(df: pd.DataFrame):
    cols = ["ghi_actual", "dni_actual", "dhi_actual", "cos_zenith"]
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    sub = sub[sub["cos_zenith"] > DAYTIME]
    if len(sub) < 8:
        return None
    dni_cos = sub["dni_actual"].clip(lower=0) * sub["cos_zenith"].clip(lower=0)
    recon = (dni_cos + sub["dhi_actual"].clip(lower=0)).to_numpy()
    ghi = sub["ghi_actual"].to_numpy()
    resid = ghi - recon
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((ghi - ghi.mean()) ** 2))
    return {
        "ghi": ghi, "recon": recon,
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "rmse": float(np.sqrt(ss_res / len(resid))),
        "mae": float(np.mean(np.abs(resid))),
        "n": int(len(resid)),
    }


def main() -> int:
    import matplotlib.pyplot as plt

    set_style("light")
    loader = SolarUniBenchLoader("D:/monica/workspace/solar-unibench/data/standardized")
    data = {}
    for ds in DATASETS:
        try:
            c = closure_arrays(loader.load_tabular(ds))
        except Exception as e:  # pragma: no cover
            print(f"  {ds} skipped: {e!r}", flush=True)
            c = None
        if c is not None:
            data[ds] = c
            print(f"  {ds}: R2={c['r2']:.4f} RMSE={c['rmse']:.2f} n={c['n']:,}", flush=True)
    if not data:
        print("no closure data; aborting", flush=True)
        return 1

    n_panels = len(data)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 5.8), squeeze=False)
    for ax, (ds, c) in zip(axes[0], data.items()):
        hb = ax.hexbin(c["ghi"], c["recon"], gridsize=90, cmap="viridis",
                       mincnt=1, alpha=0.95)
        lim = float(max(c["ghi"].max(), c["recon"].max())) * 1.02
        ax.plot([0, lim], [0, lim], color=PALETTE[3], linestyle="--", linewidth=1.5,
                label="y = x (perfect closure)")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"Measured GHI  (W/m$^2$)")
        ax.set_ylabel(r"Reconstructed  DNI$\cdot\cos(\theta_z)$ + DHI  (W/m$^2$)")
        sub = ("single-station, 1-min" if ds == "folsom"
               else "6-station, 30-min wide-area")
        ax.set_title(f"({chr(ord('a') + list(data).index(ds))}) {ds}\n{sub}")
        ax.text(0.04, 0.96,
                f"R$^2$ = {c['r2']:.4f}\n"
                f"RMSE = {c['rmse']:.2f} W/m$^2$\n"
                f"MAE = {c['mae']:.2f} W/m$^2$\n"
                f"n = {c['n']:,}",
                transform=ax.transAxes, fontsize=9.5, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#888888", alpha=0.9))
        ax.legend(loc="lower right", fontsize=8.5)
        cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("point density")
    fig.suptitle("Physical-closure parity: measured GHI vs its direct+diffuse "
                 "decomposition", fontsize=12)
    out = save_fig(fig, "fig_physics_closure_parity", Path("figures_full"))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
