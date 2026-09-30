"""Export each panel of fig01 (v3) as a SEPARATE image, WITHOUT titles.

Outputs (300 dpi, figures/fig01_dataset_showcase_panels/):
  a_map.png             station map (axes + legend, no title)
  b1_skippd_rgb.png     64x64 sky frame
  b2_folsom_goes.png    10x10 GOES patch
  b3_sunlake_seviri.png 64x64 VIS006
  b4_gefcom_nwp.png     12 vars x 24 h heatmap (+colorbar, ticks; no title)
  c_timeseries.png      GHI + kt aligned pair (axis labels/legend; no title)

Run (conda env solar_uni):
    python scripts/export_fig01_panels.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_fig1_showcase as V

mpl.rcParams.update({
    "font.family": "Arial", "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.7,
    "axes.facecolor": "white", "figure.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42,
})

OUT = _REPO_ROOT / "figures" / "fig01_dataset_showcase_panels"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)
    print(f"  {name}.png")


# -- (a) map, title stripped ------------------------------------------------
fig = plt.figure(figsize=(10.5, 3.4))
V.draw_map(fig, [0.01, 0.02, 0.98, 0.96])
fig.gca().set_title("")                      # strip the (a) title
save(fig, "a_map")

# -- (b1..b4) single modalities, no tags/captions ---------------------------
def bare(ax, border=True):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    if border:
        for sp in ax.spines.values():
            sp.set_color("#111111"); sp.set_linewidth(1.0)


fig, ax = plt.subplots(figsize=(2.4, 2.4)); bare(ax)
ax.imshow(V.pick_skippd_frame(), interpolation="nearest")
save(fig, "b1_skippd_rgb")

fig, ax = plt.subplots(figsize=(2.4, 2.4)); bare(ax)
ax.imshow(V.pick_folsom_goes(), cmap="gray", interpolation="nearest")
save(fig, "b2_folsom_goes")

fig, ax = plt.subplots(figsize=(2.4, 2.4)); bare(ax)
ax.imshow(V.pick_sunlake_patch(), cmap=V.CMAP_VIS, vmin=0, vmax=0.9,
          interpolation="nearest")
save(fig, "b3_sunlake_seviri")

m, _ = V.pick_gefcom_heatmap()
mn = (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-9)
fig, ax = plt.subplots(figsize=(3.0, 3.0))
ax.imshow(mn, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2,
          interpolation="nearest")
ax.set_xticks([0, 11, 23], ["0", "12", "23"], fontsize=7)
ax.set_yticks([0, 5, 11], ["CW", "RH", "TP"], fontsize=7)
ax.set_xlabel("hour of day", fontsize=7.5, labelpad=1.5)
cb = fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.03)
cb.set_ticks([-2, 0, 2]); cb.ax.tick_params(labelsize=6.5)
cb.set_label("z-score", fontsize=7)
save(fig, "b4_gefcom_nwp")

# -- (c) GHI series only (no kt panel) ---------------------------------------
fig = plt.figure(figsize=(12.0, 2.3))
ax = fig.add_axes([0.07, 0.20, 0.90, 0.72])
df = pd.read_parquet(V.STD / "folsom" / "time_series.parquet")
df = df.set_index("timestamp_utc").sort_index()
sl = df.loc["2014-02-26 08:00:00+00:00":"2014-02-28 08:00:00+00:00"]
local = sl.index.tz_convert("America/Los_Angeles")
xh = (local - local[0]) / np.timedelta64(1, "h")
cs = sl["ghi_clearsky"].to_numpy()
for s0, e0 in ((0, 6), (18, 30), (42, 47.983)):
    ax.axvspan(s0, e0, color="#e6e6e6", alpha=0.6, zorder=0)
ax.text(24, 950, "night", ha="center", va="center", fontsize=7,
        color=V.INK_MUTED)
ax.plot(xh, sl["ghi_actual"].to_numpy(), color=V.DS_COLORS["folsom"],
        lw=0.8, label="Observed GHI")
ax.plot(xh, cs, color=V.INK_2, lw=0.7, ls=(0, (4, 2)), label="Clear-sky GHI")
ax.set_ylabel(r"GHI (W m$^{-2}$)")
ax.set_ylim(0, 1000); ax.set_yticks([0, 500, 1000])
ax.set_xlim(0, 48); ax.set_xticks(np.arange(0, 49, 6))
ax.set_xlabel("Local time (PST)")
ax.legend(loc="upper left", frameon=False, handlelength=1.8)
save(fig, "c_timeseries")

print(f"done -> {OUT}")
