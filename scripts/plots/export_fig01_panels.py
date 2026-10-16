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

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(Path(__file__).resolve().parent)):
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
ax.set_yticks([0, 3, 4, 11], ["CW", "RH", "TCC", "TP"], fontsize=7)
ax.set_xlabel("hour of day", fontsize=7.5, labelpad=1.5)
cb = fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.03)
cb.set_ticks([-2, 0, 2]); cb.ax.tick_params(labelsize=6.5)
cb.set_label("z-score", fontsize=7)
save(fig, "b4_gefcom_nwp")

# -- (c) 1x2 side-by-side: Folsom GHI | SKIPP'D normalized PV -------------
import numpy as np

TEAL = "#008B8B"
NIGHT_BANDS = ((0, 6), (18, 30), (42, 47.983))

fig, (axc1, axc2) = plt.subplots(1, 2, figsize=(12.0, 2.6),
                                 sharex=True)

def _night(ax):
    for s0, e0 in NIGHT_BANDS:
        ax.axvspan(s0, e0, color="#e6e6e6", alpha=0.6, zorder=0)

def _frame_marker(ax, h, label):
    ax.axvline(h, color="black", lw=0.8, ls=(0, (3, 2)), zorder=5)
    ax.annotate(label, xy=(h, 0.06), xycoords=("data", "axes fraction"),
                xytext=(h - 1.2, 0.30), textcoords=("data", "axes fraction"),
                fontsize=7, color="#0B0B0B", ha="right",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#0B0B0B"))

# (c1) Folsom GHI — 48 h, trough marker at h=31.0
fdf = pd.read_parquet(V.STD / "folsom" / "time_series.parquet")
fdf = fdf.set_index("timestamp_utc").sort_index()
f0 = pd.Timestamp("2014-02-26 08:00:00", tz="UTC")
fw = fdf.loc[f0:f0 + pd.Timedelta(hours=48)]
fxh = (fw.index - f0).total_seconds() / 3600.0
_night(axc1)
axc1.text(24, 950, "Night", ha="center", va="center", fontsize=7,
          color=V.INK_MUTED)
axc1.plot(fxh, fw["ghi_actual"].to_numpy(), color=V.DS_COLORS["folsom"],
          lw=0.7, label="Observed GHI")
axc1.plot(fxh, fw["ghi_clearsky"].to_numpy(), color=V.INK_2, lw=0.6,
          ls=(0, (4, 2)), label="Clear-sky GHI")
_frame_marker(axc1, 38.43, "frame in (b2)")
axc1.set_ylabel(r"GHI (W m$^{-2}$)")
axc1.set_ylim(0, 1000); axc1.set_yticks([0, 500, 1000])
axc1.set_xlim(0, 48); axc1.set_xticks(np.arange(0, 49, 6))
axc1.set_xlabel("Local time (PST)")
axc1.legend(loc="upper left", frameon=False, fontsize=7, handlelength=1.6)

# (c2) SKIPP'D normalized PV — 48 h from 2018-06-10 07:00 UTC, dip at 19.7 h
sdf = pd.read_parquet(V.STD / "skippd" / "time_series.parquet")
sdf["ts"] = pd.to_datetime(sdf["timestamp_utc"])
sdf = sdf.set_index("ts").sort_index()
s0 = pd.Timestamp("2018-06-10 07:00:00", tz="UTC")
sw = sdf.loc[s0:s0 + pd.Timedelta(hours=48)]
sxh = (sw.index - s0).total_seconds() / 3600.0
_night(axc2)
axc2.text(24, 0.93, "Night", ha="center", va="center", fontsize=7,
          color=V.INK_MUTED)
axc2.plot(sxh, sw["pv_power_normalized"].astype(float).to_numpy(),
          color="#0072B2", lw=0.5, label="Normalized PV power")
_frame_marker(axc2, 12.03, "frame in (b1)")
axc2.set_ylabel("Normalized PV power (0–1)")
axc2.set_ylim(0, 1); axc2.set_yticks([0, 0.5, 1.0])
axc2.set_xlabel("Local time (PST)")
axc2.legend(loc="upper left", frameon=False, fontsize=7, handlelength=1.6)

fig.subplots_adjust(wspace=0.22, left=0.06, right=0.99, top=0.97,
                    bottom=0.20)
save(fig, "c_timeseries")

print(f"done -> {OUT}")
