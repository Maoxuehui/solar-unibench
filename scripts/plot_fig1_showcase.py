"""Figure 1 — dataset showcase (3-panel stack, ALL REAL DATA).

  (a) Station spatial distribution  cropped Robinson map, 50S-70N, light
      grey land/ocean (#f4f6f8); distinct marker shapes, CVD-safe colors.
  (b) Representative core observation modalities (1x4):
      (b1) SKIPP'D sky camera, native 64x64 RGB frame          [zarr]
      (b2) Folsom GOES-15 satellite patch, native 10x10 pixels
           reshaped from the real 100-pixel row of Folsom_satellite.csv
      (b3) SunLake SEVIRI RSS patch, 64x64 VIS006 channel       [zarr]
      (b4) GEFCom2014 NWP heatmap, the 12 real ECMWF VAR columns
           x 24 h of one zone-day (predictors15.csv)
  (c) Folsom 1-min GHI + clearness index, 26-27 Feb 2014 (unchanged v1).

Output: scratch/fig01_dataset_showcase.png/pdf at 300 dpi.

Run (conda env solar_uni):
    python scripts/plot_fig1_showcase.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Circle
from PIL import Image

mpl.rcParams.update({
    "font.family": "Arial", "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.7,
    "axes.facecolor": "white", "figure.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42,
})

STD = _REPO_ROOT / "data" / "standardized"
RAW = Path("D:/monica/workspace/solar_benchmark_dataset")
OUT_DIR = _REPO_ROOT / "scratch"
OUT_STEM = "fig01_dataset_showcase"
W_IN, H_IN = 13.0, 9.5                      # user spec (12-14 x 9.5)

DS_COLORS = {"gefcom2014": "#56B4E9", "folsom": "#E69F00",
             "skippd": "#009E73", "sunlake": "#CC79A7"}
DS_MARKERS = {"gefcom2014": "^", "folsom": "s", "skippd": "o", "sunlake": "D"}
DS_LABELS = {"gefcom2014": "GEFCom2014 (zones 1–3)", "folsom": "Folsom",
             "skippd": "SKIPP'D", "sunlake": "SunLake (BSRN)"}
INK_2, INK_MUTED = "#52514E", "#898781"
ANCHOR = "#0072B2"
CMAP_VIS = LinearSegmentedColormap.from_list(
    "vis_blue", ["#0D366B", "#1C5CAB", "#3987E5", "#9EC5F4", "#E8F1FC"])
CMAP_NRBR = LinearSegmentedColormap.from_list(
    "nrbr", ["#D55E00", "#F7F7F5", "#0072B2"])
PT_TITLE = 11
PT_TAG = 10


# ===================== (a) station map ====================================
def draw_map(fig, rect) -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    meta = pd.concat([pd.read_parquet(STD / s / "metadata.parquet")
                      for s in ("skippd", "folsom", "sunlake", "gefcom2014")],
                     ignore_index=True)
    ax = fig.add_axes(rect, projection=ccrs.Robinson())
    ax.set_extent((-175, 180, -50, 70), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#f4f6f8",
                   zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#f4f6f8",
                   edgecolor="#c9cdd1", linewidth=0.4, zorder=1)
    gl = ax.gridlines(crs=ccrs.PlateCarree(),
                      xlocs=[-120, -60, 0, 60, 120, 180],
                      ylocs=[-30, 0, 30, 60], linewidth=0.3, color="#dfe3e6")
    gl.rotate_labels = False
    for src, grp in meta.groupby("dataset_source"):
        ax.scatter(grp.longitude, grp.latitude, s=34, c=DS_COLORS[src],
                   marker=DS_MARKERS[src], edgecolors="white",
                   linewidths=0.6, zorder=5, transform=ccrs.PlateCarree(),
                   label=DS_LABELS[src])
    def _site(src, ent):
        r = meta[(meta.dataset_source == src)
                 & (meta.entity_id == ent)].iloc[0]
        return r.longitude, r.latitude
    akw = dict(arrowprops=dict(arrowstyle="-", color="gray", lw=0.6),
               fontsize=8, fontweight="bold", color="#000000",
               ha="center", va="center", zorder=6,
               transform=ccrs.PlateCarree())
    ax.annotate("SKIPP'D", xy=_site("skippd", "SKIPPD_STFORD"),
                xytext=(-137, 31), **akw)
    ax.annotate("Folsom", xy=_site("folsom", "FOLSOM_STN"),
                xytext=(-112, 45), **akw)
    ax.text(163, -30.0, "GEFCom2014\n(zones 1–3)", fontsize=8,
            fontweight="bold", color="#000000", ha="center", va="center",
            zorder=6, transform=ccrs.PlateCarree())
    for ent, code, lx, ly in (("SUNLAKE_CAB", "CAB", -22.0, 51.5),
                              ("SUNLAKE_PAL", "PAL", -26.0, 47.5),
                              ("SUNLAKE_PAY", "PAY", -20.0, 43.5),
                              ("SUNLAKE_CNR", "CNR", -15.0, 39.5),
                              ("SUNLAKE_IZA", "IZA", 24.0, 30.5),
                              ("SUNLAKE_TAM", "TAM", -11.0, 19.5)):
        ax.annotate(code, xy=_site("sunlake", ent), xytext=(lx, ly), **akw)
    ax.legend(loc="lower left", bbox_to_anchor=(0.045, 0.02), frameon=True,
              facecolor="white", framealpha=0.85, edgecolor="#D5D5D1",
              handletextpad=0.2, labelspacing=0.3, borderaxespad=0.4,
              fontsize=8)
    ax.set_title("(a) Station spatial distribution", fontsize=PT_TITLE,
                 loc="left", pad=6)


# ===================== (b) observation modalities =========================
def pick_skippd_frame() -> np.ndarray:
    g = zarr.open_group(str(STD / "skippd" / "sky_images.zarr"), mode="r")
    i = 3600 * 24                                    # a full-sky mid-record
    fr = np.asarray(g["images"][i])                  # (4, 64, 64) RGB+BRR
    return np.transpose(fr[:3], (1, 2, 0))           # -> (64, 64, 3) RGB


def pick_folsom_goes() -> np.ndarray:
    """One real row of Folsom_satellite.csv -> native 10x10 GOES patch."""
    df = pd.read_csv(RAW / "Folsom_data" / "Folsom_satellite.csv",
                     header=None, nrows=200)
    row = df.iloc[100, 1:101].to_numpy(dtype=float)  # 100 pixel columns
    return row.reshape(10, 10)


def pick_sunlake_patch() -> np.ndarray:
    g = zarr.open_group(str(STD / "sunlake" / "satellite_future.zarr"),
                        mode="r")
    vis = g["context"][:, 7]                         # VIS006 channel
    return vis[int(np.argmax(vis.std(axis=(1, 2))))]


def pick_gefcom_heatmap() -> tuple[np.ndarray, list[str]]:
    """The 12 real ECMWF VAR columns over one zone-day (24 hours)."""
    df = pd.read_csv(RAW / "GEFCom2014_data" / "Solar" / "Task 15"
                     / "predictors15.csv")
    zone1 = df[df.ZONEID == 1]
    day = zone1.iloc[:24]                            # first full day, 24 rows
    vars12 = ["VAR78", "VAR79", "VAR134", "VAR157", "VAR164", "VAR165",
              "VAR166", "VAR167", "VAR169", "VAR175", "VAR178", "VAR228"]
    m = day[vars12].to_numpy(dtype=float).T          # (12 vars, 24 hours)
    return m, vars12


def draw_modalities(fig, rect) -> None:
    x0, y0, w, h = rect
    fig.text(x0, y0 + h + 0.006,
             "(b) Representative core observation modalities",
             fontsize=PT_TITLE, fontweight="bold", ha="left", va="bottom")
    gap, iw = 0.008, (w - 3 * 0.008) / 4
    ih = min(iw * 1.55, h)
    y_img = y0 + h - ih

    def _mk(k):
        ax = fig.add_axes([x0 + k * (iw + gap), y_img, iw, ih])
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_color("#111111"); sp.set_linewidth(1.0)
        return ax

    # (b1) SKIPP'D sky RGB
    _mk(0).imshow(pick_skippd_frame(), interpolation="nearest")
    # (b2) Folsom GOES 10x10, honest pixelated look
    _mk(1).imshow(pick_folsom_goes(), cmap="gray", interpolation="nearest")
    # (b3) SunLake SEVIRI VIS006
    _mk(2).imshow(pick_sunlake_patch(), cmap=CMAP_VIS, vmin=0, vmax=0.9,
                  interpolation="nearest")
    # (b4) GEFCom NWP heatmap 12 vars x 24 h, normalized per variable
    m, vars12 = pick_gefcom_heatmap()
    mn = (m - m.mean(axis=1, keepdims=True)) \
        / (m.std(axis=1, keepdims=True) + 1e-9)
    ax4 = _mk(3)
    im = ax4.imshow(mn, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2,
                    interpolation="nearest")
    ax4.set_xticks([0, 11, 23], ["0", "12", "23"], fontsize=7)
    ax4.set_yticks([0, 5, 11], ["CW", "RH", "TP"], fontsize=7)
    ax4.set_xlabel("hour of day", fontsize=7.5, labelpad=1.5)
    cax = fig.add_axes([x0 + 3 * (iw + gap) + 0.62 * iw,
                        y_img - 0.022, 0.34 * iw, 0.010])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([-2, 0, 2]); cb.ax.tick_params(labelsize=6.5, length=2,
                                                width=0.4, pad=1.0)
    cax.set_xlabel("z-score", fontsize=7, labelpad=1.0)
    for k, tag in enumerate(("(b1)", "(b2)", "(b3)", "(b4)")):
        fig.text(x0 + k * (iw + gap) + 0.004, y_img + ih - 0.006, tag,
                 fontsize=PT_TAG, fontweight="bold", color="white",
                 ha="left", va="top",
                 bbox=dict(boxstyle="square,pad=0.18", facecolor="black",
                           alpha=0.55, edgecolor="none"))
    caps = ["SKIPP'D sky RGB\n64 × 64", "Folsom GOES-15\n10 × 10 px",
            "SunLake SEVIRI RSS\n64 × 64, 11-ch", "GEFCom2014 NWP\n12 vars × 24 h"]
    for k, c in enumerate(caps):
        fig.text(x0 + k * (iw + gap) + iw / 2, y_img - 0.030, c,
                 fontsize=7.5, color=INK_2, ha="center", va="top")


# ===================== (c) aligned time series ============================
def draw_timeseries(fig, rect) -> None:
    x0, y0, w, h = rect
    fig.text(x0, y0 + h + 0.006,
             "(c) Aligned multi-day series — Folsom 1-min GHI and clearness "
             "index, 26–27 Feb 2014 (PST)", fontsize=PT_TITLE,
             fontweight="bold", ha="left", va="bottom")
    frac_kt, gap = 0.36, 0.010
    ax1 = fig.add_axes([x0, y0 + h * frac_kt + gap, w, h * (1 - frac_kt - gap)])
    ax2 = fig.add_axes([x0, y0, w, h * frac_kt - gap], sharex=ax1)

    df = pd.read_parquet(STD / "folsom" / "time_series.parquet")
    df = df.set_index("timestamp_utc").sort_index()
    sl = df.loc["2014-02-26 08:00:00+00:00":"2014-02-28 08:00:00+00:00"]
    local = sl.index.tz_convert("America/Los_Angeles")
    xh = (local - local[0]) / np.timedelta64(1, "h")
    cs = sl["ghi_clearsky"].to_numpy()
    for s, e in ((0, 6), (18, 30), (42, 47.983)):     # fixed night bands
        for ax in (ax1, ax2):
            ax.axvspan(s, e, color="#e6e6e6", alpha=0.6, zorder=0)
    ax1.text(24, 950, "night", ha="center", va="center", fontsize=7,
             color=INK_MUTED)
    ax1.plot(xh, sl["ghi_actual"].to_numpy(), color=DS_COLORS["folsom"],
             lw=0.8, label="Observed GHI")
    ax1.plot(xh, cs, color=INK_2, lw=0.7, ls=(0, (4, 2)),
             label="Clear-sky GHI")
    ax1.set_ylabel(r"GHI (W m$^{-2}$)")
    ax1.set_ylim(0, 1000); ax1.set_yticks([0, 500, 1000])
    ax1.tick_params(labelbottom=False)
    ax1.legend(loc="upper left", frameon=False, handlelength=1.8)
    # (anchor frame lives outside this Feb window — no marker plotted)
    kt = sl["clearness_index"].to_numpy(dtype=float).copy()
    kt[cs < 10.0] = np.nan
    ax2.plot(xh, kt, color=INK_2, lw=0.6)
    ax2.axhline(1.0, color=INK_MUTED, lw=0.5, ls=(0, (4, 2)))
    ax2.text(47.6, 1.03, r"$k_t=1$", fontsize=7, color=INK_MUTED,
             ha="right", va="bottom")
    ax2.set_ylabel(r"$k_t$ (–)")
    ax2.set_ylim(0, 1.6); ax2.set_yticks([0, 0.8, 1.6])
    ax2.set_xlim(0, 48); ax2.set_xticks(np.arange(0, 49, 6))
    ax2.set_xlabel("Local time (PST)")


# ===================== assemble ===========================================
fig = plt.figure(figsize=(W_IN, H_IN))
draw_map(fig, [0.045, 0.665, 0.94, 0.295])
draw_modalities(fig, [0.045, 0.355, 0.94, 0.235])
draw_timeseries(fig, [0.055, 0.075, 0.925, 0.205])

OUT_DIR.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=300, bbox_inches="tight")
    print(f"  {OUT_STEM}.{ext}")
print(f"done -> {OUT_DIR / (OUT_STEM + '.png')}")
