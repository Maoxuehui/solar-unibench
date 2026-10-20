"""Figure 1 — dataset showcase: composite render + bare panel exports.

Merged from plot_fig1_showcase.py + export_fig01_panels.py (self-contained).
ALL REAL DATA, no synthetic arrays.

Two modes of output:
  1. Composite reference figure (3-panel stack) -> scratch/fig01_dataset_showcase.png
  2. Bare panels for the paper composite (no titles, 300 dpi)
     -> figures/fig01_dataset_showcase_panels/:
       a_map.png             station map (axes + legend, no title)
       b1_skippd_rgb.png     64x64 sky frame
       b2_folsom_goes.png    10x10 GOES patch
       b3_sunlake_seviri.png 64x64 VIS006
       b4_gefcom_nwp.png     12 vars x 24 h heatmap (+colorbar, ticks)
       c_timeseries.png      GHI + normalized-PV aligned 1x2 pair

Run (conda env solar_uni):
    python scripts/plots/plot_fig01.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
import os
_REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = Path(os.environ.get(
    "SOLAR_RAW_ROOT",
    "D:/monica/workspace/solar_benchmark_dataset"))

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

mpl.rcParams.update({
    "font.family": "Arial", "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.7,
    "axes.facecolor": "white", "figure.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42,
})

STD = _REPO_ROOT / "data" / "standardized"
RAW = RAW_ROOT
PANEL_DIR = _REPO_ROOT / "figures" / "fig01_dataset_showcase_panels"
COMPOSITE_DIR = _REPO_ROOT / "scratch"
W_IN, H_IN = 13.0, 9.5

DS_COLORS = {"gefcom2014": "#56B4E9", "folsom": "#E69F00",
             "skippd": "#009E73", "sunlake": "#CC79A7"}
DS_MARKERS = {"gefcom2014": "^", "folsom": "s", "skippd": "o", "sunlake": "D"}
DS_LABELS = {"gefcom2014": "GEFCom2014 (zones 1–3)", "folsom": "Folsom",
             "skippd": "SKIPP'D", "sunlake": "SunLake (BSRN)"}
INK_2, INK_MUTED = "#52514E", "#898781"
CMAP_VIS = LinearSegmentedColormap.from_list(
    "vis_blue", ["#0D366B", "#1C5CAB", "#3987E5", "#9EC5F4", "#E8F1FC"])
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


# ===================== data pickers (modality panels) =====================
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

    _mk(0).imshow(pick_folsom_goes(), cmap="gray", interpolation="nearest")
    m, _ = pick_gefcom_heatmap()
    mn = (m - m.mean(axis=1, keepdims=True)) \
        / (m.std(axis=1, keepdims=True) + 1e-9)
    ax4 = _mk(1)
    im = ax4.imshow(mn, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2,
                    interpolation="nearest")
    ax4.set_xticks([0, 11, 23], ["0", "12", "23"], fontsize=7)
    ax4.set_yticks([0, 3, 4, 11], ["CW", "RH", "TCC", "TP"], fontsize=7)
    ax4.set_xlabel("hour of day", fontsize=7.5, labelpad=1.5)
    _mk(2).imshow(pick_sunlake_patch(), cmap=CMAP_VIS, vmin=0, vmax=0.9,
                  interpolation="nearest")
    _mk(3).imshow(pick_skippd_frame(), interpolation="nearest")
    cax = fig.add_axes([x0 + 1 * (iw + gap) + 0.33 * iw,
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
    caps = ["Folsom GOES-15\n10 × 10 px", "GEFCom2014 NWP\n12 vars × 24 h",
            "SunLake SEVIRI RSS\n64 × 64, 11-ch", "SKIPP'D sky RGB\n64 × 64"]
    for k, c in enumerate(caps):
        fig.text(x0 + k * (iw + gap) + iw / 2, y_img - 0.030, c,
                 fontsize=7.5, color=INK_2, ha="center", va="top")


# ===================== (c) aligned time series (composite) ================
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


# ===================== bare-panel exports =================================
def _save(fig, name: str) -> None:
    fig.savefig(PANEL_DIR / f"{name}.png", dpi=300, bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)
    print(f"  {name}.png", flush=True)


def _bare(ax, border=True):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    if border:
        for sp in ax.spines.values():
            sp.set_color("#111111"); sp.set_linewidth(1.0)


NIGHT_BANDS = ((0, 6), (18, 30), (42, 47.983))


def _night(ax):
    for s0, e0 in NIGHT_BANDS:
        ax.axvspan(s0, e0, color="#e6e6e6", alpha=0.6, zorder=0)


def _frame_marker(ax, h, label):
    ax.axvline(h, color="black", lw=0.8, ls=(0, (3, 2)), zorder=5)
    ax.annotate(label, xy=(h, 0.06), xycoords=("data", "axes fraction"),
                xytext=(h - 1.2, 0.30), textcoords=("data", "axes fraction"),
                fontsize=7, color="#0B0B0B", ha="right",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#0B0B0B"))


def export_panels() -> None:
    PANEL_DIR.mkdir(parents=True, exist_ok=True)

    # (a) map, title stripped
    fig = plt.figure(figsize=(10.5, 3.4))
    draw_map(fig, [0.01, 0.02, 0.98, 0.96])
    fig.gca().set_title("")
    _save(fig, "a_map")

    # (b1..b4) single modalities, no tags/captions
    fig, ax = plt.subplots(figsize=(2.4, 2.4)); _bare(ax)
    ax.imshow(pick_skippd_frame(), interpolation="nearest")
    _save(fig, "b1_skippd_rgb")

    fig, ax = plt.subplots(figsize=(2.4, 2.4)); _bare(ax)
    ax.imshow(pick_folsom_goes(), cmap="gray", interpolation="nearest")
    _save(fig, "b2_folsom_goes")

    fig, ax = plt.subplots(figsize=(2.4, 2.4)); _bare(ax)
    ax.imshow(pick_sunlake_patch(), cmap=CMAP_VIS, vmin=0, vmax=0.9,
              interpolation="nearest")
    _save(fig, "b3_sunlake_seviri")

    m, _ = pick_gefcom_heatmap()
    mn = (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.imshow(mn, aspect="auto", cmap="RdBu_r", vmin=-1.8, vmax=1.8,
              interpolation="nearest")
    ax.set_xticks([0, 11, 23], ["0", "12", "23"], fontsize=7)
    ax.set_yticks([0, 3, 4, 11], ["CW", "RH", "TCC", "TP"], fontsize=7)
    ax.set_xlabel("hour of day", fontsize=7.5, labelpad=1.5)
    cb = fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.03)
    cb.set_ticks([-2, 0, 2]); cb.ax.tick_params(labelsize=6.5)
    cb.set_label("z-score", fontsize=7)
    _save(fig, "b4_gefcom_nwp")

    # (c) 1x2: Folsom GHI | SKIPP'D normalized PV, 48 h each
    fig, (axc1, axc2) = plt.subplots(1, 2, figsize=(12.0, 2.6), sharex=True)

    fdf = pd.read_parquet(STD / "folsom" / "time_series.parquet")
    fdf = fdf.set_index("timestamp_utc").sort_index()
    f0 = pd.Timestamp("2014-02-26 08:00:00", tz="UTC")
    fw = fdf.loc[f0:f0 + pd.Timedelta(hours=48)]
    fxh = (fw.index - f0).total_seconds() / 3600.0
    _night(axc1)
    axc1.text(24, 950, "Night", ha="center", va="center", fontsize=7,
              color=INK_MUTED)
    axc1.plot(fxh, fw["ghi_actual"].to_numpy(), color=DS_COLORS["folsom"],
              lw=0.7, label="Observed GHI")
    axc1.plot(fxh, fw["ghi_clearsky"].to_numpy(), color=INK_2, lw=0.6,
              ls=(0, (4, 2)), label="Clear-sky GHI")
    _frame_marker(axc1, 38.43, "frame in (b2)")
    axc1.set_ylabel(r"GHI (W m$^{-2}$)")
    axc1.set_ylim(0, 1000); axc1.set_yticks([0, 500, 1000])
    axc1.set_xlim(0, 48); axc1.set_xticks(np.arange(0, 49, 6))
    axc1.set_xlabel("Local time (PST)")
    axc1.legend(loc="upper left", frameon=False, fontsize=7, handlelength=1.6)

    sdf = pd.read_parquet(STD / "skippd" / "time_series.parquet")
    sdf["ts"] = pd.to_datetime(sdf["timestamp_utc"])
    sdf = sdf.set_index("ts").sort_index()
    s0 = pd.Timestamp("2018-06-10 07:00:00", tz="UTC")
    sw = sdf.loc[s0:s0 + pd.Timedelta(hours=48)]
    sxh = (sw.index - s0).total_seconds() / 3600.0
    _night(axc2)
    axc2.text(24, 0.93, "Night", ha="center", va="center", fontsize=7,
              color=INK_MUTED)
    axc2.plot(sxh, sw["pv_power_normalized"].astype(float).to_numpy(),
              color="#0072B2", lw=0.5, label="Normalized PV power")
    _frame_marker(axc2, 12.03, "frame in (b1)")
    axc2.set_ylabel("Normalized PV power (0–1)")
    axc2.set_ylim(0, 1); axc2.set_yticks([0, 0.5, 1.0])
    axc2.set_xlabel("Local time (PST)")
    axc2.legend(loc="upper left", frameon=False, fontsize=7, handlelength=1.6)

    fig.subplots_adjust(wspace=0.22, left=0.06, right=0.99, top=0.97,
                        bottom=0.20)
    _save(fig, "c_timeseries")


def export_composite() -> None:
    fig = plt.figure(figsize=(W_IN, H_IN))
    draw_map(fig, [0.045, 0.665, 0.94, 0.295])
    draw_modalities(fig, [0.045, 0.355, 0.94, 0.235])
    draw_timeseries(fig, [0.055, 0.075, 0.925, 0.205])
    COMPOSITE_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(COMPOSITE_DIR / f"fig01_dataset_showcase.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  fig01_dataset_showcase.png/.pdf (composite)", flush=True)


def main() -> int:
    export_panels()
    export_composite()
    export_c_set()
    print(f"done -> panels in {PANEL_DIR}")
    return 0


# ============ c-set (paper fig01 750x750 panels, simplified REAL DATA) ====
# For each dataset: one day of the TARGET variable + the other modality
# extracted at the event moment on the curve. Square canvas -> isotropic
# 750x750 (no anisotropic distortion).
C_DIR = _REPO_ROOT / "figures" / "fig01_c1_set_rebuilt"


def _save750(fig, name: str) -> None:
    import io

    from PIL import Image
    C_DIR.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    side = max(im.size)
    canvas = Image.new("RGB", (side, side), "white")
    canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    canvas = canvas.resize((750, 750), Image.LANCZOS)
    canvas.save(C_DIR / f"{name}.png")
    print(f"  {name}.png (750x750, square)", flush=True)


def _curve(ax, hours, values, color, bands, vline_h, ylabel, ylim, yticks,
           label, cs=None, legend_pt=7.0, marker=None, ms=3.0):
    for s0, e0 in bands:
        ax.axvspan(s0, e0, color="#e6e6e6", alpha=0.6, zorder=0)
    mk = dict(marker=marker, markersize=ms, markerfacecolor=color,
              markeredgecolor="white", markeredgewidth=0.5) if marker else {}
    ax.plot(hours, values, color=color, lw=0.9, label=label, zorder=3, **mk)
    if cs is not None:
        ax.plot(hours, cs, color=INK_2, lw=0.7, ls=(0, (4, 2)),
                label="Clear-sky GHI", zorder=2)
    if vline_h is not None:
        ax.axvline(vline_h, color="black", lw=0.8, ls=(0, (3, 2)), zorder=5)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(*ylim); ax.set_yticks(yticks)
    ax.set_xlim(0, 24); ax.set_xticks(np.arange(0, 25, 4))
    ax.tick_params(labelsize=10)
    ax.legend(loc="upper left", frameon=False, fontsize=legend_pt,
              handlelength=1.6)


def _with_gap_breaks(times, hours, values, max_gap_min=2.0):
    """Insert NaN where consecutive samples are > max_gap_min apart, so
    the line breaks at real data gaps instead of drawing wrap-around
    connectors (e.g. the SKIPP'D table has no night rows)."""
    h, v = list(hours), list(values)
    out_h, out_v = [], []
    for i in range(len(h)):
        if i > 0:
            dt_min = (times[i] - times[i - 1]).total_seconds() / 60.0
            if dt_min > max_gap_min:
                out_h.append(np.nan); out_v.append(np.nan)
        out_h.append(h[i]); out_v.append(v[i])
    return np.array(out_h, dtype=float), np.array(out_v, dtype=float)


def _img_panel(name, array, **imshow_kw):
    fig, ax = plt.subplots(figsize=(2.5, 2.5)); _bare(ax)
    ax.imshow(array, interpolation="nearest", **imshow_kw)
    _save750(fig, name)


def export_c_set() -> None:
    C_DIR.mkdir(parents=True, exist_ok=True)

    # ---- (c1) Folsom 2014-04-26: GHI + fisheye + GOES at crash moment --
    d1 = pd.read_parquet(STD / "folsom" / "time_series.parquet")
    d1 = d1.set_index("timestamp_utc").sort_index()
    day1 = d1.loc["2014-04-26"]
    loc1 = day1.index.tz_convert("America/Los_Angeles")   # PDT (real DST)
    xh1 = loc1.hour + loc1.minute / 60.0
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.set_xlabel("Local time (PDT)", fontsize=11)
    _curve(ax, xh1.to_numpy(), day1["ghi_actual"].to_numpy(),
           DS_COLORS["folsom"], [(0.0, 6.05), (19.85, 24.0)],
           vline_h=12.47, ylabel=r"GHI (W m$^{-2}$)",
           ylim=(0, 1500), yticks=[0, 500, 1000, 1500],
           label="Observed GHI", cs=day1["ghi_clearsky"].to_numpy())
    _save750(fig, "c1_curve_24h")

    g = zarr.open_group(str(STD / "folsom" / "sky_images_2014.zarr"),
                        mode="r")
    ts = pd.to_datetime(pd.Series(g["images_timestamps"][:]), utc=True)
    idx = int(np.argmin(np.abs(
        (ts - pd.Timestamp("2014-04-26 19:28:00+00:00"))
        .dt.total_seconds().to_numpy())))
    blob = g["images"][idx]
    fr = np.asarray(Image.open(io.BytesIO(blob))) \
        if isinstance(blob, (bytes, str)) else np.asarray(blob)
    if fr.ndim == 3 and fr.shape[0] in (3, 4):
        fr = np.transpose(fr[:3], (1, 2, 0))
    _img_panel("c1_fisheye", fr)

    sat = pd.read_csv(RAW / "Folsom_data" / "Folsom_satellite.csv",
                      header=None)
    tt = pd.to_datetime(sat.iloc[:, 0], utc=True, errors="coerce")
    gi = int(np.argmin(np.abs(
        (tt - pd.Timestamp("2014-04-26 19:15:00+00:00")).dt.total_seconds()
        .to_numpy())))
    _img_panel("c1_goes", sat.iloc[gi, 1:101].to_numpy(float).reshape(10, 10),
               cmap="gray")

    # ---- (c2) SKIPP'D 2018-06-10: PV + sky frame at dip ------------------
    d2 = pd.read_parquet(STD / "skippd" / "time_series.parquet")
    d2["ts"] = pd.to_datetime(d2["timestamp_utc"])
    d2 = d2.set_index("ts").sort_index()
    day2 = d2.loc["2018-06-10"]
    loc2 = day2.index.tz_convert("America/Los_Angeles")   # PDT
    xh2 = loc2.hour + loc2.minute / 60.0
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.set_xlabel("Local time (PDT)", fontsize=11)
    xb2, yb2 = _with_gap_breaks(list(day2.index), xh2.to_numpy(),
                                 day2["pv_power_normalized"]
                                 .astype(float).to_numpy())
    _curve(ax, xb2, yb2,
           "#0072B2", [(0.0, 5.87), (20.42, 24.0)],
           vline_h=12.03, ylabel="Normalized PV power (0–1)",
           ylim=(0, 1), yticks=[0, 0.5, 1.0],
           label="Normalized PV power")
    _save750(fig, "c2_curve_24h")

    g2 = zarr.open_group(str(STD / "skippd" / "sky_images.zarr"), mode="r")
    _img_panel("c2_sky_frame",
               np.transpose(np.asarray(g2["images"][152480])[:3],
                            (1, 2, 0)))

    # ---- (c3) GEFCom2014 zone 3, 2012-10-07: PV + NWP heatmap ------------
    pred = pd.read_csv(RAW / "GEFCom2014_data" / "Solar" / "Task 15"
                       / "predictors15.csv")
    vars12 = ["VAR78", "VAR79", "VAR134", "VAR157", "VAR164", "VAR165",
              "VAR166", "VAR167", "VAR169", "VAR175", "VAR178", "VAR228"]
    z3 = pred[pred.ZONEID == 3]
    day3 = z3[z3.TIMESTAMP.str.startswith("20121007")]
    m = day3[vars12].to_numpy(dtype=float).T
    mn = (m - m.mean(axis=1, keepdims=True)) \
        / (m.std(axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.imshow(mn, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2,
              interpolation="nearest")
    ax.set_xticks([0, 11, 23], ["0", "12", "23"], fontsize=10)
    ax.set_yticks([0, 3, 4, 11], ["CW", "RH", "TCC", "TP"], fontsize=10)
    ax.set_xlabel("hour of day", fontsize=11)
    cb = fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.03)
    cb.set_ticks([-2, 0, 2]); cb.ax.tick_params(labelsize=10)
    cb.set_label("z-score", fontsize=11)
    _save750(fig, "c3_nwp_24h")

    gt = pd.read_parquet(STD / "gefcom2014" / "time_series.parquet")
    gt = gt.set_index("timestamp_utc").sort_index()
    loc_all = gt.index.tz_convert("Australia/Sydney")
    m3 = ((gt.entity_id == "GEFCOM_ZONE_3")
          & (loc_all.date == pd.Timestamp("2012-10-07").date()))
    g3 = gt.loc[m3]
    loc3 = g3.index.tz_convert("Australia/Sydney")
    xh3 = loc3.hour + loc3.minute / 60.0
    y3 = g3["pv_power_normalized"].astype(float).to_numpy()
    x3 = xh3.to_numpy()
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.set_xlabel("Local time (AEDT)", fontsize=11)
    _curve(ax, x3, y3,
           "#0072B2", [(0.0, 5.7), (18.32, 24.0)],
           vline_h=17.0, ylabel="Normalized PV power (0–1)",
           ylim=(0, 1), yticks=[0, 0.5, 1.0],
           label="Normalized PV power", marker="o", ms=3.2)
    _save750(fig, "c3_pv_24h")

    # ---- (c4) SunLake CAB 2021-08-03: GHI + satellite at drop moment -----
    g4 = zarr.open_group(str(STD / "sunlake" / "sunlake_satellite_full.zarr"),
                         mode="r")
    fr4 = np.asarray(g4["images"][202196]) if "images" in g4 \
        else np.asarray(g4[list(g4.array_keys())[0]][202196])
    vis = fr4 if fr4.ndim == 2 else fr4[7]
    _img_panel("c4_satellite_patch", vis, cmap="PuBu", vmin=0, vmax=0.75)
    _img_panel("c4_satellite_seviri", vis, cmap=CMAP_VIS, vmin=0, vmax=0.9)

    s4 = pd.read_parquet(STD / "sunlake" / "time_series.parquet")
    s4 = s4.set_index("timestamp_utc").sort_index()
    m4 = ((s4.entity_id == "SUNLAKE_CAB")
          & (s4.index >= "2021-08-03") & (s4.index < "2021-08-04"))
    c4 = s4.loc[m4]
    xh4 = c4.index.hour + c4.index.minute / 60.0
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.set_xlabel("Hour (UTC)", fontsize=11)
    _curve(ax, xh4.to_numpy(), c4["ghi_actual"].to_numpy(),
           DS_COLORS["folsom"], [(0.0, 4.08), (19.25, 24.0)],
           vline_h=11.5, ylabel=r"GHI (W m$^{-2}$)",
           ylim=(0, 1000), yticks=[0, 500, 1000],
           label="Observed GHI", cs=c4["ghi_clearsky"].to_numpy())
    _save750(fig, "c4_ghi_24h")


if __name__ == "__main__":
    sys.exit(main())


run = main
