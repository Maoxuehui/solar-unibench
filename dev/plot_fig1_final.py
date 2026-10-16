"""Figure 1 (final variant) — composite showcase per the user's reference
script, with REAL data only.

Implements the reference visual conventions:
* matplotlib tab10 palette bindings + firebrick ALERT anchor (#b22222)
* map gridlines WITH 8 pt labels (top/right off), dashed gray
* station legend from scatter labels, white translucent box
* panel titles 9.5 pt bold, SENTENCE CASE (first letter only)
* (c) annotation "Frame in B (16:09 PST)" with arrow, red, both panels vline
* output Figure1_Final.png (300 dpi, tight) + vector PDF twin

Real-data corrections vs the reference placeholders (as instructed):
* GEFCom2014 zones: the reference "North America (-97, 35)" point is
  fabricated — the real three zones are at (150.0 E, -33.4/-33.5/-34.1 S),
  read from metadata.parquet.
* SunLake BSRN coordinates: the reference six coords are shifted/wrong;
  real coordinates are read from metadata.parquet.
* Sky frame, satellite patch and the 26-27 Feb 2014 Folsom series come from
  the standardized packages (same loaders as plot_fig1_showcase.py).

Run (conda env solar_uni):
    python scripts/plot_fig1_final.py
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
from matplotlib.lines import Line2D
from PIL import Image

# ===========================================================================
# Global style (reference script block) + palette bindings
# ===========================================================================
mpl.rcParams.update({
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.family": "sans-serif",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#E5E5E5",
    "grid.linewidth": 0.3,
    "axes.axisbelow": True,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})

COLOR_SKIPPD = "#2ca02c"
COLOR_FOLSOM = "#ff7f0e"
COLOR_GEFCOM = "#1f77b4"
COLOR_SUNLAKE = "#c567bf"
COLOR_ALERT = "#b22222"
DS_COLORS = {"skippd": COLOR_SKIPPD, "folsom": COLOR_FOLSOM,
             "gefcom2014": COLOR_GEFCOM, "sunlake": COLOR_SUNLAKE}
DS_MARKERS = {"gefcom2014": "^", "folsom": "s", "skippd": "o", "sunlake": "D"}
DS_LABELS = {"gefcom2014": "GEFCom2014 (zones 1–3)", "folsom": "Folsom",
             "skippd": "SKIPP'D", "sunlake": "SunLake (BSRN)"}

INK = "#0B0B0B"
INK_2 = "#52514E"
INK_MUTED = "#898781"
COAST = "#9A9A96"
LAND = "#E4E4E0"      # gray land
OCEAN = "#CFE0F0"     # blue ocean
CMAP_NRBR = LinearSegmentedColormap.from_list(
    "nrbr", ["#D55E00", "#F7F7F5", "#0072B2"])
CMAP_VIS = LinearSegmentedColormap.from_list(
    "vis_blue", ["#0D366B", "#1C5CAB", "#3987E5", "#9EC5F4", "#E8F1FC"])
LENS_RADIUS_FRAC = 0.95
MAP_EXTENT = (-180, 180, -48, 70)      # full longitude, latitude cropped
                                        # (keeps GEFCom S hemisphere + Europe)
PT_TITLE = 9.5
PT_AXIS = 8.5
PT_TICK = 7.5
PT_TAG = 8

WIDTH_MM, HEIGHT_MM = 180, 140
STD = _REPO_ROOT / "data" / "standardized"
OUT_DIR = _REPO_ROOT / "scratch"
OUT_STEM = "Figure1_Final"

TS_START = "2014-02-26 08:00:00+00:00"
TS_END = "2014-02-28 08:00:00+00:00"
FRAME_WIN = ("2014-02-26T14", "2014-02-28T02")
FRAME_CLOUD_TARGET = 0.35
SEVIRI_VIS006 = 7
mm = 1 / 25.4


# ===========================================================================
# Real-data loaders (identical to plot_fig1_showcase.py)
# ===========================================================================
def load_metadata() -> pd.DataFrame:
    frames = [pd.read_parquet(STD / s / "metadata.parquet")
              for s in ("skippd", "folsom", "sunlake", "gefcom2014")]
    meta = pd.concat(frames, ignore_index=True)
    if len(meta) != 11:
        raise ValueError(f"expected 11 entities, got {len(meta)}")
    return meta


def pick_sky_frame() -> tuple[np.ndarray, str]:
    g = zarr.open_group(STD / "folsom" / "sky_images_2014.zarr", mode="r")
    ts = [str(t) for t in g["images_timestamps"][:]]
    best = None
    for i, t in enumerate(ts):
        if not (FRAME_WIN[0] <= t[:13] <= FRAME_WIN[1]):
            continue
        rgb = np.asarray(Image.open(io.BytesIO(bytes(g["images"][i]))))
        if rgb.ndim != 3 or rgb.mean() < 50:
            continue
        r = rgb[..., 0].astype(np.float32)
        b = rgb[..., 2].astype(np.float32)
        cloudy = float(((b - r) / (b + r + 1e-6) < -0.1).mean())
        score = abs(cloudy - FRAME_CLOUD_TARGET)
        if best is None or score < best[0]:
            best = (score, rgb, cloudy, t, i)
    if best is None:
        raise RuntimeError("no daylight Folsom frame found in window")
    _, rgb, cloudy, t, i = best
    print(f"[frame] idx={i} ts={t} cloudy_frac={cloudy:.2f}")
    return rgb, t


def pick_satellite_patch() -> np.ndarray:
    g = zarr.open_group(STD / "sunlake" / "satellite_future.zarr", mode="r")
    vis = g["context"][:, SEVIRI_VIS006]
    i = int(np.argmax(vis.std(axis=(1, 2))))
    print(f"[satellite] context idx={i}")
    return vis[i]


def load_folsom_slice() -> pd.DataFrame:
    df = pd.read_parquet(STD / "folsom" / "time_series.parquet")
    df = df.set_index("timestamp_utc").sort_index()
    return df.loc[TS_START:TS_END].copy()


# ===========================================================================
# (a) map — real coordinates, gridline labels, scatter legend
# ===========================================================================
def draw_map(fig, rect) -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    ax = fig.add_axes(rect, projection=ccrs.Robinson())
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor=OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor=LAND, zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"),
                   linewidth=0.4, edgecolor=COAST, zorder=1.5)
    ax.spines["geo"].set_edgecolor("#333333")
    ax.spines["geo"].set_linewidth(0.8)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      xlocs=[-120, -60, 0, 60, 120, 180],
                      ylocs=[-30, 0, 30, 60], linewidth=0.3, color="gray",
                      alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8, "color": "#222222"}
    gl.ylabel_style = {"size": 8, "color": "#222222"}

    # REAL station coordinates from metadata.parquet (reference placeholders
    # replaced): scatter per regime carries the legend labels.
    for src, grp in meta.groupby("dataset_source"):
        ax.scatter(grp.longitude, grp.latitude, color=DS_COLORS[src],
                   marker=DS_MARKERS[src], s=32, edgecolor="white",
                   linewidth=0.5, zorder=5, transform=ccrs.PlateCarree(),
                   label=DS_LABELS[src])

    def _site(src, entity):
        row = meta[(meta.dataset_source == src)
                   & (meta.entity_id == entity)].iloc[0]
        return row.longitude, row.latitude

    akw = dict(arrowprops=dict(arrowstyle="-", color="gray", lw=0.6),
               fontsize=6.5, fontweight="bold", color="#000000",
               ha="center", va="center", zorder=6,
               transform=ccrs.PlateCarree())
    ax.annotate("SKIPP'D", xy=_site("skippd", "SKIPPD_STFORD"),
                xytext=(-137, 31), **akw)
    ax.annotate("Folsom", xy=_site("folsom", "FOLSOM_STN"),
                xytext=(-112, 45), **akw)
    ax.text(163, -38, "GEFCom2014\n(zones 1–3)", fontsize=6.5,
            fontweight="bold", color="#000000", ha="center", va="center",
            zorder=6, transform=ccrs.PlateCarree())
    for ent, code, lx, ly in (
            ("SUNLAKE_CAB", "CAB", -22.0, 51.5),
            ("SUNLAKE_PAL", "PAL", -26.0, 47.5),
            ("SUNLAKE_PAY", "PAY", -20.0, 43.5),
            ("SUNLAKE_CNR", "CNR", -15.0, 39.5),
            ("SUNLAKE_IZA", "IZA", 24.0, 30.5),
            ("SUNLAKE_TAM", "TAM", -11.0, 19.5)):
        ax.annotate(code, xy=_site("sunlake", ent), xytext=(lx, ly), **akw)

    ax.legend(loc="lower center", frameon=True,
              facecolor="white", framealpha=0.85, edgecolor="#D5D5D1",
              handletextpad=0.2, labelspacing=0.3, borderaxespad=0.4,
              fontsize=7)
    ax.set_title("(a) Station distribution across the four benchmark datasets",
                 fontsize=PT_TITLE, fontweight="bold", loc="left", pad=5)


# ===========================================================================
# (b) three uniform squares, bold tags, thin colorbars, red anchor border
# ===========================================================================
def draw_images(fig, rect) -> None:
    """(b) cell: title at cell top, three square images directly beneath it
    (minimal gaps, maximal size), colorbars + timestamp INSIDE the cell so
    the (b) block aligns top-to-bottom with the (c) block."""
    x0, y0, w, h = rect
    fig.text(x0, y0 + h + 0.008, "(b) Multi-modal observations",
             fontsize=PT_TITLE, fontweight="bold", ha="left", va="bottom")
    n, gap = 3, 0.006
    title_h, foot_h = 0.050, 0.040
    iw = (w - (n - 1) * gap) / n
    ih = min(iw, h - title_h - foot_h)
    y_img = y0 + h - title_h - ih
    imgs = [fig.add_axes([x0 + k * (iw + gap), y_img, iw, ih])
            for k in range(n)]
    for ax in imgs:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_color("#333333"); sp.set_linewidth(0.8)

    imgs[0].imshow(rgb_frame)
    for sp in imgs[0].spines.values():
        sp.set_color(COLOR_ALERT); sp.set_linewidth(1.4)
    r = rgb_frame[..., 0].astype(np.float32)
    b = rgb_frame[..., 2].astype(np.float32)
    nrbr = (b - r) / (b + r + 1e-6)
    hh, ww = nrbr.shape
    yy, xx = np.mgrid[0:hh, 0:ww]
    lens_r = LENS_RADIUS_FRAC * min(hh, ww) / 2.0
    nrbr[(yy - (hh - 1) / 2) ** 2 + (xx - (ww - 1) / 2) ** 2
         > lens_r ** 2] = np.nan
    im_n = imgs[1].imshow(nrbr, cmap=CMAP_NRBR,
                          norm=TwoSlopeNorm(vcenter=0.0, vmin=-0.4, vmax=0.8))
    im_v = imgs[2].imshow(vis_patch, cmap=CMAP_VIS, vmin=0.0, vmax=0.9,
                          interpolation="nearest")
    for ax, tag in zip(imgs, ("(b1)", "(b2)", "(b3)")):
        ax.text(0.03, 0.96, tag, transform=ax.transAxes, fontsize=PT_TAG,
                fontweight="bold", color="white", ha="left", va="top",
                bbox=dict(boxstyle="square,pad=0.18", facecolor="black",
                          edgecolor="none", alpha=0.55))
    # footer row inside the cell: timestamp under b1, colorbars under b2/b3
    ft_local = pd.Timestamp(frame_ts).tz_convert("America/Los_Angeles")
    fig.text(x0 + 0.5 * iw, y_img - 0.006,
             ft_local.strftime("%d %b %Y, %H:%M PST"), fontsize=PT_TICK,
             color=INK, ha="center", va="top", fontweight="bold")
    for im, k in ((im_n, 1), (im_v, 2)):
        cax = fig.add_axes([x0 + k * (iw + gap) + 0.08 * iw,
                            y_img - 0.022, 0.84 * iw, 0.012])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_ticks([-0.4, 0.0, 0.4, 0.8] if k == 1 else [0.0, 0.45, 0.9])
        cb.ax.tick_params(labelsize=6.5, width=0.4, length=2.0, pad=1.2)
        cb.outline.set_linewidth(0.4)
        cax.set_xlabel("NRBR" if k == 1 else "reflectance", fontsize=7,
                       labelpad=1.5)


# ===========================================================================
# (c) full-width aligned series — red anchor per reference
# ===========================================================================
def draw_timeseries(fig, rect) -> None:
    x0, y0, w, h = rect
    fig.text(x0, y0 + h + 0.008, "(c) Aligned multi-day series",
             fontsize=PT_TITLE, fontweight="bold", ha="left", va="bottom")
    frac_kt, gap = 0.36, 0.010
    h_ghi = (h - gap) * (1 - frac_kt)
    h_kt = (h - gap) * frac_kt
    ax_c1 = fig.add_axes([x0, y0 + h_kt + gap, w, h_ghi])
    ax_c2 = fig.add_axes([x0, y0, w, h_kt], sharex=ax_c1)

    local = sl.index.tz_convert("America/Los_Angeles")
    x_hours = (local - local[0]) / np.timedelta64(1, "h")
    cs = sl["ghi_clearsky"].to_numpy()

    # Night gray blocks via axvspan — fixed clock periods (user spec):
    # Feb 26 00:00-06:00, Feb 26 18:00 -> Feb 27 06:00, Feb 27 18:00-23:59,
    # expressed as hours from the window start (axis is hours, 0-48).
    night_periods_h = [(0.0, 6.0), (18.0, 30.0), (42.0, 47.983)]
    for start, end in night_periods_h:
        for ax in (ax_c1, ax_c2):
            # light-gray translucent block, zorder=0 keeps it under the lines
            ax.axvspan(start, end, color="#e6e6e6", alpha=0.6, zorder=0)
    ax_c1.text(24.0, 950, "night", ha="center", va="center", fontsize=6.5,
               color=INK_MUTED)

    ax_c1.plot(x_hours, sl["ghi_actual"].to_numpy(), color=COLOR_FOLSOM,
               lw=0.7, label="Observed GHI")
    ax_c1.plot(x_hours, cs, color=INK_2, lw=0.6, ls=(0, (4, 2)),
               label="Clear-sky GHI")
    ax_c1.set_ylabel(r"GHI (W m$^{-2}$)")
    ax_c1.set_ylim(0, 1000)
    ax_c1.set_yticks([0, 500, 1000])
    ax_c1.tick_params(labelbottom=False)
    ax_c1.legend(loc="upper left", frameon=False, borderaxespad=0.2,
                 handlelength=1.8)

    ft_local = pd.Timestamp(frame_ts).tz_convert("America/Los_Angeles")
    target_h = (ft_local - local[0]) / np.timedelta64(1, "h")   # 16:09 PST
    for ax in (ax_c1, ax_c2):
        ax.axvline(target_h, color=COLOR_ALERT, linestyle="--",
                   linewidth=1.1, zorder=10)
    ax_c1.annotate("Frame in B\n(16:09 PST)", xy=(target_h, 850),
                   xytext=(target_h - 1.5, 850),
                   arrowprops=dict(arrowstyle="->", color=COLOR_ALERT,
                                   lw=0.9),
                   fontsize=6.5, color=COLOR_ALERT, ha="right", va="center")

    kt_plot = sl["clearness_index"].to_numpy(dtype=float).copy()
    kt_plot[cs < 10.0] = np.nan
    ax_c2.plot(x_hours, kt_plot, color=INK_2, lw=0.5)
    ax_c2.axhline(1.0, color=INK_MUTED, lw=0.5, ls=(0, (4, 2)))
    ax_c2.text(47.6, 1.03, r"$k_t=1$", fontsize=6.5, color=INK_MUTED,
               ha="right", va="bottom")
    ax_c2.set_ylabel(r"$k_t$ (–)")
    ax_c2.set_ylim(0, 1.6)
    ax_c2.set_yticks([0, 0.8, 1.6])
    ax_c2.set_xlim(0, 48)
    ax_c2.set_xticks(np.arange(0, 49, 6))
    ax_c2.set_xlabel("Local time (PST)")


# ===========================================================================
# Assemble
# ===========================================================================
print("loading real dataset files ...")
meta = load_metadata()
rgb_frame, frame_ts = pick_sky_frame()
vis_patch = pick_satellite_patch()
sl = load_folsom_slice()

fig = plt.figure(figsize=(WIDTH_MM * mm, HEIGHT_MM * mm))
# Layout: two EQUAL rows (1:1) — (a) map spans the top; the bottom row
# splits (b):(c) = 3:2, both cells sharing the same top/bottom so titles
# and blocks align. gap_y leaves room for the (b)/(c) titles between the
# map's bottom gridline labels and the bottom row.
L, R, T, B = 0.030, 0.988, 0.962, 0.092
gap_y = 0.048
row_h = ((T - B) - gap_y) / 2.0
b_frac, c_frac, gap_x = 0.585, 0.385, 0.030

draw_map(fig, [L, T - row_h, R - L, row_h])
draw_images(fig, [L, B, (R - L) * b_frac, row_h])
draw_timeseries(fig, [L + (R - L) * (b_frac + gap_x), B,
                      (R - L) * c_frac, row_h])

OUT_DIR.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=300, bbox_inches="tight")
    print(f"  {OUT_STEM}.{ext}")
plt.close(fig)
print(f"done -> {OUT_DIR / (OUT_STEM + '.png')}")
