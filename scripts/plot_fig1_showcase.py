"""Figure 1 — dataset showcase: station map, multi-modal imagery, GHI time series.

Standalone ESSD/Copernicus figure for the Solar-UniBench data descriptor:

  (a)  Global distribution of the 11 benchmark entities across the four
       regimes (SKIPP'D, Folsom, GEFCom2014, SunLake BSRN), Robinson
       projection, from metadata.parquet (real lat/lon, no synthetic data).
  (b)  Real multi-modal imagery: (b1) Folsom fisheye sky-camera RGB frame,
       (b2) NRBR channel (B-R)/(B+R+eps) computed from that same frame,
       (b3) SunLake SEVIRI VIS006 satellite context patch.
  (c)  1-min Folsom GHI (observed + clear-sky model) with the clearness
       index k_t on an aligned lower panel (two stacked panels sharing one
       x-axis — never a dual y-axis). The dashed vertical marker shows the
       UTC timestamp of the (b1)/(b2) sky frame.

Design contract: follows scripts/plot_essd_figures.py (Arial 8.5 pt,
180 mm double-column, Wong/Okabe-Ito DS_COLORS binding, light grid,
no plot titles — bold panel letters + sentence-case sub-labels).

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
from matplotlib.lines import Line2D
from PIL import Image

# ===========================================================================
# MANDATORY PARAMETER BLOCK — the ONLY place sizes/colors/dimensions live
# ===========================================================================
TEXT_SIZE_PT = 8.5
SMALL_PT = 8.0          # image sub-labels / station codes
FONT_FAMILY = "Arial"
LINE_WIDTH_DATA = 0.6
WIDTH_MM = 180          # ESSD double column
HEIGHT_MM = 206
DPI_RASTER = 300
GRID_COLOR = "#E5E5E5"
GRID_WIDTH = 0.3
PANEL_LABEL_PT = 9.5

INK = "#0B0B0B"         # primary text
INK_2 = "#52514E"       # secondary text / reference curves
INK_MUTED = "#898781"   # leader lines, dates
COAST = "#B9B9B5"
LAND = "#EFEFED"
OCEAN = "#FBFCFD"

#: Dataset -> color binding (semantic, shared by every figure — house standard).
DS_COLORS = {
    "gefcom2014": "#56B4E9",   # sky blue
    "folsom": "#E69F00",       # orange
    "skippd": "#009E73",       # bluish green
    "sunlake": "#CC79A7",      # reddish purple
}
DS_MARKERS = {"gefcom2014": "^", "folsom": "s", "skippd": "o", "sunlake": "D"}
DS_LABELS = {"gefcom2014": "GEFCom2014 (zones 1–3)", "folsom": "Folsom",
             "skippd": "SKIPP'D", "sunlake": "SunLake (BSRN)"}

#: NRBR diverging pair (blue <-> vermillion, near-white neutral midpoint).
CMAP_NRBR = LinearSegmentedColormap.from_list(
    "nrbr", ["#D55E00", "#F7F7F5", "#0072B2"])
#: Satellite VIS reflectance — single-hue blue ramp, dark ground -> bright cloud.
CMAP_VIS = LinearSegmentedColormap.from_list(
    "vis_blue", ["#0D366B", "#1C5CAB", "#3987E5", "#9EC5F4", "#E8F1FC"])
#: Fisheye lens mask radius as a fraction of the inscribed-circle radius.
LENS_RADIUS_FRAC = 0.95
#: Station direct-label typography (relief for sub-3:1 palette contrast).
STATION_PT = 7.5
LEADER_KW = dict(arrowstyle="-", color="gray", lw=0.6)

#: Panel (c) data window (UTC) — two local days with real broken-cloud ramps.
TS_START = "2014-02-26 08:00:00+00:00"
TS_END = "2014-02-28 08:00:00+00:00"
#: Sky-frame search window (UTC, daylight PST) + target broken-cloud fraction.
FRAME_WIN = ("2014-02-26T14", "2014-02-28T02")
FRAME_CLOUD_TARGET = 0.35
SEVIRI_VIS006 = 7       # channel index in satellite_future.zarr context

STD = _REPO_ROOT / "data" / "standardized"
OUT_DIR = _REPO_ROOT / "scratch"
OUT_STEM = "fig01_dataset_showcase"

mpl.rcParams.update({
    "font.family": FONT_FAMILY,
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT_FAMILY,
    "mathtext.it": f"{FONT_FAMILY}:italic",
    "mathtext.bf": f"{FONT_FAMILY}:bold",
    "mathtext.default": "rm",
    "font.size": TEXT_SIZE_PT,
    "axes.labelsize": TEXT_SIZE_PT,
    "axes.titlesize": PANEL_LABEL_PT,
    "xtick.labelsize": TEXT_SIZE_PT,
    "ytick.labelsize": TEXT_SIZE_PT,
    "legend.fontsize": SMALL_PT,
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
    "pdf.fonttype": 42,
})
mm = 1 / 25.4


# ===========================================================================
# Data loading — REAL files only; every loader raises if a file is missing.
# ===========================================================================
def load_metadata() -> pd.DataFrame:
    frames = []
    for src in ("skippd", "folsom", "sunlake", "gefcom2014"):
        p = STD / src / "metadata.parquet"
        if not p.exists():
            raise FileNotFoundError(p)
        frames.append(pd.read_parquet(p))
    meta = pd.concat(frames, ignore_index=True)
    if len(meta) != 11:
        raise ValueError(f"expected 11 entities, got {len(meta)}")
    return meta


def pick_sky_frame() -> tuple[np.ndarray, str]:
    """Deterministically pick the most representative broken-cloud Folsom frame.

    Scans the 2014 sky-image zarr inside the panel-(c) UTC window (daylight
    PST), computes the cloudy-pixel fraction from the real NRBR channel and
    returns the frame whose fraction is closest to FRAME_CLOUD_TARGET.
    """
    store = STD / "folsom" / "sky_images_2014.zarr"
    g = zarr.open_group(store, mode="r")
    ts = [str(t) for t in g["images_timestamps"][:]]
    best = None
    for i, t in enumerate(ts):
        if not (FRAME_WIN[0] <= t[:13] <= FRAME_WIN[1]):
            continue
        rgb = np.asarray(Image.open(io.BytesIO(bytes(g["images"][i]))))
        if rgb.ndim != 3 or rgb.mean() < 50:          # skip night / malformed
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
    """SEVIRI VIS006 context frame with the most cloud structure (max std)."""
    store = STD / "sunlake" / "satellite_future.zarr"
    g = zarr.open_group(store, mode="r")
    ctx = g["context"][:]                     # (M, 11, 64, 64) float32
    vis = ctx[:, SEVIRI_VIS006]
    i = int(np.argmax(vis.std(axis=(1, 2))))
    print(f"[satellite] context idx={i} VIS006 std={vis[i].std():.3f} "
          f"range=[{vis[i].min():.2f}, {vis[i].max():.2f}]")
    return vis[i]


def load_folsom_slice() -> pd.DataFrame:
    p = STD / "folsom" / "time_series.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_parquet(p)
    df = df.set_index("timestamp_utc").sort_index()
    sl = df.loc[TS_START:TS_END].copy()
    if len(sl) < 2800:
        raise ValueError(f"expected ~2880 rows, got {len(sl)}")
    return sl


# ===========================================================================
# Panel (a) — station map
# ===========================================================================
def draw_map(fig, gs) -> plt.Axes:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    ax = fig.add_subplot(gs, projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor=OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor=LAND, zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"),
                   linewidth=0.4, edgecolor=COAST, zorder=1.5)
    ax.spines["geo"].set_edgecolor(COAST)
    ax.spines["geo"].set_linewidth(0.5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(),
                      xlocs=[-120, -60, 0, 60, 120, 180],
                      ylocs=[-60, -30, 0, 30, 60],
                      color=GRID_COLOR, linewidth=GRID_WIDTH, linestyle="-",
                      zorder=1.2)
    gl.rotate_labels = False

    def _site(src, entity):
        row = meta[(meta.dataset_source == src)
                   & (meta.entity_id == entity)].iloc[0]
        return row.longitude, row.latitude

    # -- markers (real coordinates; close pairs intentionally overplot) -----
    for src, grp in meta.groupby("dataset_source"):
        ax.scatter(grp.longitude, grp.latitude, s=16,
                   c=DS_COLORS[src], marker=DS_MARKERS[src],
                   edgecolors="white", linewidths=0.5, zorder=5,
                   transform=ccrs.PlateCarree())

    # -- direct labels (relief for sub-3:1 palette contrast) ----------------
    # Every displaced label: 7.5 pt bold black text, and a grey leader drawn
    # as ONE FULL LINE from the marker's CENTRE to the text's CENTRE (under
    # the glyphs, ending beneath the middle letter) — no bbox clipping.
    tkw = dict(fontsize=STATION_PT, fontweight="bold", color="#000000",
               ha="center", va="center", zorder=6,
               transform=ccrs.PlateCarree())

    def _leader(lon0, lat0, lon1, lat1):
        ax.plot([lon0, lon1], [lat0, lat1], color=LEADER_KW["color"],
                lw=LEADER_KW["lw"], zorder=4.8,
                transform=ccrs.PlateCarree())

    def _label(text, site_xy, lx, ly):
        x, y = site_xy
        _leader(x, y, lx, ly)
        ax.text(lx, ly, text, **tkw)

    # California pair — SKIPP'D (Stanford, southern marker) and Folsom
    # (~1.5 deg NE of it) nearly overplot at this scale, so their leaders
    # leave in OPPOSITE directions (SW vs NE), each from its own marker.
    _label("SKIPP'D", _site("skippd", "SKIPPD_STFORD"), -137, 31)
    _label("Folsom", _site("folsom", "FOLSOM_STN"), -112, 45)
    # GEFCom2014 zones co-located in SE Australia — one label for the
    # cluster, lifted two text-line heights so the second row clears the
    # markers (they sit at lat -33.4..-34.1). No leader (cluster label).
    ax.text(163, -18, "GEFCom2014\n(zones 1–3)", **tkw)
    # SunLake BSRN — six leader lines, marker centre -> text centre.
    # PAL and CNR sit horizontally right of their markers at the SAME
    # text-to-marker distance as the CAB/PAY Atlantic pair (~27 deg),
    # and stay vertically spread apart.
    for ent, code, lx, ly in (
            ("SUNLAKE_CAB", "CAB", -22.0, 51.5),   # ~27.0 deg from marker
            ("SUNLAKE_PAL", "PAL", 29.2, 51.0),    # ~27.1 deg from marker
            ("SUNLAKE_PAY", "PAY", -20.0, 43.5),   # ~27.1 deg from marker
            ("SUNLAKE_CNR", "CNR", 28.6, 40.5),    # ~27.1 deg from marker
            ("SUNLAKE_IZA", "IZA", 30.0, 30.0),
            ("SUNLAKE_TAM", "TAM", -11.0, 19.5)):
        _label(code, _site("sunlake", ent), lx, ly)

    handles = [Line2D([], [], color=DS_COLORS[s], marker=DS_MARKERS[s],
                      linestyle="none", markersize=4.5,
                      markeredgecolor="white", markeredgewidth=0.4,
                      label=DS_LABELS[s])
               for s in ("skippd", "folsom", "gefcom2014", "sunlake")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.005, 0.01),
              frameon=False, handletextpad=0.2, labelspacing=0.35,
              borderaxespad=0.0)
    ax.set_title("(a) Station distribution across the four benchmark datasets",
                 loc="left", fontweight="bold", pad=4)
    return ax


# ===========================================================================
# Panel (b) — multi-modal imagery
# ===========================================================================
def draw_images(fig, gs) -> None:
    # Row 0: three square image axes; row 1: horizontal colorbars under
    # (b2)/(b3) — vertical colorbars would bury their tick labels under the
    # neighbouring opaque image axes.
    sub = gs.subgridspec(2, 3, height_ratios=[1, 0.05], hspace=0.14)

    ax_rgb = fig.add_subplot(sub[0, 0])
    ax_nrbr = fig.add_subplot(sub[0, 1])
    ax_vis = fig.add_subplot(sub[0, 2])
    ax_cnrbr = fig.add_subplot(sub[1, 1])
    ax_cvis = fig.add_subplot(sub[1, 2])
    for ax in (ax_rgb, ax_nrbr, ax_vis):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_color(COAST)
            sp.set_linewidth(0.4)

    # (b1) sky camera RGB — real frame (timestamp in local time, matching c).
    ft_local = pd.Timestamp(frame_ts).tz_convert("America/Los_Angeles")
    ax_rgb.imshow(rgb_frame)
    ax_rgb.set_xlabel("(b1) Sky camera RGB\nFolsom, "
                      + ft_local.strftime("%d %b %Y, %H:%M PST"),
                      fontsize=SMALL_PT, color=INK_2, labelpad=15)

    # (b2) NRBR computed from the same real frame: (B - R) / (B + R + 1e-6).
    r = rgb_frame[..., 0].astype(np.float32)
    b = rgb_frame[..., 2].astype(np.float32)
    nrbr = (b - r) / (b + r + 1e-6)
    # Circular lens mask: fisheye sky only — pixels beyond the inscribed
    # circle (camera housing / horizon ring) become NaN and render white,
    # removing the noisy orange boundary band.
    hh, ww = nrbr.shape
    yy, xx = np.mgrid[0:hh, 0:ww]
    cy, cx = (hh - 1) / 2.0, (ww - 1) / 2.0
    lens_r = LENS_RADIUS_FRAC * min(hh, ww) / 2.0
    nrbr[(yy - cy) ** 2 + (xx - cx) ** 2 > lens_r ** 2] = np.nan
    im_n = ax_nrbr.imshow(nrbr, cmap=CMAP_NRBR,
                          norm=TwoSlopeNorm(vcenter=0.0, vmin=-0.4, vmax=0.8))
    cbar_n = fig.colorbar(im_n, cax=ax_cnrbr, orientation="horizontal")
    cbar_n.set_ticks([-0.4, 0.0, 0.4, 0.8])
    cbar_n.ax.tick_params(labelsize=7, width=0.4, length=2.5, pad=1.5)
    cbar_n.outline.set_linewidth(0.4)
    cbar_n.ax.set_xlabel("(b2) NRBR $(B-R)/(B+R)$", fontsize=SMALL_PT,
                         color=INK_2, labelpad=2)

    # (b3) SEVIRI VIS006 satellite context patch (SunLake, 64x64, nearest).
    im_v = ax_vis.imshow(vis_patch, cmap=CMAP_VIS, vmin=0.0, vmax=0.9,
                         interpolation="nearest")
    cbar_v = fig.colorbar(im_v, cax=ax_cvis, orientation="horizontal")
    cbar_v.set_ticks([0.0, 0.45, 0.9])
    cbar_v.ax.tick_params(labelsize=7, width=0.4, length=2.5, pad=1.5)
    cbar_v.outline.set_linewidth(0.4)
    cbar_v.ax.set_xlabel("(b3) SEVIRI VIS006, SunLake patch",
                         fontsize=SMALL_PT, color=INK_2, labelpad=2)

    # Panel header spans the three image axes.
    ax_rgb.set_title("(b) Multi-modal observations", loc="left",
                     fontweight="bold", pad=4)


# ===========================================================================
# Panel (c) — GHI + clearness index on stacked, x-aligned panels
# ===========================================================================
def draw_timeseries(fig, gs_ghi, gs_kt) -> None:
    ax_ghi = fig.add_subplot(gs_ghi)
    ax_kt = fig.add_subplot(gs_kt, sharex=ax_ghi)

    local = sl.index.tz_convert("America/Los_Angeles")
    x_hours = (local - local[0]) / np.timedelta64(1, "h")

    # -- GHI: observed (entity color) + clear-sky reference (dashed ink) ----
    ax_ghi.plot(x_hours, sl["ghi_actual"].to_numpy(),
                color=DS_COLORS["folsom"], lw=0.7, label="Observed GHI")
    ax_ghi.plot(x_hours, sl["ghi_clearsky"].to_numpy(),
                color=INK_2, lw=0.6, ls=(0, (4, 2)), label="Clear-sky GHI")
    ax_ghi.set_ylabel(r"GHI (W m$^{-2}$)")
    ax_ghi.set_ylim(0, 1000)
    ax_ghi.set_yticks([0, 250, 500, 750, 1000])
    ax_ghi.legend(loc="upper left", frameon=False, borderaxespad=0.2,
                  handlelength=1.8)
    ax_ghi.set_title("(c) Folsom 1-min GHI and clearness index, "
                     "26–27 Feb 2014 (local time)",
                     loc="left", fontweight="bold", pad=4)

    # Day labels along the top edge of the GHI panel.
    for d, xlab in ((0, "26 Feb"), (1, "27 Feb")):
        ax_ghi.text(d * 24 + 12, 940, xlab, ha="center", va="top",
                    fontsize=SMALL_PT, color=INK_MUTED)

    # Sky-frame marker — the (b1)/(b2) frame lives at this timestamp.
    ft_local = pd.Timestamp(frame_ts).tz_convert("America/Los_Angeles")
    xf = (ft_local - local[0]) / np.timedelta64(1, "h")
    for ax in (ax_ghi, ax_kt):
        ax.axvline(xf, color=INK, lw=0.6, ls=(0, (1, 2)), zorder=4)
    ax_ghi.annotate("frame in (b)", xy=(xf, 860), xytext=(xf - 9.5, 860),
                    fontsize=7, color=INK, va="center", ha="right",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=INK_MUTED))

    # -- clearness index: NaN below the 10 W m-2 clear-sky threshold so the
    # line is truly broken overnight (boolean-indexing would splice the dusk
    # and dawn endpoints into one straight diagonal segment).
    kt_plot = sl["clearness_index"].to_numpy(dtype=float).copy()
    kt_plot[sl["ghi_clearsky"].to_numpy() < 10.0] = np.nan
    ax_kt.plot(x_hours, kt_plot, color=INK_2, lw=0.5)
    ax_kt.axhline(1.0, color=INK_MUTED, lw=0.5, ls=(0, (4, 2)))
    ax_kt.text(47.6, 1.03, r"$k_t=1$ (clear)", fontsize=7, color=INK_MUTED,
               ha="right", va="bottom")
    ax_kt.set_ylabel(r"$k_t$ (–)")
    ax_kt.set_ylim(0, 1.6)
    ax_kt.set_yticks([0, 0.5, 1.0, 1.5])
    # One shared x-axis: both sub-axes aligned to identical limits.
    ax_kt.set_xlim(0, 48)
    ax_ghi.set_xlim(0, 48)
    ax_kt.set_xticks(np.arange(0, 49, 6))
    ax_kt.set_xticklabels(["00:00", "06:00", "12:00", "18:00", "00:00",
                           "06:00", "12:00", "18:00", "00:00"])
    ax_kt.set_xlabel("Local time (PST)")


# ===========================================================================
# Assemble
# ===========================================================================
print("loading real dataset files ...")
meta = load_metadata()
rgb_frame, frame_ts = pick_sky_frame()
vis_patch = pick_satellite_patch()
sl = load_folsom_slice()
print(f"[ts] rows={len(sl)} ghi_max={sl.ghi_actual.max():.0f} "
      f"kt_daytime_mean="
      f"{sl.clearness_index[sl.ghi_clearsky > 20].mean():.2f}")

import cartopy.crs as ccrs  # noqa: F401  (env sanity: cartopy importable)

fig = plt.figure(figsize=(WIDTH_MM * mm, HEIGHT_MM * mm))
gs = fig.add_gridspec(4, 1, height_ratios=[1.06, 0.94, 0.66, 0.36],
                      left=0.055, right=0.975, top=0.975, bottom=0.065,
                      hspace=0.52)

draw_map(fig, gs[0])
draw_images(fig, gs[1])
draw_timeseries(fig, gs[2], gs[3])

OUT_DIR.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=DPI_RASTER,
                bbox_inches="tight")
    print(f"  {OUT_STEM}.{ext}")
plt.close(fig)
print(f"done -> {OUT_DIR / (OUT_STEM + '.png')}")
