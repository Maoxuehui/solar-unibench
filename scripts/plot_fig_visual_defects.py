"""Visual defects & anomalies — REAL-frame gallery (2x2), house style.

  (a) Sun glare saturating camera channels — real Folsom frame with the
      saturated-pixel mask (RGB >= 250) overlaid in vermillion.
  (b) Non-sky occlusion — real low-sun frame whose bottom fisheye band is
      100% non-blue (buildings / horizon); occlusion mask overlaid, with the
      lens-exclusion circle (0.95 r) drawn.
  (c) Temporal discontinuity — the largest SAME-DAY DAYLIGHT store gap:
      before / after frames side by side, gap duration annotated.
      (Corrupted frames: 0 in 4,177-frame audit — the ingestion soft-fail
       skips them; documented in the caption, not shown.)
  (d) Camera orientation — monthly solar-position fits (glare-based
      rotation) across 2014-2016: a seasonal-periodic residual consistent
      with azimuth-model error, NOT mechanical drift; written to CSV.

Run (conda env solar_uni):
    python scripts/plot_fig_visual_defects.py
"""
from __future__ import annotations

import io
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib as mpl

mpl.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from PIL import Image

mpl.rcParams.update({
    "font.family": "Arial", "font.size": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": "#E5E5E5", "grid.linewidth": 0.3,
    "axes.axisbelow": True, "xtick.minor.visible": False,
    "ytick.minor.visible": False, "axes.facecolor": "white",
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})

STD = _REPO_ROOT / "data" / "standardized"
OUT_FIG = _REPO_ROOT / "Figures" / "fig_visual_defects"
ORIENT_CSV = _REPO_ROOT / "figures_full" / "results" / "results_camera_orientation.csv"
VERM = "#D55E00"
GREEN = "#009E73"
PT_TITLE = 9
GLARE_IDX = 158935            # 2014-08-14 00:00:38 UTC (fig05's frame)
OCCL_IDX = 54000              # 2014-03-27 15:54:59 UTC (bottom band 100% non-blue)


def store(year: int):
    g = zarr.open_group(str(STD / "folsom" / f"sky_images_{year}.zarr"), "r")
    ts = pd.to_datetime([str(t) for t in g["images_timestamps"][:]],
                        utc=True).tz_localize(None)
    return g, ts


def dec(g, i):
    return np.asarray(Image.open(io.BytesIO(bytes(g["images"][i]))))


def frame_axis(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_color("#333333"); sp.set_linewidth(0.6)


def overlay(ax, img, mask, alpha=0.40):
    ax.imshow(img)
    ov = img.copy()
    ov[mask] = (0xD5, 0x5E, 0x00)
    ax.imshow(ov, alpha=alpha)


def panel_a(ax):
    g, _ = store(2014)
    img = dec(g, GLARE_IDX)
    sat = (img[..., 0] >= 250) & (img[..., 1] >= 250) & (img[..., 2] >= 250)
    overlay(ax, img, sat)
    ax.set_title("(a) Sun glare", fontsize=PT_TITLE, loc="left", pad=3)


def panel_b(ax):
    g, _ = store(2014)
    img = dec(g, OCCL_IDX)
    h, w, _ = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = (xx - (w - 1) / 2) ** 2 + (yy - (h - 1) / 2) ** 2
    ring = r2 > (0.80 * min(h, w) / 2) ** 2            # outer fisheye ring
    nonblue = img[..., 2] <= img[..., 0] + 10
    occ = ring & nonblue
    overlay(ax, img, occ)
    circ = plt.Circle(((w - 1) / 2, (h - 1) / 2), 0.95 * min(h, w) / 2,
                      fill=False, ec=GREEN, lw=1.1, ls=(0, (4, 2)))
    ax.add_patch(circ)
    ax.set_title("(b) Non-sky occlusion", fontsize=PT_TITLE, loc="left",
                 pad=3)


def find_daylight_gap():
    """Largest same-day, daylight store gap in 2014 (5-120 min)."""
    g, ts = store(2014)
    tab = pd.read_parquet(STD / "folsom" / "time_series.parquet")
    tab = tab.set_index("timestamp_utc").sort_index()
    tab.index = tab.index.tz_localize(None)
    geo = {t.round("min"): float(z) for t, z in zip(
        tab.index, np.degrees(np.arccos(np.clip(tab["cos_zenith"], -1, 1))))}
    best = None
    for i in range(1, len(ts)):
        dt = (ts[i] - ts[i - 1]).total_seconds()
        if not (300 <= dt <= 7200):
            continue
        if ts[i].date() != ts[i - 1].date():
            continue
        z0 = geo.get(ts[i - 1].round("min"), 90.0)
        z1 = geo.get(ts[i].round("min"), 90.0)
        if z0 >= 85 or z1 >= 85:                        # keep it daylight
            continue
        if best is None or dt > best[0]:
            best = (dt, i)
    dt, i = best
    print(f"[gap] {ts[i-1]} -> {ts[i]}  ({dt/60:.0f} min daylight gap)")
    return g, i - 1, i, dt


def panel_c(sub):
    g, i0, i1, dt = find_daylight_gap()
    ts = store(2014)[1]
    (axL, axR) = [fig.add_subplot(s) for s in sub]
    for ax, i, tag in ((axL, i0, "before"), (axR, i1, "after")):
        frame_axis(ax)
        ax.imshow(dec(g, i))
        ax.set_xlabel(f"{tag} {ts[i]:%H:%M}", fontsize=7.5, labelpad=1.5)
    axL.annotate("", xy=(1.06, 0.5), xytext=(-0.06, 0.5),
                 xycoords=axR.axes, textcoords=axL.axes,
                 arrowprops=dict(arrowstyle="->", lw=1.0, color="#0B0B0B"))
    axL.set_title("(c) Temporal gap", fontsize=PT_TITLE, loc="left", pad=3)
    axR.set_title(rf"$\Delta t$ = {dt/60:.0f} min", fontsize=8,
                  loc="right", pad=2)


def panel_d(ax):
    """Monthly glare-based rotation fits 2014-2016 (deterministic rescan)."""
    rots = {}
    for year in (2014, 2015, 2016):
        g, ts = store(year)
        tab = pd.read_parquet(STD / "folsom" / "time_series.parquet")
        tab = tab.set_index("timestamp_utc").sort_index()
        tab.index = tab.index.tz_localize(None)
        zen = np.degrees(np.arccos(np.clip(tab["cos_zenith"], -1, 1)))
        geo = {t.round("min"): (float(a), float(z2)) for t, a, z2 in
               zip(tab.index, tab["solar_azimuth"], zen)}
        idxs = [i for i, t in enumerate(ts)
                if 30 <= geo.get(t.round("min"), (None, 999))[1] <= 70]
        per_month = {}
        for i in idxs[:: max(1, len(idxs) // 3000)]:
            im = Image.open(io.BytesIO(bytes(g["images"][i])))
            im.draft("RGB", (128, 128))
            img = np.asarray(im)
            if img.mean() < 60:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            sat = (gray >= 250).astype(np.uint8)
            n, _l, stats, cents = cv2.connectedComponentsWithStats(sat, 8)
            if n < 2:
                continue
            k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            if stats[k, cv2.CC_STAT_AREA] / gray.size < 0.003:
                continue
            az, _z = geo.get(ts[i].round("min"), (None, None))
            if az is None:
                continue
            dx, dy = cents[k][0] - 63.5, cents[k][1] - 63.5
            cb = math.degrees(math.atan2(dx, -dy)) % 360
            cp = math.degrees(math.atan2(np.sin(np.deg2rad(az)),
                                         np.cos(np.deg2rad(az)))) % 360
            per_month.setdefault((ts[i].year, ts[i].month),
                                 []).append((cb - cp) % 360)
        for ym, vals in per_month.items():
            v = np.array(vals)
            med = np.median(v)
            keep = np.abs((v - med + 180) % 360 - 180) < 25
            ang = np.deg2rad(v[keep])
            rots[ym] = (math.degrees(math.atan2(np.sin(ang).mean(),
                                                np.cos(ang).mean())) % 360,
                        int(keep.sum()))
    ys = pd.to_datetime([f"{y}-{m:02d}-01" for y, m in rots])
    rv = np.array([r[0] for r in rots.values()])
    ax.plot(ys, rv, marker="o", ms=2.5, lw=0.7, color="#0072B2")
    ax.set_ylabel("Rotation (deg)", fontsize=8)
    ax.set_ylim(0, 360)
    ax.set_yticks([0, 90, 180, 270, 360])
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title("(d) Camera orientation — monthly solar fits\n"
                 "(seasonal residual, mount stable)", fontsize=PT_TITLE,
                 loc="left", pad=3)
    pd.DataFrame([{"year": y, "month": m, "rotation_deg": r, "n_frames": n}
                  for (y, m), (r, n) in sorted(rots.items())]
                 ).to_csv(ORIENT_CSV, index=False)
    print(f"wrote {ORIENT_CSV}")


def main() -> int:
    global fig
    fig = plt.figure(figsize=(180 / 25.4, 50 / 25.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.9, 1.25], wspace=0.28,
                          left=0.03, right=0.995, top=0.90, bottom=0.14)
    a = fig.add_subplot(gs[0]); frame_axis(a); panel_a(a)
    b = fig.add_subplot(gs[1]); frame_axis(b); panel_b(b)
    panel_c(gs[2].subgridspec(1, 2, wspace=0.22))
    panel_d(fig.add_subplot(gs[3]))

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_FIG}.{ext}", dpi=300, bbox_inches="tight")
        print(f"  {OUT_FIG.name}.{ext}")
    plt.close(fig)
    print(f"done -> {OUT_FIG}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
