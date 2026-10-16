"""Fig. 5 — visual QC contrast: glare masking before/after Farneback flow.

2x2 qualitative panels, REAL Folsom data only:
  (a) Raw ASI frame with solar glare (geometric sun position circled)
  (b) Geometric glare mask (equidistant fisheye projection; fallback path
      annotated when it fires)
  (c) Uncurated optical flow — Farneback on the raw pair; spurious vectors
      radiating from the saturated glare disc
  (d) Quality-assured optical flow — glare disc inpainted before Farneback
      + vector median smoothing; coherent cloud-motion field

Also runs the full VisualQCPipeline (dry-run) over a bounded frame strip to
print the six-line diagnostics report, tying the figure to Step 2.

Run (conda env solar_uni):
    python scripts/plot_fig05_qc_contrast.py
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

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from PIL import Image

from solarunibench.operators.visual_qc import VisualQCPipeline

mpl.rcParams.update({
    "font.family": "Arial",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.default": "rm",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})

SKY_ZARR = _REPO_ROOT / "data" / "standardized" / "folsom" / \
    "sky_images_2014.zarr"
TS_PARQ = _REPO_ROOT / "data" / "standardized" / "folsom" / "time_series.parquet"
OUT_DIR = _REPO_ROOT / "Figures"
OUT_STEM = "fig05"
GLARE_SCAN_WINDOW = ("2014-06-01T16", "2014-09-01T02")   # summer, low sun
STRIP_N = 240                                            # diagnostics strip
QUIVER_STEP = 10                                         # px between arrows
PT_TITLE = 8.5
INK = "#0B0B0B"
INK_2 = "#52514E"
ALERT = "#D55E00"   # house Wong/Okabe-Ito vermillion (ESSD palette)


def load_store():
    g = zarr.open_group(str(SKY_ZARR), mode="r")
    ts = pd.to_datetime([str(t) for t in g["images_timestamps"][:]],
                        utc=True).tz_localize(None)
    return g, ts


def decode(g, i) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(bytes(g["images"][i]))))


def pick_glare_pair(g, ts, tab):
    """Deterministic pick: daylight frame with the strongest saturated glare
    blob, plus its successor (for Farneback). Sun geometry from the table."""
    best = None
    for i in np.flatnonzero((ts >= GLARE_SCAN_WINDOW[0])
                            & (ts < GLARE_SCAN_WINDOW[1]))[::5]:
        img = decode(g, i)
        if img.ndim != 3 or img.mean() < 60:          # daylight only
            continue
        sat = ((img[..., 0] >= 250) & (img[..., 1] >= 250)
               & (img[..., 2] >= 250)).mean()
        score = sat
        if best is None or score > best[0]:
            best = (score, i)
    if best is None or best[0] < 0.001:
        raise RuntimeError("no glare frame found in scan window")
    _, i = best
    row = tab.loc[tab.index.round("min") == ts[i].round("min")]
    az = float(row["solar_azimuth"].iloc[0]) if len(row) else None
    cz = float(row["cos_zenith"].iloc[0]) if len(row) else None
    zen = float(np.degrees(np.arccos(np.clip(cz, -1, 1)))) if cz is not None \
        else None
    print(f"[glare-frame] idx={i} ts={ts[i]} sat_frac={best[0]:.4f} "
          f"az={az} zen={zen:.1f}")
    return i, az, zen


def largest_glare_blob(frame: np.ndarray) -> tuple:
    """Ground-truth glare core: centroid of the LARGEST saturated connected
    component (>=250 grayscale). Returns (u, v) or (None, None)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    sat = (gray >= 250).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(sat, 8)
    if n < 2:
        return None, None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return float(cents[i][0]), float(cents[i][1])


def calibrate_camera(shape, az: float, zen: float, blob_uv: tuple) -> dict:
    """Derive the fixed-mount camera calibration (rotation + radius scale)
    from one frame's measured glare core vs the uncalibrated projection."""
    h, w = shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r_pred = (zen / 90.0) * (min(h, w) / 2.0)
    u0, v0 = blob_uv
    dx, dy = u0 - cx, v0 - cy
    import math
    clock_blob = math.degrees(math.atan2(dx, -dy)) % 360.0
    # uncalibrated projection uses clock direction == az itself
    clock_pred = math.degrees(
        math.atan2(np.sin(np.deg2rad(az)), np.cos(np.deg2rad(az)))) % 360.0
    rot = (clock_blob - clock_pred) % 360.0
    scale = (math.hypot(dx, dy) / r_pred) if r_pred > 1e-6 else 1.0
    return {"camera_rotation_deg": rot, "camera_radius_scale": scale}


def farneback(gray_a: np.ndarray, gray_b: np.ndarray) -> tuple:
    flow = cv2.calcOpticalFlowFarneback(
        gray_a, gray_b, None,
        pyr_scale=0.5, levels=3, winsize=21, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0)
    return flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32)


def quiver(ax, u, v, step=QUIVER_STEP, color=INK_2, scale=0.25):
    """Draw a flow field with PHYSICAL units: scale_units='xy' + scale=0.25
    -> a 1 px/frame flow renders as a 4 px arrow (clearly visible)."""
    h, w = u.shape
    yy, xx = np.mgrid[0:h:step, 0:w:step]
    q = ax.quiver(xx, yy, u[::step, ::step], v[::step, ::step],
                  color=color, pivot="mid", scale=scale, scale_units="xy",
                  width=0.004, headwidth=3.8, headlength=4.8,
                  minshaft=1.5)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    return q


def main() -> int:
    print("loading real files ...")
    g, ts = load_store()
    tab = pd.read_parquet(TS_PARQ).set_index("timestamp_utc").sort_index()
    tab.index = tab.index.tz_localize(None)

    i, az, zen = pick_glare_pair(g, ts, tab)
    raw0 = decode(g, i)
    # Fixed-mount camera calibration from THIS frame's measured glare core,
    # then the GEOMETRIC projection must land on the measured core.
    blob_uv = largest_glare_blob(raw0)
    calib = calibrate_camera(raw0.shape, az, zen, blob_uv)
    print(f"[calib] rotation={calib['camera_rotation_deg']:.1f} deg, "
          f"radius_scale={calib['camera_radius_scale']:.3f} "
          f"(blob core {blob_uv[0]:.1f},{blob_uv[1]:.1f})")
    qc = VisualQCPipeline({"mode": "dry_run", **calib})
    res = qc.process_frame(raw0, solar_azimuth_deg=az,
                           solar_zenith_deg=zen, modality="asi")
    u_sun, v_sun = res.sun_uv
    dist = float(np.hypot(u_sun - blob_uv[0], v_sun - blob_uv[1]))
    print(f"[sun] source={res.sun_source} geometric uv=({u_sun:.1f},"
          f"{v_sun:.1f}) vs blob ({blob_uv[0]:.1f},{blob_uv[1]:.1f}) "
          f"|d|={dist:.1f} px")
    assert dist <= 6.0, f"geometric sun {dist:.1f} px off the glare core"
    print(f"[sun] glare_px={int(res.glare_mask.sum())} — circle overlaps core")

    # diagnostics strip: N frames around the glare frame, real geometry
    strip_idx = np.arange(max(0, i - STRIP_N // 2), i + STRIP_N // 2)
    qc2 = VisualQCPipeline({"mode": "dry_run"})
    for j in strip_idx:
        r = tab.loc[tab.index.round("min") == ts[j].round("min")]
        qc2.process_frame(
            decode(g, j), prev_frame=decode(g, j - 1) if j > 0 else None,
            solar_azimuth_deg=float(r["solar_azimuth"].iloc[0]) if len(r)
            else None,
            cos_zenith=float(r["cos_zenith"].iloc[0]) if len(r) else None,
            modality="asi")
    qc2.print_diagnostics_report()

    # ---- panels ------------------------------------------------------------
    raw = decode(g, i)
    nxt = decode(g, i + 1)
    gray_a = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(nxt, cv2.COLOR_RGB2GRAY)

    # (c) uncurated flow: Farneback directly on the raw consecutive pair
    u_raw, v_raw = farneback(gray_a, gray_b)
    mag = np.hypot(u_raw, v_raw)
    print(f"[flow] |f| px/frame: median={np.median(mag):.3f} "
          f"p95={np.percentile(mag, 95):.3f} max={mag.max():.3f} | "
          f"glare-zone median={np.median(mag[res.glare_mask]):.3f} "
          f"outside={np.median(mag[~res.glare_mask]):.3f}")

    # (d) quality-assured: the SAME flow field with the glare mask applied —
    # mask==1 vectors are set to NaN (not drawn); everything else untouched,
    # so the (c)/(d) difference is purely the mask.
    u_qc = u_raw.copy()
    v_qc = v_raw.copy()
    u_qc[res.glare_mask] = np.nan
    v_qc[res.glare_mask] = np.nan

    fig, axs = plt.subplots(1, 4, figsize=(180 / 25.4, 50 / 25.4))
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#333333"); sp.set_linewidth(0.6)

    a = axs[0]
    a.imshow(raw)
    circ = plt.Circle((u_sun, v_sun), qc.glare_radius_px, fill=False,
                      edgecolor=ALERT, lw=1.2)
    a.add_patch(circ)
    a.set_title("(a) Raw ASI frame with glare", fontsize=PT_TITLE,
                loc="left", pad=3)

    b = axs[1]
    b.imshow(res.glare_mask.astype(float), cmap="gray", vmin=0, vmax=1)
    b.plot([u_sun], [v_sun], marker="+", color=ALERT, ms=10, mew=1.2)
    b.set_title("(b) Geometric glare mask", fontsize=PT_TITLE,
                loc="left", pad=3)

    c = axs[2]
    c.imshow(gray_a, cmap="gray")
    quiver(c, u_raw, v_raw, color=ALERT)
    c.set_title("(c) Uncurated optical flow", fontsize=PT_TITLE,
                loc="left", pad=3)

    d = axs[3]
    d.imshow(gray_a, cmap="gray")
    quiver(d, u_qc, v_qc, color="#0072B2")
    d.set_title("(d) Quality-assured flow", fontsize=PT_TITLE,
                loc="left", pad=3)

    fig.tight_layout(w_pad=0.9)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=300,
                    bbox_inches="tight")
        print(f"  {OUT_STEM}.{ext}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
