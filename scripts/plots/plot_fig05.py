"""Paper fig05 — QC flow on ONE selected Folsom frame (paper sample).

Deterministic regeneration of the chosen batch candidate
QC_flow_folsom_20140226_005013.png (2014-02-26 00:50:13 UTC) from
dev/plot_fig05_qc_candidates.py's v2 method:
  (a) raw ASI frame with the 25-px glare circle on the MEASURED whiteout
      centre (calibrated geometric projection rot=172.8/scale=0.880 is the
      prior/fallback);
  (b) grey frame + vermillion glare-mask overlay;
  (c) uncurated Farneback optical flow (vermillion quiver);
  (d) quality-assured flow (glare region masked, blue quiver).

Outputs (300 dpi):
  figures/paper/fig05.png / .pdf        the single merged 1x4 figure (a-d)

Run (conda env solar_uni):
    python scripts/plots/plot_fig05.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
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
    "font.family": "Arial", "font.size": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6,
    "axes.facecolor": "white", "figure.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42,
})

STD = _REPO_ROOT / "data" / "standardized"
OUT = _REPO_ROOT / "figures" / "paper"
TARGET_TS = pd.Timestamp("2014-02-26 00:50:13")      # the paper sample
CALIB = {"camera_rotation_deg": 172.8, "camera_radius_scale": 0.880}
PT_TITLE, STEP, SCALE = 8.5, 8, 0.25
VERM, BLUE = "#D55E00", "#0072B2"
VERM_RGB = (0xD5, 0x5E, 0x00)
GLARE_R = 25                                         # full-res px


def decode(g, i):
    return np.asarray(Image.open(io.BytesIO(bytes(g["images"][i]))))


def glare_blob(gray):
    sat = (gray >= 250).astype(np.uint8)
    n, _l, stats, cents = cv2.connectedComponentsWithStats(sat, 8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return stats[k, cv2.CC_STAT_AREA] / gray.size, cents[k]


def sun_uv_prior(img, az, z, radius):
    qc = VisualQCPipeline({"mode": "dry_run", "modality": "asi",
                           "glare_radius_px": radius, **CALIB})
    return qc.process_frame(img, solar_azimuth_deg=az,
                            solar_zenith_deg=z, modality="asi").sun_uv


def farneback(a, b):
    fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    return fl[..., 0].astype(np.float32), fl[..., 1].astype(np.float32)


def quiver(ax, u, v, color):
    h, w = u.shape
    yy, xx = np.mgrid[0:h:STEP, 0:w:STEP]
    ax.quiver(xx, yy, u[::STEP, ::STEP], v[::STEP, ::STEP], color=color,
              pivot="mid", scale=SCALE, scale_units="xy", width=0.004,
              headwidth=3.8, headlength=4.8, minshaft=1.5)
    ax.set_xlim(0, w); ax.set_ylim(h, 0)


def build_mask(img, az, z, radius=GLARE_R):
    """Glare mask centred EXACTLY on the measured whiteout (prior fallback)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blob = glare_blob(gray)
    h, w = gray.shape
    if blob is not None:
        cx, cy = float(blob[1][0]), float(blob[1][1])
    else:
        cx, cy = sun_uv_prior(img, az, z, radius)
    mask = np.zeros((h, w), dtype=bool)
    cv2.circle(mask.view(np.uint8), (int(round(cx)), int(round(cy))),
               radius, 1, thickness=-1)
    return mask, (cx, cy)


def _frame_ax(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#333333"); sp.set_linewidth(0.6)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    g = zarr.open_group(str(STD / "folsom" / "sky_images_2014.zarr"), "r")
    ts = pd.to_datetime([str(t) for t in g["images_timestamps"][:]],
                        utc=True).tz_localize(None)
    i = int(np.argmin(np.abs((ts - TARGET_TS).total_seconds().to_numpy())))
    t = ts[i]
    print(f"[frame] idx {i} @ {t} (dt="
          f"{abs((t - TARGET_TS).total_seconds()):.0f} s)", flush=True)

    tab = pd.read_parquet(STD / "folsom" / "time_series.parquet")
    tab = tab.set_index("timestamp_utc").sort_index()
    tab.index = tab.index.tz_localize(None)
    zen = np.degrees(np.arccos(np.clip(tab["cos_zenith"], -1, 1)))
    geo = {tt.round("min"): (float(a), float(z2)) for tt, a, z2 in
           zip(tab.index, tab["solar_azimuth"], zen)}
    az, z = geo[t.round("min")]

    img = decode(g, i)
    mask, (cx, cy) = build_mask(img, az, z)
    u, v = farneback(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
                     cv2.cvtColor(decode(g, i + 1), cv2.COLOR_RGB2GRAY))
    u_q, v_q = u.copy(), v.copy()
    u_q[mask], v_q[mask] = np.nan, np.nan
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    ov = gray_rgb.copy()
    ov[mask] = VERM_RGB

    def _save(fig, name, pdf=False):
        fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight",
                    pad_inches=0.02)
        if pdf:
            fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  {name}.png" + (" + .pdf" if pdf else ""), flush=True)

    # ---- the single merged 1x4 figure ----
    fig, axs = plt.subplots(1, 4, figsize=(180 / 25.4, 50 / 25.4))
    for ax in axs:
        _frame_ax(ax)
    axs[0].imshow(img)
    axs[0].add_patch(plt.Circle((cx, cy), GLARE_R, fill=False, ec=VERM,
                                lw=1.2))
    axs[0].set_title("(a) Raw ASI frame with glare", fontsize=PT_TITLE,
                     loc="left", pad=3)
    axs[1].imshow(gray_rgb); axs[1].imshow(ov, alpha=0.4)
    axs[1].set_title("(b) Geometric glare mask", fontsize=PT_TITLE,
                     loc="left", pad=3)
    axs[2].imshow(gray, cmap="gray"); quiver(axs[2], u, v, VERM)
    axs[2].set_title("(c) Uncurated optical flow", fontsize=PT_TITLE,
                     loc="left", pad=3)
    axs[3].imshow(gray, cmap="gray"); quiver(axs[3], u_q, v_q, BLUE)
    axs[3].set_title("(d) Quality-assured flow", fontsize=PT_TITLE,
                     loc="left", pad=3)
    fig.tight_layout(w_pad=0.9)
    _save(fig, "fig05", pdf=True)
    print(f"done -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


run = main
