"""Batch QC-flow candidate figures v2 — 30 Folsom samples, exact sun alignment
+ real cloud advection mining.

Fixes over v1 (user review):
  Flaw 1 (mask misalignment): the glare-mask CENTRE now uses the measured
    centre of the intense solar whiteout (largest saturated connected
    component, full-res), so (a)/(b) align EXACTLY with the observed sun.
    The calibrated geometric projection (rot=172.8 deg, scale=0.880) remains
    the PRIOR for qualification and the fallback when no blob is measurable.
  Flaw 2 (clear-sky frames): cloud metrics exclude a dilated sun
    neighbourhood; qualification requires (i) non-sun cloud fraction in
    [0.15, 0.6] AND (ii) mean Farneback velocity over the CLOUD region
    > 2.0 px/frame (full-res equivalent; measured on the half-res prefilter
    pair with threshold 1.0 px) — distinct moving cloud structure only.

Per user flow rules: collect until >30 qualify (early stop), random pick 30
with month/day diversity quotas; regenerate everything into
Figures/QC_Flow_Candidates/. For EVERY selected sample TWO figures are
saved: the current overlay version (+PDF) and one extra PNG using the
ORIGINAL fig05 method (panel (b) as a binary mask).

Run (conda env solar_uni):
    python scripts/plot_fig05_qc_candidates.py
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
    "font.family": "Arial", "font.size": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6,
    "axes.facecolor": "white", "figure.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42,
})

STD = _REPO_ROOT / "data" / "standardized"
OUT_DIR = _REPO_ROOT / "Figures" / "QC_Flow_Candidates"
SKYS = [STD / "folsom" / f"sky_images_{y}.zarr" for y in (2014, 2015, 2016)]
N_SAMPLES = 30
SEED = 0
ZEN_RANGE = (45.0, 65.0)
CLOUD_NS_RANGE = (0.15, 0.6)      # non-sun cloud fraction band
FLOW_MEAN_MIN = 1.0               # px/frame at HALF res (2.0 full-res equiv)
BLOB_MIN_FRAC = 0.0015
BLOB_MAX_DIST = 40.0              # prior-consistency gate (half-res /2)
SUN_EXCLUDE_R = 75.0              # dilated sun neighbourhood, full-res px
PER_MONTH, PER_DAY = 3, 2
CALIB = {"camera_rotation_deg": 172.8, "camera_radius_scale": 0.880}
PT_TITLE, STEP, SCALE = 8.5, 8, 0.25
VERM, BLUE = "#D55E00", "#0072B2"
VERM_RGB = (0xD5, 0x5E, 0x00)


def decode(g, i):
    return np.asarray(Image.open(io.BytesIO(bytes(g["images"][i]))))


def decode_small(g, i):
    im = Image.open(io.BytesIO(bytes(g["images"][i])))
    im.draft("RGB", (128, 128))
    return np.asarray(im)


def nrbr_cloud_mask(img):
    r = img[..., 0].astype(np.float32)
    b = img[..., 2].astype(np.float32)
    return (b - r) / (b + r + 1e-6) < -0.1


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


def qualifies(img_s, nxt_s, az, z):
    """Half-res qualification: non-sun cloud band + glare blob + prior
    consistency + MEAN FLOW over the cloud region above threshold."""
    if img_s.ndim != 3 or img_s.mean() < 60 or nxt_s is None:
        return None
    gray = cv2.cvtColor(img_s, cv2.COLOR_RGB2GRAY)
    blob = glare_blob(gray)
    if blob is None or blob[0] < BLOB_MIN_FRAC:
        return None
    bfrac, cent = blob
    uv = sun_uv_prior(img_s, az, z, 13)
    if uv is None:
        return None
    if np.hypot(cent[0] - uv[0], cent[1] - uv[1]) > BLOB_MAX_DIST / 2:
        return None
    h, w = gray.shape
    # non-sun cloud fraction (exclude dilated sun disc + fisheye corners)
    yy, xx = np.mgrid[0:h, 0:w]
    lens = (yy - (h - 1) / 2) ** 2 + (xx - (w - 1) / 2) ** 2
    ok_zone = lens <= (0.95 * min(h, w) / 2) ** 2
    sun_zone = ((xx - cent[0]) ** 2 + (yy - cent[1]) ** 2
                <= (SUN_EXCLUDE_R / 2) ** 2)
    cloud = nrbr_cloud_mask(img_s) & ok_zone & ~sun_zone
    frac = float(cloud.sum() / max(1, (ok_zone & ~sun_zone).sum()))
    if not (CLOUD_NS_RANGE[0] <= frac <= CLOUD_NS_RANGE[1]):
        return None
    # cloud-region mean flow (dilate cloud mask, measure on the pair)
    u, v = farneback(gray, cv2.cvtColor(nxt_s, cv2.COLOR_RGB2GRAY))
    kernel = np.ones((5, 5), np.uint8)
    region = cv2.dilate(cloud.astype(np.uint8), kernel).astype(bool)
    if region.sum() < 200:
        return None
    vel = float(np.hypot(u, v)[region].mean())
    if vel < FLOW_MEAN_MIN:
        return None
    return {"cloud_ns": frac, "flare": bfrac, "vel": vel}


def quiver(ax, u, v, color):
    h, w = u.shape
    yy, xx = np.mgrid[0:h:STEP, 0:w:STEP]
    ax.quiver(xx, yy, u[::STEP, ::STEP], v[::STEP, ::STEP], color=color,
              pivot="mid", scale=SCALE, scale_units="xy", width=0.004,
              headwidth=3.8, headlength=4.8, minshaft=1.5)
    ax.set_xlim(0, w); ax.set_ylim(h, 0)


def build_mask(img, az, z, radius=25):
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


def _frame_axes(img):
    fig, axs = plt.subplots(1, 4, figsize=(180 / 25.4, 50 / 25.4))
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#333333"); sp.set_linewidth(0.6)
    return fig, axs


def render(img, mask, u, v, out_png, overlay: bool):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    u_q, v_q = u.copy(), v.copy()
    u_q[mask], v_q[mask] = np.nan, np.nan
    ys, xs = np.where(mask)
    cy, cx = (ys.mean(), xs.mean()) if len(ys) else (128.0, 128.0)
    fig, axs = _frame_axes(img)
    axs[0].imshow(img)
    axs[0].add_patch(plt.Circle((cx, cy), 25, fill=False, ec=VERM, lw=1.2))
    axs[0].set_title("(a) Raw ASI frame with glare", fontsize=PT_TITLE,
                     loc="left", pad=3)
    if overlay:                                   # current method (v2)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        ov = gray_rgb.copy()
        ov[mask] = VERM_RGB
        axs[1].imshow(gray_rgb)
        axs[1].imshow(ov, alpha=0.4)
    else:                                         # ORIGINAL fig05 method
        axs[1].imshow(mask.astype(float), cmap="gray", vmin=0, vmax=1)
    axs[1].set_title("(b) Geometric glare mask", fontsize=PT_TITLE,
                     loc="left", pad=3)
    axs[2].imshow(gray, cmap="gray"); quiver(axs[2], u, v, VERM)
    axs[2].set_title("(c) Uncurated optical flow", fontsize=PT_TITLE,
                     loc="left", pad=3)
    axs[3].imshow(gray, cmap="gray"); quiver(axs[3], u_q, v_q, BLUE)
    axs[3].set_title("(d) Quality-assured flow", fontsize=PT_TITLE,
                     loc="left", pad=3)
    fig.tight_layout(w_pad=0.9)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    if overlay:                                   # PDF twin for the main set
        fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    for old in OUT_DIR.glob("QC_flow_folsom_*"):
        old.unlink()                              # regenerate everything
    stores = [zarr.open_group(str(p), mode="r") for p in SKYS]
    tss = [pd.to_datetime([str(t) for t in g["images_timestamps"][:]],
                          utc=True).tz_localize(None) for g in stores]
    tab = pd.read_parquet(STD / "folsom" / "time_series.parquet")
    tab = tab.set_index("timestamp_utc").sort_index()
    tab.index = tab.index.tz_localize(None)
    zen = np.degrees(np.arccos(np.clip(tab["cos_zenith"], -1, 1)))
    geo = {t.round("min"): (float(a), float(z2))
           for t, a, z2 in zip(tab.index, tab["solar_azimuth"], zen)}

    cand = [(si, i, t)
            for si, ts in enumerate(tss)
            for i, t in enumerate(ts)
            if (lambda gz: gz is not None and ZEN_RANGE[0] <= gz <= ZEN_RANGE[1])
            (geo.get(t.round("min"), (None, None))[1])]
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(cand))
    print(f"[stage1] zenith {ZEN_RANGE}: {len(cand)} frames (2014-2016); "
          f"scanning (non-sun cloud + flow>2px/frm + glare)...", flush=True)
    quals, scanned = [], 0
    for k in order:
        if len(quals) >= N_SAMPLES:
            break
        si, i, t = cand[k]
        az, z = geo[t.round("min")]
        if i + 1 >= len(tss[si]):
            continue
        img_s = decode_small(stores[si], i)
        q = qualifies(img_s, decode_small(stores[si], i + 1), az, z)
        scanned += 1
        if q is None:
            continue
        quals.append((si, i, t, q))
    print(f"[stage2-3] scanned {scanned} -> {len(quals)} qualify "
          f"(non-sun cloud {CLOUD_NS_RANGE}, cloud-region flow > "
          f"{2 * FLOW_MEAN_MIN:.1f} px/frm full-res)", flush=True)

    picked, per_month, per_day = [], {}, {}
    for si, i, t, q in quals:
        if len(picked) >= N_SAMPLES:
            break
        mkey, dkey = (t.year, t.month), t.date()
        if per_month.get(mkey, 0) >= PER_MONTH \
                or per_day.get(dkey, 0) >= PER_DAY:
            continue
        picked.append((si, i, t, q))
        per_month[mkey] = per_month.get(mkey, 0) + 1
        per_day[dkey] = per_day.get(dkey, 0) + 1
    if len(picked) < N_SAMPLES:
        print(f"[warn] only {len(picked)} samples met ALL criteria "
              f"(pool={len(quals)})", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for n, (si, i, t, q) in enumerate(picked, 1):
        img = decode(stores[si], i)
        az, z = geo[t.round("min")]
        mask, (cx, cy) = build_mask(img, az, z)
        u, v = farneback(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
                         cv2.cvtColor(decode(stores[si], i + 1),
                                      cv2.COLOR_RGB2GRAY))
        stem = f"QC_flow_folsom_{t:%Y%m%d_%H%M%S}"
        render(img, mask, u, v, OUT_DIR / f"{stem}.png", overlay=True)
        render(img, mask, u, v, OUT_DIR / f"{stem}_origmethod.png",
               overlay=False)                     # extra: original method
        print(f"  [{n:2d}/{len(picked)}] {stem} cloud_ns={q['cloud_ns']:.2f} "
              f"vel={2 * q['vel']:.1f}px/frm flare={q['flare'] * 100:.2f}% "
              f"sun=({cx:.0f},{cy:.0f})", flush=True)
    pngs = sorted(OUT_DIR.glob("QC_flow_folsom_*.png"))
    print(f"[done] {len(pngs)} PNGs in {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
