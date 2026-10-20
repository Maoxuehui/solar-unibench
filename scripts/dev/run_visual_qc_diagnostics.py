"""Large-scale Visual-QC dry-run diagnostics for paper Section 3.2.

Year-scale sweeps over the REAL standardized stores, per-dataset camera
calibration, honest ratios (nothing forced):

  ASI (Folsom 2014, PRIMARY)     full calendar year of daytime frames,
                                 deterministic stride cap ~40k frames
  ASI (SKIPP'D 2018, SECONDARY)  1-year contiguous window (its record spans
                                 2017-03..2019-10; 2018 chosen), low-res
                                 camera with its OWN calibration + scaled
                                 glare radius (25 px / 256 -> 6 px / 64)
  Satellite (SunLake 2008)       full-year context sequence from the full
                                 local mirror; theta_z from the standardized
                                 table's CAB rows (30-min aligned)

Q3 note: frame-diff variance needs TEMPORALLY ADJACENT pairs, so the drop-
frame ratio is measured on a dedicated contiguous block (2,000 frames) per
ASI dataset (the strided sweep would inflate the variance) and on the full
consecutive satellite sequence.

Prints the six-line VisualQCPipeline diagnostics report per dataset plus a
final four-metric summary table (Q0/Q1/Q2/Q3) for the paper.

Run (conda env solar_uni):
    python scripts/run_visual_qc_diagnostics.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
import pandas as pd
import zarr
from PIL import Image

from solarunibench.operators.visual_qc import VisualQCPipeline

STD = _REPO_ROOT / "data" / "standardized"
FULL_SAT = STD / "sunlake" / "sunlake_satellite_full.zarr"
OUT_DIR = _REPO_ROOT / "scratch" / "qc_diagnostics"
MAX_FRAMES = 40_000            # per-ASI sweep cap (deterministic stride)
Q3_BLOCK = 2_000               # contiguous frames for the Q3 statistic
DAYTIME_COSZ = 0.05
DAYTIME_MEAN = 60              # image fallback daylight filter
CALIB_SCAN = ("{}-06-01T14", "{}-09-01T02")   # summer low-sun glare window


def minute_geometry(tab: pd.DataFrame) -> dict:
    """{rounded minute: (azimuth, zenith_deg)} lookup from the table."""
    zen = np.degrees(np.arccos(np.clip(tab["cos_zenith"], -1, 1)))
    return {ts.round("min"): (float(a), float(z))
            for ts, a, z in zip(tab.index, tab["solar_azimuth"], zen)}


def calibrate_multiframe(frames_fetch, idxs, ts, geo, shape_hw,
                         max_frames: int = 80):
    """Multi-frame camera calibration: collect strong-glare frames across the
    sweep, then fit (rotation, radius_scale) by circular mean + least squares
    with outlier trimming (robust to cloud-edge saturated false positives)."""
    import math
    cands = []
    for i in idxs[:: max(1, len(idxs) // 1200)]:
        img = frames_fetch(i)
        if img is None or img.mean() < DAYTIME_MEAN:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sat = (g >= 250).astype(np.uint8)
        n, _lab, stats, cents = cv2.connectedComponentsWithStats(sat, 8)
        if n < 2:
            continue
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        frac = stats[k, cv2.CC_STAT_AREA] / g.size
        if frac < 0.002:
            continue
        az, zen = geo.get(ts[i].round("min"), (None, None))
        if az is None or zen is None or zen >= 85:
            continue
        h, w = shape_hw
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        dx, dy = cents[k][0] - cx, cents[k][1] - cy
        cands.append((math.degrees(math.atan2(dx, -dy)) % 360.0,
                      math.degrees(math.atan2(np.sin(np.deg2rad(az)),
                                              np.cos(np.deg2rad(az)))) % 360.0,
                      math.hypot(dx, dy), (zen / 90.0) * (min(h, w) / 2.0)))
    if len(cands) < 5:
        print(f"  [calib] only {len(cands)} glare candidates -> default")
        return {"camera_rotation_deg": 0.0, "camera_radius_scale": 1.0}
    diffs = np.array([(c - p) % 360.0 for c, p, _, _ in cands])
    med = np.median(diffs)
    keep = np.abs((diffs - med + 180) % 360.0 - 180) < 25.0
    d = diffs[keep]
    ang = np.deg2rad(d)
    rot = math.degrees(math.atan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360.0
    arr = np.array([(c[2], c[3]) for c in cands])[keep]
    rb, rp = arr[:, 0], arr[:, 1]
    scale = float((rb * rp).sum() / (rp * rp).sum())
    print(f"  [calib] {int(keep.sum())}/{len(cands)} frames -> rot={rot:.1f} deg, "
          f"scale={scale:.3f}")
    return {"camera_rotation_deg": rot, "camera_radius_scale": scale}


# -- Folsom -----------------------------------------------------------------
def run_folsom():
    g = zarr.open_group(str(STD / "folsom" / "sky_images_2014.zarr"), "r")
    ts = pd.to_datetime([str(t) for t in g["images_timestamps"][:]],
                        utc=True).tz_localize(None)
    tab = pd.read_parquet(STD / "folsom" / "time_series.parquet")
    tab = tab.set_index("timestamp_utc").sort_index()
    tab.index = tab.index.tz_localize(None)
    tab = tab.loc['2014']
    geo = minute_geometry(tab)

    def fetch(i):
        return np.asarray(Image.open(io.BytesIO(bytes(g["images"][i]))))

    day_idx = [i for i, t in enumerate(ts)
               if geo.get(t.round('min'), (None, 90.0))[1] < 90]
    stride = max(1, len(day_idx) // MAX_FRAMES + 1)
    sweep = day_idx[::stride]
    print(f"[folsom] 2014 daytime frames={len(day_idx)} sweep={len(sweep)} "
          f"(stride {stride})")
    calib = calibrate_multiframe(fetch, sweep, ts, geo, (256, 256))
    print(f"[folsom] calibration: rot={calib['camera_rotation_deg']:.1f} "
          f"scale={calib['camera_radius_scale']:.3f}")
    qc = VisualQCPipeline({"mode": "dry_run", "modality": "asi",
                           "glare_radius_px": 25, **calib})
    for i in sweep:
        az, zen = geo.get(ts[i].round("min"), (None, None))
        qc.process_frame(fetch(i), solar_azimuth_deg=az,
                         solar_zenith_deg=zen, modality="asi")
    qc.print_diagnostics_report()
    # Q3 contiguous block (mid-year)
    qc3 = VisualQCPipeline({"mode": "dry_run", "modality": "asi",
                            "glare_radius_px": 25, "var_frozen": 1e-6,
                            **calib})
    mid = day_idx[len(day_idx) // 2] if day_idx else 0
    block = [i for i in range(mid, min(mid + 4 * Q3_BLOCK, len(ts)))
             if geo.get(ts[i].round("min"), (None, 90.0))[1] < 90
             ][:Q3_BLOCK]
    prev = None
    for i in block:
        az, zen = geo.get(ts[i].round("min"), (None, None))
        fr = fetch(i)
        qc3.process_frame(fr, prev_frame=prev, solar_azimuth_deg=az,
                          solar_zenith_deg=zen, modality="asi")
        prev = fr
    print(f"[folsom] Q3 contiguous block ({Q3_BLOCK} frames): "
          f"drop ratio = {100.0 * qc3.q3_drop / max(1, qc3.n_frames):.4f}%")
    return {"dataset": "folsom_2014",
            "n": qc.n_frames,
            "q0_oob_pct": 100.0 * qc.q0_oob_pixels / max(1, qc.q0_total_pixels),
            "q1_glare_pct": 100.0 * qc.q1_glare_pixels / max(1, qc.q1_total_pixels),
            "q1_overexp_pct": 100.0 * qc.q1_overexposed / max(1, qc.n_frames),
            "q2_pct": float("nan"),
            "q3_pct": 100.0 * qc3.q3_drop / max(1, qc3.n_frames),
            "calib": calib}


# -- SKIPP'D ----------------------------------------------------------------
def run_skippd():
    g = zarr.open_group(str(STD / "skippd" / "sky_images.zarr"), "r")
    arr = g["images"]
    ts = pd.to_datetime([str(t) for t in g["images_timestamps"][:]],
                        utc=True).tz_localize(None)
    tab = pd.read_parquet(STD / "skippd" / "time_series.parquet")
    tab = tab.set_index("timestamp_utc").sort_index()
    tab.index = tab.index.tz_localize(None)
    tab = tab.loc["2018-01-01":"2018-12-31"]
    geo = minute_geometry(tab)

    def fetch(i):
        f = np.asarray(arr[i])                     # (4, H, W) uint8
        return np.transpose(f[:3], (1, 2, 0))      # RGB (H, W, 3)

    day_idx = [i for i, t in enumerate(ts)
               if "2018" in str(t.date()) and t.round("min") in geo]
    stride = max(1, len(day_idx) // MAX_FRAMES + 1)
    sweep = day_idx[::stride]
    print(f"[skippd] 2018 frames={len(day_idx)} sweep={len(sweep)}")
    calib = calibrate_multiframe(fetch, sweep, ts, geo, (64, 64))
    print(f"[skippd] calibration: rot={calib['camera_rotation_deg']:.1f} "
          f"scale={calib['camera_radius_scale']:.3f}")
    qc = VisualQCPipeline({"mode": "dry_run", "modality": "asi",
                           "glare_radius_px": 6, **calib})
    for i in sweep:
        az, zen = geo.get(ts[i].round("min"), (None, None))
        qc.process_frame(fetch(i), solar_azimuth_deg=az,
                         solar_zenith_deg=zen, modality="asi")
    qc.print_diagnostics_report()
    qc3 = VisualQCPipeline({"mode": "dry_run", "modality": "asi",
                            "glare_radius_px": 6, "var_frozen": 1e-6,
                            **calib})
    c0 = day_idx[len(day_idx) // 2] if day_idx else 0
    prev = None
    for i in range(c0, min(c0 + Q3_BLOCK, len(ts))):
        az, zen = geo.get(ts[i].round("min"), (None, None))
        qc3.process_frame(fetch(i), prev_frame=prev, solar_azimuth_deg=az,
                          solar_zenith_deg=zen, modality="asi")
        prev = fetch(i)
    print(f"[skippd] Q3 contiguous block: "
          f"drop ratio = {100.0 * qc3.q3_drop / max(1, qc3.n_frames):.4f}%")
    return {"dataset": "skippd_2018",
            "n": qc.n_frames,
            "q0_oob_pct": 100.0 * qc.q0_oob_pixels / max(1, qc.q0_total_pixels),
            "q1_glare_pct": 100.0 * qc.q1_glare_pixels / max(1, qc.q1_total_pixels),
            "q1_overexp_pct": 100.0 * qc.q1_overexposed / max(1, qc.n_frames),
            "q2_pct": float("nan"),
            "q3_pct": 100.0 * qc3.q3_drop / max(1, qc3.n_frames),
            "calib": calib}


# -- SunLake satellite (Q2) ---------------------------------------------------
def run_sunlake():
    g = zarr.open_group(str(FULL_SAT), mode="r")
    ts = pd.to_datetime([str(t) for t in g["time_utc"][:]], utc=True)
    ts = ts.tz_localize(None)
    tab = pd.read_parquet(STD / "sunlake" / "time_series.parquet")
    tab = tab.set_index("timestamp_utc").sort_index()
    tab.index = tab.index.tz_localize(None)
    cab = tab[tab.entity_id == "SUNLAKE_CAB"].loc["2008"]
    geo = minute_geometry(cab)
    ctx = g["context"]
    qc = VisualQCPipeline({"mode": "dry_run", "modality": "satellite"})
    prev = None
    n_year = int(np.searchsorted(ts, pd.Timestamp("2009-01-01")))
    for i in range(n_year):
        az, zen = geo.get(ts[i].round("min"), (None, None))
        fr = np.transpose(np.asarray(ctx[i]), (1, 2, 0))   # (H, W, C)
        qc.process_frame(fr, prev_frame=prev, solar_azimuth_deg=az,
                         solar_zenith_deg=zen, modality="satellite")
        prev = fr
    qc.print_diagnostics_report()
    return {"dataset": "sunlake_2008_satellite",
            "n": qc.n_frames,
            "q0_oob_pct": 100.0 * qc.q0_oob_pixels / max(1, qc.q0_total_pixels),
            "q1_glare_pct": float("nan"),
            "q1_overexp_pct": float("nan"),
            "q2_pct": 100.0 * qc.q2_high_zenith / max(1, qc.n_frames),
            "q3_pct": 100.0 * qc.q3_drop / max(1, qc.n_frames),
            "calib": {}}


# -- helpers -----------------------------------------------------------------
    return fetch(i)


def main() -> int:
    rows = [run_folsom(), run_skippd(), run_sunlake()]
    print("\n" + "=" * 72)
    print("FINAL SUMMARY — Section 3.2 visual-QC empirical statistics")
    print("=" * 72)
    print(f"{'dataset':28s} {'n':>7s} {'Q0 oob%':>9s} {'Q1 glare%':>10s} "
          f"{'Q1 overexp%':>12s} {'Q2 zenith%':>10s} {'Q3 drop%':>9s}")
    for r in rows:
        fmt = lambda v: "—" if not np.isfinite(v) else f"{v:.4f}"
        print(f"{r['dataset']:28s} {r['n']:7d} {fmt(r['q0_oob_pct']):>9s} "
              f"{fmt(r['q1_glare_pct']):>10s} {fmt(r['q1_overexp_pct']):>12s} "
              f"{fmt(r['q2_pct']):>10s} {fmt(r['q3_pct']):>9s}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "results_visual_qc_diagnostics.csv"
    pd.DataFrame([{k: v for k, v in r.items() if k != "calib"}
                  for r in rows]).to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
