"""Bulk-render EVERY frame of two complete Folsom days (no filtering).

Reuses the candidate renderer (plot_fig05_qc_candidates.render — the fig05
1x4 composite: (a) raw + sun circle, (b) glare-mask overlay alpha 0.4,
(c) vermillion raw Farneback, (d) blue masked flow; 300 DPI PNG) for every
frame of 2014-02-26 and 2014-02-27 UTC (the paper showcase window), saved
as Figures/QC_Flow_Candidates/QC_flow_folsom_<YYYYMMDD_HHMMSS>.png.

Days are configurable via --dates 2014-02-26,2014-02-27.

Run (conda env solar_uni):
    python scripts/plot_fig05_bulk_2days.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import pandas as pd
import zarr

from plot_fig05_qc_candidates import (  # reuse the OLD renderer verbatim
    OUT_DIR, STD, decode, farneback, render,
)

SKY = STD / "folsom" / "sky_images_2014.zarr"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", default="2014-02-26,2014-02-27",
                    help="comma-separated calendar days (UTC)")
    args = ap.parse_args()
    days = [d.strip() for d in args.dates.split(",")]

    g = zarr.open_group(str(SKY), mode="r")
    ts = pd.to_datetime([str(t) for t in g["images_timestamps"][:]],
                        utc=True).tz_localize(None)
    tab = pd.read_parquet(STD / "folsom" / "time_series.parquet")
    tab = tab.set_index("timestamp_utc").sort_index()
    tab.index = tab.index.tz_localize(None)
    zen = np.degrees(np.arccos(np.clip(tab["cos_zenith"], -1, 1)))
    geo = {t.round("min"): (float(a), float(z2))
           for t, a, z2 in zip(tab.index, tab["solar_azimuth"], zen)}

    idx = [i for i, t in enumerate(ts)
           if str(t.date()) in days
           and geo.get(t.round("min"), (None, 90.0))[1] < 90]   # daytime only
    print(f"[bulk] days={days} DAYTIME frames={len(idx)} -> {OUT_DIR}",
          flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # local import to avoid cycle at module load
    from plot_fig05_qc_candidates import build_mask
    for n, i in enumerate(idx, 1):
        if i + 1 >= len(ts):
            break
        t = ts[i]
        az, z = geo.get(t.round("min"), (None, None))
        img = decode(g, i)
        mask, _ = build_mask(img, az, z)
        u, v = farneback(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
                         cv2.cvtColor(decode(g, i + 1), cv2.COLOR_RGB2GRAY))
        stem = f"QC_flow_folsom_{t:%Y%m%d_%H%M%S}"
        render(img, mask, u, v, OUT_DIR / f"{stem}.png", overlay=True)
        if n % 100 == 0:
            print(f"  [{n}/{len(idx)}] {stem}", flush=True)
    print(f"[done] rendered frames for {days} into {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
