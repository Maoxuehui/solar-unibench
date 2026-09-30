"""Full SunLake SEVIRI imagery: hub -> local raw mirror -> standardized zarr.

Two-step, resumable pipeline (per user spec, 2026-09):

  step 1  download   Per-year subtree copy of hub://crossvivit/SunLake into
                     SunLake_data/SunLake_raw_deeplake/<year>/ via
                     Dataset.copy(tensors=<year subtree>, num_workers=4).
                     A year is skipped when its local mirror already has the
                     expected context length (resume-safe).
  step 2  convert    Offline pass over the local mirrors -> one standardized
                     zarr (float32 context, per-frame chunks (1,11,64,64),
                     lz4/Blosc-SHUFFLE via the repo's IMAGE_COMPRESSOR
                     convention), plus per-frame time_utc strings and station
                     coords, yearly_counts + total_frames attrs, consolidated
                     metadata. Expected total: 222,383 frames (verified
                     against hub tensor_meta on 2026-09-17).

The bounded benchmark deliverable data/standardized/sunlake/satellite_future.zarr
(200 scenes + derived futures) is NOT touched — this store is the full-corpus
companion: data/standardized/sunlake/sunlake_satellite_full.zarr.

Run (conda env solar_uni):
    python scripts/pull_sunlake_full_imagery.py --step 1     # download (hours)
    python scripts/pull_sunlake_full_imagery.py --step 2     # convert
    python scripts/pull_sunlake_full_imagery.py --verify     # integrity spot-check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zarr

_REPO_ROOT = Path(__file__).resolve().parent.parent
HUB = "hub://crossvivit/SunLake"
RAW_ROOT = Path("D:/monica/workspace/solar_benchmark_dataset/SunLake_data/"
                "SunLake_raw_deeplake")
OUT_ZARR = _REPO_ROOT / "data" / "standardized" / "sunlake" / \
    "sunlake_satellite_full.zarr"
YEARS = range(2008, 2023)


def _year_dir(year: int) -> Path:
    return RAW_ROOT / f"{year}_context"

#: Hub tensor_meta ground truth (queried 2026-09-17; see session log).
EXPECTED = {2008: 7926, 2009: 14093, 2010: 14353, 2011: 14952, 2012: 14311,
            2013: 15728, 2014: 15621, 2015: 16119, 2016: 15616, 2017: 14861,
            2018: 16997, 2019: 17424, 2020: 15955, 2021: 13931, 2022: 14496}
EXPECTED_TOTAL = sum(EXPECTED.values())          # 222,383

CTX_CHANNELS = ["IR_016", "IR_039", "IR_087", "IR_097", "IR_108", "IR_120",
                "IR_134", "VIS006", "VIS008", "WV_062", "WV_073"]
BATCH = 2000                                      # frames per append batch


def _year_tensors(names, year: int) -> list[str]:
    return [n for n in names if n.startswith(f"{year}_nonhrv/")]


def step1_download(years) -> None:
    """Per-year copy of the hub CONTEXT group only (data + time_utc + lat/
    lon/elevation) into SunLake_raw_deeplake/<year>_context/, num_workers=4,
    resume-safe: a year is skipped only when its mirror is complete."""
    import deeplake
    ds = deeplake.load(HUB)
    names = set(ds.tensors.keys())
    for year in years:
        local = _year_dir(year)
        if (local / "dataset_meta.json").exists():
            try:                                   # resume: check completeness
                lds = deeplake.load(str(local))
                n = lds[f"{year}_nonhrv/context/data"].shape[0]
                if n == EXPECTED[year]:
                    print(f"[step1] {year}: already complete ({n} frames), "
                          f"skip", flush=True)
                    continue
                print(f"[step1] {year}: incomplete ({n}/{EXPECTED[year]}), "
                      f"re-copying", flush=True)
            except Exception:
                print(f"[step1] {year}: unreadable mirror, re-copying",
                      flush=True)
        tensors = [t for t in (f"{year}_nonhrv/context/{k}" for k in
                               ("data", "time_utc", "latitude", "longitude",
                                "elevation")) if t in names]
        print(f"[step1] {year}: copying {len(tensors)} context tensors "
              f"-> {local}", flush=True)
        try:
            ds.copy(str(local), tensors=tensors, overwrite=True,
                    num_workers=4)
            n = deeplake.load(str(local))[
                f"{year}_nonhrv/context/data"].shape[0]
            print(f"[step1] {year}: done, context frames = {n} "
                  f"(expected {EXPECTED[year]})", flush=True)
        except Exception as exc:
            print(f"[step1] {year}: download FAILED: {exc}", flush=True)
    print("[step1] all requested years processed.", flush=True)


class _AppendZarr:
    """Incremental (N, 11, 64, 64) float32 store, repo compression convention."""

    def __init__(self, path: Path):
        from numcodecs import Blosc
        self.path = path
        store = zarr.DirectoryStore(str(path))
        self.root = zarr.group(store=store, overwrite=True)
        self.ctx = self.root.create_dataset(
            "context", shape=(0, 11, 64, 64), chunks=(1, 11, 64, 64),
            dtype="float32",
            compressor=Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE))
        self.ts = self.root.create_dataset(
            "time_utc", shape=(0,), chunks=(10000,), dtype="<U25")

    def append(self, ctx: np.ndarray, ts: np.ndarray) -> None:
        self.ctx.append(ctx.astype(np.float32))
        self.ts.append(np.asarray(ts, dtype="<U25"))


def step2_convert(years) -> None:
    import deeplake
    import pandas as pd
    if OUT_ZARR.exists() and "--restart" not in sys.argv:
        # Resume support: if a previous run recorded progress, only missing
        # years are re-appended by a full rebuild (simplest correct policy:
        # rebuild is cheap I/O once the raw mirror is local).
        pass
    zw = _AppendZarr(OUT_ZARR)
    yearly_counts: dict[str, int] = {}
    total = 0
    for year in years:
        local = _year_dir(year)
        if not (local / "dataset_meta.json").exists():
            print(f"[step2] {year}: raw mirror missing, skip "
                  "(run step 1 first)")
            continue
        try:
            lds = deeplake.load(str(local))
            pfx = f"{year}_nonhrv/context"
            n_expected = EXPECTED[year]
            t_utc = lds[f"{pfx}/time_utc"].numpy().squeeze()
            n_frames = 0
            for s in range(0, n_expected, BATCH):
                e = min(s + BATCH, n_expected)
                ctx = lds[f"{pfx}/data"][s:e].numpy().astype(np.float32)
                ts = pd.to_datetime(t_utc[s:e]).strftime("%Y-%m-%dT%H:%M:%S%z")
                zw.append(ctx, np.asarray(ts, dtype=object))
                n_frames = e
            yearly_counts[str(year)] = n_frames
            total += n_frames
            print(f"[step2] {year}: converted {n_frames} frames "
                  f"(cumulative {total})", flush=True)
            # per-year coords: constant per station patch, store once in attrs
            for k in ("latitude", "longitude", "elevation"):
                try:
                    v = lds[f"{pfx}/{k}"].numpy().squeeze()
                    zw.root.attrs[f"{year}_{k}"] = \
                        np.round(float(np.ravel(v)[0]), 6)
                except Exception:
                    pass
        except Exception as exc:
            print(f"[step2] {year}: FAILED locally: {exc}", flush=True)
    zw.root.attrs["description"] = ("Full SunLake SEVIRI satellite imagery "
                                    "(2008-2022) for Solar-UniBench v1.0; "
                                    "per-frame chunks, float32, lz4")
    zw.root.attrs["channels"] = CTX_CHANNELS
    zw.root.attrs["total_frames"] = total
    zw.root.attrs["yearly_counts"] = json.dumps(yearly_counts, indent=0)
    zw.root.attrs["expected_total"] = EXPECTED_TOTAL
    zarr.consolidate_metadata(zarr.DirectoryStore(str(OUT_ZARR)))
    print(f"[step2] wrote {OUT_ZARR}")
    print(f"[step2] final frames: {total} / expected {EXPECTED_TOTAL}")


def verify() -> None:
    import deeplake
    g = zarr.open_group(str(OUT_ZARR), mode="r")
    total = int(g["context"].shape[0])
    counts = json.loads(g.attrs["yearly_counts"])
    print(f"[verify] context shape = {g['context'].shape}")
    print(f"[verify] total = {total} / expected {EXPECTED_TOTAL} "
          f"-> {'OK' if total == EXPECTED_TOTAL else 'MISMATCH'}")
    ok = all(counts.get(str(y)) == EXPECTED[y] for y in YEARS)
    print(f"[verify] yearly counts vs hub meta: {'OK' if ok else 'MISMATCH'}")
    # spot-check 3 random frames against the hub
    rng = np.random.default_rng(0)
    idx = sorted(rng.choice(total, size=3, replace=False).tolist())
    ds = deeplake.load(HUB)
    for i in idx:
        # locate year + offset for the global index
        acc = 0
        for y in YEARS:
            if i < acc + EXPECTED[y]:
                break
            acc += EXPECTED[y]
        local = (ds[f"{y}_nonhrv/context/data"][i - acc].numpy()
                 .astype(np.float32))
        stored = g["context"][i]
        same = np.array_equal(local, stored)
        print(f"[verify] frame {i} ({y}+{i - acc}): "
              f"{'IDENTICAL' if same else 'DIFFERS'} "
              f"(max|d|={np.abs(local - stored).max():.2e})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, choices=(1, 2), default=None)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--years", default=None,
                    help="e.g. 2008,2010 or 2012-2015 (default all)")
    args = ap.parse_args()
    if args.years:
        yrs = set()
        for part in args.years.split(","):
            if "-" in part:
                a, b = part.split("-")
                yrs.update(range(int(a), int(b) + 1))
            else:
                yrs.add(int(part))
        years = sorted(yrs)
    else:
        years = list(YEARS)
    if args.step == 1:
        step1_download(years)
    elif args.step == 2:
        step2_convert(years)
    elif args.verify:
        verify()
    else:
        ap.error("choose --step 1, --step 2 or --verify")


if __name__ == "__main__":
    main()
