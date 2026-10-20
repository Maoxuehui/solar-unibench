"""Recover timestamps for a zarr by scanning the tarball, EXCLUDING skipped members.

This is the PRECISE variant of recover_timestamps.py: it reads the SKIP list
(extracted from build logs) so the emitted timestamps align 1:1 with the images
that were actually written. The plain recover_timestamps.py includes skipped
(truncated-JPEG) members, causing a count mismatch.

Usage:
    python scripts/recover_timestamps_filtered.py --year 2014
    python scripts/recover_timestamps_filtered.py --year 2015

Requires logs/skiplists/folsom_<year>.txt (member names, one per line).
"""
import sys
import tarfile
from pathlib import Path

import os

_REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = Path(os.environ.get(
    "SOLAR_RAW_ROOT",
    "D:/monica/workspace/solar_benchmark_dataset"))

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from solarunibench.imaging.folsom_sky import _ts_from_name, _to_utc_iso
from solarunibench.operators.zarr_io import write_array
_REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = RAW_ROOT / "Folsom_data"
ZARR_PATH = Path("data/standardized/folsom/sky_images_{year}.zarr")
SKIPLIST_DIR = Path("logs/skiplists")


def main():
    year = sys.argv[sys.argv.index("--year") + 1] if "--year" in sys.argv else "2014"
    zarr_path = ZARR_PATH.parent / ZARR_PATH.name.format(year=year)
    import zarr

    z = zarr.open(str(zarr_path), mode="r")
    n_images = z["images"].shape[0]
    print(f"[recover_filtered] zarr {zarr_path.name}: {n_images} images", flush=True)

    # Load skip list (member names that were truncated during decode and thus
    # absent from the images array).
    skip_file = SKIPLIST_DIR / f"folsom_{year}.txt"
    skip_set = set()
    if skip_file.exists():
        skip_set = {l.strip() for l in skip_file.read_text().splitlines() if l.strip()}
    print(f"[recover_filtered] skip list: {len(skip_set)} members to exclude", flush=True)

    tar = DATA_ROOT / f"Folsom_sky_images_{year}.tar.bz2"
    timestamps = []
    skipped = 0
    with tarfile.open(tar, "r|bz2") as tf:  # streaming mode
        for m in tf:
            if not m.isfile() or not m.name.lower().endswith(".jpg"):
                continue
            if m.name in skip_set:
                skipped += 1
                continue
            timestamps.append(_to_utc_iso(_ts_from_name(m.name)))
            if len(timestamps) % 50000 == 0:
                print(f"  scanned {len(timestamps)} kept (+{skipped} skipped)...",
                      flush=True)

    print(f"[recover_filtered] {year}: {len(timestamps)} kept, {skipped} skipped "
          f"by filter", flush=True)

    if len(timestamps) != n_images:
        print(f"[recover_filtered] WARNING: count mismatch! "
              f"timestamps={len(timestamps)} vs images={n_images}", flush=True)
        # Trim or pad is NOT done here — investigate manually if this fires.
        sys.exit(1)

    # Single large chunk (NOT chunks=(1,) which creates 250k tiny NTFS files).
    write_array(zarr_path, "images_timestamps", np.array(timestamps),
                chunks=(max(1, len(timestamps)),))
    sidecar = zarr_path.parent / (zarr_path.name + ".timestamps.txt")
    sidecar.unlink(missing_ok=True)

    z2 = zarr.open(str(zarr_path), mode="r")
    ts = z2["images_timestamps"]
    print(f"[recover_filtered] VERIFIED: {ts.shape} | first: {ts[0]} | last: {ts[-1]}",
          flush=True)
    print(f"[recover_filtered] {year} DONE", flush=True)


if __name__ == "__main__":
    main()
