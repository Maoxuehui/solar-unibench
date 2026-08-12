"""Recover timestamps for a zarr that has images but no images_timestamps.

Streams the tarball(s) WITHOUT decoding images (just reads member names),
derives timestamps from filenames, and writes the array. Much faster than full
materialization (no Pillow decode/resize/JPEG-encode).

Usage: python scripts/recover_timestamps.py --year 2014
"""
import sys, tarfile
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solarunibench.imaging.folsom_sky import _ts_from_name, _to_utc_iso
from solarunibench.operators.zarr_io import write_array

DATA_ROOT = Path("D:/monica/workspace/solar_benchmark_dataset/Folsom_data")
ZARR_PATH = Path("data/standardized/folsom/sky_images_{year}.zarr")


def main():
    year = sys.argv[sys.argv.index("--year") + 1] if "--year" in sys.argv else "2014"
    zarr_path = ZARR_PATH.parent / ZARR_PATH.name.format(year=year)
    import zarr
    z = zarr.open(str(zarr_path), mode="r")
    n_images = z["images"].shape[0]
    print(f"zarr {zarr_path} has {n_images} images, checking tar {year}...", flush=True)

    tar = DATA_ROOT / f"Folsom_sky_images_{year}.tar.bz2"
    timestamps = []
    with tarfile.open(tar, "r|bz2") as tf:  # streaming mode (faster for sequential reads)
        for m in tf:
            if not m.isfile() or not m.name.lower().endswith(".jpg"):
                continue
            ts = _to_utc_iso(_ts_from_name(m.name))
            timestamps.append(ts)
            if (len(timestamps)) % 50000 == 0:
                print(f"  scanned {len(timestamps)} members...", flush=True)

    print(f"tar {year}: {len(timestamps)} .jpg members", flush=True)
    # Write timestamps array with SINGLE large chunk (fast).
    write_array(zarr_path, "images_timestamps", np.array(timestamps),
                chunks=(max(1, len(timestamps)),))
    # Clean up sidecar if present
    sidecar = zarr_path.parent / (zarr_path.name + ".timestamps.txt")
    sidecar.unlink(missing_ok=True)
    print(f"wrote {len(timestamps)} timestamps -> {zarr_path}/images_timestamps", flush=True)

    # Verify
    z2 = zarr.open(str(zarr_path), mode="r")
    ts = z2["images_timestamps"]
    print(f"verified: {ts.shape} | first: {ts[0]} | last: {ts[-1]}", flush=True)


if __name__ == "__main__":
    main()
