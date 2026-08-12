"""Rebuild ONLY the ``sky_images.zarr`` for one dataset at a bounded frame limit.

Calls :func:`build_folsom_sky_zarr` / :func:`build_skippd_sky_zarr` DIRECTLY into
``<output_root>/<ds>/sky_images.zarr`` — no tabular re-run, no splits rebuild,
no metadata rewrite. The destination store constructors open with ``overwrite=True``
which wipes only that zarr group, so nothing else under ``<output_root>/<ds>/``
is touched (metadata.parquet / time_series.parquet / splits are preserved).

Storage formats (per the storage-optimization plan):
  - folsom: JPEG-q90 RGB bytes (zarr VLenBytes object array), BRR on load.
  - skippd: uint8 RGB+BRR (4,H,W) lossless, native resolution (64x64).

Usage:
    python scripts/rebuild_sky_zarr.py --dataset folsom --limit 2000
    python scripts/rebuild_sky_zarr.py --dataset skippd --limit 2000
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solarunibench.imaging.folsom_sky import build_folsom_sky_zarr
from solarunibench.imaging.skippd_sky import build_skippd_sky_zarr

# Mirrors scripts/build_dataset.RAW_SUBDIRS so this script stays decoupled from
# the full adapter stack.
RAW_SUBDIRS = {"folsom": "Folsom_data", "skippd": "SKIPPD_all"}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rebuild sky_images.zarr (compact format) for one dataset")
    ap.add_argument("--dataset", required=True, choices=["folsom", "skippd"])
    ap.add_argument("--limit", type=int, default=200, help="frame cap (default 200; 0 = all)")
    ap.add_argument("--only-year", type=int, default=None,
                    help="folsom only: restrict to Folsom_sky_images_<year>.tar.bz2 (e.g. 2014)")
    ap.add_argument("--data-root", default=None, help="override configs/paths.yaml data_root")
    ap.add_argument("--output-root", default=None, help="override configs/paths.yaml output_root")
    args = ap.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((repo / "configs" / "paths.yaml").read_text())
    data_root = Path(args.data_root or cfg["data_root"])
    output_root = Path(args.output_root or cfg["output_root"])
    # Per-year output paths for Folsom (--only-year): each year gets its own
    # zarr store (sky_images_2014.zarr etc.) so they never overwrite each other.
    if args.dataset == "folsom" and args.only_year is not None:
        out_path = output_root / args.dataset / f"sky_images_{args.only_year}.zarr"
    else:
        out_path = output_root / args.dataset / "sky_images.zarr"

    if args.dataset == "folsom":
        tar_paths = sorted((data_root / RAW_SUBDIRS["folsom"]).glob("Folsom_sky_images_*.tar.bz2"))
        if args.only_year is not None:
            tar_paths = [tp for tp in tar_paths if f"_{args.only_year}." in tp.name]
            if not tar_paths:
                raise SystemExit(f"--only-year {args.only_year}: no matching Folsom tarball "
                                 f"(expected Folsom_sky_images_{args.only_year}.tar.bz2)")
        if not tar_paths:
            raise SystemExit(f"No Folsom sky tarballs under {data_root / RAW_SUBDIRS['folsom']}")
        print(f"[rebuild_sky_zarr] using {len(tar_paths)} tarball(s): "
              f"{', '.join(p.name for p in tar_paths)}", flush=True)
        result = build_folsom_sky_zarr(tar_paths, out_path, limit=args.limit)
    else:  # skippd
        parquet_dir = data_root / RAW_SUBDIRS["skippd"] / "default"
        if not parquet_dir.exists():
            raise SystemExit(f"SKIPPD image parquets not found at {parquet_dir}")
        result = build_skippd_sky_zarr(parquet_dir, out_path, limit=args.limit)

    print(f"[rebuild_sky_zarr] {args.dataset}: wrote {result['n_frames']} frames -> {out_path}")


if __name__ == "__main__":
    main()
