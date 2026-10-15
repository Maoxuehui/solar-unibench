"""Split a large file into raw byte slices named <stem>_NNN.zip.

Matches the convention already on the Zenodo deposit (skippd_001.zip,
folsom_001.zip, ...): part 1 carries the original's leading bytes, and
concatenating all parts (copy /b on Windows, cat on unix) restores the
original file byte-for-byte. Verifies the part-sum equals the source size
and is a no-op if the slices already exist and sum correctly.

Usage:
    python scripts/split_zip.py --src "data/standardized/Solar-UniBench v1.0/sunlake.zip" --part-mb 450
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="file to slice")
    ap.add_argument("--part-mb", type=float, default=450.0,
                    help="slice size in MiB (default 450)")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_file():
        print(f"source not found: {src}")
        return 1
    part = int(args.part_mb * 1024 * 1024)
    size = src.stat().st_size
    n = (size + part - 1) // part

    existing = sorted(src.parent.glob(f"{src.stem}_[0-9][0-9][0-9].zip"))
    # any complete existing set wins, whatever its part size (e.g. the
    # hand-sliced 1 GiB sunlake parts) — never re-slice over it
    if existing and sum(p.stat().st_size for p in existing) == size:
        print(f"{src.stem}: {len(existing)} parts already present "
              f"(sizes differ from --part-mb, using them), sum matches — skip")
        return 0

    print(f"{src.stem}: {size:,} bytes -> {n} parts of <= {part:,}", flush=True)
    with open(src, "rb") as f:
        for i in range(1, n + 1):
            out = src.parent / f"{src.stem}_{i:03d}.zip"
            left = min(part, size - (i - 1) * part)
            with open(out, "wb") as o:
                while left > 0:
                    chunk = f.read(min(16 * 1024 * 1024, left))
                    if not chunk:
                        break
                    o.write(chunk)
                    left -= len(chunk)
            if i % 10 == 0 or i == n:
                print(f"  {i:3d}/{n} written ({out.name})", flush=True)

    parts = sorted(src.parent.glob(f"{src.stem}_[0-9][0-9][0-9].zip"))
    ok = len(parts) == n and sum(p.stat().st_size for p in parts) == size
    print(f"{src.stem}: {len(parts)} parts, sum match = {ok}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
