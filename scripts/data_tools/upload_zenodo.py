"""Upload the four Solar-UniBench v1.0 zips to the Zenodo deposit.

Deposit: 23201168 (draft). Files (real, as built by the packaging step):
  gefcom2014.zip     0.005 TB
  skippd.zip         4.7 TB
  folsom.zip         9.9 TB
  sunlake.zip       21.4 TB

Streams each file in 8 MB chunks with a tqdm progress bar (no full-file
RAM load). Retries each chunk-range on connection errors; resumes are
re-run-safe (Zenodo file PUTs are idempotent per filename). Small files
first so partial progress is visible early.

The multi-GB originals are uploaded as raw byte slices (`skippd_001.zip`,
`skippd_002.zip`, ... — concat to restore): each slice is one independent
HTTP PUT, so a connection reset only re-sends that slice instead of the
whole file. Zenodo's deposit API has no true multipart (POST ?uploads ->
405), so per-slice PUTs are the only restartable unit.

Usage:
    python scripts/upload_zenodo.py --parts    # serial upload of *_NNN.zip slices
    python scripts/upload_zenodo.py --only gefcom2014.zip
    python scripts/upload_zenodo.py --list     # show files already on Zenodo

Env override: ZENODO_TOKEN (falls back to scratch/zenodo_secret.py,
which is gitignored — never hardcode the token here).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_token() -> str:
    env = os.environ.get("ZENODO_TOKEN")
    if env:
        return env
    secret = _REPO_ROOT / "scratch" / "zenodo_secret.py"
    if secret.is_file():
        ns: dict = {}
        exec(secret.read_text(), ns)      # local credential file, gitignored
        return ns["ACCESS_TOKEN"]
    sys.exit("no token: set ZENODO_TOKEN or create scratch/zenodo_secret.py")


ACCESS_TOKEN = _load_token()
DEPOSIT_ID = "23201168"
API = f"https://zenodo.org/api/deposit/depositions/{DEPOSIT_ID}"
DIR = Path(r"D:\monica\workspace\solar-unibench\data\standardized\Solar-UniBench v1.0")
CHUNK = 8 * 1024 * 1024          # 8 MB stream chunks
RETRIES = 200
TIMEOUT = 120                     # per-chunk read timeout (s)


def deposit_info() -> dict:
    r = requests.get(API, params={"access_token": ACCESS_TOKEN}, timeout=60)
    if r.status_code != 200:
        print("deposit fetch failed:", r.status_code, r.text[:500])
        sys.exit(1)
    return r.json()


class ProgressFile:
    """File wrapper: exposes read()/__len__ so requests streams it with a
    known Content-Length (Zenodo rejects chunked PUTs without length)."""

    def __init__(self, path: Path, pbar: tqdm):
        self._f = open(path, "rb")
        self._size = path.stat().st_size
        self._pbar = pbar

    def read(self, n=-1):
        data = self._f.read(CHUNK if n is None or n < 0 else min(n, CHUNK))
        if data:
            self._pbar.update(len(data))
        return data

    def __len__(self):
        return self._size

    def close(self):
        self._f.close()


def upload_one(bucket: str, path: Path) -> bool:
    size = path.stat().st_size
    remote = path.name
    hsize = f"{size/1e9:.2f} GB" if size >= 1e9 else f"{size/1e6:.1f} MB"
    print(f"\n=== {remote}  ({hsize}) ===", flush=True)
    pbar = tqdm(total=size, unit="B", unit_scale=True, unit_divisor=1024,
                desc=remote, mininterval=2)
    for attempt in range(1, RETRIES + 1):
        try:
            pbar.reset()
            body = ProgressFile(path, pbar)
            r = requests.put(
                f"{bucket}/{remote}",
                data=body,
                params={"access_token": ACCESS_TOKEN},
                headers={"Content-Type": "application/octet-stream",
                         "Content-Length": str(size)},
                timeout=TIMEOUT,
            )
            body.close()
            if r.status_code in (200, 201):
                pbar.close()
                print(f"  OK ({remote})", flush=True)
                return True
            print(f"\n  HTTP {r.status_code}: {r.text[:300]}", flush=True)
        except requests.RequestException as e:
            print(f"\n  attempt {attempt}/{RETRIES} failed: {e}", flush=True)
        time.sleep(min(15, 3 * attempt))   # fast retry cadence (night push)
    pbar.close()
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", action="store_true",
                    help="serially upload the *_NNN.zip slices "
                         "(skippd first, then folsom; skips originals)")
    ap.add_argument("--dataset", default=None,
                    help="with --parts: restrict to one dataset's slices "
                         "(e.g. skippd / folsom / sunlake)")
    ap.add_argument("--only", default=None,
                    help="upload just this filename (e.g. gefcom2014.zip)")
    ap.add_argument("--list", action="store_true",
                    help="list files already in the deposit and exit")
    args = ap.parse_args()

    info = deposit_info()
    bucket = info["links"]["bucket"]
    existing = {f["filename"] for f in info.get("files", [])}
    print(f"deposit {DEPOSIT_ID}: {len(existing)} file(s) already present")
    for f in sorted(existing):
        print("  -", f)
    if args.list:
        return 0

    if args.parts:
        # slices only (skippd_001.zip ...), originals excluded; dataset
        # order by first-slice size so the smaller set completes first
        datasets = sorted({p.name.rsplit("_", 1)[0]
                           for p in DIR.glob("*_[0-9][0-9][0-9].zip")},
                          key=lambda d: next(DIR.glob(f"{d}_[0-9][0-9][0-9].zip")).stat().st_size)
        if args.dataset:
            datasets = [d for d in datasets if d == args.dataset]
        todo = [p for d in datasets
                for p in sorted(DIR.glob(f"{d}_[0-9][0-9][0-9].zip"))]
    else:
        zips = sorted(DIR.glob("*.zip"), key=lambda p: p.stat().st_size)
        todo = [p for p in zips if args.only is None or p.name == args.only]
    if not todo:
        print("no matching zip files");  return 1

    failed = []
    n_parts = len(todo)
    for i, p in enumerate(todo, 1):
        if p.name in existing:
            print(f"\n=== [{i}/{n_parts}] {p.name}: already on Zenodo, skipping "
                  "(delete it there first to re-upload) ===")
            continue
        print(f"\n>>> slice {i}/{n_parts}", flush=True)
        if not upload_one(bucket, p):
            failed.append(p.name)
    print("\nsummary:", "all uploaded" if not failed else f"FAILED: {failed}")
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
