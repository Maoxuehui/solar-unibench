# Solar-UniBench Phase 4: Image/Zarr Layer (sky images BRR + satellite optical-flow) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the high-dimensional tensor layer: a BRR-augmented sky-image zarr store (Folsom, SKIPPD) and a satellite optical-flow extrapolation + uncertainty zarr store (SunLake), all config-driven for full scale and validated on bounded subsets.

**Architecture:** Three pure operators (`brr`, `optical_flow`, `zarr_io`) + three dataset builders that stream each dataset's raw images through the operators into zarr stores. Builders are invoked from each adapter's `process_and_save_tensors` (already a no-op hook on `BaseSolarPipeline`). All builders are **bounded by a `limit`/`year` config** so Phase 4 validates on a subset; full scale (Folsom 49 GB, full SunLake imagery) is a resumable batch job, not an in-session run.

**Tech Stack:** Python 3.10 (`solar_uni`), zarr 2.18.3, opencv-python-headless 5.0.0 (`cv2`), Pillow, numpy, deeplake (SunLake). Windows 11 / Git Bash.

## Global Constraints

- **Env:** conda env **`solar_uni`** (3.10.20) via `conda run -n solar_uni ...`. zarr + opencv-python-headless + Pillow already installed. **Never use multi-line `python -c` with `conda run`** — write probe scripts to files.
- **Branch:** `feat/phase4-images-zarr` (created from master). Commit per task.
- **TDD:** pure operators get synthetic-fixture unit tests; builders get a small bounded real-data integration test (`RUN_SLOW`-gated where slow).
- **Scale = config-driven.** Every builder accepts a `limit` (max frames) or `year` bound. Full-scale runs are background batch jobs; the plan validates on bounded subsets (e.g. ≤ a few thousand frames).
- **Contract (spec §4.3/4.4):** `sky_images.zarr` shape `(N, 4, 256, 256)` float32, channels = R, G, B (normalized 0–1), **BRR** = `(B−R)/(B+R)`; chunks `(1, 4, 256, 256)`. `satellite_future.zarr` → `context (N, C, H, W)`, `future (K, N, C, H, W)`, `uncertainty_map (K, N, H, W)`; default `K=12`.
- **Missing modalities are absent files** (Honest NaN principle): SKIPPD has sky images only (no satellite → no `satellite_future.zarr`); GEFCom has no images.
- **UTC:** Folsom sky filenames are UTC (`YYYYMMDD_HHMMSS.jpg`); SKIPPD `time` is US/Pacific → UTC; SunLake `time_utc` is UTC.

---

## File Structure

```
solar-unibench/
├── solarunibench/operators/
│   ├── brr.py                 # Task 1
│   ├── zarr_io.py             # Task 2
│   └── optical_flow.py        # Task 3
├── solarunibench/imaging/     # new subpackage
│   ├── __init__.py            # Task 1
│   ├── skippd_sky.py          # Task 4
│   ├── folsom_sky.py          # Task 5
│   └── sunlake_satellite.py   # Task 6
└── tests/
    ├── test_brr.py            # Task 1
    ├── test_zarr_io.py        # Task 2
    ├── test_optical_flow.py   # Task 3
    ├── test_skippd_sky.py     # Task 4
    ├── test_folsom_sky.py     # Task 5
    └── test_sunlake_satellite.py  # Task 6
```

---

### Task 1: BRR channel operator

**Files:**
- Create: `solarunibench/imaging/__init__.py`, `solarunibench/operators/brr.py`, `tests/test_brr.py`

**Interfaces:**
- Produces: `generate_brr_channel(rgb: np.ndarray) -> np.ndarray` — input `(H, W, 3)` uint8 (or float 0–255), output `(4, H, W)` float32 = `[R/255, G/255, B/255, BRR]` where `BRR = (B−R)/(B+R+eps)` clipped to `[−1, 1]`.

- [ ] **Step 1: Write the failing test `tests/test_brr.py`**

```python
import numpy as np
from solarunibench.operators.brr import generate_brr_channel


def test_output_shape_and_channels():
    img = np.zeros((10, 12, 3), dtype=np.uint8)
    out = generate_brr_channel(img)
    assert out.shape == (4, 10, 12)
    assert out.dtype == np.float32


def test_brr_formula_and_bounds():
    # pure-blue image: BRR = (B-R)/(B+R) = 1 ; pure-red: -1
    blue = np.zeros((4, 4, 3), dtype=np.uint8); blue[..., 2] = 200
    red = np.zeros((4, 4, 3), dtype=np.uint8); red[..., 0] = 200
    ob = generate_brr_channel(blue)[3]
    orn = generate_brr_channel(red)[3]
    assert np.allclose(ob, 1.0, atol=1e-3)
    assert np.allclose(orn, -1.0, atol=1e-3)


def test_rgb_normalized_to_unit():
    img = np.full((2, 2, 3), 255, dtype=np.uint8)
    out = generate_brr_channel(img)
    assert np.allclose(out[0], 1.0) and np.allclose(out[1], 1.0) and np.allclose(out[2], 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n solar_uni python -m pytest tests/test_brr.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `solarunibench/imaging/__init__.py` (empty docstring) and `solarunibench/operators/brr.py`**

`solarunibench/imaging/__init__.py`:
```python
"""Image/tensor builders: sky images (BRR) + satellite optical-flow zarr stores."""
```

`solarunibench/operators/brr.py`:
```python
"""BRR (Blue-to-Red Ratio) channel operator: RGB -> 4-channel [R,G,B,BRR]."""
import numpy as np


def generate_brr_channel(rgb: np.ndarray) -> np.ndarray:
    """rgb: (H, W, 3) uint8 (or float 0-255) -> (4, H, W) float32 [R/255,G/255,B/255,BRR]."""
    img = np.asarray(rgb, dtype=np.float32) / 255.0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    eps = 1e-6
    brr = np.clip((b - r) / (b + r + eps), -1.0, 1.0)
    return np.stack([r, g, b, brr], axis=0).astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n solar_uni python -m pytest tests/test_brr.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add solarunibench/imaging/__init__.py solarunibench/operators/brr.py tests/test_brr.py
git commit -m "feat(operators): BRR (blue-red ratio) channel operator

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Zarr I/O helpers

**Files:**
- Create: `solarunibench/operators/zarr_io.py`, `tests/test_zarr_io.py`

**Interfaces:**
- Produces: `SkyImageZarr` class — opens/creates a `(N,4,256,256)` float32 zarr array (chunks `(1,4,256,256)`) and appends frames; and `write_array(path, name, data, chunks)` generic helper.

- [ ] **Step 1: Write the failing test `tests/test_zarr_io.py`**

```python
import numpy as np
import zarr
from solarunibench.operators.zarr_io import SkyImageZarr, write_array


def test_sky_zarr_append_and_read(tmp_path):
    z = SkyImageZarr(tmp_path / "sky.zarr", overwrite=True)
    frame = np.random.rand(4, 256, 256).astype(np.float32)
    z.append(frame)
    z.append(frame)
    arr = zarr.open(str(tmp_path / "sky.zarr"), mode="r")["images"]
    assert arr.shape == (2, 4, 256, 256)
    assert arr.chunks == (1, 4, 256, 256)
    assert arr.dtype == np.float32


def test_write_array(tmp_path):
    data = np.arange(2 * 3, dtype=np.float32).reshape(2, 3)
    write_array(tmp_path / "x.zarr", "ctx", data, chunks=(1, 3))
    arr = zarr.open(str(tmp_path / "x.zarr"), mode="r")["ctx"]
    assert arr.shape == (2, 3)
    np.testing.assert_array_equal(arr[:], data)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n solar_uni python -m pytest tests/test_zarr_io.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `solarunibench/operators/zarr_io.py`**

```python
"""Zarr tensor I/O for sky images and satellite cubes (zarr 2.x API)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import zarr


class SkyImageZarr:
    """Append-only (N, 4, 256, 256) float32 sky-image store, chunked (1,4,256,256)."""

    def __init__(self, path, overwrite: bool = True):
        self.path = Path(path)
        self.store = zarr.DirectoryStore(str(self.path))
        self.root = zarr.group(store=self.store, overwrite=overwrite)
        self._arr = None
        self._n = 0

    def _ensure(self):
        if self._arr is None:
            self._arr = self.root.empty("images", shape=(0, 4, 256, 256),
                                        chunks=(1, 4, 256, 256), dtype="float32")

    def append(self, frame: np.ndarray) -> None:
        """frame: (4, 256, 256) float32."""
        self._ensure()
        f = np.asarray(frame, dtype=np.float32)[None, ...]  # (1,4,256,256)
        self._arr.append(f)
        self._n += 1

    @property
    def n(self) -> int:
        return self._n


def write_array(path, name: str, data: np.ndarray, chunks) -> None:
    path = Path(path)
    store = zarr.DirectoryStore(str(path))
    root = zarr.group(store=store, overwrite=True)
    root.array(name, data=np.asarray(data), chunks=chunks, overwrite=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n solar_uni python -m pytest tests/test_zarr_io.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add solarunibench/operators/zarr_io.py tests/test_zarr_io.py
git commit -m "feat(operators): zarr tensor I/O (sky images + generic array)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Optical-flow extrapolation + uncertainty

**Files:**
- Create: `solarunibench/operators/optical_flow.py`, `tests/test_optical_flow.py`

**Interfaces:**
- Produces: `extrapolate_with_uncertainty(flow_u, flow_v, frame_t1, steps_k=12) -> (future, uncertainty)`:
  - `flow_u`, `flow_v`: `(H, W)` float32 per-pixel motion (precomputed; e.g. SunLake `ctx_opt_flow`).
  - `frame_t1`: `(H, W)` float32 current frame (single channel).
  - Returns `future (steps_k, H, W)` float32 (semi-Lagrangian warps) and `uncertainty (steps_k, H, W)` float32 in `[0,1]` (divergence + lead-time decay).

- [ ] **Step 1: Write the failing test `tests/test_optical_flow.py`**

```python
import numpy as np
from solarunibench.operators.optical_flow import extrapolate_with_uncertainty


def _shifted_pair(shift=2):
    # a bright square on a dark background, shifted right by `shift` px
    f0 = np.zeros((32, 32), np.float32); f0[10:20, 10:20] = 1.0
    f1 = np.zeros((32, 32), np.float32); f1[10:20, 10+shift:20+shift] = 1.0
    u = np.full((32, 32), -float(shift), np.float32)  # motion: frame moves right => sample left
    v = np.zeros((32, 32), np.float32)
    return f0, f1, u, v


def test_extrapolation_shape_and_bounds():
    f0, f1, u, v = _shifted_pair()
    future, unc = extrapolate_with_uncertainty(u, v, f1, steps_k=4)
    assert future.shape == (4, 32, 32) and future.dtype == np.float32
    assert unc.shape == (4, 32, 32)
    assert np.all(unc >= 0.0) and np.all(unc <= 1.0)


def test_uncertainty_grows_with_lead_time():
    f0, f1, u, v = _shifted_pair()
    _, unc = extrapolate_with_uncertainty(u, v, f1, steps_k=5)
    mean_by_step = unc.mean(axis=(1, 2))
    assert np.all(np.diff(mean_by_step) <= 1e-6)  # non-increasing confidence (uncertainty rises or flat)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n solar_uni python -m pytest tests/test_optical_flow.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `solarunibench/operators/optical_flow.py`**

```python
"""Optical-flow extrapolation + uncertainty (semi-Lagrangian warp + divergence decay).

Takes a PRECOMPUTED flow field (u, v) so the extrapolation is decoupled from how the
flow was obtained (SunLake ships ctx_opt_flow; synthetic tests pass a known flow).
"""
import numpy as np
import cv2


def extrapolate_with_uncertainty(flow_u: np.ndarray, flow_v: np.ndarray,
                                 frame_t1: np.ndarray, steps_k: int = 12):
    """Extrapolate `steps_k` future frames by repeated semi-Lagrangian warp of frame_t1.

    Returns (future (steps_k,H,W) float32, uncertainty (steps_k,H,W) float32 in [0,1]).
    Uncertainty rises with lead step and with local flow divergence.
    """
    u = np.asarray(flow_u, dtype=np.float32)
    v = np.asarray(flow_v, dtype=np.float32)
    frame = np.asarray(frame_t1, dtype=np.float32)
    h, w = frame.shape

    div = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3) + cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3)
    divergence = np.abs(div)

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    alpha, beta = 0.02, 0.05  # lead-time decay / divergence penalty (per methodology)
    futures, unc = [], []
    warped = frame.copy()
    for k in range(1, steps_k + 1):
        map_x = (xx + k * u).astype(np.float32)
        map_y = (yy + k * v).astype(np.float32)
        warped = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        futures.append(warped)
        u_map = np.exp(-alpha * k) * (1.0 - np.minimum(1.0, beta * k * divergence))
        unc.append(np.clip(u_map, 0.0, 1.0).astype(np.float32))
    return np.asarray(futures, dtype=np.float32), np.asarray(unc, dtype=np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n solar_uni python -m pytest tests/test_optical_flow.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add solarunibench/operators/optical_flow.py tests/test_optical_flow.py
git commit -m "feat(operators): optical-flow extrapolation + uncertainty

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: SKIPPD sky-image builder

**Files:**
- Create: `solarunibench/imaging/skippd_sky.py`, `tests/test_skippd_sky.py`

**Interfaces:**
- Consumes: `operators.brr.generate_brr_channel`, `operators.zarr_io.SkyImageZarr`.
- Produces: `build_skippd_sky_zarr(parquet_dir, out_path, limit=500) -> dict` — reads `default/{train,test}/*.parquet`, decodes the PNG-bytes `image` dicts, resizes to 256×256, applies BRR, writes `sky_images.zarr`. Returns `{n_frames, out_path}`. `limit` caps frames (full-scale = omit/raise).

- [ ] **Step 1: Write the failing test `tests/test_skippd_sky.py`**

```python
import io
import numpy as np
import pandas as pd
import zarr
from pathlib import Path
from PIL import Image
from solarunibench.imaging.skippd_sky import build_skippd_sky_zarr


def _write_fixture(root: Path, n=3):
    for split in ("train", "test"):
        d = root / "default" / split
        d.mkdir(parents=True)
        rows = []
        for i in range(n):
            arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
            buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="PNG")
            rows.append({"image": {"bytes": buf.getvalue(), "path": f"{i}.png"},
                         "time": pd.Timestamp(f"2018-06-01 12:{i:02d}", tz="US/Pacific"), "pv": 0.5})
        pd.DataFrame(rows).to_parquet(d / "0000.parquet")


def test_build_skippd_sky_zarr(tmp_path):
    _write_fixture(tmp_path, n=3)
    out = tmp_path / "sky.zarr"
    res = build_skippd_sky_zarr(tmp_path / "default", out, limit=4)
    arr = zarr.open(str(out), mode="r")["images"]
    assert arr.shape[0] == 4  # 3 train + 1 test (capped at limit=4)
    assert arr.shape[1:] == (4, 256, 256)
    assert arr.dtype == np.float32
    assert res["n_frames"] == arr.shape[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n solar_uni python -m pytest tests/test_skippd_sky.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `solarunibench/imaging/skippd_sky.py`**

```python
"""SKIPPD sky-image -> sky_images.zarr (RGB+BRR, 256x256). Reads default/{train,test} parquet."""
from __future__ import annotations
from pathlib import Path
import io
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from ..operators.brr import generate_brr_channel
from ..operators.zarr_io import SkyImageZarr


def _decode_resize(img_entry, size=256):
    buf = io.BytesIO(img_entry["bytes"])
    arr = np.array(Image.open(buf).convert("RGB"))
    if arr.shape[:2] != (size, size):
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    return arr


def build_skippd_sky_zarr(parquet_dir, out_path, limit=500) -> dict:
    parquet_dir = Path(parquet_dir)
    store = SkyImageZarr(out_path, overwrite=True)
    n = 0
    for split in ("train", "test"):
        d = parquet_dir / split
        if not d.exists():
            continue
        for f in sorted(d.glob("*.parquet")):
            df = pd.read_parquet(f)
            for img in df["image"]:
                if img is None or not isinstance(img, dict) or not img.get("bytes"):
                    continue
                frame = generate_brr_channel(_decode_resize(img))
                store.append(frame)
                n += 1
                if limit and n >= limit:
                    return {"n_frames": n, "out_path": str(out_path)}
    return {"n_frames": n, "out_path": str(out_path)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n solar_uni python -m pytest tests/test_skippd_sky.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add solarunibench/imaging/skippd_sky.py tests/test_skippd_sky.py
git commit -m "feat(imaging): SKIPPD sky-image -> sky_images.zarr builder

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Folsom sky-image builder (stream tarball)

**Files:**
- Create: `solarunibench/imaging/folsom_sky.py`, `tests/test_folsom_sky.py`

**Interfaces:**
- Produces: `iter_folsom_sky_frames(tar_path, year=None)` generator yielding `(timestamp_utc, frame(4,256,256))`, and `build_folsom_sky_zarr(tar_paths, out_path, limit=500) -> dict`. Timestamps parsed from the filename `YYYYMMDD_HHMMSS.jpg` as UTC.

- [ ] **Step 1: Write the failing test `tests/test_folsom_sky.py`**

```python
import io, tarfile, numpy as np, zarr
from pathlib import Path
from PIL import Image
from solarunibench.imaging.folsom_sky import build_folsom_sky_zarr, iter_folsom_sky_frames
import pandas as pd


def _make_tar(path, names):
    with tarfile.open(path, "w:bz2") as tf:
        for nm in names:
            arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
            buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="JPEG")
            data = buf.getvalue()
            ti = tarfile.TarInfo(nm); ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))


def test_iter_frames_parses_utc_timestamp(tmp_path):
    tar = tmp_path / "2014.tar.bz2"
    _make_tar(tar, ["2014/06/01/20140601_120000.jpg", "2014/06/01/20140601_120100.jpg"])
    frames = list(iter_folsom_sky_frames(tar))
    assert len(frames) == 2
    ts0, f0 = frames[0]
    assert ts0 == pd.Timestamp("2014-06-01 12:00:00", tz="UTC")
    assert f0.shape == (4, 256, 256)


def test_build_folsom_sky_zarr(tmp_path):
    tar = tmp_path / "2014.tar.bz2"
    _make_tar(tar, [f"2014/06/01/20140601_1200{i:02d}.jpg" for i in range(5)])
    out = tmp_path / "sky.zarr"
    res = build_folsom_sky_zarr([tar], out, limit=3)
    arr = zarr.open(str(out), mode="r")["images"]
    assert arr.shape == (3, 4, 256, 256)
    assert res["n_frames"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n solar_uni python -m pytest tests/test_folsom_sky.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `solarunibench/imaging/folsom_sky.py`**

```python
"""Folsom sky-image tarball -> sky_images.zarr. Streams bz2 tar; timestamps from filename (UTC)."""
from __future__ import annotations
import io
import tarfile
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from ..operators.brr import generate_brr_channel
from ..operators.zarr_io import SkyImageZarr


def _decode(member_bytes, size=256):
    arr = np.array(Image.open(io.BytesIO(member_bytes)).convert("RGB"))
    if arr.shape[:2] != (size, size):
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    return arr


def _ts_from_name(name: str):
    stem = Path(name).stem  # YYYYMMDD_HHMMSS
    return pd.to_datetime(stem, format="%Y%m%d_%H%M%S").tz_localize("UTC")


def iter_folsom_sky_frames(tar_path, size=256):
    """Yield (timestamp_utc, frame(4,size,size)) for every .jpg in the tar.bz2."""
    with tarfile.open(tar_path, "r:bz2") as tf:
        for m in tf:
            if not m.isfile() or not m.name.lower().endswith(".jpg"):
                continue
            data = tf.extractfile(m).read()
            yield _ts_from_name(m.name), generate_brr_channel(_decode(data, size))


def build_folsom_sky_zarr(tar_paths, out_path, limit=500) -> dict:
    store = SkyImageZarr(out_path, overwrite=True)
    n = 0
    for tp in tar_paths:
        for _ts, frame in iter_folsom_sky_frames(tp):
            store.append(frame)
            n += 1
            if limit and n >= limit:
                return {"n_frames": n, "out_path": str(out_path)}
    return {"n_frames": n, "out_path": str(out_path)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n solar_uni python -m pytest tests/test_folsom_sky.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add solarunibench/imaging/folsom_sky.py tests/test_folsom_sky.py
git commit -m "feat(imaging): Folsom sky-image tarball -> sky_images.zarr builder

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: SunLake satellite builder (context + optical-flow extrapolation)

**Files:**
- Create: `solarunibench/imaging/sunlake_satellite.py`, `tests/test_sunlake_satellite.py`

**Interfaces:**
- Consumes: `operators.optical_flow.extrapolate_with_uncertainty`, `operators.zarr_io.write_array`, deeplake (lazy).
- Produces: `build_sunlake_satellite_zarr(hub_path, out_path, station="CAB", year="2008", limit_frames=200, steps_k=12) -> dict`. Pulls a bounded slice of SunLake `context/data` + `ctx_opt_flow/data` from the hub, writes `context` `(M,11,64,64)`, and for consecutive frame pairs computes `future (K,M-1,11,64,64)` + `uncertainty_map (K,M-1,64,64)` using the precomputed flow.

- [ ] **Step 1: Write the failing test `tests/test_sunlake_satellite.py`** (pure logic — does NOT hit the hub)

```python
import numpy as np
from solarunibench.imaging.sunlake_satellite import extrapolate_context_series


def test_extrapolate_context_series_shape():
    # 5 frames, 2 channels, 8x8; flow is 2*(8,8)
    ctx = np.random.rand(5, 2, 8, 8).astype(np.float32)
    flow = np.random.rand(5, 4, 8, 8).astype(np.float32)  # (vx,vy) per channel pair -> use 2 here
    future, unc = extrapolate_context_series(ctx, flow, channels=[0, 1], steps_k=3)
    assert future.shape == (3, 4, 2, 8, 8)   # K, M-1, C, H, W
    assert unc.shape == (3, 4, 8, 8)          # K, M-1, H, W
    assert np.all(unc >= 0) and np.all(unc <= 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n solar_uni python -m pytest tests/test_sunlake_satellite.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `solarunibench/imaging/sunlake_satellite.py`**

```python
"""SunLake satellite context -> satellite_future.zarr (optical-flow extrapolation + uncertainty).

Uses SunLake's PRECOMPUTED ctx_opt_flow (vx/vy per SEVIRI channel). Bounded by limit_frames.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from ..operators.optical_flow import extrapolate_with_uncertainty
from ..operators.zarr_io import write_array

CTX_CHANNELS = ["IR_016", "IR_039", "IR_087", "IR_097", "IR_108", "IR_120", "IR_134",
                "VIS006", "VIS008", "WV_062", "WV_073"]


def extrapolate_context_series(context: np.ndarray, flow: np.ndarray,
                               channels=None, steps_k: int = 12):
    """context (M, C, H, W); flow (M, 2C, H, W) with [vx_c0, vy_c0, vx_c1, vy_c1, ...].
    Returns future (steps_k, M-1, C, H, W), uncertainty (steps_k, M-1, H, W).
    For each consecutive pair (t, t+1), warp frame t+1 by the flow at t+1 for K steps.
    """
    M, C, H, W = context.shape
    channels = list(range(C)) if channels is None else channels
    futures, uncs = [], []
    for t in range(M - 1):
        frame_next = context[t + 1]            # (C,H,W)
        f_t = flow[t + 1]                       # (2C, H, W); layout [vx0,vy0,vx1,vy1,...]
        # representative (shared) uncertainty from channel 0 (per spec: one map per timestep)
        vx0, vy0 = f_t[2 * channels[0]], f_t[2 * channels[0] + 1]
        _, unc_per_ch = extrapolate_with_uncertainty(vx0, vy0, frame_next[channels[0]], steps_k=steps_k)
        fut_per_ch = []
        for ci in channels:
            vx, vy = f_t[2 * ci], f_t[2 * ci + 1]
            f, _ = extrapolate_with_uncertainty(vx, vy, frame_next[ci], steps_k=steps_k)
            fut_per_ch.append(f)                # (K,H,W) per channel
        futures.append(np.stack(fut_per_ch, axis=1))  # (K, C, H, W)
        uncs.append(unc_per_ch)                       # (K, H, W)
    future = np.stack(futures, axis=1)   # (K, M-1, C, H, W)
    unc = np.stack(uncs, axis=1)         # (K, M-1, H, W)
    return future.astype(np.float32), unc.astype(np.float32)


def build_sunlake_satellite_zarr(hub_path: str, out_path, station="CAB", year="2008",
                                 limit_frames=200, steps_k=12) -> dict:
    import deeplake  # lazy
    ds = deeplake.load(hub_path)
    pfx = f"{year}_nonhrv"
    ctx = ds[f"{pfx}/context/data"][0].numpy()        # (M, 11, 64, 64)
    flow = ds[f"{pfx}/ctx_opt_flow/data"][0].numpy()  # (M, 22, 64, 64)
    if limit_frames:
        ctx = ctx[:limit_frames]; flow = flow[:limit_frames]
    future, unc = extrapolate_context_series(ctx, flow, channels=list(range(ctx.shape[1])),
                                             steps_k=steps_k)
    out_path = Path(out_path)
    write_array(out_path / "satellite_future.zarr", "context", ctx, chunks=(1, 11, 64, 64))
    write_array(out_path / "satellite_future.zarr", "future", future, chunks=(1, 1, 11, 64, 64))
    write_array(out_path / "satellite_future.zarr", "uncertainty_map", unc, chunks=(1, 1, 64, 64))
    return {"station": station, "year": year, "frames": int(ctx.shape[0]),
            "steps_k": steps_k, "out_path": str(out_path / "satellite_future.zarr")}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n solar_uni python -m pytest tests/test_sunlake_satellite.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add solarunibench/imaging/sunlake_satellite.py tests/test_sunlake_satellite.py
git commit -m "feat(imaging): SunLake satellite context -> optical-flow extrapolation zarr

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Wire builders into adapters + bounded real-data integration test

**Files:**
- Modify: `solarunibench/pipeline/adapters/skippd.py` (`process_and_save_tensors`), `solarunibench/pipeline/adapters/folsom.py` (`process_and_save_tensors`), `solarunibench/pipeline/adapters/sunlake.py` (`process_and_save_tensors`), `solarunibench/pipeline/base.py` (pass `output_dir`/config to the hook if needed).
- Create: `tests/test_imaging_integration.py` (RUN_SLOW).

**Interfaces:**
- Each adapter's `process_and_save_tensors(df_ts)` now calls its builder with a bounded `limit` (from `self.config.get("image_limit", 200)`) and writes `<output_dir>/<dataset>/sky_images.zarr` (or `satellite_future.zarr`) next to the parquet.

- [ ] **Step 1: Read the three adapters' `process_and_save_tensors` + `base.run_pipeline`** to confirm the hook receives `df_ts` and the instance has `self.output_dir`/`self.config`. Use the `Read` tool on the adapter files and `solarunibench/pipeline/base.py`.

- [ ] **Step 2: Wire SKIPPD** — in `skippd.py` `process_and_save_tensors`, call the builder on the SKIPPD_all `default` dir (derive from `self.raw_data_dir`) with `limit=self.config.get("image_limit", 200)`:
```python
def process_and_save_tensors(self, df_ts):
    from ...imaging.skippd_sky import build_skippd_sky_zarr
    limit = self.config.get("image_limit", 200)
    build_skippd_sky_zarr(Path(self.raw_data_dir) / "default", self.output_dir / "sky_images.zarr", limit=limit)
```
(Analogous for Folsom — call `build_folsom_sky_zarr` on the year tarballs present in `self.raw_data_dir`, `limit=config`; and SunLake — call `build_sunlake_satellite_zarr` on `hub://crossvivit/SunLake`, bounded.)

- [ ] **Step 3: Write `tests/test_imaging_integration.py`** (RUN_SLOW-gated). Runs each builder on a tiny bounded real slice and asserts the zarr exists with the right shape:
```python
import os, pytest, zarr
DATA = os.environ.get("SOLAR_UNIBENCH_DATA_ROOT", "D:/monica/workspace/solar_benchmark_dataset")
slow = pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="set RUN_SLOW=1")

@slow
def test_real_skippd_sky(tmp_path):
    from solarunibench.imaging.skippd_sky import build_skippd_sky_zarr
    res = build_skippd_sky_zarr(f"{DATA}/SKIPPD_all/default", tmp_path/"sky.zarr", limit=50)
    arr = zarr.open(str(tmp_path/"sky.zarr"), mode="r")["images"]
    assert arr.shape == (res["n_frames"], 4, 256, 256)

@slow
def test_real_folsom_sky(tmp_path):
    from solarunibench.imaging.folsom_sky import build_folsom_sky_zarr
    res = build_folsom_sky_zarr([f"{DATA}/Folsom_data/Folsom_sky_images_2014.tar.bz2"], tmp_path/"sky.zarr", limit=50)
    arr = zarr.open(str(tmp_path/"sky.zarr"), mode="r")["images"]
    assert arr.shape[0] == 50 and arr.shape[1:] == (4, 256, 256)

@slow
def test_real_sunlake_satellite(tmp_path):
    from solarunibench.imaging.sunlake_satellite import build_sunlake_satellite_zarr
    res = build_sunlake_satellite_zarr("hub://crossvivit/SunLake", tmp_path, limit_frames=20, steps_k=4)
    z = zarr.open(str(tmp_path/"satellite_future.zarr"), mode="r")
    assert z["context"].shape[0] == 20 and z["future"].shape[0] == 4
```

- [ ] **Step 4: Verify** — `conda run -n solar_uni python -m pytest -q` (fast green, slow skipped). Then `RUN_SLOW=1 conda run -n solar_uni python -m pytest tests/test_imaging_integration.py -v` → all 3 PASS on real data. Report frame counts + timing.

- [ ] **Step 5: Commit**

```bash
git add solarunibench/pipeline/adapters/ tests/test_imaging_integration.py
git commit -m "feat(imaging): wire sky/satellite builders into adapters + integration tests

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Definition of Done (Phase 4)

- `conda run -n solar_uni python -m pytest -q` → all fast unit tests green.
- The 3 operators (`brr`, `zarr_io`, `optical_flow`) are pure + unit-tested.
- `RUN_SLOW=1` integration: SKIPPD sky (≥50 frames), Folsom sky (50 frames from the 2014 tarball), SunLake satellite (20 context frames → extrapolation) each write a correctly-shaped zarr.
- Each adapter's `process_and_save_tensors` writes the image zarr (bounded by `image_limit` config).

## Scale note (full batch = later)
Full-scale (Folsom 49 GB every frame; full SunLake imagery) is a resumable background batch job — the builders are `limit`-bounded and streaming, so raising/removing the limit + running as a background job covers it. This phase validates correctness on bounded subsets.

## Open items (out of scope, noted)
- SunLake imagery is pulled on-demand from the hub per-build (not materialized locally); a full local materialization is a Phase 4+ ops task if offline repro is needed.
- SKIPPD has no satellite in this snapshot → no `satellite_future.zarr` for SKIPPD (Honest absence).
- Optical-flow uses SunLake's precomputed `ctx_opt_flow`; the spec's TV-L1 compute path is available via `cv2.calcOpticalFlowFarneback` if a compute-it-yourself variant is ever needed.
