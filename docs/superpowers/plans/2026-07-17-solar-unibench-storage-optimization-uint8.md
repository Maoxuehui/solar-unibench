# Sky-Image Zarr Storage Optimization (uint8 + JPEG) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace float32 sky-image zarr with per-dataset compact storage so full materialization is feasible AND downloadable: **Folsom** (photographic, source=JPEG) → JPEG-q90 RGB bytes @256² (~12 GB); **SKIPPD** (source=PNG) → lossless uint8 RGB+BRR @64² (~6 GB). Zero-or-source-fidelity loss; simple/fast reads.

**Architecture (heterogeneous — each format matches its source):**
- `image_codec` (DONE): uint8 encode/decode for the lossless path (SKIPPD).
- `SkyImageZarr` (DONE): uint8 (4,H,W) + lz4 SHUFFLE; infers (H,W) from first frame.
- `JPEGSkyStore` (NEW): append-only store of Pillow-JPEG-encoded RGB bytes (zarr `VLenBytes` object array) — for Folsom. BRR is NOT stored; computed on load.
- Resolution policy (NEW): `downsample_if_larger(arr, 256)` — native Folsom 1536→256; SKIPPD 64 stays 64.
- Unified read (NEW): `load_sky_frame(cube, idx) -> (4,H,W) float32` branches on storage — JPEG bytes → Pillow decode → `generate_brr_channel`; uint8 4ch → `decode_image`. `SolarUniBenchDataset` uses it, so models always see a uniform `(lookback,4,H,W)` float tensor.

**Tech Stack:** zarr 2.18.3, numcodecs 0.13.1 (Blosc, VLenBytes — **no JPEG codec**, so Pillow does JPEG), numpy, Pillow 10.4.0, pytorch. Env `solar_uni`.

## Global Constraints

- **Per-dataset storage (decided with user):**
  - Folsom: JPEG-q90 RGB bytes @256×256, zarr `VLenBytes` object array `images` shape `(N,)`; BRR computed on load. Lossy ≈ source (source is JPEG). Target ~12 GB full.
  - SKIPPD: uint8 RGB+BRR (4ch) @64×64, lz4 SHUFFLE, lossless. `images` shape `(N,4,64,64)`. Target ~6 GB full.
  - SunLake satellite stays float32; GEFCom has no images.
- **Resolution policy:** `downsample_if_larger(rgb_hwc, target=256)` — downsample only if H>target or W>target (INTER_AREA); never upsample. Folsom 1536→256, SKIPPD 64→64.
- **Single source of truth:** uint8 maps in `image_codec`; JPEG encode/decode in `JPEGSkyStore` + `load_sky_frame` (reuses `generate_brr_channel` for BRR — do not duplicate the BRR formula).
- **Unified model input:** `SolarUniBenchDataset.__getitem__["image"]` is ALWAYS `(lookback,4,H,W)` float32 [RGB∈0,1, BRR∈-1,1], regardless of the per-dataset storage. The branch lives in `load_sky_frame`.
- **Env:** `conda run -n solar_uni ...`; run pytest from repo root (or `PYTHONPATH=repo`). TDD, DRY, YAGNI, frequent commits; messages end with `Co-Authored-By: Claude <noreply@anthropic.com>`.

**Branch:** `feat/uint8-zarr-storage` (from `master` @ `72ca89d`).

---

### Task 1: uint8 image codec — ✅ DONE (`ef4b720`)
`encode_image`/`decode_image`, lossless RGB/BRR maps, round-trip ≤1/255.

### Task 2: SkyImageZarr uint8 + lz4 Blosc — ✅ DONE (`876c849`)
uint8 store, configurable compressor (default lz4 SHUFFLE).

### Task 3: Dataset decode + fail-fast — ✅ DONE (`2dc862a`)
`decode_image` on read for the 4ch uint8 path (SKIPPD); fail-fast `ValueError` on missing columns. (Decode-test strength depends on the fixture fix in Task 6.)

---

### Task 4: Resolution policy — store at native-or-256

**Files:**
- Create: `solarunibench/imaging/resize.py` (`downsample_if_larger`)
- Modify: `solarunibench/operators/zarr_io.py` (`SkyImageZarr` infers (H,W) from first appended frame)
- Modify: `solarunibench/imaging/folsom_sky.py`, `solarunibench/imaging/skippd_sky.py` (apply policy before encode)
- Test: `tests/test_resolution_policy.py`

**Interfaces:**
- Produces: `downsample_if_larger(rgb_hwc, target=256) -> rgb_hwc`; `SkyImageZarr` array shape inferred from first `append` (no hardcoded 256).

- [ ] **Step 1: Write failing test**

```python
# tests/test_resolution_policy.py
import numpy as np, zarr
from solarunibench.imaging.resize import downsample_if_larger
from solarunibench.operators.zarr_io import SkyImageZarr

def test_downsample_only_when_larger():
    assert downsample_if_larger(np.zeros((1536,1536,3),np.uint8),256).shape[:2]==(256,256)
    assert downsample_if_larger(np.zeros((64,64,3),np.uint8),256).shape[:2]==(64,64)   # unchanged
    assert downsample_if_larger(np.zeros((256,256,3),np.uint8),256).shape[:2]==(256,256)

def test_sky_image_zarr_infers_size(tmp_path):
    s = SkyImageZarr(tmp_path/"a.zarr", overwrite=True)
    s.append(np.zeros((4,64,64),np.float32))
    assert zarr.open(str(tmp_path/"a.zarr"),mode="r")["images"].shape == (1,4,64,64)
```

- [ ] **Step 2: Run to verify it fails** — `ImportError: solarunibench.imaging.resize` / SkyImageZarr hardcodes 256.
- [ ] **Step 3: Implement** — `solarunibench/imaging/resize.py`:

```python
"""Resolution policy: keep native if <= target, else downsample (never upsample)."""
from __future__ import annotations
import cv2, numpy as np

def downsample_if_larger(rgb_hwc: np.ndarray, target: int = 256) -> np.ndarray:
    h, w = rgb_hwc.shape[:2]
    if h <= target and w <= target:
        return rgb_hwc
    return cv2.resize(rgb_hwc, (target, target), interpolation=cv2.INTER_AREA)
```

`SkyImageZarr._ensure(frame)` infers `_,h,w = frame.shape`; array `(0,4,h,w)`, chunks `(1,4,h,w)`. `append` passes frame to `_ensure`. In `folsom_sky._decode` and `skippd_sky._decode_resize`, wrap decoded RGB with `downsample_if_larger(arr, 256)` before `generate_brr_channel`.
- [ ] **Step 4: Run to verify pass** — `pytest tests/test_resolution_policy.py tests/test_zarr_io.py -v`.
- [ ] **Step 5: Commit** — `feat(imaging): resolution policy (native if ≤256 else downsample); SkyImageZarr infers size`

---

### Task 5: JPEGSkyStore (Folsom lossy RGB bytes) + folsom_sky wiring

**Files:**
- Modify: `solarunibench/operators/zarr_io.py` (add `JPEGSkyStore`)
- Modify: `solarunibench/imaging/folsom_sky.py` (store RGB JPEG bytes via `JPEGSkyStore`; drop BRR-at-store — BRR computed on load)
- Test: `tests/test_jpeg_store.py`

**Interfaces:**
- Produces: `JPEGSkyStore(path, quality=90, overwrite=True)` with `.append(rgb_frame)` (rgb: (3,H,W) or (H,W,3) uint8/float ∈[0,1]) and `.n`; persists `images` as a zarr `VLenBytes` object array of JPEG bytes.

- [ ] **Step 1: Write failing test**

```python
# tests/test_jpeg_store.py
import numpy as np, zarr
from numcodecs import VLenBytes
from solarunibench.operators.zarr_io import JPEGSkyStore

def test_jpeg_store_roundtrip(tmp_path):
    st = JPEGSkyStore(tmp_path/"f.zarr", quality=90, overwrite=True)
    rgb = (np.random.rand(3,256,256)*255).astype(np.uint8)
    st.append(rgb); st.append(rgb)
    arr = zarr.open(str(tmp_path/"f.zarr"), mode="r")["images"]
    assert arr.shape == (2,) and arr.dtype == object   # VLenBytes
    b0 = bytes(arr[0]); assert b0[:2] == b"\xff\xd8"    # JPEG SOI marker
    from PIL import Image; import io
    dec = np.array(Image.open(io.BytesIO(b0)))          # (H,W,3)
    assert dec.shape == (256,256,3) and dec.dtype == np.uint8
    assert st.n == 2
```

- [ ] **Step 2: Run to verify it fails** — `ImportError: cannot import JPEGSkyStore`.
- [ ] **Step 3: Implement** — add to `zarr_io.py`:

```python
import io
from PIL import Image
from numcodecs import VLenBytes

class JPEGSkyStore:
    """Append-only store of Pillow-JPEG-encoded RGB frames (variable-length bytes).

    For photographic sky imagery (Folsom) where lossless compresses poorly.
    Each frame stored as JPEG bytes in a zarr VLenBytes object array ``images``.
    BRR is computed on READ (load_sky_frame), not stored."""
    def __init__(self, path, quality: int = 90, overwrite: bool = True):
        self.path = Path(path); self.quality = quality
        self.store = zarr.DirectoryStore(str(self.path))
        self.root = zarr.group(store=self.store, overwrite=overwrite)
        self._arr = self.root.array("images", shape=(0,), dtype=object,
                                    object_codec=VLenBytes(), chunks=(1,), overwrite=True)
        self._n = 0

    @staticmethod
    def _to_hwc_uint8(rgb):
        a = np.asarray(rgb)
        if a.ndim == 3 and a.shape[0] <= 4:        # (C,H,W) -> (H,W,C)
            a = np.transpose(a, (1, 2, 0))
        if a.dtype != np.uint8:
            a = np.clip(a * 255.0, 0, 255).astype(np.uint8) if a.max() <= 1.0 else a.astype(np.uint8)
        return a[..., :3]                           # RGB only
    def append(self, rgb) -> None:
        hwc = self._to_hwc_uint8(rgb)
        buf = io.BytesIO(); Image.fromarray(hwc).save(buf, format="JPEG", quality=self.quality)
        self._arr.append([buf.getvalue()]); self._n += 1
    @property
    def n(self): return self._n
```

`folsom_sky.build_folsom_sky_zarr`: replace `SkyImageZarr` with `JPEGSkyStore`; in the loop store `downsample_if_larger(_decode(data),256)` (the RGB array) directly (NOT `generate_brr_channel` — BRR deferred to load). Keep `images_timestamps` write via `write_array`. Update docstring (RGB JPEG bytes, BRR-on-load, ~12GB full).
- [ ] **Step 4: Run to verify pass** — `pytest tests/test_jpeg_store.py -v`.
- [ ] **Step 5: Commit** — `feat(zarr): JPEGSkyStore (Pillow JPEG RGB bytes) + wire folsom_sky`

---

### Task 6: Unified `load_sky_frame` + Dataset wiring + fixture fix

**Files:**
- Create: `solarunibench/imaging/load.py` (`load_sky_frame`)
- Modify: `solarunibench/dataset.py` (use `load_sky_frame` in the image-attach branch)
- Modify: `tests/test_loader.py::_build_pkg` (write uint8 non-trivial sky images so decode tests are genuine) + `tests/test_dataset.py` (add JPEG→BRR load test)

**Interfaces:**
- Produces: `load_sky_frame(cube, idx) -> (4,H,W) float32 [RGB∈0,1, BRR∈-1,1]` — branches: object/JPEG bytes → Pillow decode → `generate_brr_channel`; uint8 4ch → `decode_image`; uint8 3ch → `generate_brr_channel`.

- [ ] **Step 1: Write failing tests** (append to `tests/test_dataset.py`; update `_build_pkg`)

```python
# in tests/test_loader.py _build_pkg with_sky branch — write uint8 NON-trivial so decode is exercised:
g.array("images", data=(np.random.rand(n,4,256,256)*255).astype(np.uint8), chunks=(1,4,256,256), overwrite=True)
```

```python
# tests/test_dataset.py
def test_dataset_loads_jpeg_rgb_and_computes_brr(tmp_path):
    # build a Folsom-style JPEG store + timestamps, assert Dataset yields (lb,4,H,W) float with BRR∈[-1,1]
    import numpy as np, zarr
    from numcodecs import VLenBytes
    from solarunibench.operators.zarr_io import JPEGSkyStore
    from solarunibench.loader import SolarUniBenchLoader
    d = tmp_path/"folsom"; d.mkdir()
    # minimal metadata + time_series with a target + split
    ... (use _build_pkg-style meta/ts, then JPEGSkyStore for images)
    st = JPEGSkyStore(d/"sky_images.zarr", quality=90, overwrite=True)
    times = pd.date_range("2020-06-01 06:00", periods=6, freq="1h", tz="UTC")
    for _ in range(6): st.append((np.random.rand(3,256,256)*255).astype(np.uint8))
    zarr.open(d/"sky_images.zarr", mode="a").array("images_timestamps",
        np.array([t.isoformat() for t in times]), chunks=(1,), overwrite=True)
    ld = SolarUniBenchLoader(str(tmp_path))
    ds = SolarUniBenchDataset(ld, ["folsom"], split="train", target="ghi_actual",
                              feature_cols=["ghi_clearsky"], lookback=3, horizon=2, return_images=True)
    s = ds[0]
    assert s["image"].shape == (3,4,256,256) and s["image"].dtype == torch.float32
    assert float(s["image"][:,3].min()) >= -1.0-1e-6 and float(s["image"][:,3].max()) <= 1.0+1e-6
```

- [ ] **Step 2: Run to verify it fails** — JPEG load path missing (Dataset only does decode_image) / decode test now genuine.
- [ ] **Step 3: Implement** — `solarunibench/imaging/load.py`:

```python
"""Unified sky-frame loader: any storage -> (4,H,W) float32 [RGB 0..1, BRR -1..1]."""
from __future__ import annotations
import io
import numpy as np
from PIL import Image
from ..operators.brr import generate_brr_channel
from ..operators.image_codec import decode_image

def load_sky_frame(cube, idx) -> np.ndarray:
    """Return one (4,H,W) float32 frame from a sky-image cube, regardless of storage.

    - JPEG bytes (object array, Folsom): Pillow decode -> generate_brr_channel.
    - uint8 4ch (SKIPPD): decode_image.
    - uint8 3ch: generate_brr_channel.
    """
    if cube.dtype == object:                       # JPEG bytes (Folsom)
        rgb = np.array(Image.open(io.BytesIO(bytes(cube[idx]))).convert("RGB"))
        return generate_brr_channel(rgb)           # (4,H,W) float32 incl. BRR
    arr = np.asarray(cube[idx])
    if arr.shape[0] == 4:                           # uint8 RGB+BRR (SKIPPD)
        return decode_image(arr)
    return generate_brr_channel(arr)               # uint8 RGB only -> +BRR
```

In `dataset.py` image-attach branch, replace the `decode_image(...)` per-frame decode with `load_sky_frame(cube, r)`:

```python
        if (self.return_images and self._img_index[ds] is not None and self._img_cube[ds] is not None):
            cube = self._img_cube[ds]; idx = self._img_index[ds]
            rows = [idx.get(str(t)) for t in tss[lb]]
            if all(r is not None for r in rows) and len(rows) == self.lookback:
                from ..imaging.load import load_sky_frame
                dec = np.stack([load_sky_frame(cube, r) for r in rows])
                sample["image"] = torch.from_numpy(dec.astype(np.float32))
```
- [ ] **Step 4: Run to verify pass** — `pytest tests/test_dataset.py tests/test_loader.py tests/test_image_codec.py tests/test_zarr_io.py -v`.
- [ ] **Step 5: Commit** — `feat(imaging): unified load_sky_frame (JPEG|uint8) + genuine decode tests`

---

### Task 7: Rebuild bounded Folsom(JPEG) + SKIPPD(uint8@64) + measure + integration

**Files:**
- Create: `scripts/rebuild_sky_zarr.py` (rebuild ONLY sky zarr per dataset at bounded limit)
- Modify: `tests/test_loader_integration.py` (SKIPPD uint8@64 dtype+decode; Folsom JPEG bytes+decode)

- [ ] **Step 1: Create `scripts/rebuild_sky_zarr.py`** — args `--dataset {folsom,skippd} --limit 200`; calls `build_folsom_sky_zarr` / `build_skippd_sky_zarr` directly into `data/standardized/<ds>/sky_images.zarr` (no full tabular re-run).
- [ ] **Step 2: Rebuild + measure** —
```bash
du -sh data/standardized/folsom/sky_images.zarr data/standardized/skippd/sky_images.zarr   # BEFORE (float32@256)
conda run -n solar_uni python scripts/rebuild_sky_zarr.py --dataset folsom --limit 2000     # extrapolate ratio
conda run -n solar_uni python scripts/rebuild_sky_zarr.py --dataset skippd --limit 2000
du -sh data/standardized/folsom/sky_images.zarr data/standardized/skippd/sky_images.zarr    # AFTER
```
Report extrapolated full sizes (Folsom JPEG ~12 GB; SKIPPD uint8@64 ~6 GB) vs the float32 baseline.
- [ ] **Step 3: Integration tests** — append to `tests/test_loader_integration.py`: SKIPPD `images` dtype uint8 shape `(N,4,64,64)` + Dataset decode→float; Folsom `images` object/JPEG + Dataset `(lb,4,256,256)` float with BRR∈[-1,1]. Both RUN_SLOW-gated.
- [ ] **Step 4: Verify + commit** — fast suite green; `RUN_SLOW=1 pytest tests/test_loader_integration.py -v` passes. Commit: `feat(imaging): rebuild sky zarr (Folsom JPEG / SKIPPD uint8@64) + integration`.

---

## Definition of Done

- Folsom sky stored as JPEG-q90 RGB bytes @256² (~12 GB full); SKIPPD as uint8 RGB+BRR @64² lossless (~6 GB).
- `load_sky_frame` unifies both → `(lookback,4,H,W)` float for models.
- `SolarUniBenchDataset` image attach works for both formats; fail-fast validation present.
- Resolution policy: native ≤256 kept, >256 downsampled (Folsom 1536→256, SKIPPD 64→64).
- Bounded rebuild measured; fast suite green; RUN_SLOW integration passes for both formats.

## Notes / out of scope
- SunLake satellite stays float32 (NWP bands). GEFCom has no images.
- Full (unbounded) materialization + final compress/uncompress per-dataset confirmation = Phase 8 batch (now feasible: Folsom ~12 GB, SKIPPD ~6 GB).
- `numcodecs.JPEG` unavailable (0.13.1) → Pillow JPEG via `VLenBytes` object array (documented).
