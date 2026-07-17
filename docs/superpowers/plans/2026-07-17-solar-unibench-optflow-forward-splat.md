# Optical-Flow Revision: Forward-Splat Warp — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace `extrapolate_with_uncertainty`'s `cv2.remap` backward-sample warp with the authoritative **forward-splat** method (collision-averaged bilinear forward warp + hole-fill), preserving the existing `k*flow` advection contract so the SunLake builder + all current tests keep working. Apply P0 perf wins (10–100×) and make the result-changing quality options configurable.

**Architecture:** `operators/optical_flow.py` gains `_bilinear_forward_splat` (P0: `np.bincount` + merged 4-corner pass), a cached `_meshgrid`, and `forward_warp(frame, u, v, k, hole_fill=...)`. `extrapolate_with_uncertainty` switches to `forward_warp` (incremental chaining by default — smaller holes; direct mode optional) and exposes flags: `incremental`, `hole_fill ∈ {nearest, inpaint}`, `decay_tau` (direct mode only). Signature unchanged → SunLake builder untouched.

**Reference:** `docs/references/crossvivit_tscontext_dataset_ref.py` (`warp_image` / `_bilinear_forward_splat`).

**Tech Stack:** numpy, cv2 (opencv-python-headless), `functools.lru_cache`. Env `solar_uni`.

## Global Constraints

- **Advection contract preserved:** `future[k-1]` = frame advected forward by `k*flow` (k=1→+u px, k=2→+2u px, …). Matches SunLake `ctx_opt_flow` forward-displacement convention. The 4 existing `test_optical_flow.py` tests MUST still pass.
- **P0 (identical output, do unconditionally):** `_bilinear_forward_splat` uses `np.bincount` (NOT `np.add.at`) and merges the 4 corner contributions into ONE bincount pass.
- **P1 (include):** incremental chaining (default `incremental=True`); meshgrid cached via `lru_cache`.
- **P2/P3 (configurable flags, ablated later in E4):** `hole_fill={'nearest','inpaint'}` (nearest = current `distanceTransformWithLabels`; inpaint = `cv2.inpaint` Telea); `decay_tau` (scales effective displacement `k*exp(-k/tau)`, direct mode only).
- **Env:** `conda run -n solar_uni ...`; pytest from repo root (or `PYTHONPATH=repo`). TDD, DRY, YAGNI; commit trailer `Co-Authored-By: Claude <noreply@anthropic.com>`.

**Branch:** `feat/optflow-forward-splat` (from `master` @ `ed07333`).

---

### Task 1: `_bilinear_forward_splat` (P0 bincount) + meshgrid cache

**Files:** Modify `solarunibench/operators/optical_flow.py` (add helpers, keep `extrapolate_with_uncertainty` as-is for now); Test `tests/test_optical_flow.py` (append).

**Interfaces:** `_bilinear_forward_splat(values[H,W], dst_x[H,W], dst_y[H,W], valid_mask=None) -> (output[H,W], weight_sum[H,W])`; `_meshgrid(h,w) -> (yy,xx)` cached.

- [ ] **Step 1: Write failing test** (append to `tests/test_optical_flow.py`):

```python
def test_forward_splat_uniform_destination():
    from solarunibench.operators.optical_flow import _bilinear_forward_splat
    # All source points land exactly on integer dst (no fractional) -> identity-ish
    h = w = 8
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    vals = (xx.astype(np.float32) + 1.0)  # value = col+1
    out, weight = _bilinear_forward_splat(vals, xx.astype(np.float32), yy.astype(np.float32))
    assert out.shape == (h, w) and weight.shape == (h, w)
    assert np.allclose(out, vals)            # integer dst -> exact copy
    assert np.all(weight > 0)                # fully covered

def test_forward_splat_matches_add_at():
    # P0 correctness: bincount path must equal the naive np.add.at accumulation.
    from solarunibench.operators.optical_flow import _bilinear_forward_splat
    rng = np.random.default_rng(0)
    h = w = 12
    vals = rng.random((h, w)).astype(np.float32)
    dst_x = rng.uniform(0, w - 1.0, (h, w)).astype(np.float32)
    dst_y = rng.uniform(0, h - 1.0, (h, w)).astype(np.float32)
    out, _ = _bilinear_forward_splat(vals, dst_x, dst_y)
    # naive reference
    x0 = np.floor(dst_x).astype(int); y0 = np.floor(dst_y).astype(int)
    ref = np.zeros((h, w), np.float32); wsum = np.zeros((h, w), np.float32)
    wx = dst_x - x0; wy = dst_y - y0
    for dx, dy, ww in [(0,0,(1-wx)*(1-wy)),(1,0,wx*(1-wy)),(0,1,(1-wx)*wy),(1,1,wx*wy)]:
        xi, yi = x0+dx, y0+dy
        ok = (xi>=0)&(xi<w)&(yi>=0)&(yi<h)&(ww>0)
        np.add.at(ref, (yi[ok], xi[ok]), (vals*ww)[ok])
        np.add.at(wsum, (yi[ok], xi[ok]), ww[ok])
    covered = wsum > 1e-6; ref[~covered] = 0
    np.putmask(ref, covered, ref/np.maximum(wsum,1e-12))
    assert np.allclose(out, ref, atol=1e-5)
```

- [ ] **Step 2: Run → fails** (`ImportError` / `AttributeError`).
- [ ] **Step 3: Implement** (add to `optical_flow.py`):

```python
from functools import lru_cache

@lru_cache(maxsize=64)
def _meshgrid(h: int, w: int):
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    return yy, xx

def _bilinear_forward_splat(values, dst_x, dst_y, valid_mask=None):
    """Bilinear forward splat. P0: single np.bincount pass over all 4 corners."""
    values = np.asarray(values, dtype=np.float32)
    dst_x = np.asarray(dst_x, dtype=np.float32); dst_y = np.asarray(dst_y, dtype=np.float32)
    h, w = values.shape
    valid_mask = np.ones((h, w), dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    finite = np.isfinite(values) & np.isfinite(dst_x) & np.isfinite(dst_y) & valid_mask
    x0 = np.floor(dst_x).astype(np.int64); y0 = np.floor(dst_y).astype(np.int64)
    wx = dst_x - x0; wy = dst_y - y0
    n = h * w
    flats, wlist, vlist = [], [], []
    for dx, dy, ww in [(0, 0, (1 - wx) * (1 - wy)), (1, 0, wx * (1 - wy)),
                       (0, 1, (1 - wx) * wy), (1, 1, wx * wy)]:
        xi = x0 + dx; yi = y0 + dy
        ok = finite & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h) & (ww > 0)
        if not np.any(ok):
            continue
        flats.append(yi[ok] * w + xi[ok]); wlist.append(ww[ok]); vlist.append((values * ww)[ok])
    if not flats:
        return np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    flat = np.concatenate(flats); ws = np.concatenate(wlist); vs = np.concatenate(vlist)
    weight = np.bincount(flat, weights=ws, minlength=n).astype(np.float32)
    accum = np.bincount(flat, weights=vs, minlength=n).astype(np.float32)
    out = np.zeros(n, np.float32)
    covered = weight > 1e-6
    out[covered] = accum[covered] / weight[covered]
    return out.reshape(h, w), weight.reshape(h, w)
```

- [ ] **Step 4: Run → pass** (2 new + existing 4).
- [ ] **Step 5: Commit** — `feat(optflow): _bilinear_forward_splat (P0 bincount) + meshgrid cache`

---

### Task 2: `forward_warp` + revise `extrapolate_with_uncertainty`

**Files:** Modify `solarunibench/operators/optical_flow.py`; Test `tests/test_optical_flow.py`.

**Interfaces:** `forward_warp(frame, u, v, k=1, hole_fill="nearest", inpaint_radius=3) -> (H,W)`; `extrapolate_with_uncertainty(flow_u, flow_v, frame_t1, steps_k=12, incremental=True, hole_fill="nearest", decay_tau=None)` (signature backward-compatible — new params keyword-only with defaults).

- [ ] **Step 1: Tests** — the 4 existing tests MUST pass unchanged (advection preserved). Add:

```python
def test_forward_warp_advects_right_and_leaves_disocclusion():
    from solarunibench.operators.optical_flow import forward_warp
    f1 = np.zeros((32, 32), np.float32); f1[10:20, 12:22] = 1.0
    u = np.full((32, 32), 2.0, np.float32); v = np.zeros((32, 32), np.float32)
    out = forward_warp(f1, u, v, k=1)
    assert out[10:20, 14:24].max() > 0.9        # square moved right by 2
    assert out[10:20, 12:14].max() < 0.1         # old position now dark (bg covered)

def test_extrapolate_uses_forward_splat_by_default():
    # incremental default still yields k*flow advection (k=2 -> +4px)
    f0, f1, u, v = _shifted_pair(shift=2)
    future, _ = extrapolate_with_uncertainty(u, v, f1, steps_k=2)
    assert future[1][10:20, 16:26].max() > 0.9   # k=2 -> +4
```

- [ ] **Step 2: Run → existing 4 pass (unchanged behavior), 2 new fail (forward_warp missing).**
- [ ] **Step 3: Implement**:

```python
def forward_warp(frame, u, v, k=1, hole_fill="nearest", inpaint_radius=3):
    frame = np.asarray(frame, dtype=np.float32)
    u = np.asarray(u, dtype=np.float32); v = np.asarray(v, dtype=np.float32)
    h, w = frame.shape
    yy, xx = _meshgrid(h, w)
    dst_x = xx + k * u; dst_y = yy + k * v
    valid = np.isfinite(dst_x) & np.isfinite(dst_y)
    predicted, weight = _bilinear_forward_splat(frame, dst_x, dst_y, valid_mask=valid)
    covered = weight > 1e-6
    if np.all(covered):
        return predicted
    if not np.any(covered):
        return frame.copy()
    if hole_fill == "inpaint":
        mask = (~covered).astype(np.uint8) * 255
        return cv2.inpaint(predicted, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA).astype(np.float32)
    hole_map = (~covered).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(hole_map, cv2.DIST_L2, cv2.DIST_MASK_5, cv2.DIST_LABEL_PIXEL)
    lv = np.zeros(int(labels.max()) + 1, dtype=np.float32)
    lv[labels[covered]] = predicted[covered]
    predicted[~covered] = lv[labels[~covered]]
    return predicted
```

Revise `extrapolate_with_uncertainty` to use it (keep divergence/uncertainty formula + signature; add keyword-only flags):

```python
def extrapolate_with_uncertainty(flow_u, flow_v, frame_t1, steps_k=12, *,
                                 incremental=True, hole_fill="nearest", decay_tau=None):
    u = np.asarray(flow_u, dtype=np.float32); v = np.asarray(flow_v, dtype=np.float32)
    frame = np.asarray(frame_t1, dtype=np.float32)
    div = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3) + cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3)
    divergence = np.abs(div)
    alpha, beta = 0.02, 0.05
    futures, unc = [], []
    current = frame
    for k in range(1, steps_k + 1):
        if incremental:
            current = forward_warp(current, u, v, 1, hole_fill=hole_fill)   # +1*flow/step -> k*flow
            fut = current
        else:
            eff_k = k * (float(np.exp(-k / decay_tau)) if decay_tau else 1.0)
            fut = forward_warp(frame, u, v, eff_k, hole_fill=hole_fill)
        futures.append(fut)
        u_map = np.exp(-alpha * k) * (1.0 - np.minimum(1.0, beta * k * divergence))
        unc.append(np.clip(u_map, 0.0, 1.0).astype(np.float32))
    return np.asarray(futures, np.float32), np.asarray(unc, np.float32)
```

- [ ] **Step 4: Run → all pass** (4 existing + 4 new). Verify `RUN_SLOW=1 pytest tests/test_optical_flow.py -v` (the real-SunLake direction test still passes — forward-splat must advect right).
- [ ] **Step 5: Commit** — `feat(optflow): forward-splat warp + incremental extrapolation (configurable hole-fill/decay)`

---

### Task 3 (optional verify): real-SunLake forward-splat ≥ cv2.remap

**Files:** `tests/test_optical_flow.py` (append RUN_SLOW test).

- [ ] Add `@slow test_forward_splat_beats_or_matches_remapproach`: on SunLake CAB/2008 (8 frames, IR_108/VIS006/VIS008), compute Pearson corr of `forward_warp(base, vx, vy, 1)` vs real `ctx[t+2]` and assert it is **≥ the cv2.remap(minus) corr − 0.02** (forward-splat should be at least comparable; report both). Print both means.
- [ ] Commit: `test(optflow): real-SunLake forward-splat corr check`.

---

## Definition of Done
- `_bilinear_forward_splat` uses `np.bincount` + merged corners (P0; matches `np.add.at` reference).
- `extrapolate_with_uncertainty` warps via forward-splat (incremental default); preserves `k*flow` advection → all 4 existing tests pass + new ones.
- Configurable `hole_fill`/`decay_tau`/`incremental` for E4 ablation.
- Real-SunLake direction test passes; forward-splat corr ≈/≥ cv2.remap.

## Notes / out of scope
- Batch multi-channel warp (P1 #6) deferred — per-channel is correct; batching needs 3D `_bilinear_forward_splat` (later optimization).
- SunLake satellite builder unchanged (signature preserved).
- Full E4 extrapolation-quality experiment = Phase 7.
