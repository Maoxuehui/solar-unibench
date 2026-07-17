# Phase 6b — Deep/Sequence Baselines + Optical-Flow Revision + Multimodal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Extend the forecast suite with **LSTM**, **PatchTST-style Transformer**, and **CrossViViT** (multimodal); revise the optical-flow extrapolation to the authoritative **forward-splat** method; fix Folsom cross-modal image-attach (nearest-neighbor) + mixed-dataset resize. All deep models plug into the existing `ForecastModel` ABC + `SolarForecastPipeline` (with `PhysicalFeatureScaler` + `daytime_metrics`).

**Architecture:** All new models subclass `ForecastModel` (`fit(X_past, Y, times, X_future)`, `predict(...)->(n,horizon)`, `name`) and consume the 5-tuple windows from `EndogenousFeatureEngineer`. LSTM/PatchTST use `X_past` (sequence); CrossViViT additionally consumes attached sky/satellite images (via `SolarUniBenchDataset` / a multimodal FeatureEngineer). Optical-flow revision is contained in `operators/optical_flow.py` (preserves the `k*flow` advection contract).

**Tech Stack:** torch 2.6.0+cu124 (CUDA avail), numpy, pandas. Env `solar_uni`.

## Global Constraints
- **ForecastModel ABC:** every deep model implements `fit(X_past, Y, times=None, X_future=None)` / `predict(...) -> (n, horizon)` / `name`. No bespoke training loops leaking outside the class.
- **Scaling default-on:** the pipeline's `PhysicalFeatureScaler` already applies (irradiance /1500, NWP z-score train-fit) — deep models receive scaled features; do NOT re-scale inside models.
- **Daytime eval:** headline metric is `daytime_metrics` (cos_zenith>0.05).
- **Optical-flow advection contract preserved:** `future[k-1]` = frame advected by `k*flow` (the 4 existing `test_optical_flow.py` tests must still pass); switch the warp core to forward-splat (collision-averaged bilinear + hole-fill).
- **Honest NaN:** absent modalities (GEFCom no images, Folsom/SunLake no PV) handled gracefully.
- **Env:** `conda run -n solar_uni ...`; pytest from repo root (or `PYTHONPATH=repo`). TDD, DRY, YAGNI; trailer `Co-Authored-By: Claude <noreply@anthropic.com>`. Deep-model tests use TINY configs (few epochs, small hidden) on synthetic data so the fast suite stays quick; real-data training is RUN_SLOW.

**Branch:** `feat/phase6b-deep-baselines` (from `master` @ `58ecc13`).

---

### Task 1: Optical-flow forward-splat revision
**Detailed sub-plan:** `docs/superpowers/plans/2026-07-17-solar-unibench-optflow-forward-splat.md` (execute its Task 1 → Task 2 → optional Task 3).
- `_bilinear_forward_splat` (P0: `np.bincount` + merged 4-corner pass) + cached `_meshgrid`.
- `forward_warp(frame,u,v,k,hole_fill,decay)` + revise `extrapolate_with_uncertainty` to forward-splat, incremental-default, configurable `hole_fill`/`decay_tau`/`incremental`. Preserve `k*flow` advection → existing 4 tests pass. RUN_SLOW real-SunLake corr check.
- Files: `solarunibench/operators/optical_flow.py`; `tests/test_optical_flow.py`.

### Task 2: LSTM baseline
**Files:** `solarunibench/forecast/models/lstm_model.py`; `tests/test_lstm.py`.
- `LSTMModel(hidden_size=64, num_layers=2, dropout=0.2, lr=1e-3, epochs=8, batch_size=32)`(ForecastModel). Input `X_past[n,lb,Fp]` → `(n,lb,Fp)` tensor → LSTM → last hidden → Linear(horizon). predict `(n,horizon)`.
- TINY config in fast tests (epochs=2, hidden=8) on synthetic; assert shape + finite + beats-persistence on a learnable synthetic trend.

### Task 3: PatchTST-style Transformer baseline
**Files:** `solarunibench/forecast/models/patchtst_model.py`; `tests/test_patchtst.py`.
- `PatchTSTModel(d_model=64, n_heads=4, patch_len=4, stride=2, depth=2, epochs=8, …)` (ForecastModel). Patch-tokenize `X_past` along the lookback dim (per-channel patches), Transformer encoder, linear head → `(n,horizon)`. Config flag to swap a vanilla TS-Transformer.
- TINY fast-test config; assert shape + finite + learns a synthetic pattern.

### Task 4: Folsom nearest-neighbor image-attach + mixed-dataset resize
**Files:** `solarunibench/dataset.py` (image-attach); `solarunibench/imaging/load.py` or a new `imaging/resize.py` helper (per-dataset resize to a common size for batched image models).
- **Nearest-neighbor attach:** Folsom sky images are second-cadence; tabular irradiance hourly → exact-match finds ~0. Change the Dataset image-attach to **nearest timestamp within a tolerance** (e.g., ≤30 min) when exact match misses, controlled by a `match_tolerance` param (default keeps exact for SKIPPD which aligns).
- **Mixed-dataset resize:** a helper `resize_image_batch(imgs, target_hw)` so a multimodal model can batch Folsom (256²) + SKIPPD (64²) by resizing to a common size on load.
- Tests: Folsom-style fixture (images at finer cadence than tabular) attaches via NN within tolerance; mixed-size resize correctness.

### Task 5: CrossViViT baseline (multimodal)
**Files:** `solarunibench/forecast/models/crossvivit_model.py`; `tests/test_crossvivit.py`.
- `CrossViViTModel(...)` (ForecastModel) — tabular tokens + sky/satellite image tokens, cross-attention, horizon head. Reference: `docs/references/crossvivit_tscontext_dataset_ref.py` (the `TSContextDataset` pattern + forward-splat extrapolation). Configurable; default a SMALL variant for tests (full 145M is a RUN_SLOW/Phase-8 training decision).
- Consumes `X_past` + attached images (via a multimodal FeatureEngineer or the Dataset). Image branch uses the revised forward-splat future frames where available.
- TINY fast-test (synthetic images + tabular) → shape + finite. Document that full training is Phase 8.

### Task 6: Integration — deep baselines on real data (RUN_SLOW)
**Files:** `tests/test_forecast_integration.py` (extend).
- Run LSTM + PatchTST on GEFCom (tabular); CrossViViT on a multimodal dataset (SKIPPD or SunLake) via the pipeline + scaler + daytime eval. Print the daytime skill table including the new deep baselines vs Smart Persistence.

---

## Definition of Done
- Optical-flow = forward-splat (P0 bincount); real-SunLake corr ≈/≥ cv2.remap.
- LSTM, PatchTST, CrossViViT all subclass ForecastModel + run via the pipeline; fast tests green; RUN_SLOW real-data skill table printed.
- Folsom NN image-attach works (within tolerance); mixed-dataset resize helper present.
- Framework stays extensible (drop-in ForecastModel).

## Notes / out of scope
- Full CrossViViT (145M) training + hyperparameter tuning = Phase 8 (compute-budget decision).
- E4 (optical-flow quality experiment), E8 (full benchmark heatmap), E9 (governance vs complexity) = Phase 7.
