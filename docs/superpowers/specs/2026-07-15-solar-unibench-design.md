# Solar-UniBench — Design Specification

- **Date:** 2026-07-15
- **Project:** Solar-UniBench-v1.0 — A Multi-modal Standardized Dataset and Distributed Processing Pipeline for Harmonized Solar Forecasting
- **Target venue:** ESSD (Earth System Science Data)
- **Status:** Approved (as-is) — pending written-spec review

## 1. Goal

Take four heterogeneous public solar-forecasting datasets (GEFCom2014, Folsom, SKIPPD, SunLake), run each through one object-oriented "operator assembly line" (`BaseSolarPipeline` + a per-dataset adapter), and emit physically-independent but **logically-identical** standardized packages following a single data contract. Provide a downstream loader, PyTorch dataset, train/eval splits, and baselines so reviewers can reproduce results. Core intellectual contribution: a unified, physics-grounded feature system (clear-sky anchors, clearness index, reverse-geolocation, quality flags, BRR channel, optical-flow extrapolation with uncertainty) — "治理重于模型" (governance over modeling).

**Scope decision (approved):** Build the complete framework **and** execute it end-to-end on all four datasets, processing **every** sky/satellite frame into zarr (full optical-flow extrapolation).

## 2. Guiding principles

1. **物理独立存储 + 逻辑大一统** — independent per-dataset storage; one abstract base class enforces identical output schema.
2. **Reproducibility** — deterministic, logged, idempotent pipeline. Every run appends an audit log.
3. **Honest NaN** — missing modalities are absent files / NaN columns, never fabricated. The loader switches modality by probing existence.
4. **Physics in the base, parsing in the adapters** — all天文/质量/视觉/动力学 operators live in `BaseSolarPipeline`; adapters only translate raw → standard frames.
5. **Config-driven scale** — a single `--limit` / config flag bounds processing for tests and for scaling up later.

## 3. Repository layout

```
solar-unibench/
├── pyproject.toml              # package metadata + pinned deps
├── requirements.txt            # pinned deps for the conda env "solar-unibench"
├── configs/                    # per-dataset YAML (paths, scale, horizons)
├── solarunibench/
│   ├── contract.py             # DataContract: canonical columns, dtypes, zarr shapes
│   ├── pipeline/
│   │   ├── base.py             # BaseSolarPipeline: operators + run_pipeline()
│   │   ├── operators/          # clearsky, quality_audit, brr, optical_flow, reverse_geolocate, align, denoise
│   │   └── adapters/           # gefcom.py, folsom.py, skippd.py, sunlake.py
│   ├── orchestrator.py         # staged decision workflow (deterministic, auditable)
│   ├── loader.py               # SolarUniBenchLoader + SolarUniBenchDataset (PyTorch)
│   ├── splits.py               # temporal_split / spatial_split
│   ├── forecast/
│   │   ├── core.py             # FeatureEngineer / ForecastModel ABCs, PipelineConfig, SolarForecastPipeline
│   │   ├── features.py         # Endogenous, SkyCam(NRBR), multimodal engineers
│   │   └── models/             # mlp.py, lstm.py, smart_persistence.py, cross_vivit.py
│   └── metrics.py              # MAE/RMSE/MAPE/Forecast Skill/FS/pinball/CRPS
├── scripts/
│   ├── build_dataset.py        # CLI: run one/all adapters → standardized packages
│   └── run_baseline.py         # CLI: train+eval a baseline on standardized data
├── tests/                      # synthetic-fixture unit tests per operator/adapter
├── data/standardized/{gefcom2014,folsom,skippd,sunlake}/   # outputs (gitignored)
└── docs/                       # methodology + dataset docs (existing) + this spec
```

The existing `extensions.py` scaffold is retired into `forecast/`:
- `SmartPersistenceModel`, `SkyCamFeatureEngineer`, `LSTMModel` → `forecast/models` and `forecast/features`
- its missing `.core` (referenced as `FeatureEngineer, ForecastModel, PipelineConfig, SolarForecastPipeline, MLPModel, EndogenousFeatureEngineer`) → `forecast/core.py`

## 4. Data contract (single source of truth — `contract.py`)

`contract.py` defines canonical schemas; every adapter must emit conformant artifacts. A `test_contract_conformance` must pass per adapter.

### 4.1 `metadata.parquet` (Table 1 — static metadata)
| column | dtype | note |
|---|---|---|
| `entity_id` | str | unique per station/zone |
| `dataset_source` | str | one of gefcom2014 / folsom / skippd / sunlake |
| `latitude`, `longitude` | float32 | degrees; inferred for GEFCom |
| `elevation_meters` | float32 | |
| `local_timezone` | str | IANA |
| `surface_albedo` | float32 | default NaN→0.2 |
| `capacity_kw` | float32 | NaN if pure irradiance task |
| `slope`, `azimuth` | float32 | panel/plane orientation, NaN if unknown |
| `geo_inferred` | bool | True for reverse-geolocated GEFCom |

### 4.2 `time_series.parquet` (Tables 2–4 — keyed by `timestamp_utc, entity_id`)
- **Anchors (Table 2):** `cos_zenith, solar_azimuth, ghi_clearsky, dni_clearsky, dhi_clearsky`
- **Ground/target (Table 3):** `ghi_actual, dni_actual, dhi_actual, pv_power_normalized` (NaN where the dataset lacks them)
- **Derived:** `clearness_index` (= `ghi_actual / ghi_clearsky`, clipped [0, 1.5]), `quality_flag` (0 good / 1 curtailment-clipping / 2 instrument-fault / 3 night)
- **NWP (Table 4):** dataset-specific, namespaced `nwp_*`; union schema, NaN-filled

Completeness (per methodology, reproduced here as the contract's expectation):
- Folsom: T1✓ T2✓ T3(partial: irradiance+kt, no PV) T4✓
- SKIPPD: T1✓ T2✓ T3✓(full: irradiance+kt+PV) T4✓ — the all-modality set
- SunLake: T1✓ T2✓ T3(partial: GHI+kt, no PV) T4✓
- GEFCom2014: T1(partial: inferred coords) T2✓ T3(partial: normalized PV only) T4✓

### 4.3 `sky_images.zarr` (Table 5)
- shape `(N, 4, 256, 256)` float32, channels = R, G, B, **BRR** = `(B−R)/(B+R)`
- chunks `(1, 4, 256, 256)` for lazy single-frame access
- present for **Folsom, SKIPPD** only

### 4.4 `satellite_future.zarr` (Tables 6–7)
- `context`: `(N, C, H, W)` — 50 km × 50 km site-centered crop, VIS/IR/WV
- `future`: `(K, N, C, H, W)` — K-step optical-flow extrapolation
- `uncertainty_map`: `(K, N, H, W)` float32 in [0,1] — divergence + lead-time decay
- present for **SKIPPD, SunLake** only

Missing modalities → file absent. Loader probes existence/NaN to switch modality.

### 4.5 Config defaults (pin the under-specified parameters)
- `K` (extrapolation steps): default **12** (= 15-min horizons up to ~3 h at native cadence; config-driven per dataset).
- `ε_t` (cross-modal alignment tolerance): **±1 native step** (e.g., ±1 min for SKIPPD/Folsom 1-min, ±30 min for SunLake 30-min).
- Satellite grid `(H, W)`: derived from the 50 km × 50 km crop at source resolution; default target **128×128** after resample (config-driven).
- `surface_albedo` default **0.2** when unknown.
- clearness_index clip **[0, 1.5]**; physical clamp to `[0, capacity_kw]`.

## 5. Base pipeline & operators (`pipeline/base.py`, `pipeline/operators/`)

`BaseSolarPipeline` (ABC) — constructor `(raw_data_dir, output_dir, dataset_name, config)`. Adapters implement `load_and_standardize_raw() -> (df_meta, df_ts)` and `process_and_save_tensors(df_ts)`.

Operators (in the base):
- `_compute_clearsky_and_astronomy()` — pvlib `solarposition.get_solarposition` + `clearsky.ineichen` (correct API: pass `linke_turbidity` via `lookup_linke_turbidity` or use defaults; fix the methodology's erroneous `appointment_time=` call).
- `_apply_quality_audit_and_features()` — night-zeroing (`cos_zenith <= 0`), `clearness_index`, `quality_flag`, physical clamp to `[0, capacity]`.
- `generate_brr_channel()` — RGB→4-channel with BRR (from `extensions.py`).
- `extrapolate_with_uncertainty()` — TV-L1 optical flow + semi-Lagrangian warp + divergence-based uncertainty (from `extensions.py`, moved to `operators/optical_flow.py`).
- `_reverse_geolocate()` — grid search minimizing residual between observed clearness pattern and clearsky; validated against existing `GEFCom2014_data/geo_location.py`.
- `run_pipeline()` — load → clearsky → quality → (tensors) → save + manifest + audit log.

**Corrections to the methodology's reference code:** fix the Ineichen call signature; index `solpos`/merge on `timestamp_utc` correctly; ensure `entity_id` is preserved through merges.

## 6. Per-dataset adapters

| Adapter | Raw parsing | Tabular target | Tensors produced |
|---|---|---|---|
| **gefcom** | `GEFCom2014_data/Solar/Task *` (train/predictors CSVs + xlsx); reverse-geolocate 3 anonymous zones | `pv_power_normalized` | none |
| **folsom** | `Folsom_data/Folsom_{irradiance,weather,NAM_*,satellite}.csv` + `Folsom_sky_images_*.tar.bz2` | GHI/DNI/DHI + kt | `sky_images.zarr` |
| **skippd** | HF parquet (`default/era5/gfs/labels`) after `download_skippd.py` | PV + irradiance + kt (full) | `sky_images.zarr`, `satellite_future.zarr` |
| **sunlake** | per-station PCCI (zarr/parquet) + `context/` + `ctx_opt_flow/` | GHI + kt | `satellite_future.zarr` (reuse existing opt-flow where present) |

## 7. Orchestrator — the "agent" workflow (`orchestrator.py`)

**Decision (approved):** Implement as a **deterministic, fully-logged state machine**, not live LLM calls. Stages:
1. **Predictability analysis** — signal-strength check; route near-dead series (e.g., all-night) to a simple baseline and skip multimodal fusion.
2. **Quality diagnosis & repair** — map defects (taxonomy in §8) to repair primitives; record trigger + action.
3. **Denoise & dimensionality reduction** — drop constant channels, high-missingness vars, redundant features.
4. **Spatiotemporal alignment** — systematic-lag correction + cross-modal nearest-frame match within tolerance ε_t.

Each stage appends `{stage, entity_id, trigger, action, ts}` to `audit.jsonl`. An optional LLM *narration* of the audit log can be added later for the paper's "agent" framing without changing any output.

## 8. Defect taxonomy → repair primitives (from methodology)

- **Structured:** missing values, irregular sampling, duplicate/non-monotonic timestamps, outliers/physical inconsistency, timezone inconsistency, missing geo-metadata.
- **Unstructured (image):** missing/corrupt frames, quality degradation (Q score), invalid-region/occlusion (ρ_invalid), local pixel anomalies, geometric misalignment.
- **Cross-modal:** async (image vs. structured timestamp diff > ε_t).

Repair primitives: delete+realign, physical-context interpolation, outlier clamp+physical correction, timezone normalization/inferral, geo-metadata inference, quality diagnostics, window QC, frame-level filtering+CLAHE, temporal alignment, visual feature extraction, window-level exclusion + modality mask, systematic-lag correction, diagnostic logging.

## 9. Loader & downstream API (`loader.py`)

- `SolarUniBenchLoader`:
  - `load_tabular(dataset, split=None) -> df` — merged metadata + time_series.
  - `get_tensor_cube(dataset, kind) -> zarr.Array` — lazy `DirectoryStore`; HTTP-mappable later.
- `SolarUniBenchDataset(torch.utils.data.Dataset)` — fixed column/tensor names so a batch can mix all four datasets; modality auto-switch.
- Splits (`splits.py`): `temporal_split` (time cutoff) and `spatial_split` (leave-one-dataset-out for cross-domain zero-shot).

## 10. Forecasting framework & baselines (`forecast/`)

- `forecast/core.py` — `PipelineConfig`, `SolarForecastPipeline`, `FeatureEngineer`/`ForecastModel` ABCs (the `.core` that `extensions.py` imports).
- **Canonical task (approved):** deterministic + probabilistic forecasting of the available target — **PV** for SKIPPD/GEFCom, **GHI** for Folsom/SunLake — at three horizons: intra-hour (15-min), intra-day (1–6 h), day-ahead (24 h).
- **Feature engineers:** Endogenous (PV lags), SkyCam/NRBR stats, multimodal (images+PV).
- **Baselines:** 7-neuron MLP (PV-only), MLP+SkyCam, Smart Persistence (physical skill reference), LSTM, CrossViViT (multimodal).
- **Metrics (`metrics.py`):** MAE, RMSE, MAPE, Forecast Skill vs. Smart Persistence, pinball loss / CRPS.

## 11. Execution, error handling, robustness

- `scripts/build_dataset.py` — resumable, idempotent (`--force` to overwrite); per-stage checkpoints; `tqdm` + log progress; heavy zarr stages run as background jobs. All runs append to `audit.jsonl` + emit `manifest.json`.
- Robustness: physical clamping; NaN propagation tracked in `quality_flag`; unreadable frames logged + skipped; UTC enforced at ingest; defensive raw-format parsing with explicit dtypes.
- Environment: dedicated conda env `solar-unibench` from `requirements.txt` (base Anaconda lacks pandas/pvlib/zarr/pyarrow/sklearn/xarray).

## 12. Testing strategy (TDD)

Synthetic-fixture unit tests for each operator:
- clearsky vs. known pvlib reference values at a known site/time
- BRR channel bounds ∈ [−1, 1]
- optical-flow extrapolation sanity (identity on identical frames; translation on shifted frames)
- reverse-geolocation recovery on a synthetic known-coord site (MAE bound)
- quality_flag assignment (night, clipping, outlier)

Plus a **contract-conformance test** each adapter must pass (schema, dtypes, key uniqueness, file-presence rules). Tests run on synthetic data so CI is fast; real-data execution is the final batch phase.

## 13. Phased plan (build strategy C — vertical spike then layers)

1. **Env + scaffold + contract + tests harness.**
2. **Vertical spike:** GEFCom2014 tabular end-to-end (incl. reverse-geolocation) — locks base class, contract, CLI.
3. **Tabular layer:** Folsom, SKIPPD, SunLake standardized parquet (Tables 1–4).
4. **zarr/image layer:** Folsom sky (RGB+BRR), SKIPPD sky+satellite, SunLake satellite (full optical-flow + uncertainty).
5. **Loader + splits + PyTorch Dataset.**
6. **Forecast framework + baselines + metrics.**
7. **Full batch execution + manifest/audit + README / figure-ready outputs.**

Each phase ends with a review checkpoint (per superpowers workflow).

## 14. Open / deferred

- HuggingFace push of standardized packages — deferred; outputs local for now (config flag to add later).
- LLM narration of audit log — deferred (paper "agent" framing), does not affect outputs.
- CrossViViT full training — implemented and runnable; full training-time tuning deferred to a compute-budget decision.
