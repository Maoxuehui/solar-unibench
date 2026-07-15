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
│   │   └── models/             # smart_persistence.py, gbt.py, mlp.py, lstm.py, transformer.py, cross_vivit.py
│   ├── metrics.py              # MAE/RMSE/MAPE/Forecast Skill/FS/pinball/CRPS
│   └── viz/                    # experiment + diagnostic figures (uses the dataviz skill)
│       ├── style.py            # shared matplotlib style + palette
│       ├── defect_diagnostics.py  # missingness, irregularity, outlier heatmaps
│       ├── geo_recovery.py     # recovered vs true coords, blind-test error
│       ├── clearsky_value.py   # skill gain from clearsky/clearness features
│       ├── brr_viz.py          # BRR channel + thin-cloud contrast
│       ├── optical_flow_viz.py # extrapolation frames, uncertainty, PSNR/SSIM decay
│       ├── quality_flag_viz.py # flag distribution, before/after cleaning
│       └── benchmark_results.py # baselines × datasets results table + figure
├── scripts/
│   ├── build_dataset.py        # CLI: run one/all adapters → standardized packages
│   ├── run_baseline.py         # CLI: train+eval a baseline on standardized data
│   └── run_experiments.py      # CLI: run the §11 experiment suite → figures + tables
├── tests/                      # synthetic-fixture unit tests per operator/adapter
├── data/standardized/{gefcom2014,folsom,skippd,sunlake}/   # outputs (gitignored)
├── figures/                    # experiment figures + result tables (gitignored)
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

### 4.6 Splits — frozen train/val/test (`split` column + manifest)
To guarantee **fair horizontal comparison**, the benchmark — not the user — owns the split. Each standardized dataset carries a frozen, versioned split as a **`split` column inside `time_series.parquet`** (`split ∈ {train, val, test}`), with a thin `splits_manifest.json` for provenance. Single source of truth, zero joins, self-consistent (no drift between data and split).
- `SolarUniBenchLoader.load_tabular(dataset, split=...)` filters the column; rows are labeled at build time, so test data cannot leak into training by accident. The `split` column is optional in the contract (present after `build_splits`, absent from `--raw` packages).
- **Default strategy — temporal, per-entity, leak-free:** chronological split within each `entity_id`. `test` is taken first from a dataset-specific cutoff (`test_start`), then the remainder is split train/val by `val_ratio`.
- **Per-dataset natural test sets (honor the source, stronger than arbitrary ratios):**
  - **GEFCom** → `test_start = 2014-06-01` (the competition's held-out `Solution to Task 15` future month); Tasks 1–14 split train/val.
  - **Folsom** → `test_start = 2016-01-01` (2016 held out).
  - **SunLake** → `test_start = 2021-01-01` (2021–2022 held out).
  - **SKIPPD** → reuse its official HF train/test (`strategy: predefined`); carve val from train.
- `splits_manifest.json` records cutoffs, ratios, counts, and a `splits_version` tag so any re-run reproduces the exact same split.
- **Spatial protocol (cross-domain zero-shot), separate** — `configs/splits.yaml` also defines leave-one-dataset-out folds (train on 3 → zero-shot test on the 4th); this is benchmark-level, not a per-row label.

## 5. Base pipeline & operators (`pipeline/base.py`, `pipeline/operators/`)

`BaseSolarPipeline` (ABC) — constructor `(raw_data_dir, output_dir, dataset_name, config)`. Adapters implement `load_and_standardize_raw() -> (df_meta, df_ts)` and `process_and_save_tensors(df_ts)`.

Operators (in the base):
- `_compute_clearsky_and_astronomy()` — pvlib `solarposition.get_solarposition` + `clearsky.ineichen` (correct API: pass `linke_turbidity` via `lookup_linke_turbidity` or use defaults; fix the methodology's erroneous `appointment_time=` call).
- `_apply_quality_audit_and_features()` — night-zeroing (`cos_zenith <= 0`), `clearness_index`, `quality_flag`, physical clamp to `[0, capacity]`.
- `generate_brr_channel()` — RGB→4-channel with BRR (from `extensions.py`).
- `extrapolate_with_uncertainty()` — TV-L1 optical flow + semi-Lagrangian warp + divergence-based uncertainty (from `extensions.py`, moved to `operators/optical_flow.py`).
- `_reverse_geolocate()` — **recovery via the clear-sky method**: grid-search the lat/lon minimizing residual between the observed clearness/surface-radiation diurnal pattern and the pvlib clear-sky model. **Correctness validated by the TSR method**: independently verify recovered coords by comparing theoretical top-of-atmosphere solar radiation (pure solar geometry from recovered coords, cloud-free by construction) against the dataset's `VAR178 (TSR)` field; low residual confirms the recovery. Reuse/validate against existing `GEFCom2014_data/geo_location.py`, `clearsky_output.txt`, `geo_location_results.csv`.
- `run_pipeline()` — load → clearsky → quality → (tensors) → save + manifest + audit log.

**Corrections to the methodology's reference code:** fix the Ineichen call signature; index `solpos`/merge on `timestamp_utc` correctly; ensure `entity_id` is preserved through merges.

## 6. Per-dataset adapters

| Adapter | Raw parsing | Tabular target | Tensors produced |
|---|---|---|---|
| **gefcom** | **Solar track only** — `GEFCom2014_data/Solar/Task *` (train/predictors CSVs + xlsx); the Load/Wind/Price tracks are **not** used. Recover the 3 anonymous zones' geo via the **clear-sky method**, validated by the **TSR method** | `pv_power_normalized` | none |
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
  - `load_tabular(dataset, split=None) -> df` — merged metadata + time_series, **filtered by the frozen split** when `split ∈ {train, val, test}` (§4.6). Unsplit rows are excluded.
  - `get_tensor_cube(dataset, kind) -> zarr.Array` — lazy `DirectoryStore`; HTTP-mappable later.
- `SolarUniBenchDataset(torch.utils.data.Dataset)` — fixed column/tensor names so a batch can mix all four datasets; modality auto-switch.
- Splits (`splits.py` + `scripts/build_splits.py`, config `configs/splits.yaml`): `make_splits()` (per-entity chronological, dataset-specific `test_start`/`val_ratio`; `predefined` strategy for SKIPPD in Phase 3) and `spatial_split` (leave-one-dataset-out for cross-domain zero-shot). `build()` stamps the `split` column **into** `time_series.parquet` and writes `splits_manifest.json`.

## 10. Forecasting framework & baselines (`forecast/`)

- `forecast/core.py` — `PipelineConfig`, `SolarForecastPipeline`, `FeatureEngineer`/`ForecastModel` ABCs (the `.core` that `extensions.py` imports).
- **Canonical task (approved):** deterministic + probabilistic forecasting of the available target — **PV** for SKIPPD/GEFCom, **GHI** for Folsom/SunLake — at three horizons: intra-hour (15-min), intra-day (1–6 h), day-ahead (24 h).
- **Feature engineers:** Endogenous (PV lags), SkyCam/NRBR stats, multimodal (images+PV).
- **Baselines** (tiered suite of standard solar-forecasting references; all hyper-params config-driven):
  1. **Smart Persistence** — physical skill reference. Target-agnostic: clear-sky-scaled persistence works on both PV (SKIPPD/GEFCom) and GHI (Folsom/SunLake), so it is the skill denominator for every dataset.
  2. **Gradient Boosting (LightGBM)** — the standard strong *tabular* baseline in solar/energy forecasting (hand-crafted lag + NWP + clear-sky features).
  3. **MLP** — fully configurable architecture (default `[128, 64]` hidden layers; the 7-neuron config is preserved as a `--preset skippd_paper` replication of Nie et al. 2020). PV/tabular features; an `MLP+SkyCam/NRBR` variant covers image features.
  4. **LSTM** — recurrent sequence baseline.
  5. **Transformer (PatchTST-style)** — transformer-family sequence baseline (patch tokenization + encoder); the modern deep-learning reference. Config flag allows swapping to a vanilla time-series Transformer or Informer.
  6. **CrossViViT** — multimodal (sky/satellite + tabular) cross-attention ViT.
  - Adding/swappping a model is a drop-in `ForecastModel` subclass, so the set is extensible without touching the harness.
- **Metrics (`metrics.py`):** MAE, RMSE, MAPE, Forecast Skill vs. Smart Persistence, pinball loss / CRPS.

## 11. Experiments & visualizations — evidence of value (`viz/`, `scripts/run_experiments.py`)

**Goal:** empirically demonstrate that each pipeline operation (alignment, feature engineering, quality audit) improves downstream utility, plus diagnostic figures of the data itself. Every figure is rendered through the **dataviz skill** design system (consistent palette, light/dark, accessible). Figures → `figures/` (gitignored); numeric results → `figures/results/*.csv`.

Each experiment is specified as **hypothesis → method → metric → figure**, and implemented as a pure function `(loader, out_dir) -> {figures, metrics}` that reads only the standardized packages (fully reproducible). `scripts/run_experiments.py` runs E1–E9 (subset via `--only E1,E3`).

**E1 — Geo-recovery (clear-sky) accuracy + TSR validation + downstream value (GEFCom).** *Hypothesis: the clear-sky method recovers accurate coords (confirmed independently by TSR), and the features those coords unlock improve forecasting.*
- **Blind test:** run clear-sky recovery on **Folsom** (known coords) → recovered-vs-true lat/lon error + great-circle MAE (km). Then recover GEFCom's 3 anonymous zones.
- **TSR validation:** for each recovered coord, compare theoretical TOA solar radiation (from recovered geometry) against the dataset's `VAR178 (TSR)` field; report residual / R² as the independent correctness check.
- **Downstream ablation:** MLP on GEFCom (a) raw PV lags vs (b) + `cos_zenith / ghi_clearsky / clearness_index` from recovered coords → ΔMAE / Forecast Skill.
- Figures: `fig_geo_blindtest.png` (recovered vs true on map + error bars), `fig_geo_tsr_validation.png` (TSR theoretical vs observed, per zone), `fig_geo_skill_gain.png`.

**E2 — Clear-sky / clearness-index feature value (all 4 datasets).** *Hypothesis: physics anchors beat raw endogenous features.*
- Ablation: baseline features vs +clear-sky anchors, per dataset.
- Figure: `fig_clearsky_skill.png` (grouped bars, skill gain per dataset).

**E3 — BRR thin-cloud enhancement (Folsom / SKIPPD).** *Hypothesis: the BRR 4th channel amplifies thin-cloud edges and improves image forecasting.*
- Qualitative gallery (clear / thin-cloud / overcast): RGB vs BRR.
- Quantitative: image-baseline skill RGB-only vs RGB+BRR; thin-cloud subset skill.
- Figures: `fig_brr_gallery.png`, `fig_brr_skill.png`.

**E4 — Optical-flow extrapolation quality + uncertainty calibration (SKIPPD / SunLake).** *Hypothesis: extrapolated future frames track truth within the predicted uncertainty envelope.*
- Metrics: PSNR/SSIM of extrapolated frames vs real future frames across lead-time K; uncertainty reliability (prediction-interval coverage).
- Figures: `fig_optflow_sequence.png` (context → extrapolated → truth + uncertainty row), `fig_optflow_psnr_decay.png`, `fig_uncertainty_reliability.png`.

**E5 — Quality-audit cleaning value.** *Hypothesis: `quality_flag` (night-zero, de-clip, fault removal) cuts label noise and lifts skill.*
- Ablation: raw vs quality-audited training data → skill + target-distribution shift.
- Figures: `fig_quality_flags.png` (flag composition per dataset), `fig_quality_cleaning_effect.png`.

**E6 — Cross-modal alignment value (Folsom / SKIPPD).** *Hypothesis: temporal alignment + systematic-lag correction reduces image↔scalar async label noise.*
- Metric: timestamp-diff distribution before/after; multimodal baseline skill before/after.
- Figures: `fig_align_lag_hist.png`, `fig_align_skill.png`.

**E7 — Data-defect diagnostics (descriptive, all datasets).** *The methodology's "缺陷热力图" diagnostic reports — no model.*
- Figures: `fig_defect_missingness.png`, `fig_defect_irregularity.png`, `fig_defect_outliers.png` + a defect summary table.

**E8 — Unified benchmark results (baselines × datasets).** *One standardized eval proving the benchmark is uniform.*
- All baselines on all datasets at the three horizons, skill vs Smart Persistence.
- Figures: `fig_benchmark_heatmap.png` (model × dataset × horizon), `results_benchmark.csv`.

**E9 — Data governance vs. model complexity (the "治理重于模型" thesis; centerpiece).** *Hypothesis: cleaning + alignment buys more metric gain than scaling model complexity — i.e., a simple model on well-governed data can match or beat a complex model on raw data.*
- **2-axis ablation:** data state {**raw / unaligned / uncleaned** vs **pipeline-cleaned + aligned**} × model complexity {Smart Persistence → LightGBM → MLP → LSTM → Transformer}, on all datasets at the chosen horizon. The "raw" state is the standardized-but-pre-quality/unaligned frame (a `--raw` snapshot the base pipeline emits *before* the quality-audit + alignment stages), so both arms share identical parsing and split boundaries — the only difference is governance.
- **Key contrasts:** (i) Δ from *data hygiene* = same model, raw→cleaned; (ii) Δ from *complexity* = same data, simplest→most-complex model. Report the **crossover** where "simple + clean ≥ complex + raw" — the empirical proof that governance dominates complexification.
- Figures: `fig_governance_vs_complexity.png` (skill heatmap over {data state × model}), `fig_cleaning_delta.png` (per-dataset skill gain attributable to cleaning/alignment alone vs. to model upgrade).

## 12. Execution, error handling, robustness

- `scripts/build_dataset.py` — resumable, idempotent (`--force` to overwrite); per-stage checkpoints; `tqdm` + log progress; heavy zarr stages run as background jobs. All runs append to `audit.jsonl` + emit `manifest.json`.
- Robustness: physical clamping; NaN propagation tracked in `quality_flag`; unreadable frames logged + skipped; UTC enforced at ingest; defensive raw-format parsing with explicit dtypes.
- Environment: dedicated conda env `solar-unibench` from `requirements.txt` (base Anaconda lacks pandas/pvlib/zarr/pyarrow/sklearn/xarray).

## 13. Testing strategy (TDD)

Synthetic-fixture unit tests for each operator:
- clearsky vs. known pvlib reference values at a known site/time
- BRR channel bounds ∈ [−1, 1]
- optical-flow extrapolation sanity (identity on identical frames; translation on shifted frames)
- reverse-geolocation recovery on a synthetic known-coord site (MAE bound)
- quality_flag assignment (night, clipping, outlier)

Plus a **contract-conformance test** each adapter must pass (schema, dtypes, key uniqueness, file-presence rules). Tests run on synthetic data so CI is fast; real-data execution is the final batch phase.

## 14. Phased plan (build strategy C — vertical spike then layers)

1. **Env + scaffold + contract + tests harness.**
2. **Vertical spike:** GEFCom2014 tabular end-to-end (incl. reverse-geolocation) **+ frozen train/val/test split** — locks base class, contract, CLI, and splits.
3. **Tabular layer:** Folsom, SKIPPD, SunLake standardized parquet (Tables 1–4).
4. **zarr/image layer:** Folsom sky (RGB+BRR), SKIPPD sky+satellite, SunLake satellite (full optical-flow + uncertainty).
5. **Loader + splits + PyTorch Dataset.**
6. **Forecast framework + baselines + metrics.**
7. **Experiments & visualizations (§11, E1–E8)** — evidence-of-value figures + result tables; feeds the README and paper figures.
8. **Full batch execution + manifest/audit + README / figure-ready outputs.**

Each phase ends with a review checkpoint (per superpowers workflow).

## 15. Open / deferred

- HuggingFace push of standardized packages — deferred; outputs local for now (config flag to add later).
- LLM narration of audit log — deferred (paper "agent" framing), does not affect outputs.
- CrossViViT full training — implemented and runnable; full training-time tuning deferred to a compute-budget decision.
