# Solar-UniBench — A Multi-modal Standardized Benchmark for Solar Forecasting

**Target venue:** ESSD (Earth System Science Data)
**Status:** Framework complete (Phases 1–7); full-config publishable numbers are in preparation (Phase 8).

Solar-UniBench takes four heterogeneous, publicly-available solar-forecasting
datasets and routes each through a single object-oriented **standardization
pipeline** (`BaseSolarPipeline` + one per-dataset adapter) so that every output
follows the same data contract. On top of those harmonized packages it provides
a loader, frozen train/val/test splits, a tiered baseline suite, and a
reproducible experiment harness (E1–E9).

The benchmark's intellectual centerpiece — experiment **E9** — is the empirical
test of the thesis **"governance dominates complexification"** (治理重于模型):
cleaning + cross-modal alignment buys more forecast skill than scaling model
complexity. In other words, *a simple model on well-governed data can match or
beat a complex model on raw data.* The E9 crossover — the point where
"simple + clean ≥ complex + raw" — is the headline result.

> ### Honest dev-config vs. full-config distinction (please read)
> All numbers and figures currently in the repository are **dev-config**: a
> bounded, fast configuration (small image caps, short training) used to validate
> the framework end-to-end. They are **honest** (see *Honest NaN* below) but are
> **not the publishable numbers**. The publication set — full image
> materialization and full model configs on all data — is produced by Phase 8
> Stream B. Wherever this README would report a headline metric, it uses a
> `<!-- TODO: full-config numbers -->` placeholder rather than a dev-config value,
> so that no bounded number is mistaken for a final result.

> ### Honest NaN
> Missing modalities are **absent files or NaN columns, never fabricated.** The
> loader switches modality by probing file existence / NaN columns. Each dataset
> exposes only the channels its source actually contains (see
> [Datasets](#datasets)).

---

## Overview

- **One pipeline, four datasets.** Physics-grounded operators — clear-sky
  anchors, clearness index, reverse-geolocation, quality flags, the BRR
  (color-ratio) image channel, and optical-flow extrapolation with uncertainty —
  live in the shared base class. Adapters only translate raw → standard frames.
- **A single data contract.** Every dataset emits the same `metadata.parquet`,
  `time_series.parquet`, and (where imagery exists) `sky_images.zarr` /
  `satellite_future.zarr` with canonical columns and shapes.
- **Governed forecasting.** Deterministic quality audit + cross-modal alignment
  (a fully-logged state machine, `orchestrator.py`) feeds a tiered baseline suite
  evaluated against Smart Persistence.
- **Reproducible by construction.** Deterministic pipeline, frozen splits,
  per-run audit log, config-driven scale.

The core idea is **governance over modeling**: the same operators that clean and
align the data are what unlock downstream skill, and E9 isolates that effect.

---

## Datasets

| Dataset | Target | Modalities (standardized) | Notes |
|---|---|---|---|
| **GEFCom2014** (solar track) | PV power | Tabular (PV, NWP); coordinates **recovered** via the clear-sky method and independently validated by the TSR (top-of-atmosphere radiation) method | 3 anonymous zones; coords inferred (not in source) |
| **Folsom** | GHI / DNI / DHI | Tabular (irradiance, weather, NAM NWP) + sky images (`sky_images.zarr`) | **Image experiments use 2014 data only.** |
| **SKIPPD** | PV power | Full: PV + irradiance + sky images + satellite (`satellite_future.zarr`) | The all-modality set |
| **SunLake** | GHI | Tabular (GHI) + satellite context + optical-flow extrapolation | Multi-station |

Modalities that a dataset does not contain are simply absent (Honest NaN) — the
contract does not synthesize them.

---

## Installation

Solar-UniBench requires Python ≥ 3.10 and a dedicated conda environment named
**`solar_uni`**.

```bash
# 1. Create the environment
conda create -n solar_uni python=3.10
conda activate solar_uni

# 2. Install core dependencies
pip install -r requirements-core.txt

# 3. Additional packages for the full stack (imagery + deep models)
#    torch, lightgbm, opencv-python-headless, zarr, numcodecs, Pillow, xarray,
#    huggingface_hub
#    (install per your platform / CUDA setup)
```

The standardized data root is configured in `configs/paths.yaml`
(`output_root`). Raw sources live under `data/raw/` (gitignored).

---

## Quickstart

### 1. Build standardized packages

```bash
conda activate solar_uni
# One dataset (writes data/standardized/<dataset>/ + manifest + audit log)
python scripts/build_dataset.py --dataset gefcom2014
# ...repeat for folsom, skippd, sunlake

# Stamp the frozen train/val/test splits into time_series.parquet
python scripts/build_splits.py
```

### 2. Run the experiment suite (E1–E9)

`scripts/run_experiments.py` is the single entrypoint. It discovers each
experiment lazily and writes figures + per-experiment CSVs to `figures/` and
`figures/results/`.

```bash
# All implemented experiments (dev config)
python scripts/run_experiments.py

# Fast smoke run (< 1 min): E1, E2, E7
python scripts/run_experiments.py --quick

# Subset
python scripts/run_experiments.py --only E1,E9

# E9/E5 raw-vs-cleaned axis needs a bounded RAW snapshot (real build, cached):
RUN_SLOW=1 python scripts/run_experiments.py --only E9
```

Key flags: `--only`, `--quick`, `--data-root`, `--out`, `--raw-source`,
`--raw-cache`.

### 3. Export paper figures

```bash
# Export ~7 print-quality figures (PDF + 300 dpi PNG) to figures/paper/
python scripts/select_figures.py
```

---

## Reproducibility

- **Deterministic pipeline.** Every `build_dataset.py` run appends to
  `audit.jsonl` and writes a `manifest.json` (row/frame counts, provenance).
- **Frozen splits.** Splits are owned by the benchmark (not the user): a `split`
  column stamped into `time_series.parquet` + a `splits_manifest.json` recording
  cutoffs, ratios, counts, and a `splits_version` tag. Default strategy is
  per-entity chronological with dataset-specific `test_start`; a leave-one-out
  spatial protocol is also defined in `configs/splits.yaml`.
- **Config-driven scale.** A single `--limit` / config flag bounds processing for
  tests and controls scaling.
- **Single-command experiment entrypoint.** `python scripts/run_experiments.py`
  runs the full E1–E9 suite against the standardized packages.

> A one-shot `--full` entrypoint that re-runs every baseline × dataset × horizon
> at full config is part of Phase 8 packaging.

---

## Results

> **Phase 8 full-config results (populated).** The tables below are generated
> from `figures_full/results/` (`SUB_FULL_CONFIG=1`: full-capacity GPU models,
> 50k-row/entity cap, 3-horizon grid; Folsom imagery = 2014). The dev-config
> `figures/` remains for the fast test suite and should not be cited as the
> publishable set.

### E8 — Unified benchmark (Forecast Skill vs. Smart Persistence)

Baselines × datasets × horizons. Skill is reported against Smart Persistence
(daylight-filtered). Horizons: intra-hour (15 min), intra-day (1–6 h), day-ahead
(24 h).

| Model | GEFCom2014 (PV) | Folsom (GHI) | SKIPPD (PV) | SunLake (GHI) |
|---|---|---|---|---|
| Smart Persistence | 0.00 (ref) | 0.00 (ref) | 0.00 (ref) | 0.00 (ref) |
| LightGBM | +0.16 | −0.15 | −0.12 | +0.23 |
| MLP | +0.17 | −0.29 | −0.34 | +0.15 |
| LSTM | +0.18 | −0.56 | −0.55 | +0.11 |
| Transformer (PatchTST) | −0.39 | −1.64 | −3.03 | −0.12 |
| CrossViViT (multimodal) | n/a | −0.63 | −1.01 | n/a |

Daytime skill vs Smart Persistence, mean over the 1 h / 6 h / 24 h horizons
(`figures_full/results/results_benchmark.csv`). Negative = below the persistence
baseline — Smart Persistence is a very strong reference on smooth PV/GHI at
short horizons (e.g. SKIPPD h1 SP MAE ≈ 0.010), so even well-trained models can
sit below it in skill terms. CrossViViT runs only where sky images resolve
(Folsom/SKIPPD); it is omitted on tabular-only packages where its image modality
has no signal.

> **Folsom imagery note:** Folsom sky-image experiments use **2014 data only**
> (250,598 frames materialized into `sky_images.zarr`, JPEG-q90 RGB @256² + BRR-on-load).

### E9 — Governance vs. complexity (the centerpiece)

Two-axis ablation: data state {**raw / unaligned / uncleaned** vs.
**pipeline-cleaned + aligned**} × model complexity {Smart Persistence → LightGBM
→ MLP → LSTM → Transformer}. The headline is the **crossover** where
"simple + clean ≥ complex + raw".

| Contrast | What it measures | Result |
|---|---|---|
| Δ from **data hygiene** | same model, raw → cleaned | **+0.045 mean** (up to **+0.115** for PatchTST) |
| Δ from **complexity** | same data, simplest → most-complex (cleaned) | +0.269 (PatchTST over SmartPersistence, cleaned) |
| **Crossover** (simple + clean ≥ complex + raw) | governance dominates? | **YES** — MLP + cleaned (skill +0.221) beats the most complex model on raw (PatchTST +0.155) |

`figures_full/results/results_e9_governance.csv`. The real-data crossover
reproduces the thesis demonstrated on the synthetic axis in Phase 7: **governance
(cleaning + alignment) buys more skill than scaling model complexity** — a simple
model on well-governed data matches or beats a complex model on raw data.

### Supporting experiments

| ID | Question | Figure(s) |
|---|---|---|
| E1 | Does clear-sky geo-recovery (validated by TSR) unlock downstream skill? | `fig_geo_blindtest`, `fig_geo_tsr_validation`, `fig_geo_skill_gain` |
| E2 | Do physics anchors (clear-sky / clearness) beat raw endogenous features? | `fig_clearsky_skill` |
| E3 | Does the BRR channel enhance thin-cloud contrast? | `fig_brr_gallery`, `fig_brr_skill` |
| E4 | Do extrapolated frames track truth within predicted uncertainty? | `fig_optflow_sequence`, `fig_optflow_psnr_decay`, `fig_uncertainty_reliability` |
| E5 | Does quality-audit cleaning cut label noise? | `fig_quality_flags`, `fig_quality_cleaning_effect` |
| E6 | Does cross-modal alignment reduce image↔scalar async? | `fig_align_lag_hist`, `fig_align_skill` |
| E7 | What do data-defect diagnostics reveal? | `fig_defect_missingness`, `fig_defect_irregularity`, `fig_defect_outliers` |

All figures are rendered through a shared, colorblind-safe design system
(`solarunibench/viz/`); print-quality exports are produced by
`scripts/select_figures.py`.

---

## Repository layout

```
solar-unibench/
├── configs/                 # paths.yaml, splits.yaml (per-dataset config)
├── solarunibench/
│   ├── contract.py          # canonical schemas (single source of truth)
│   ├── pipeline/            # base.py + operators/ + adapters/
│   ├── orchestrator.py      # deterministic quality/alignment state machine
│   ├── loader.py            # SolarUniBenchLoader + PyTorch Dataset
│   ├── splits.py            # frozen train/val/test splits
│   ├── forecast/            # core, features, models (baselines + deep)
│   ├── metrics.py           # MAE/RMSE/MAPE/Skill/pinball/CRPS
│   ├── experiments/         # E1-E9 (pure (loader, out_dir) -> figures+metrics)
│   └── viz/                 # shared style + figure I/O
├── scripts/                 # build_dataset, build_splits, run_experiments,
│                            # run_baseline, rebuild_sky_zarr, select_figures
├── tests/                   # synthetic-fixture unit + contract-conformance tests
├── data/standardized/       # generated packages (gitignored)
├── figures/                 # generated figures + result CSVs (gitignored)
└── docs/                    # methodology + dataset docs + design spec
```

---

## Citation

`<!-- TODO: citation — update with the ESSD DOI once published -->`

```bibtex
@article{solarunibench2026,
  title   = {Solar-UniBench: A Multi-modal Standardized Dataset and Distributed
             Processing Pipeline for Harmonized Solar Forecasting},
  journal = {Earth System Science Data},
  year    = {2026},
  note    = {DOI: TODO}
}
```

---

## License

`<!-- TODO: license — to be finalized (code and data license). -->`

License terms will be confirmed prior to release. Standard practice for ESSD
code is an OSI-approved license (e.g., MIT) with data under a CC-BY-4.0 license;
the final choice will be recorded here and in a `LICENSE` file.
