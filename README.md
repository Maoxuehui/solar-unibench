# Solar-UniBench — A Multi-modal Standardized Benchmark for Solar Forecasting

<!-- TODO(dataset-release): swap XXXXXXX for the Zenodo record number once
     the dataset deposit (draft 23201168) is published, then uncomment. -->
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

**Target venue:** ESSD (Earth System Science Data)
**Status:** Framework complete (Phases 1–8); full-config results populated for all experiments. E8 rebuilt on the physics-time-aligned grid with CrossViViT v2 (2026-08). Folsom sky images materialized for 2014–2016 (775,845 frames).

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

> ### Full-config vs. dev-config
> The tables below report **full-config** results (`SUB_FULL_CONFIG=1`:
> convergence-tuned GPU models, 50k-row/entity cap, 3-horizon grid; Folsom
> imagery = 2014–2016, 775,845 frames in per-year zarr stores). The dev-config
> `figures/` remains for the fast test suite and should not be cited as the
> publishable set. All numbers are **honest** (see *Honest NaN* below) — negative
> results are reported faithfully, never hidden.

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
| **Folsom** | GHI / DNI / DHI | Tabular (irradiance, weather, NAM NWP) + sky images (per-year `sky_images_YYYY.zarr`) | 775,845 frames (2014–2016), JPEG-q90 RGB @256² + BRR-on-load |
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

# 2. Install the verified full environment (pinned; reproduces the paper)
pip install -r requirements.txt

#    ...or the minimal core stack for the tabular pipeline only:
#    pip install -r requirements-core.txt
```

The standardized data root is configured in `configs/paths.yaml`
(`output_root`). Raw sources live under `data/raw/` (gitignored).

---

## Quickstart

### 0. One-minute experience (no data mirror needed)

The repo ships a 1-day Folsom sample (`data/sample/`, ~16 MB). Rebuild it
through the REAL curation pipeline, then run an experiment on it:

```bash
python scripts/build_dataset.py --sample     # raw -> standardized via the real pipeline
python scripts/run_experiments.py --sample   # E7 defect diagnostics on the sample
python scripts/run_agent_curation.py         # the same curation, driven by an agent
```

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

# Reproduce the paper's core experiments (benchmark + governance crossover)
python scripts/run_experiments.py --only E8,E9

# Subset
python scripts/run_experiments.py --only E1,E9

# E9/E5 raw-vs-cleaned axis needs a bounded RAW snapshot (real build, cached):
RUN_SLOW=1 python scripts/run_experiments.py --only E9
```

Key flags: `--only`, `--quick`, `--data-root`, `--out`, `--raw-source`,
`--raw-cache`.

### 3. Agent-mode curation

```bash
# Natural-language instruction -> tool dispatch (geolocation / physical QC /
# NRBR + optical-flow features) with a ReAct-style trace; see SKILL.md
python scripts/run_agent_curation.py
python scripts/run_agent_curation.py --instruction "recover the site coordinates only"
```

### 4. Export the paper figures

```bash
# All 9 code-generated figures (fig04 is hand-drawn) -> figures/paper/
python scripts/generate_paper_figures.py            # all
python scripts/generate_paper_figures.py --fig 8    # just Figure 8
```

Notebooks: `notebooks/01_quickstart_data_pipeline.ipynb` (pipeline on the
sample) and `notebooks/02_reproduce_fig8.ipynb` (Figure 8 from the
result CSVs).

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

> The `--full` flag (`scripts/run_experiments.py --full`) re-runs every baseline ×
> dataset × horizon at full config (`SUB_FULL_CONFIG=1`).

---

## Results

> **Full-config results.** E8 tables below come from the rebuilt
> physics-time-aligned benchmark in `figures/results/` (FULL training: 50
> epochs, AdamW+Cosine, early stopping, daylight-masked Skill%). The E9/E2/E6
> ablation tables come from `figures_full/results/` (`SUB_FULL_CONFIG=1`),
> re-verified on the merged code tree 2026-08 (E9/E2 bit-identical to the
> pre-merge archive; E6 re-based on the fully materialized Folsom imagery).
> The dev-config `figures/` remains for the fast test suite.

### E8 — Unified benchmark (physics-time-aligned, Forecast Skill vs. Smart Persistence)

The E8 grid was **rebuilt** (2026-08) on a physics-time-aligned design: every
dataset is forecast at its **native operational horizons** with a lookback
matched to its physical memory scale, under a fixed F=4 input
`[target_lag, cos_zenith, ghi_clearsky, clearness_index_aligned]` plus the
known future astronomical trajectory `X_future=[ghi_clearsky, cos_zenith]`,
scored by daylight-masked Skill% (grid rationale:
`docs/results/experiment_grid_design.md`).

| Dataset | Lookback | Horizons (physical) |
|---|---|---|
| Folsom (GHI, 1-min) | 4 h | 5 / 10 / 15 min |
| SKIPPD (PV, 1-min) | 4 h | 5 / 10 / 15 min |
| SunLake (GHI, 30-min) | 12 h | 30 / 60 / 90 min |
| GEFCom2014 (PV, 1-h) | 24 h | 1 / 2 / 3 h |

Daytime Skill_MAE% per horizon (0 = parity with Smart Persistence; +=better):

| Model | Folsom 5/10/15min | SKIPPD 5/10/15min | SunLake 30/60/90min | GEFCom 1/2/3h |
|---|---|---|---|---|
| LightGBM | −9.5 / −6.1 / −3.3 | −4.6 / +9.3 / +18.4 | +30.8 / +26.0 / +24.0 | +31.7 / +29.2 / +30.8 |
| MLP | −93.8 / −48.2 / −45.0 | −200.3 / −122.4 / −51.9 | +22.9 / +26.6 / +22.0 | +27.6 / +27.2 / +25.4 |
| LSTM | −10.4 / +1.9 / +5.9 | −12.7 / +1.8 / +17.0 | **+32.1** / +28.1 / +23.2 | **+33.9** / +23.8 / +22.6 |
| iTransformer | −106.3 / −25.4 / −20.9 | −60.6 / −33.4 / −63.2 | +29.9 / +25.5 / +23.6 | +29.0 / +23.5 / +27.6 |
| PatchTST-CI | −112.6 / −46.9 / −45.4 | −272.0 / −161.0 / −100.4 | +15.3 / +12.8 / +5.7 | +28.8 / +14.1 / +21.2 |
| PatchTST-CD | −21.4 / −10.0 / +0.1 | −26.6 / +0.8 / +13.2 | +30.2 / +26.2 / +24.7 | +33.1 / +26.8 / +25.3 |
| CrossViViT v2 | −13.8 / −14.9 / −3.9 | −12.0 / +8.1 / **+19.9** | n/a | n/a |

Daytime Skill_RMSE% per horizon (same sign convention):

| Model | Folsom 5/10/15min | SKIPPD 5/10/15min | SunLake 30/60/90min | GEFCom 1/2/3h |
|---|---|---|---|---|
| LightGBM | **+8.5 / +10.7 / +12.1** | **+3.9 / +9.1 / +12.0** | **+18.9 / +18.2 / +18.6** | **+27.4 / +28.2 / +31.7** |
| MLP | −8.3 / +4.2 / +6.3 | −43.3 / −31.5 / −6.6 | +17.0 / +17.4 / +18.6 | +25.6 / +26.0 / +29.7 |
| LSTM | +5.5 / +7.6 / +9.7 | −10.6 / −1.4 / +5.1 | +17.4 / +16.5 / +14.8 | +24.4 / +22.4 / +25.8 |
| iTransformer | −10.7 / +5.9 / +6.4 | −16.8 / −10.0 / −24.1 | +16.6 / +14.8 / +15.5 | +23.0 / +20.5 / +24.0 |
| PatchTST-CI | −19.8 / −1.6 / −1.7 | −95.9 / −62.1 / −41.7 | +9.0 / +10.7 / +6.3 | +20.2 / +13.2 / +21.4 |
| PatchTST-CD | +4.5 / +7.5 / +9.2 | −2.3 / +5.1 / +9.2 | +16.4 / +15.0 / +16.0 | +24.3 / +24.1 / +24.0 |
| CrossViViT v2 | +4.0 / +6.7 / +9.5 | −0.4 / +4.1 / +9.4 | n/a | n/a |

Daytime MAE per horizon (normalized units):

| Model | Folsom 5/10/15min | SKIPPD 5/10/15min | SunLake 30/60/90min | GEFCom 1/2/3h |
|---|---|---|---|---|
| LightGBM | **0.014 / 0.017 / 0.019** | 0.015 / 0.019 / 0.022 | 0.032 / 0.036 / 0.039 | 0.072 / 0.084 / 0.090 |
| MLP | 0.025 / 0.024 / 0.027 | 0.044 / 0.047 / 0.041 | 0.036 / 0.036 / 0.040 | 0.077 / 0.086 / 0.097 |
| LSTM | 0.014 / 0.016 / **0.017** | 0.017 / 0.021 / 0.022 | **0.031** / 0.035 / 0.039 | **0.070** / 0.090 / 0.100 |
| iTransformer | 0.026 / 0.020 / 0.022 | 0.024 / 0.028 / 0.044 | 0.032 / 0.037 / 0.039 | 0.075 / 0.091 / 0.094 |
| PatchTST-CI | 0.027 / 0.024 / 0.027 | 0.055 / 0.055 / 0.054 | 0.039 / 0.043 / 0.048 | 0.076 / 0.102 / 0.102 |
| PatchTST-CD | 0.015 / 0.018 / 0.019 | 0.019 / 0.021 / 0.023 | 0.032 / 0.036 / 0.039 | 0.071 / 0.087 / 0.097 |
| CrossViViT v2 | 0.014 / 0.019 / 0.019 | **0.017 / 0.020 / 0.021** | n/a | n/a |

Daytime RMSE per horizon (normalized units):

| Model | Folsom 5/10/15min | SKIPPD 5/10/15min | SunLake 30/60/90min | GEFCom 1/2/3h |
|---|---|---|---|---|
| LightGBM | **0.036 / 0.042 / 0.044** | **0.041 / 0.048 / 0.052** | **0.052 / 0.058 / 0.061** | **0.107 / 0.123 / 0.132** |
| MLP | 0.043 / 0.045 / 0.047 | 0.061 / 0.069 / 0.063 | 0.053 / 0.058 / 0.061 | 0.109 / 0.127 / 0.136 |
| LSTM | 0.037 / 0.043 / 0.046 | 0.047 / 0.053 / 0.056 | 0.053 / 0.059 / 0.064 | 0.111 / 0.133 / 0.143 |
| iTransformer | 0.044 / 0.044 / 0.047 | 0.050 / 0.058 / 0.074 | 0.054 / 0.060 / 0.064 | 0.113 / 0.136 / 0.147 |
| PatchTST-CI | 0.047 / 0.048 / 0.051 | 0.084 / 0.085 / 0.084 | 0.059 / 0.063 / 0.071 | 0.117 / 0.149 / 0.152 |
| PatchTST-CD | 0.038 / 0.043 / 0.046 | 0.044 / 0.050 / 0.054 | 0.054 / 0.060 / 0.063 | 0.111 / 0.130 / 0.147 |
| CrossViViT v2 | 0.038 / 0.044 / 0.046 | 0.043 / 0.050 / 0.054 | n/a | n/a |

(SmartPersistence reference MAE/RMSE per dataset: Folsom 0.013/0.016/0.019 and
0.040/0.047/0.050 · SKIPPD 0.015/0.021/0.027 and 0.043/0.052/0.059 · SunLake
0.046/0.049/0.051 and 0.064/0.071/0.075 · GEFCom 0.106/0.119/0.130 and
0.147/0.171/0.193.)

Canonical file: `figures/results/e8_summary.csv` (48 rows: 4 datasets × 4
metrics × 3 horizons; raw per-cell values in `e8_full_checkpoint*.csv`).
CrossViViT runs only where sky images resolve (Folsom/SKIPPD). Figures:
`figures/results/fig_e8_skill_heatmap.png` (two-panel diverging skill heatmap,
MAE | RMSE) and `fig_e8_horizon_lines.png` (per-dataset skill-vs-horizon
crossover lines), generated by `scripts/plots/` (paper set) directly from the
CSV (figures cannot drift from the table).

**Honest readings.**
- On 1-min data at 5-min lead, persistence is near-unbeatable on MAE (clouds
  barely move): every model is negative on Folsom h5; LSTM turns positive from
  10 min (+1.9 → +5.9), CrossViViT v2 reaches +19.9 on SKIPPD-15min. On RMSE
  skill, however, LightGBM/CrossViViT/PatchTST-CD are already positive at
  5 min — the models trim the large errors first even when the median error
  lags persistence.
- At coarser cadences the models dominate: **LSTM is the best MAE model** on
  SunLake (+32.1) and GEFCom (+33.9); **LightGBM is the best RMSE model on
  every dataset** (Skill_RMSE +8.5…+31.7 across all cells where it is positive,
  i.e. all but the 1-min MAE cells).
- **DLinear is excluded from the displayed tables/figures** (user decision):
  its trend/seasonal + linear head collapses on the 1-min datasets
  (Skill_MAE −590%…−382% on Folsom), compressing every shared scale; its full
  results remain in `e8_summary.csv` for completeness (it does turn positive on
  GEFCom at longer leads, +7.8 at 3 h). PatchTST-CI (channel-independent) is
  systematically weak on low-channel tabular signals; the CD variant recovers
  most of the gap.

> **CrossViViT v2 note (honest):** the benchmark's CrossViViT is a
> **lightweight variant (~224 K params)** inspired by Boussif et al. 2023
> (arXiv:2306.01112), **not** the official 145 M RoCrossViViT. v2 (2026-08)
> keeps the image temporal axis (frame-position embeddings; no temporal
> mean-pool) and fuses the known future trajectory `X_future` as extra tokens.
> Single-seed MAE regressed slightly vs v1 while RMSE improved; kept as
> official per the recorded decision + limitation statement
> (`docs/results/e8_crossvivit_v2_notes.md`; v1 results archived under
> `figures/results/archive_e8_pre_crossvivit_v2/`). The image-enabled variant
> (`e8_crossvivit_image_v2.csv`) is reported separately on image-covered windows.

> **Folsom imagery note:** Folsom sky-image experiments use **2014–2016 data**
> (775,845 frames materialized into per-year stores `sky_images_2014.zarr`,
> `sky_images_2015.zarr`, `sky_images_2016.zarr`, JPEG-q90 RGB @256² + BRR-on-load).
> The `ConcatenatedSkyCube` wrapper (`loader.py`) virtually concatenates per-year
> stores so downstream code is unaware of the partitioning.

### E9 — Governance vs. complexity (the centerpiece)

Two-axis ablation: data state {**raw / unaligned / uncleaned** vs.
**pipeline-cleaned + aligned**} × model complexity {Smart Persistence → LightGBM
→ MLP → LSTM → Transformer}. The headline is the **crossover** where
"simple + clean ≥ complex + raw".

| Contrast | What it measures | Result |
|---|---|---|
| Δ from **data hygiene** | same model, raw → cleaned | **+0.045 mean** (up to **+0.118** for PatchTST) |
| Δ from **complexity** | same data, simplest → most-complex (cleaned) | +0.172 (PatchTST over SmartPersistence, cleaned) |
| **Crossover** (simple + clean ≥ complex + raw) | governance dominates? | **YES** — MLP + cleaned (skill +0.221) beats the most complex model on raw (PatchTST +0.054) |

`figures_full/results/results_e9_governance.csv`. The real-data crossover
reproduces the thesis demonstrated on the synthetic axis in Phase 7: **governance
(cleaning + alignment) buys more skill than scaling model complexity** — a simple
model on well-governed data matches or beats a complex model on raw data.

### Phase 3 — cross-dataset replication, governance waterfall, granularity

**Cross-dataset crossover (physics-aligned 1h-ahead, `results_e9_cross_dataset.csv`):**
the cadence-mismatch artifact in the first pass (1-min datasets stuck in a
degenerate all-negative-skill regime because horizon=1 row = 1-minute-ahead) was
fixed by aligning a 12 h lookback / 1 h-ahead forecast in physical time
(GEFCom L=12/H=1, SunLake L=24/H=2, Folsom/SKIPPD L=720/H=60). With that, the
crossover holds on **all 4 datasets** with margins: GEFCom +0.012, Folsom +0.097,
SKIPPD +0.013, SunLake +0.198 (`fig_e9_cross_dataset_crossover.png`).

**Governance waterfall (Folsom, LightGBM probe, `results_e9_waterfall.csv`):**
label-governance (night-zero / range-clamp / UTC) is a near-no-op on
already-clean raw (MAE 0.0532 → 0.0542), while adding clear-sky anchor
features cuts MAE to 0.0462 (**−13.1%**) — the physical anchors dominate the
governance gain, now measured with the same "simplest non-trivial" model the
crossover criterion uses (`fig_e9_waterfall.png`; the earlier DLinear probe
showed the same ordering at −2.0%).

**Temporal granularity (Folsom 1min→1h, dual-metric `results_e9_granularity.csv`):**
Smart-Persistence physical MAE rises 47.1 → 52.1 W/m² (nMAE 0.035 → 0.046) as
cadence coarsens from 1 min to 1 h — the SP-skill-collapse hypothesis confirmed
in unscaled units (`fig_e9_granularity_sp_collapse.png`). Honest caveat: even
the LightGBM probe (swapped for DLinear 2026-08) does not exploit the collapse
at this bounded grid (skill −6.1 at 1-min, −4.3 at 1-h) — disclosed as a
setup limitation (lookback=12-row windows at every granularity) rather than
re-run to force a flip.

**Physical-closure parity (ESSD data validation, `fig_physics_closure_parity.png` / paper fig11):** measured GHI vs DNI·cos(z)+DHI — Folsom R²=0.9998 (single-station, 1-min) and SunLake R²=0.9398 (6-station wide-area), confirming the irradiance modality is physically consistent.

**Geo-recovery honest limitation (ESSD data validation, `fig_geo_validation.png` / paper fig3b; `results_geo_ablation_phase_shift.csv`, `results_folsom_blind_inversion.csv`):** three disclosed negative/mixed results, reported faithfully rather than hidden. (1) Folsom blind clear-sky inversion (3° global grid, measured GHI as the envelope) recovers (-36, 66) vs the known truth (38.651, -121.169) — a **19,316 km** great-circle error (near-antipodal), the inherent hemisphere ambiguity of fitting a clear-sky shape to cloudy irradiance; the seasonal-amplitude signal is corrupted by cloud noise, so TSR (pure geometry) remains the only hemisphere-unambiguous method. (2) A timestamp phase-shift ablation (dt ∈ {-60…+60} min) shows the recovered longitude drifts at the correct solar-geometry rate (~15°/60 min) in both GEFCom and Folsom, but the absolute location is severely biased on cloudy PV/GHI. The clearness-index corruption metric confirms the clear-sky features are genuinely phase-misaligned (super-physical kt>1.2 fraction: 0.117 at dt=0 rising to 0.201 at dt=-60). (3) Despite feature corruption, a bounded 1h-ahead LightGBM's daytime skill is essentially flat across dt (max drop ≈ 0.001) — the expected "avalanche parabola" did not appear, because autoregressive lags dominate at 1 h horizon and the model is robust to the corrupted clear-sky/cos_zenith anchors.

**BRR representation × capacity ablation (4-arm, `fig_brr_ablation_comparison.png` / paper fig12; `results_brr_capacity_ablation.csv`):** the E3 finding ("scalar BRR stats are honestly negative for irradiance forecasting") is decomposed along two orthogonal axes — **representation form** (scalar stats vs a 2D spatial tensor) × **model capacity** (shallow LightGBM vs a deep ResNet-18 feature extractor with ImageNet pretraining + Channel Padding Alignment). Four arms share the same scalar base (target lag + ghi_clearsky + cos_zenith) and differ only in image representation: ① Baseline (scalar only), ② Scalar BRR (+ BRR mean/std), ③ Deep RGB (+ 512-d ResNet-18 embedding from 3-channel RGB, ImageNet pretrained, stacked into LightGBM), ④ Deep BRR (identical to ③ but ResNet-18 `conv1` widened to 4 channels via Channel Padding Alignment: `conv1.weight[:,:3]` = pretrained RGB, `conv1.weight[:,3]` = mean(R,B) + 1% noise). Arm 4 − Arm 3 on the thin-cloud subset is the headline net gain of letting a CNN read BRR's spatial structure (capacity held fixed). Setup: ImageNet-pretrained ResNet-18 (`ResNet18_Weights.DEFAULT`), supervised 25 epochs with cosine-annealing LR + early stopping (patience=3), then frozen and stacked; ~700 daytime image-bearing windows per dataset; thin-cloud = middle third by mean BRR (same as E3); skill vs Smart Persistence, daytime. Three findings: (i) **capacity dominates** — deep arms cut Folsom overall MAE ~120 → ~92–96 W/m² (skill −1.05 → −0.56..−0.63), the only robust large signal. (ii) **Scalar BRR is not robustly helpful** — marginal on Folsom overall but *hurts* on the Folsom thin-cloud subset (skill −1.91 vs −1.72), reproducing E3's scalar-compression negative. (iii) **Deep BRR > Deep RGB on thin-cloud (Folsom)** — Arm 4 − Arm 3 thin-cloud Δskill is **Folsom +0.097** (the Channel Padding Alignment + ImageNet pretraining flipped the sign from the random-init −0.030 to a stable positive); SKIPPD Δskill is −0.013 (64×64 low-resolution limit, n=35 thin subset too small to trust). The dialectic: scalar compression collapses BRR's spatial topology (shallow trees can't use it); 2D structure lets a pretrained CNN clear the shallow-tree ceiling, and with proper channel alignment (CPA) the 4th BRR channel adds a genuine marginal gain for thin-cloud detection on high-resolution imagery.


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
`dev/select_figures.py` (untracked dev tree).

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
├── figures/                 # generated figures (results/ E8 deliverables ARE tracked)
├── figures_full/            # full-config ablation results (results/ tracked)
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

Code is released under the [MIT License](LICENSE). Dataset slices retain their upstream licenses (see the dataset table above).

License terms will be confirmed prior to release. Standard practice for ESSD
code is an OSI-approved license (e.g., MIT) with data under a CC-BY-4.0 license;
the final choice will be recorded here and in a `LICENSE` file.
