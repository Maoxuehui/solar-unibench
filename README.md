# Solar-UniBench — A Multi-modal Standardized Benchmark for Solar Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
<!-- Zenodo DOI Badge (Uncomment after publishing Zenodo deposit) -->
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

**Solar-UniBench v1.0** is an open, analysis-ready, multimodal data product and benchmark suite for reproducible short-term solar forecasting. It integrates four widely used public archives — spanning **11 sites and 91 station-years** across California, Europe, and Australia — into a single physics-guided curation pipeline.

> **Key Thesis — Governance over Complexity:** Experimental evaluations (E9) demonstrate that physics-guided data governance (quality auditing + clear-sky physical anchoring) yields greater forecast skill improvements than scaling model architectural complexity alone.

---

## Key Features

- **Standardized Data Contract:** Emits unified `metadata.parquet`, `time_series.parquet`, and chunked `sky_images.zarr` tensors across all observing sites.
- **Physics-Guided Curation:** Reconstructs missing geospatial parameters, enforces physical boundaries ($k_t \in [0, 1.5]$), and derives physics-informed visual features (NRBR color-ratio maps and optical-flow advection reliability maps).
- **Agentic Workflow Integration:** Curation operators are packaged as an autonomous agentic skill (`SKILL.md`), enabling natural-language tool dispatch for automated data harmonization.
- **Fair & Frozen Benchmark:** Provides standardized chronological splits and a tiered baseline suite (Smart Persistence, LightGBM, MLP, LSTM, iTransformer, PatchTST, and CrossViViT).

---

## Dataset Overview

| Dataset | Target | Modalities (Standardized) | Coverage |
| :--- | :--- | :--- | :--- |
| **Folsom** | GHI / DNI / DHI | Tabular (GHI, weather, NAM NWP) + Ground Sky Imager (`.zarr`) | 1-min resolution, 2014–2016 (775k frames) |
| **SKIPP'D** | PV Power | Tabular (PV, Irradiance) + Sky Imager + Satellite Patches (`.zarr`) | 1-min resolution, 2017–2019 |
| **SunLake** | GHI | Tabular (GHI) + EUMETSAT Satellite Patches (`.zarr`) | 30-min resolution, 6 BSRN stations |
| **GEFCom2014** | PV Power | Tabular (PV, ECMWF NWP); Recovered WGS-84 Coordinates | 1-hour resolution, 3 Australia zones |

*Full datasets are hosted on Hugging Face: [Solar-UniBench-v1.0](https://huggingface.co/datasets/Monicahhhhh06/Solar-UniBench-v1.0).*

---

## Installation

Solar-UniBench requires **Python ≥ 3.10**.

```bash
# 1. Create and activate conda environment
conda create -n solar_uni python=3.10 -y
conda activate solar_uni

# 2. Option A: Core light install (tabular curation pipeline & ML baselines)
pip install -r requirements-core.txt

#    Option B: Full pinned install (deep learning models & Zarr visual tensors)
pip install -r requirements.txt
```

---

## Quickstart

### 1. One-Minute Quick Verification (No External Download Needed)

The repository includes a lightweight 1-day sample dataset (`data/sample/`, ~15 MB) for instant sanity checks:

```bash
# Test the raw -> standardized pipeline on the sample
python scripts/build_dataset.py --sample

# Run agentic-workflow curation on the sample
python scripts/run_agent_curation.py

# Run fast defect diagnostics
python scripts/run_experiments.py --quick
```

### 2. Full Standardization Pipeline

To process the complete raw archives into standardized Parquet/Zarr formats:

```bash
# Standardize each dataset
python scripts/build_dataset.py --dataset gefcom2014
python scripts/build_dataset.py --dataset folsom
python scripts/build_dataset.py --dataset skippd
python scripts/build_dataset.py --dataset sunlake

# Stamp frozen train/val/test splits
python scripts/build_splits.py
```

### 3. Run Benchmark & Reproduce Paper Figures

Execute the benchmark suite and generate print-quality figures for the paper:

```bash
# Run the core experiment suite (E1–E9)
python scripts/run_experiments.py --only E8,E9

# Generate all 10 paper figures -> exported to figures/paper/
python scripts/generate_paper_figures.py
```

Interactive walkthroughs: `notebooks/01_data_quickstart.ipynb` (pipeline on the sample) and `notebooks/02_reproduce_fig8.ipynb` (one-click paper Figure 8).

---

## Repository Structure

```
solar-unibench/
├── config/                  # YAML configs (paths, curation defaults, model hyperparameters)
├── data/sample/             # 1-day Folsom sample (raw + standardized) for instant demos
├── solarunibench/           # Core Python package
│   ├── curation/            # Eq.(1) geo-alignment, Eq.(2-3) physical bounds, Eq.(4-5) NRBR/flow
│   ├── agent/               # Agentic curation layer (tools + skill entry)
│   ├── models/              # Benchmark suite (Smart Persistence, LightGBM, MLP, LSTM,
│   │                        #   iTransformer, PatchTST, DLinear, CrossViViT)
│   ├── pipeline/            # Standardization adapters & contract definitions
│   └── utils/               # Evaluation metrics (MAE, RMSE, Skill%)
├── scripts/                 # CLI entry points
│   ├── run_curation.py      #   run the curation pipeline
│   ├── run_benchmark.py     #   1-h dispatch evaluation & multi-horizon sensitivity
│   └── run_agent_curation.py#   Agentic pipeline entry
├── figures/paper/           # The 10 final paper figures (PNG + PDF)
├── results/                 # Experiment result CSVs (plots/ = figure inputs)
├── notebooks/               # Quickstart & one-click figure reproduction
├── SKILL.md                 # Autonomous Agent Skill definition
├── environment.yml          # Conda environment definition
└── requirements.txt         # Pinned dependencies
```

---

## Citation

If you use Solar-UniBench in your research, please cite our paper:

```bibtex
@article{qi2026solarunibench,
  title   = {Solar-UniBench v1.0: Harmonizing heterogeneous public solar data into a spatiotemporally aligned and quality-aware data product},
  author  = {Qi, Keyu and Mao, Xuehui and Yang, Dazhi and Chu, Yinghao},
  journal = {Earth System Science Data},
  year    = {2026}
}
```

---

## License

Code is licensed under the MIT License. Data files follow the original licenses of their respective host repositories.
