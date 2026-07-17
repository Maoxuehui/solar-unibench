# Phase 7 — Experiments E1–E9 + Visualizations — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Produce the evidence-of-value experiments (spec §11, E1–E9) that justify every pipeline operation + the "governance > complexity" thesis (E9, the ESSD centerpiece), each as a pure function `(loader, out_dir) -> {figures, metrics}` reading ONLY standardized packages, rendered through a shared `viz/` design system, orchestrated by `scripts/run_experiments.py`.

**Architecture:** `viz/style.py` (palette, light/dark, colorblind-safe, rcParams) + `viz/io.py` (save fig + results CSV). Each experiment `E{n}` = a module `solarunibench/experiments/e{n}_*.py` exposing `run(loader, out_dir) -> dict` (figures + numeric metrics). `scripts/run_experiments.py` runs E1–E9 (subset via `--only E1,E3`). Figures → `figures/` (gitignored); numeric → `figures/results/*.csv`. Experiments reuse the Phase 6 forecast framework + Phase 5 loader.

**Tech Stack:** matplotlib (colorblind-safe palettes), pandas, numpy, the forecast framework. Env `solar_uni`.

## Global Constraints
- **Pure + reproducible:** each experiment reads only standardized packages (via `SolarUniBenchLoader`); outputs deterministic; no mutation of `data/standardized/`.
- **Hypothesis → method → metric → figure** for each E (spec §11).
- **Consistent viz:** all figures via `viz/style.py` (one palette, light default + dark variant, colorblind-safe, clear labels/legends, DPI for print).
- **Bounded for dev:** experiments run on the bounded standardized packages (200-frame images) where images are needed; full-scale runs = Phase 8. Model-using experiments use TINY configs for fast iteration; note full-config numbers come at Phase 8.
- **Daytime eval** where a model is scored (cos_zenith>0.05).
- **Env:** `conda run -n solar_uni ...`; pytest from repo root (or `PYTHONPATH=repo`). TDD, DRY, YAGNI; trailer `Co-Authored-By: Claude <noreply@anthropic.com>`.

**Branch:** `feat/phase7-experiments` (from `master` @ `f739755`).

---

### Task 1: viz design system + run_experiments skeleton
**Files:** `solarunibench/viz/__init__.py`, `viz/style.py`, `viz/io.py`; `scripts/run_experiments.py`; `solarunibench/experiments/__init__.py`.
- `style.py`: `set_style(theme="light")`, a 6-color colorblind-safe palette + dataset/model colormap, rcParams (font, DPI, axes). `io.py`: `save_fig(fig, name, out_dir)` (PNG + optional PDF) + `save_metrics(df, name, out_dir)` (CSV).
- `run_experiments.py`: discover `E1..E9` run() callables; `--only`, `--data-root`, `--out` (`figures/`); call each with the loader + out_dir; print a summary.
- TDD: style applies rcParams; save_fig writes a file; run_experiments `--only E1` dispatches (mock the experiment).

### Task 2: E1 — Geo-recovery accuracy + TSR validation + downstream value (GEFCom)
**Files:** `experiments/e1_geo.py`; test.
- Blind test: run clear-sky recovery on **Folsom** (known coords) → recovered-vs-true lat/lon error + great-circle MAE (km). TSR validation: theoretical TOA (recovered geometry) vs `VAR178` residual/R². Downstream: MLP on GEFCom (a) raw PV lags vs (b) +`cos_zenith/ghi_clearsky/clearness_index` → ΔMAE / skill. Figures: `fig_geo_blindtest`, `fig_geo_tsr_validation`, `fig_geo_skill_gain`.

### Task 3: E2 — Clear-sky / clearness-index feature value (all 4 datasets)
**Files:** `experiments/e2_clearsky.py`; test.
- Ablation: baseline features vs +clear-sky anchors, per dataset (MLP/LightGBM). Figure: `fig_clearsky_skill` (grouped bars, skill gain per dataset).

### Task 4: E7 — Data-defect diagnostics (descriptive, all datasets, NO model)
**Files:** `experiments/e7_defects.py`; test.
- Missingness/irregularity/outlier diagnostics from the standardized packages + `quality_flag` composition. Figures: `fig_defect_missingness`, `fig_defect_irregularity`, `fig_defect_outliers` + a defect summary CSV.

### Task 5: E5 — Quality-audit cleaning value
**Files:** `experiments/e5_quality.py`; test.
- Ablation: raw vs quality-audited training data → skill + target-distribution shift. Uses the `--raw` pre-governance snapshot if available (else synthesize). Figures: `fig_quality_flags`, `fig_quality_cleaning_effect`.

### Task 6: E3 — BRR thin-cloud enhancement (Folsom / SKIPPD)
**Files:** `experiments/e3_brr.py`; test.
- Qualitative gallery (clear/thin-cloud/overcast): RGB vs BRR. Quantitative: image-baseline skill RGB-only vs RGB+BRR; thin-cloud subset skill. Figures: `fig_brr_gallery`, `fig_brr_skill`.

### Task 7: E4 — Optical-flow extrapolation quality + uncertainty calibration (SKIPPD / SunLake)
**Files:** `experiments/e4_optflow.py`; test.
- PSNR/SSIM of forward-splat-extrapolated frames vs real future frames across lead-time K; uncertainty reliability (prediction-interval coverage). Figures: `fig_optflow_sequence`, `fig_optflow_psnr_decay`, `fig_uncertainty_reliability`.

### Task 8: E6 — Cross-modal alignment value (Folsom / SKIPPD)
**Files:** `experiments/e6_alignment.py`; test.
- Timestamp-diff distribution before/after (NN-attach); multimodal baseline skill before/after alignment. Figures: `fig_align_lag_hist`, `fig_align_skill`.

### Task 9: E8 — Unified benchmark results (baselines × datasets × horizons)
**Files:** `experiments/e8_benchmark.py`; test.
- All baselines on all datasets at the 3 horizons, skill vs Smart Persistence (daytime). Figures: `fig_benchmark_heatmap` (model × dataset × horizon) + `results/results_benchmark.csv`. (TINY configs for dev; full at Phase 8.)

### Task 10: E9 — Data governance vs model complexity (CENTERPIECE)
**Files:** `experiments/e9_governance.py`; test.
- 2-axis ablation: data state {raw/unaligned/uncleaned vs pipeline-cleaned+aligned} × model complexity {SmartPersistence → LightGBM → MLP → LSTM → Transformer}, all datasets at chosen horizon. Key contrasts: Δ from data hygiene (same model, raw→cleaned) vs Δ from complexity (same data, simplest→most-complex). Report the crossover where "simple+clean ≥ complex+raw". Figures: `fig_governance_vs_complexity`, `fig_cleaning_delta`.

### Task 11: Integration — run_experiments end-to-end
**Files:** `scripts/run_experiments.py`; test.
- `RUN_SLOW` run of E1–E9 (or a `--quick` subset) on the standardized packages; assert figures + result CSVs produced; print a results summary table.

---

## Definition of Done
- `viz/` design system (colorblind-safe, light/dark); `run_experiments.py` orchestrator (`--only`).
- E1–E9 each a pure `(loader, out_dir)->{figures,metrics}`; figures render; results CSVs written.
- E9 (governance vs complexity) is the centerpiece, demonstrating "data cleaning/alignment > model complexity".
- Fast tests green; RUN_SLOW produces the figure set on standardized packages.

## Notes / out of scope
- Full-scale (all datasets × all horizons × full-config models × full images) = Phase 8 batch.
- README + paper-figure selection = Phase 8.
