# E8 Unified Benchmark — Physics-Time-Aligned Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the step-locked base E8 with a **physics-time-aligned, top-journal-standard unified benchmark**: 9 pure-time-series models × 4 datasets × 3 physical horizons, fixed F=4 input channels, known-future `X_future`, daytime-masked Forecast-Skill evaluation, and a LaTeX `\multirow` results table.

**Base:** `feat/e8-unified-benchmark` from **`abe0508`** (user-verified Phases 1–7). Insulated from the contaminated `main`.

**Architecture (reuse the Phase-6 framework heavily):**
- **Config** (`benchmark/config.py`): per-dataset physical→step mapping. `folsom/skippd` 1-min → lookback 240 (4h), H=[5,10,15]; `sunlake` 30-min → lookback 24 (12h), H=[1,2,3]; `gefcom2014` 1-h → lookback 24 (24h), H=[1,2,3].
- **Feature prep:** reuse `EndogenousFeatureEngineer(target, feature_cols=[cos_zenith, ghi_clearsky, clearness_index_aligned])` → `X_past[Batch,LB,4]=[target,cos_zenith,ghi_clearsky,clearness_index_aligned]`, `X_future[Batch,H,2]=[ghi_clearsky,cos_zenith]`, `Y[Batch,H]=target`. Add a **`clearness_index_aligned`** column: GHI datasets = `ghi_actual/ghi_clearsky`; **PV datasets = `target/ghi_clearsky` proxy (0 if clearsky==0)** to force F=4.
- **Eval:** reuse `metrics.daytime_metrics` (cos_zenith>0.05 mask on metrics, NOT input). `Skill% = 1 − Model_metric/Persistence_metric` per metric.
- **Models:** 6 exist at `abe0508` (SmartPersistence, LightGBM, MLP, LSTM, PatchTST, CrossViViT). **Port + validate 3** from `main` (DLinear, iTransformer, PatchTST-CD variant) — reference the agent's impl, clean to the ABC.

**Tech Stack:** PyTorch, scikit-learn, LightGBM, pandas. Env `solar_uni`.

## Global Constraints
- **F=4 fixed input** `[target_lag, cos_zenith, ghi_clearsky, clearness_index_aligned]` for ALL datasets (PV clearness = `target/ghi_clearsky` proxy). All multivariate models share one architecture.
- **`X_future = [ghi_clearsky_future, cos_zenith_future]`** fed to every model (known astronomical trajectory — physical boundary).
- **Daytime mask (`cos_zenith>0.05`) on metrics/loss ONLY** — input stays continuous (incl. night).
- **Physical-time horizons** per dataset (not step-locked): the step-counts follow from sampling rate × physical duration.
- **CrossViViT only on folsom/skippd** (sky images); other 8 models pure-tabular everywhere.
- **Targets:** `pv_power_normalized` (gefcom2014/skippd), `ghi_actual` (folsom/sunlake).
- **Zero leakage:** `PhysicalFeatureScaler` fit on TRAIN only (already in framework).
- **Env:** `conda run -n solar_uni ...`; pytest from repo root. TDD, DRY, YAGNI; commit trailer `Co-Authored-By: Claude <noreply@anthropic.com>`.

---

### Task 1: Benchmark config + clearness-index-aligned feature prep
**Files:** `solarunibench/benchmark/__init__.py`, `benchmark/config.py`, `benchmark/features.py`; test `tests/test_benchmark_config.py`.
- `config.py`: `DATASET_CONFIG = {name: DatasetConfig(target, sampling_mins, lookback_steps, horizons_steps, image_dataset)}` per the matrix. `horizons_steps` derived from physical horizons ÷ sampling. Helper `physical_horizons_for(dataset)`.
- `features.py`: `add_clearness_aligned(df, target)` → adds `clearness_index_aligned` (GHI: `ghi_actual/ghi_clearsky` clipped; PV: `target/ghi_clearsky`, 0 if clearsky==0). Returns df with the column.
- TDD: config maps correct (folsom 1-min→240/[5,10,15]; sunlake 30-min→24/[1,2,3]; gefcom 1-h→24/[1,2,3]); clearness proxy correct for PV (0 when clearsky==0) + real for GHI.

### Task 2: Port + validate the 3 missing models (DLinear, iTransformer, PatchTST-CD)
**Files:** `forecast/models/dlinear_model.py`, `itransformer_model.py`, extend `patchtst_model.py` (CD variant); tests.
- Reference the agent's impls on `main` (`git show main:solarunibench/forecast/models/dlinear_model.py`, etc.), but **clean to the `abe0508` `ForecastModel` ABC + `_torch_base` mixin** (verify the ABC signature at `abe0508` — `fit(X_past,Y,times=None,X_future=None)`, `predict(...)->(n,horizon)`, `name`).
- DLinear: trend/season decomposition + linear head. iTransformer: inverted attention across variable channels. PatchTST-CD: channel-dependent patching variant ( alongside existing CI).
- TDD (TINY config): each constructs, `predict` shape `(n,horizon)`, finite, learns a synthetic pattern, seeded determinism; PatchTST both CI/CD variants construct.

### Task 3: Daytime-masked evaluator + Forecast Skill%
**Files:** `benchmark/evaluator.py`; test.
- `evaluate_predictions(y_true, y_pred, cos_zenith, persistence_pred)` → `{MAE, RMSE, Skill_MAE%, Skill_RMSE%}` with daytime mask (reuse `metrics.daytime_metrics`). Skill% = `1 − model_metric/persistence_metric` × 100, signed.
- TDD: night rows masked (cos_zenith≤0.05 excluded); Skill 0 when model==persistence; negative when model worse; MAE/RMSE finite.

### Task 4: 9-model benchmark runner
**Files:** `benchmark/run_e8.py`; test.
- For each dataset × horizon: load via `SolarUniBenchLoader`; `add_clearness_aligned`; build `EndogenousFeatureEngineer(target, [cos_zenith,ghi_clearsky,clearness_index_aligned])` with the dataset's lookback/horizon steps; run SmartPersistence (reference) + 8 models (skip CrossViViT on non-image datasets) via `SolarForecastPipeline`; collect `(dataset, horizon, model) → {MAE,RMSE,Skill_MAE%,Skill_RMSE%}`.
- TDD: mock loader + 2 tiny models → runner produces the right result-dict shape; CrossViViT skipped on gefcom/sunlake.

### Task 5: LaTeX `\multirow` summary
**Files:** `benchmark/summary.py`; test.
- Build a multi-index DataFrame: rows = `(Dataset, Metric, Horizon)`, cols = the 9 models; values = Skill% (or MAE/RMSE). Export to LaTeX via `df.to_latex(multirow=True)` (requires `booktabs`/`multirow`); also CSV.
- TDD: from a synthetic result-dict, the DataFrame has the right shape + the LaTeX string contains `\multirow` + `\cmidrule`.

### Task 6: Integration run on real data (RUN_SLOW)
**Files:** `tests/test_e8_integration.py` (RUN_SLOW); optional `scripts/run_e8_unified.py`.
- Run the full benchmark on the 4 standardized datasets × 3 horizons (TINY model configs for tractability; note full-config is a later run); assert the LaTeX table + CSV produced; print the skill table.
- TDD: RUN_SLOW-gated; produces `figures/results/e8_unified_skill.tex` + `.csv`.

---

## Definition of Done
- Physics-time-aligned config (4 datasets × 3 horizons, step-counts from sampling rate).
- F=4 fixed input + PV clearness proxy; `X_future` known-future; daytime-masked Skill% eval.
- 9 models run (3 ported+validated); CrossViViT only on image datasets.
- Multi-index LaTeX `\multirow` table (Dataset→Metric→Horizon × 9 models) + CSV.
- Fast tests green; RUN_SLOW integration produces the table on real data.

## Notes / out of scope
- Full-config (large-epoch) training of deep models = a later run; this deliverable uses TINY configs for the framework + a tractable first table.
- Exogenous-variable ablations (E2 clear-sky, etc.) intentionally DROPPED (pure-TS benchmark).
- The agent's `main` work (unverified) is NOT merged; this branch is clean from `abe0508`.
