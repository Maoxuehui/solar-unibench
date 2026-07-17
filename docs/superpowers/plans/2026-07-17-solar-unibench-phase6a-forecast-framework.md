# Phase 6a — Forecast Framework + Metrics + Tabular Baselines — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the forecasting harness (`forecast/core.py` ABCs + pipeline), `metrics.py`, and 3 tabular baselines (**Smart Persistence, LightGBM, configurable MLP**) that consume Phase 5 standardized data. Clean ABC design (NOT bound to the discarded `extensions.py`).

**Architecture:** `FeatureEngineer.create_features(df, lookback, horizon, stride) -> (X_past[n,lb,Fp], Y[n,horizon], times[n], X_future[n,horizon,Ff])` — lookback + future-context features (future clearsky/cos_zenith) so Smart Persistence (physical) and tabular models share one interface. `ForecastModel.fit(X_past, Y, times, X_future)` / `predict(...) -> (n, horizon)`. `SolarForecastPipeline(config, feature_engineer, model).run() -> metrics` loads via `SolarUniBenchLoader`, windows, fits on train, predicts on test, scores via `metrics.py` (incl. skill vs Smart Persistence).

**Canonical task (spec §10):** deterministic forecasting of the target — **PV** (`pv_power_normalized`) for SKIPPD/GEFCom, **GHI** (`ghi_actual`) for Folsom/SunLake — at horizons 15-min / 1–6h / 24h (config-driven).

**Tech Stack:** numpy, pandas, scikit-learn (MLPRegressor), lightgbm 4.6, pvlib (solar geometry for Smart Persistence). Env `solar_uni`.

## Global Constraints

- **Consume Phase 5 data:** `SolarUniBenchLoader(data_root).load_tabular(dataset, split)` returns the merged df (sorted by `entity_id` then `timestamp_utc`). Feature engineering windows from that df.
- **Honest NaN:** absent columns (e.g., GEFCom has no `ghi_actual`; Folsom/SunLake no `pv_power_normalized`) → the chosen `target` must exist or the pipeline raises a clear error (don't fabricate).
- **Per-dataset target:** PV (`pv_power_normalized`) for SKIPPD/GEFCom; GHI (`ghi_actual`) for Folsom/SunLake. Caller sets `PipelineConfig.target`.
- **Splits frozen** in the `split` column (train/val/test). Framework trains on `train`, evaluates on `test` (val for early-stop/tuning where applicable).
- **Smart Persistence = skill denominator** (target-agnostic): uses future `ghi_clearsky` (clear-sky-scaled persistence works for both PV and GHI).
- **MLP configurable:** default hidden `[128, 64]`; `--preset skippd_paper` → `[7]` logistic (Nie et al. 2020 replication). Not fixed 7-neuron.
- **Env:** `conda run -n solar_uni ...`; pytest from repo root (or `PYTHONPATH=repo`). TDD, DRY, YAGNI; commit trailer `Co-Authored-By: Claude <noreply@anthropic.com>`.

**Branch:** `feat/phase6a-forecast-framework` (from `master`).

---

### Task 1: `forecast/core.py` — ABCs + PipelineConfig + EndogenousFeatureEngineer + SolarForecastPipeline

**Files:** Create `solarunibench/forecast/__init__.py`, `solarunibench/forecast/core.py`; Test `tests/test_forecast_core.py`.

**Interfaces:**
- `PipelineConfig(dataset, target, lookback=24, horizon=24, stride=1, feature_cols=None, capacity=None, data_root=None)`.
- `FeatureEngineer` (ABC): `create_features(df, lookback, horizon, stride) -> (X_past, Y, times, X_future)`.
- `ForecastModel` (ABC): `fit(X_past, Y, times=None, X_future=None)`, `predict(X_past, times=None, X_future=None) -> (n,horizon)`, `name`.
- `EndogenousFeatureEngineer(FeatureEngineer)`: X_past = `[target, ghi_clearsky, cos_zenith, clearness_index]` over lookback; X_future = `[ghi_clearsky, cos_zenith]` over horizon; Y = target over horizon.
- `SolarForecastPipeline(config, feature_engineer, model)`: `.run() -> dict` (loads train+test, engineers, fits, predicts, returns `{pred, y_true, times, model_name}`; metrics computed in Task 2's caller or here once metrics exist).

- [ ] **Step 1: Failing test** (`tests/test_forecast_core.py`):

```python
import numpy as np, pandas as pd, pytest
from solarunibench.forecast.core import (PipelineConfig, EndogenousFeatureEngineer,
                                          SolarForecastPipeline, ForecastModel)

def _toy_df(n=60, target="pv_power_normalized"):
    t = pd.date_range("2020-06-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"timestamp_utc": t, "entity_id": "E1",
        target: np.sin(np.arange(n)/6)**2 + 0.1,
        "ghi_clearsky": np.linspace(0, 800, n), "cos_zenith": np.linspace(0, 0.9, n),
        "clearness_index": np.linspace(0, 0.9, n), "split": ["train"]*(n-12) + ["val"]*6 + ["test"]*6})

def test_endogenous_windows_shapes():
    fe = EndogenousFeatureEngineer(target="pv_power_normalized",
        feature_cols=["ghi_clearsky", "cos_zenith", "clearness_index"])
    Xp, Y, T, Xf = fe.create_features(_toy_df(), lookback=6, horizon=3, stride=1)
    assert Xp.ndim == 3 and Xp.shape[1] == 6 and Y.shape[1] == 3
    assert Xf is not None and Xf.shape[1] == 3          # future clearsky+cos_zenith
    assert len(T) == Xp.shape[0] == Y.shape[0]

class _Const(ForecastModel):
    def fit(self, Xp, Y, times=None, Xf=None): pass
    def predict(self, Xp, times=None, Xf=None): return np.zeros((len(Xp), 3), np.float32)
    @property
    def name(self): return "const"

def test_pipeline_runs_end_to_end():
    cfg = PipelineConfig(dataset="toy", target="pv_power_normalized", lookback=6, horizon=3)
    res = SolarForecastPipeline(cfg, EndogenousFeatureEngineer(target="pv_power_normalized",
        feature_cols=["ghi_clearsky","cos_zenith","clearness_index"]), _Const()).run(df_provider=lambda c: _toy_df())
    assert res["pred"].shape[0] == res["y_true"].shape[0] and res["pred"].shape[1] == 3
```

- [ ] **Step 2: Run → fails** (`ModuleNotFoundError: solarunibench.forecast.core`).
- [ ] **Step 3: Implement** — ABCs + dataclass `PipelineConfig`; `EndogenousFeatureEngineer.create_features` (groupby entity, sort by time, sliding window; X_past stacks `[target]+feature_cols`; X_future stacks `[ghi_clearsky, cos_zenith]` over horizon if present else None); `SolarForecastPipeline.run(df_provider=None)` (uses `SolarUniBenchLoader` if no provider — provider hook enables synthetic tests; loads train+test, engineers both, fits, predicts test).
- [ ] **Step 4: Run → pass** (2 tests).
- [ ] **Step 5: Commit** — `feat(forecast): core ABCs + EndogenousFeatureEngineer + pipeline`

---

### Task 2: `forecast/metrics.py` — MAE/RMSE/MAPE/Skill/pinball/CRPS

**Files:** Create `solarunibench/forecast/metrics.py`; Test `tests/test_forecast_metrics.py`.

- [ ] **Step 1: Failing test**:

```python
import numpy as np
from solarunibench.forecast.metrics import mae, rmse, mape, skill_vs_reference, pinball_loss, crps

def test_point_metrics():
    y = np.array([[1.0, 2.0],[3.0,4.0]]); p = np.array([[1.5,2.0],[2.0,5.0]])
    assert abs(mae(y,p) - 0.625) < 1e-6
    assert rmse(y,p) > 0

def test_skill_zero_when_equal():
    y = np.array([[1.0,2.0]]); p = np.array([[1.0,2.0]])
    assert abs(skill_vs_reference(y, p, p)) < 1e-6      # same preds -> 0 skill

def test_pinball_crps_shapes():
    y = np.array([[1.0,2.0]]); q = np.array([[0.9,1.8],[1.1,2.2]])  # 2 quantiles
    assert pinball_loss(y, q, quantiles=[0.1,0.9]) >= 0
    assert crps(y, q, quantiles=[0.1,0.9]) >= 0
```

- [ ] **Step 2:** Run → fails.
- [ ] **Step 3: Implement** — `mae/rmse/mape` (NaN-safe via masks; MAPE guards zero denom); `skill_vs_reference(y, pred, ref) = 1 - mae(y,pred)/mae(y,ref)`; `pinball_loss(y, quantile_pred, quantiles)`; `crps(y, quantile_pred, quantiles)` (numerical-integral form over quantile pairs).
- [ ] **Step 4:** Run → pass (3 tests).
- [ ] **Step 5:** Commit — `feat(metrics): MAE/RMSE/MAPE/Skill/pinball/CRPS`

---

### Task 3: `forecast/models/smart_persistence.py` — physical skill reference

**Files:** Create `solarunibench/forecast/models/__init__.py`, `smart_persistence.py`; Test `tests/test_smart_persistence.py`.

**Interface:** `SmartPersistenceModel(ForecastModel)` — uses X_future's `ghi_clearsky` column + X_past's last target step. κ = target_now / max(csky_now, ε); pred = clip(κ × csky_future, 0, capacity or ∞). Target-agnostic (PV or GHI). `name="SmartPersistence"`.

- [ ] **Step 1: Failing test** — synthetic: target tracks clearsky scaled by κ=0.5 → recovered κ ≈ 0.5, pred ≈ 0.5·csky_future.
- [ ] **Step 2:** Run → fails.
- [ ] **Step 3: Implement** — `fit` estimates a stable κ (median of target/clearsky over daylight, clearness>threshold) OR per-sample from X_past last step; `predict` uses X_future clearsky × κ. Physical clip ≥0.
- [ ] **Step 4:** Run → pass.
- [ ] **Step 5:** Commit — `feat(baseline): Smart Persistence (clear-sky skill reference)`

---

### Task 4: `forecast/models/lightgbm_model.py`

**Files:** Create `lightgbm_model.py`; Test `tests/test_lightgbm.py`.

**Interface:** `LightGBMModel(params=None)` (ForecastModel). Flattens X_past (+X_future) to `(n, lb*Fp + horizon*Ff)`; fits one LGBMRegressor per horizon step (multi-output via per-step models); `predict` stacks.

- [ ] **Step 1: Failing test** — synthetic linear-y trend → LightGBM MAE < persistence MAE.
- [ ] **Step 2:** Run → fails.
- [ ] **Step 3: Implement** — `_flatten(X_past, X_future)`; fit list of `lightgbm.LGBMRegressor(**params)`; predict → `(n, horizon)`.
- [ ] **Step 4:** Run → pass.
- [ ] **Step 5:** Commit — `feat(baseline): LightGBM (tabular, per-horizon)`

---

### Task 5: `forecast/models/mlp_model.py` — configurable

**Files:** Create `mlp_model.py`; Test `tests/test_mlp.py`.

**Interface:** `MLPModel(hidden_layer_sizes=(128,64), activation="relu", preset=None, **kw)`; `preset="skippd_paper"` → `hidden_layer_sizes=(7,), activation="logistic"` (Nie et al. 2020). Wraps `sklearn.neural_network.MLPRegressor`.

- [ ] **Step 1: Failing test** — default config trains; `preset="skippd_paper"` yields (7,)+logistic; MAE finite on synthetic.
- [ ] **Step 2:** Run → fails.
- [ ] **Step 3: Implement** — flatten like LightGBM; single MLPRegressor (multi-output) `predict` → `(n, horizon)`; preset override in `__init__`.
- [ ] **Step 4:** Run → pass.
- [ ] **Step 5:** Commit — `feat(baseline): configurable MLP (+ skippd_paper preset)`

---

### Task 6: Integration — run all 3 baselines on a standardized dataset (RUN_SLOW)

**Files:** `tests/test_forecast_integration.py` (RUN_SLOW); optional `scripts/run_baseline.py`.

- [ ] **Step 1:** RUN_SLOW test on a small real standardized package (e.g., GEFCom train/test via loader): build `EndogenousFeatureEngineer`, run SmartPersistence + LightGBM + MLP through `SolarForecastPipeline`, assert finite MAE/RMSE + compute `skill_vs_reference` (vs SmartPersistence) for LightGBM/MLP. Print the 3×skill table.
- [ ] **Step 2:** `RUN_SLOW=1 pytest tests/test_forecast_integration.py -v` → pass; report MAE/skill.
- [ ] **Step 3:** Commit — `test(forecast): integration — 3 baselines on GEFCom + skill table`.

---

## Definition of Done
- `forecast/core.py` (ABCs + pipeline consuming Phase 5 loader), `metrics.py` (6 metrics), 3 baselines (Smart Persistence, LightGBM, configurable MLP).
- Fast unit tests green; RUN_SLOW integration runs all 3 on a real standardized dataset with a skill-vs-Smart-Persistence table.
- Framework is extensible (drop-in `ForecastModel`) for 6b (LSTM/PatchTST/CrossViViT).

## Notes / out of scope (Phase 6b)
- LSTM, PatchTST-style Transformer, CrossViViT.
- Optical-flow forward-splat revision (plan committed; needed by CrossViViT + E4).
- Folsom nearest-neighbor image-attach + mixed-dataset resize.
- Probabilistic (quantile) heads beyond the metric definitions.
