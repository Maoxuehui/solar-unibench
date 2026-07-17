# Benchmark Integrity — GEFCom Fix + Physical Scaling + Daytime Evaluator

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Fix the GEFCom target-data loss (Solution POWER dropped in dedup), add **global physical-constant feature scaling** (zero leakage), and a **daytime-window evaluator** (cos_zenith>0.05) so benchmark metrics are honest. Removes the MLP divergence artifact and aligns the eval line across all models.

**Architecture:**
- **GEFCom adapter:** rewrite POWER/NWP extraction — POWER = concat(train1..15) + Solution15 (dedup keep last); NWP = concat(predictors1..15) (dedup keep last, drop POWER col); outer-join on (ZONEID,TIMESTAMP). Verified: gives complete POWER 2012-04→2014-07.
- **PhysicalFeatureScaler** (new, in `forecast/`): `fit(df)` on TRAIN only — irradiance cols (`ghi_*`,`dni_*`,`dhi_*`) ÷ 1500.0 (solar-constant physical max, no fit); NWP cols (`nwp_*`) → StandardScaler (train μ/σ); cos_zenith/clearness_index/target untouched (already bounded). `transform(df)` applies. Wired into `SolarForecastPipeline` (fit on train, transform train+test BEFORE feature engineering) — global, not per-model.
- **Daytime evaluator:** `daytime_metrics(y_true, y_pred, cos_zenith, threshold=0.05)` in `metrics.py` (mask cos_zenith>threshold → Daytime_MAE/RMSE/skill). Pipeline passes the forecast-issue cos_zenith (from X_past last step) for test windows.

**Tech Stack:** pandas, numpy, sklearn StandardScaler, pvlib. Env `solar_uni`.

## Global Constraints
- **Zero data leakage:** physical-constant scaling needs no fit; NWP z-score fits μ/σ on TRAIN ONLY, applied to val/test. No global stats.
- **Preserve zero:** irradiance /1500 keeps nighttime 0.0 exactly (critical for daytime filter + night-zero ops).
- **GEFCom complete POWER:** after fix, every split (train/val/test) has real non-zero daytime POWER. Verify per-split.
- **Daytime eval is the headline metric** for solar (nighttime zeros would fraudulently deflate MAE/RMSE).
- **Env:** `conda run -n solar_uni ...`; pytest from repo root (or `PYTHONPATH=repo`). TDD, DRY, YAGNI; trailer `Co-Authored-By: Claude <noreply@anthropic.com>`.

**Branch:** `feat/benchmark-integrity` (from `master` @ `012d9ac`).

---

### Task 1: GEFCom adapter — clean POWER/NWP extraction + rebuild + verify

**Files:** Modify `solarunibench/pipeline/adapters/gefcom.py` (`load_and_standardize_raw`); Test `tests/test_gefcom_adapter.py` (strengthen); rebuild `data/standardized/gefcom2014`.

- [ ] **Step 1: Failing test** — synthetic 2-task GEFCom-like fixture where a forecast-period timestamp appears in `predictors2` (no POWER) AND `solution` (POWER); assert the standardized row KEEPS the Solution POWER (not NaN/0). Also assert train+val+test POWER has non-zero daytime max after a temporal split.
- [ ] **Step 2: RED.**
- [ ] **Step 3: Implement** — replace the per-task merge+concat+dedup-first with a **combine_first** patch (user-directed; Solution's non-null POWER overrides train NaN). PRESERVE the existing multi-zone (3 zones), 12-NWP-var, geo-recovery (TSR/clear-sky → Australia), correct-path, pre-normalized-POWER structure — do NOT collapse to 1 zone / drop NWP / hardcode LA coords:
```python
solar = Path(self.raw_data_dir) / "Solar"
# POWER series: all train{t}, dedup keep LAST (train15 superset). Then Solution combine_first
# so the 2014-06 blind-period real POWER overrides train's NaN rows (the bug fix).
pow_frames = [pd.read_csv(solar/f"Task {t}"/f"train{t}.csv") for t in TASKS
              if (solar/f"Task {t}"/f"train{t}.csv").exists()]
train_pow = (pd.concat(pow_frames, ignore_index=True)
             .drop_duplicates(subset=["ZONEID","TIMESTAMP"], keep="last")
             .set_index(["ZONEID","TIMESTAMP"]))
sol = solar/"Solution to Task 15"/"Solution to Task 15.csv"
if sol.exists():
    sol_df = pd.read_csv(sol).set_index(["ZONEID","TIMESTAMP"])
    power = sol_df.combine_first(train_pow).reset_index()   # Solution wins where non-null
else:
    power = train_pow.reset_index()
# NWP series: all predictors{t} (drop any POWER col), dedup keep LAST (predictors15 superset)
nwp_frames = []
for t in TASKS:
    p = solar/f"Task {t}"/f"predictors{t}.csv"
    if p.exists():
        pp = pd.read_csv(p)
        if "POWER" in pp.columns: pp = pp.drop(columns=["POWER"])
        nwp_frames.append(pp)
nwp = (pd.concat(nwp_frames, ignore_index=True)
       .drop_duplicates(subset=["ZONEID","TIMESTAMP"], keep="last"))
raw = power.merge(nwp, on=["ZONEID","TIMESTAMP"], how="outer")\
           .sort_values(["ZONEID","TIMESTAMP"]).reset_index(drop=True)
# Audit print (user-requested): confirm 2014-06 blind period POWER revived
_jun = raw[(raw["TIMESTAMP"]>="20140601 00:00") & (raw["TIMESTAMP"]<"20140701 00:00")]
print(f"[GEFCom fix audit] 2014-06 rows={len(_jun)} POWER non-null={_jun['POWER'].notna().sum()} "
      f"max={_jun['POWER'].max():.4f}")
```
Keep the rest (timestamp_utc, rename ZONEID→zone/POWER→pv_power_normalized, the full per-zone geo-recovery loop with TSR+clear-sky, NWPVAR→`nwp_VAR*` columns, meta/ts rows) UNCHANGED. GEFCom POWER is already [0,1] in source — store directly as `pv_power_normalized` (no re-normalization).
- [ ] **Step 4: GREEN**; then rebuild: `conda run -n solar_uni python scripts/build_dataset.py --dataset gefcom2014` + `scripts/build_splits.py` (or splits.build). Verify per-split POWER non-zero daytime (probe like the diagnostic): train/val/test each have POWER max > 0.5.
- [ ] **Step 5: Commit** — `fix(gefcom): clean POWER+NWP extraction (Solution no longer lost in dedup)`

---

### Task 2: PhysicalFeatureScaler + wire into SolarForecastPipeline

**Files:** Create `solarunibench/forecast/scaling.py`; Modify `solarunibench/forecast/core.py` (pipeline uses scaler); Test `tests/test_forecast_scaling.py`.

**Interfaces:** `PhysicalFeatureScaler(irradiance_max=1500.0)` with `.fit(df)` (train) / `.transform(df) -> df` / `.fit_transform(df)`. `SolarForecastPipeline(config, fe, model, scaler=None)` — if scaler given, fit on train df, transform train+test dfs before `create_features`.

- [ ] **Step 1: Failing test** — a df with `ghi_clearsky`∈[0,1100], `nwp_VAR78` (NWP), `cos_zenith`, `pv_power_normalized`. After fit_transform: ghi_clearsky max ≤ 1100/1500; nwp_VAR78 mean≈0 std≈1 (train stats); cos_zenith + target UNCHANGED. Leakage test: scaler fit on train, transform on a different test df uses TRAIN μ/σ (not test's).
- [ ] **Step 2: RED.**
- [ ] **Step 3: Implement** `PhysicalFeatureScaler`: irradiance regex cols (`^(ghi|dni|dhi)`) ÷ irradiance_max (constant); NWP cols (`^nwp_`) → per-col StandardScaler fit on train (store mean/std, NaN-safe); everything else passthrough. Wire into `SolarForecastPipeline.run` (fit on df_train, transform both before engineering). Default `scaler=PhysicalFeatureScaler()` so the pipeline scales by default (fixes MLP divergence) — but keep it injectable for ablation (raw vs scaled = an E-axis).
- [ ] **Step 4: GREEN**; run full forecast suite (no regressions; the existing tests pass a `df_provider` — ensure scaler default doesn't break them; if it does, the test df lacks irradiance cols → scaler is a no-op passthrough, fine).
- [ ] **Step 5: Commit** — `feat(forecast): PhysicalFeatureScaler (irradiance /1500 + NWP z-score train-fit), global in pipeline`

---

### Task 3: Daytime evaluator + wire into pipeline

**Files:** Modify `solarunibench/forecast/metrics.py` (add `daytime_metrics`); Modify `core.py` (pipeline computes daytime metrics for test); Test `tests/test_forecast_metrics.py`.

- [ ] **Step 1: Failing test** — `daytime_metrics(y, p, cos_zenith)` with half the samples nighttime (cos_zenith<0.05 → filtered). Assert only daytime samples counted; returns Daytime_MAE/RMSE; assert Daytime_MAE > all-sample MAE when nighttime residuals are ~0 (proves filtering removes the fraudulent zero-deflation).
- [ ] **Step 2: RED.**
- [ ] **Step 3: Implement** `daytime_metrics(y_true, y_pred, cos_zenith, threshold=0.05, reference=None)` → dict with `Daytime_MAE`, `Daytime_RMSE`, `Daytime_skill` (vs reference if given), `n_daytime`. NaN-safe. Wire pipeline: extract issue-time cos_zenith per test window (X_past[:, -1, <cos_zenith col idx>] — the FeatureEngineer must expose cos_zenith position; simplest: the pipeline reads it from the raw test df at the forecast-issue timestamps, OR EndogenousFeatureEngineer also returns a cos_zenith array). Choose: have `create_features` additionally return the issue-time `cos_zenith` array (extend the tuple) OR re-derive in the pipeline from the test df. Prefer extending the FeatureEngineer contract minimally: add a 5th return `aux` (dict with `cos_zenith_issue`) — optional, None if absent.
- [ ] **Step 4: GREEN**; full suite.
- [ ] **Step 5: Commit** — `feat(metrics): daytime-window evaluator (cos_zenith>0.05 anti-fraud filter)`

---

### Task 4: Re-run 6a integration on fixed GEFCom + scaled + daytime

**Files:** Update `tests/test_forecast_integration.py` (RUN_SLOW).

- [ ] **Step 1:** Re-run SmartPersistence + LightGBM + MLP via the pipeline (now with PhysicalFeatureScaler default + daytime eval) on the REBUILT GEFCom (real val/test with POWER). Assert: all 3 finite Daytime_MAE; **MLP no longer diverges** (Daytime_MAE < 0.5, not ~3.8); LightGBM daytime skill > 0 vs SmartPersistence. Print the daytime skill table.
- [ ] **Step 2:** `RUN_SLOW=1 pytest tests/test_forecast_integration.py -v` → pass. Report the new table.
- [ ] **Step 3:** Commit — `test(forecast): integration on fixed GEFCom + scaled + daytime (MLP no longer diverges)`.

---

## Definition of Done
- GEFCom standardized package: complete POWER 2012-04→2014-07; train/val/test each have non-zero daytime POWER.
- PhysicalFeatureScaler in pipeline (irradiance /1500, NWP train-fit z-score); zero leakage; preserves zero.
- Daytime evaluator (cos_zenith>0.05) is the headline metric.
- 3 baselines on fixed GEFCom: MLP converges (no divergence), LightGBM daytime skill > 0 vs Smart Persistence.

## Notes / out of scope
- Other datasets (Folsom/SKIPPD/SunLake) unchanged by GEFCom fix; scaler applies to all (benefits all).
- Optical-flow revision (Phase 6b), LSTM/PatchTST/CrossViViT, full-image materialization — separate.
