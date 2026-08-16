# Solar-UniBench — All-Experiments Results Summary

> **One-page canonical digest of every experiment's results** (E1–E9 + physics
> validation + BRR capacity ablation). Generated 2026-08-16 from the committed
> result CSVs; every number below traces to the file cited in its section.
>
> Config: E8 = physics-time-aligned grid, FULL training (50 epochs, AdamW+
> Cosine, early stopping), daylight-masked Skill%. Ablations (E1–E7, E9) =
> `SUB_FULL_CONFIG=1` full-capacity presets. E9/E2/E6 re-verified on the merged
> code tree (E9/E2 bit-identical to the pre-merge archive; E6 re-based on the
> fully materialized 775,845-frame Folsom imagery).
> Skill convention: `1 − metric_model / metric_SmartPersistence`; 0 = parity,
> positive = better than persistence. Negative results are reported faithfully.

---

## E8 — Unified benchmark (physics-time-aligned, CrossViViT v2) ⭐

**File:** `figures/results/e8_summary.csv` · **Figures:** `fig_e8_skill_heatmap.png`,
`fig_e8_horizon_lines.png` (paper: fig4/fig4b)

9 models × 4 datasets × 3 native horizons. Fixed F=4 input
`[target_lag, cos_zenith, ghi_clearsky, clearness_index_aligned]` + known
future `X_future=[ghi_clearsky, cos_zenith]`. DLinear excluded from the
displayed set (its −590%…−382% on Folsom compresses shared scales; full column
remains in the CSV).

### Skill_MAE% (per horizon)

| Model | Folsom 5/10/15min | SKIPPD 5/10/15min | SunLake 30/60/90min | GEFCom 1/2/3h |
|---|---|---|---|---|
| LightGBM | −9.5 / −6.1 / −3.3 | −4.6 / +9.3 / +18.4 | +30.8 / +26.0 / +24.0 | +31.7 / +29.2 / +30.8 |
| MLP | −93.8 / −48.2 / −45.0 | −200.3 / −122.4 / −51.9 | +22.9 / +26.6 / +22.0 | +27.6 / +27.2 / +25.4 |
| LSTM | −10.4 / +1.9 / +5.9 | −12.7 / +1.8 / +17.0 | **+32.1** / +28.1 / +23.2 | **+33.9** / +23.8 / +22.6 |
| iTransformer | −106.3 / −25.4 / −20.9 | −60.6 / −33.4 / −63.2 | +29.9 / +25.5 / +23.6 | +29.0 / +23.5 / +27.6 |
| PatchTST-CI | −112.6 / −46.9 / −45.4 | −272.0 / −161.0 / −100.4 | +15.3 / +12.8 / +5.7 | +28.8 / +14.1 / +21.2 |
| PatchTST-CD | −21.4 / −10.0 / +0.1 | −26.6 / +0.8 / +13.2 | +30.2 / +26.2 / +24.7 | +33.1 / +26.8 / +25.3 |
| CrossViViT v2 | −13.8 / −14.9 / −3.9 | −12.0 / +8.1 / **+19.9** | n/a | n/a |

### Skill_RMSE% (per horizon)

| Model | Folsom 5/10/15min | SKIPPD 5/10/15min | SunLake 30/60/90min | GEFCom 1/2/3h |
|---|---|---|---|---|
| LightGBM | **+8.5 / +10.7 / +12.1** | **+3.9 / +9.1 / +12.0** | **+18.9 / +18.2 / +18.6** | **+27.4 / +28.2 / +31.7** |
| MLP | −8.3 / +4.2 / +6.3 | −43.3 / −31.5 / −6.6 | +17.0 / +17.4 / +18.6 | +25.6 / +26.0 / +29.7 |
| LSTM | +5.5 / +7.6 / +9.7 | −10.6 / −1.4 / +5.1 | +17.4 / +16.5 / +14.8 | +24.4 / +22.4 / +25.8 |
| iTransformer | −10.7 / +5.9 / +6.4 | −16.8 / −10.0 / −24.1 | +16.6 / +14.8 / +15.5 | +23.0 / +20.5 / +24.0 |
| PatchTST-CI | −19.8 / −1.6 / −1.7 | −95.9 / −62.1 / −41.7 | +9.0 / +10.7 / +6.3 | +20.2 / +13.2 / +21.4 |
| PatchTST-CD | +4.5 / +7.5 / +9.2 | −2.3 / +5.1 / +9.2 | +16.4 / +15.0 / +16.0 | +24.3 / +24.1 / +24.0 |
| CrossViViT v2 | +4.0 / +6.7 / +9.5 | −0.4 / +4.1 / +9.4 | n/a | n/a |

### Absolute daytime MAE / RMSE (normalized units, per horizon)

| Model | Folsom MAE | SKIPPD MAE | SunLake MAE | GEFCom MAE |
|---|---|---|---|---|
| LightGBM | .014/.017/.019 | .015/.019/.022 | .032/.036/.039 | .072/.084/.090 |
| MLP | .025/.024/.027 | .044/.047/.041 | .036/.036/.040 | .077/.086/.097 |
| LSTM | .014/.016/**.017** | .017/.021/.022 | **.031**/.035/.039 | **.070**/.090/.100 |
| iTransformer | .026/.020/.022 | .024/.028/.044 | .032/.037/.039 | .075/.091/.094 |
| PatchTST-CI | .027/.024/.027 | .055/.055/.054 | .039/.043/.048 | .076/.102/.102 |
| PatchTST-CD | .015/.018/.019 | .019/.021/.023 | .032/.036/.039 | .071/.087/.097 |
| CrossViViT v2 | .014/.019/.019 | **.017/.020/.021** | n/a | n/a |

| Model | Folsom RMSE | SKIPPD RMSE | SunLake RMSE | GEFCom RMSE |
|---|---|---|---|---|
| LightGBM | **.036/.042/.044** | **.041/.048/.052** | **.052/.058/.061** | **.107/.123/.132** |
| MLP | .043/.045/.047 | .061/.069/.063 | .053/.058/.061 | .109/.127/.136 |
| LSTM | .037/.043/.046 | .047/.053/.056 | .053/.059/.064 | .111/.133/.143 |
| iTransformer | .044/.044/.047 | .050/.058/.074 | .054/.060/.064 | .113/.136/.147 |
| PatchTST-CI | .047/.048/.051 | .084/.085/.084 | .059/.063/.071 | .117/.149/.152 |
| PatchTST-CD | .038/.043/.046 | .044/.050/.054 | .054/.060/.063 | .111/.130/.147 |
| CrossViViT v2 | .038/.044/.046 | .043/.050/.054 | n/a | n/a |

SmartPersistence reference MAE: Folsom .013/.016/.019 · SKIPPD .015/.021/.027
· SunLake .046/.049/.051 · GEFCom .106/.119/.130 (RMSE values in the CSV).

**Key findings.** (1) 1-min data at 5-min lead is a persistence regime on MAE
(all negative) but models already win on RMSE — they trim the error tail
first. (2) At coarser cadences models dominate: LSTM best-MAE on
SunLake/GEFCom; LightGBM best-RMSE on every dataset. (3) CrossViViT v2 is the
best SKIPPD MAE model at 15 min (+19.9). (4) DLinear worst everywhere
(excluded from display, retained in CSV).

**CrossViViT image add-on** (`e8_crossvivit_image_v2.csv`, image-covered
windows): folsom Skill_MAE −26.3/−10.2/−7.8, skippd −29.3/−70.3/−1.7 — the
8-frame subsampled imagery does not beat the tabular path; honest negative.

---

## E9 — Governance vs complexity (the centerpiece) ⭐

**Files:** `results_e9_governance.csv` (dev grid, GEFCom) ·
`results_e9_cross_dataset.csv` (4-dataset physics-aligned 1h-ahead) · paper fig1/fig8

### Cross-dataset crossover (12 h lookback, 1 h-ahead, uniform 3-yr sampling)

**Crossover = LightGBM-on-cleaned ≥ PatchTST-on-raw. Holds on 4/4 datasets.**

| Dataset | LightGBM cleaned skill | PatchTST raw skill | Margin | Holds |
|---|---|---|---|---|
| gefcom2014 | +0.581 | +0.568 | **+0.012** | ✓ |
| folsom | +0.678 | +0.582 | **+0.097** | ✓ |
| skippd | +0.886 | +0.874 | **+0.013** | ✓ |
| sunlake | +0.625 | +0.426 | **+0.198** | ✓ |

Cleaned-skill ladder (all 7 models × 4 datasets in the CSV): LightGBM tops
folsom/skippd/gefcom; MLP tops gefcom at 0.608; PatchTST and iTransformer
mid-pack everywhere.

### Dev grid (GEFCom, 5-model ladder)

| Model | skill raw | skill cleaned | Δ-hygiene |
|---|---|---|---|
| LightGBM | −0.053 | −0.069 | −0.016 |
| MLP | +0.207 | **+0.221** | +0.014 |
| LSTM | +0.201 | +0.112 | −0.089 |
| PatchTST | +0.054 | +0.172 | **+0.118** |

MLP+cleaned (+0.221) beats PatchTST+raw (+0.054): the crossover. Δ-hygiene
grows with complexity (PatchTST +0.118 largest). LSTM is the honest exception
(cleaned slightly worse — disclosed).

### Governance waterfall (Folsom, LightGBM probe)

`raw 0.0532 → +night 0.0542 → +clamp 0.0542 → +UTC 0.0542 → +clearsky
features 0.0462 (−13.1%)` — LightGBM probe (the crossover criterion's
"simplest non-trivial" model; supersedes the DLinear probe, same ordering at
−2.0%). Label governance is a near-no-op on already-clean Folsom; **the
clear-sky physical anchors carry the governance gain**.

### Temporal granularity (Folsom 1 min → 1 h)

SP physical MAE rises 47.1 → 52.1 W/m² (nMAE 0.035 → 0.046) as cadence
coarsens — SP-collapse confirmed in physical units. The LightGBM probe
(swapped for DLinear 2026-08; the earlier DLinear probe scored −3.6…−5.8)
also fails to exploit the collapse on this bounded grid (skill −6.1 at
1 min → −4.3 at 1 h, MAE 279–336 W/m²): the grid uses a fixed 12-row
lookback at every granularity, so coarse cadences starve it of windows.
Honest setup limitation, disclosed.

---

## E1 — Geo-recovery

**File:** `results_e1_geo.csv` (+ `results_folsom_blind_inversion.csv`,
`results_geo_ablation_phase_shift.csv`) · paper fig3/fig3b

| Section | Result |
|---|---|
| Blind test (Folsom, known coords) | **420.7 km** great-circle error — sub-regional scale |
| TSR validation (GEFCom, 3 zones) | recovers Australia (lat ≈ −34, lon 150); R² 0.82/0.81/0.81 |
| Downstream skill | geo features (clear-sky anchors) change daytime MAE 0.0987 → 0.1047 (−0.006; honest small/negative at 1 h where autoregressive lags dominate) |
| Folsom blind inversion (3° grid) | recovers (−36, 66) vs truth (38.7, −121.2) = **19,316 km** — the hemisphere ambiguity of clear-sky fitting on cloudy GHI, honestly disclosed |
| Phase-shift ablation (±60 min) | recovered longitude drifts at the correct ~15°/h solar rate; absolute location biased (cloud noise); downstream skill flat (autoregressive robustness) |

---

## E2 — Clear-sky anchor features

**File:** `results_e2_clearsky.csv` · figure `fig_clearsky_skill.png`

LightGBM ablation, skill_gain = clearsky-arm skill − baseline-arm skill:

| Dataset | baseline skill | +clearsky skill | skill_gain |
|---|---|---|---|
| gefcom2014 | +0.205 | +0.137 | **−0.068** |
| skippd | −0.260 | −0.226 | **+0.034** |
| folsom | −0.248 | −0.263 | **−0.015** |
| sunlake | +0.236 | +0.286 | **+0.050** |

Mixed/honest: anchors help on sunlake (+0.05) and skippd (+0.034), are neutral
on folsom, and hurt on gefcom (autoregressive lags already carry the signal at
the tested horizon).

---

## E3 — BRR thin-cloud channel

**File:** `results_e3_brr.csv` (+ `results_brr_capacity_ablation.csv`) · paper fig12/fig13

Scalar BRR vs RGB-only (TINY LightGBM, 1 h):

| Dataset / subset | RGB skill | BRR skill | verdict |
|---|---|---|---|
| folsom overall | −0.647 | −0.548 | BRR better |
| folsom thin-cloud | −1.147 | −1.141 | ≈ neutral |
| skippd overall | +0.320 | +0.311 | ≈ neutral |
| skippd thin-cloud | +0.123 | +0.173 | BRR better |

**4-arm capacity ablation** (representation × capacity; ResNet-18 ImageNet +
Channel Padding Alignment): (i) capacity dominates — deep arms cut Folsom MAE
~120 → ~92–96 W/m²; (ii) scalar BRR is not robustly helpful (hurts on Folsom
thin-cloud: −1.91 vs −1.72); (iii) **Deep BRR > Deep RGB on Folsom thin-cloud,
Δskill +0.097** (CPA flips the sign); SKIPPD −0.013 (64² resolution, n=35
subset too small).

---

## E4 — Optical-flow extrapolation

**File:** `results_e4_optflow.csv` · figures `fig_optflow_*`, `fig_uncertainty_reliability.png`

Forward-splat extrapolation quality decays monotonically with lead (SunLake
satellite): PSNR **30.6 dB** / SSIM 0.91 at k=1 → 19.7 dB / 0.57 at k=12.
Uncertainty ECE ≈ 0.0585 (slightly over-confident, honestly reported).

| lead k | 1 | 3 | 6 | 9 | 12 |
|---|---|---|---|---|---|
| PSNR (dB) | 30.6 | 25.9 | 22.5 | 20.8 | 19.7 |
| SSIM | 0.91 | 0.79 | 0.68 | 0.61 | 0.57 |

---

## E5 — Quality audit

**File:** `results_e5_quality.csv` · figures `fig_quality_flags.png`,
`fig_cleaning_delta.png`

Quality-flag distribution across datasets (0 = clean pass-through, 3 =
governed rows): gefcom 49.1% clean / 50.9% governed · folsom 50.3/49.7 ·
skippd **96.1/3.9** (cleanest source) · sunlake 51.1/48.9. The raw-vs-cleaned
downstream value is quantified by E9's hygiene axis (see above). Physics
closure (validation): Folsom GHI≈DNI·cos(z)+DHI **R²=0.9998**, SunLake
**R²=0.9398** (`fig_physics_closure_parity.png`, paper fig11).

---

## E6 — Cross-modal alignment

**File:** `results_e6_alignment.csv` · figures `fig_align_*` · paper fig7

Full Folsom imagery basis (775,845 frames): nearest-neighbor hit rate 98.6%,
median image↔scalar lag **1.0 s**. Downstream 1 h LightGBM:

| Arm | Daytime MAE | Skill |
|---|---|---|
| unaligned | 139.2 | −0.398 |
| **aligned** | **101.1** | **−0.212** |

Alignment halves the skill deficit (**skill_gain +0.186**); conclusion
re-verified on the merged tree with 3× more imagery than the first pass.

---

## E7 — Data-defect diagnostics

**File:** `defect_summary.csv` (+ `fig_defect_*`) · paper fig5

Descriptive only (no model): missingness heatmap across datasets, timestamp
irregularity (cadence gaps), outlier distributions vs physical limits. Feeds
the ESSD "data quality" section; SKIPPD is the cleanest source (E5 flag
distribution above), Folsom/SunLake/GEFCom carry ~50% governed rows.

---

## Superseded records

- `results_benchmark.csv` + `fig_benchmark_heatmap.png` (figures/ and
  figures_full/) — the agent-era fixed-step E8 grid (lookback=12, horizons
  1/6/24 h, CrossViViT v1). Replaced by the physics-time-aligned benchmark
  above; kept as a historical record (see `figures/results/README.md`).
- `results_e9_cross_dataset_legacy_fixedstep.csv` — the pre-cadence-fix E9
  first pass, kept for the methodological record (PHASE3_NOTES.md).
- CrossViViT v1 E8 rows — archived in
  `figures/results/archive_e8_pre_crossvivit_v2/` (decision + limitation
  statement in `e8_crossvivit_v2_notes.md`).
