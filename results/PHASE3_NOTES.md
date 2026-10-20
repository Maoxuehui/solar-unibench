# Phase 3 Results — Honest Record (with caveats)

> Snapshot of the FIRST Phase-3 pass, saved for provenance. A known methodological
> flaw (below) is being fixed; this record documents what was found and why the
> cross-dataset crossover needs the cadence-alignment refinement.

## Item 7 — E9 cross-dataset crossover (raw vs cleaned × 7-model ladder)
Results in `results_e9_cross_dataset.csv`. Crossover = simplest non-trivial model
(LightGBM) on CLEANED vs most-complex (PatchTST) on RAW.

| dataset | cadence | margin | holds | honest caveat |
|---|---|---|---|---|
| gefcom2014 | hourly | −0.534 | ✗ | LightGBM too weak as the "simple" model |
| folsom | 1-min | +0.822 | ✓ | **all models skill −3..−5 (degenerate regime)** |
| skippd | 1-min | +3.603 | ✓ | **all models skill −1..−9 (degenerate regime)** |
| sunlake | 30-min | +0.812 | ✓ | reasonable |

### ⚠️ Known methodological flaw (being fixed)
E9's grid uses FIXED row params (lookback=12, horizon=1). Per dataset cadence
this means a DIFFERENT physical forecast horizon:
- hourly GEFCom → 1 h ahead (meaningful; persistence beatable)
- 1-min Folsom/SKIPPD → **1 minute ahead** (persistence near-perfect → every
  model loses badly; skill −3..−9)

So the Folsom/SKIPPD "crossover holds" is a **relative ordering inside a regime
where all models catastrophically lose to persistence** — a cadence artifact, not
a governance win. **Fix in progress: cadence-aligned horizon** (1-hour-ahead for
all datasets → horizon=1 row hourly, =60 rows at 1-min) so the comparison is fair.

Also: raw snapshots are bounded first-N rows (Folsom/SKIPPD hit a hard early
regime); a representative slice is a secondary improvement.

## Item 8 — Governance waterfall (Folsom, DLinear) — `results_e9_waterfall.csv`
`raw 0.0250 → +night 0.0250 → +clamp 0.0250 → +utc 0.0250 → +clearsky_features 0.0081 (−68%)`
Label-governance operators are near-no-ops on Folsom's already-clean raw; the
**clearsky physical anchors are the dominant governance contributor**. Honest.

## Item 9 — Temporal granularity (Folsom 1min→1h) — `results_e9_granularity.csv`
Hypothesis "SP skill collapses at coarse cadence, models win more" **REFUTED**:
DLinear skill vs SP is highest at fine cadence (+0.55 @1min), lowest at coarse
(−2.15 @1h); SP MAE ~flat across granularities. At fine cadence the model
captures minute-scale dynamics; at coarse cadence data scarcity hurts it.
Caveat: MAEs are per-granularity scaled (physical-W/m² comparison is a TODO).

## Constraints honored
Honest NaN (no fabrication), zero-leakage scaling (train-only), CUDA-cache
clearing between fits. All scripts committed: `scripts/e9_cross_dataset.py`,
`scripts/e9_waterfall.py`, `scripts/e9_granularity.py`.
