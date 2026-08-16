# Experiment Window Grids — Design Rationale

**Status:** user-approved design (option 1, 2026-08-16) · applies to all post-merge experiments.

Solar-UniBench deliberately runs TWO window grids. They answer different
questions; neither is a bug.

## Grid A — E8 unified benchmark (per-dataset native horizons)

Defined in `solarunibench/benchmark/config.py::DATASET_CONFIG`. Each dataset is
evaluated at the forecast lead times its operational community actually uses,
with a lookback matched to its physical memory scale:

| dataset | sampling | lookback | horizons (steps) | horizons (physical) | rationale |
|---|---|---|---|---|---|
| folsom | 1 min | 240 (4 h) | 5/10/15 | 5–15 min | sky-camera nowcasting; 4 h captures cloud-statistical inertia without diluting attention |
| skippd | 1 min | 240 (4 h) | 5/10/15 | 5–15 min | same regime, PV target |
| sunlake | 30 min | 24 (12 h) | 1/2/3 | 30–90 min | satellite-GHI intra-day; 12 h daytime cover |
| gefcom2014 | 60 min | 24 (24 h) | 1/2/3 | 1–3 h | day-ahead PV; 24 h full diurnal cycle |

**Question answered:** "at each dataset's native lead times, which model wins?"

## Grid B — E9 governance ablation (fixed cross-dataset lead)

Defined in `scripts/e9_cross_dataset.py::ROW_PARAMS`. Every dataset forecasts the
SAME physical lead (1 h ahead, 12 h lookback), with row-counts scaled by cadence:

| dataset | lookback | horizon | rows (L, H, stride) |
|---|---|---|---|
| gefcom2014 | 12 h | 1 h | (12, 1, 1) |
| sunlake | 12 h | 1 h | (24, 2, 2) |
| folsom | 12 h | 1 h | (720, 60, 60) |
| skippd | 12 h | 1 h | (720, 60, 60) |

**Question answered:** "is governance > complexity universal, holding the task
fixed across datasets?"

## Why not one grid

- On Grid A, 1-min data at 5–15 min leads sits in the persistence-dominated
  regime (E8 result: nearly all models score negative MAE skill at 5 min). A
  governance crossover evaluated there would rank models inside a regime where
  every model loses — the exact cadence artifact diagnosed and fixed in
  `figures_full/results/PHASE3_NOTES.md` (commit `b8775a7`).
- On Grid B, the 1 h lead is chosen because it is meaningful at EVERY cadence:
  it is day-ahead-adjacent for hourly data and a genuine forecast (not
  persistence-clone) for 1-min data.

## Paper-facing statement

When reporting, state each experiment's grid in its setup table and cite this
note: E8 evaluates native-lead skill; E9 fixes the lead to isolate the
governance axis. The two grids are a feature of the benchmark (task-sensitivity
of the model ranking), not an inconsistency.

## Refresh provenance

Post-merge (`5d121d7`) refresh runs: E9 uses Grid B unchanged; E2/E6 do not
depend on the E8 window matrix (they carry their own ablation windows). Old
results archived under `figures_full/results/archive_pre_merge_refresh/`.
