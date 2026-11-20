# Legacy bounded E9 (deprecated)

**Status: DEPRECATED — archived for historical traceability only. Do not run.**

This directory preserves the first-draft E9 governance experiments and their
outputs, superseded by the full-data contiguous protocol promoted to the main
pipeline (`scripts/experiments/e9_contiguous.py` +
`scripts/experiments/e10b_waterfall_contiguous.py`).

Why deprecated:

* Data regime — the raw arm was rebuilt then **uniform-row-subsampled**
  (sunlake→4 000 rows, gefcom2014→2 000, folsom/skippd→40 000), so sliding
  windows spanned non-contiguous timestamps (hours-to-days jumps). Absolute
  MAE/Skill therefore differ substantially from the E8 full-data benchmark
  (e.g. sunlake LightGBM MAE 0.075 here vs 0.036 in E8) and the two are not
  directly comparable.
* Splits — a bespoke 70/30 temporal cut, not the frozen `build_splits`
  membership E8 uses.
* Feature set — `(ghi_clearsky, cos_zenith)` only, without
  `clearness_index_aligned`; GEFCom lookback 12 h (E8 uses 24 h).
* No per-window contiguity check (windows silently crossed data holes).

Contents:

* `scripts/` — the archived runner scripts (`e9_cross_dataset.py`,
  `e9_waterfall.py`, `e10b_*` waterfall variants).
* `results/` — the corresponding result CSVs, including the CSV that fed the
  original Figure 8 (`results_e9_cross_dataset.csv`) and the original
  waterfall (`results_e10b_waterfall_met.csv`).

The official successors live in `scripts/experiments/e9_contiguous.py`
(48-cell matrix, chi margins, markdown summary) and
`scripts/experiments/e10b_waterfall_contiguous.py` (waterfall), with outputs
under `results/results_e9_contiguous*.csv` and
`results/results_e10b_waterfall_contiguous.csv`.
