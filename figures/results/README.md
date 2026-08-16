# Results Directory Guide

## Canonical E8 result (post-2026-08-16)

- `e8_summary.csv` — the unified-benchmark FULL grid (4 datasets × 4 metrics ×
  3 horizons × 9 models). CrossViViT rows on folsom/skippd are the **v2
  architecture** (image temporal axis + X_future fusion; see
  `docs/results/e8_crossvivit_v2_notes.md`).
- `e8_crossvivit_v2.csv` / `e8_crossvivit_image_v2.csv` — the v2 reruns
  (tabular main grid; image add-on).
- `e8_full_checkpoint*.csv` — per-cell checkpoints of the FULL-grid run
  (raw long format; `e8_summary.csv` is the wide view).
- `archive_e8_pre_crossvivit_v2/` — complete pre-v2 snapshot (v1 CrossViViT
  results + the same checkpoints), kept for the v1-vs-v2 comparison in the
  decision note.
- `final_table.txt` — pretty-printed v1-era skill table (superseded by
  `e8_summary.csv`; kept for the session record).

## Superseded: figures_full/results/results_benchmark.csv

The agent-era "E8" table (fixed lookback=12, horizons 1/6/24h, CrossViViT v1)
is **superseded** by `figures/results/e8_summary.csv` (physics-time-aligned
windows, F=4 unified inputs, daytime-masked Skill%, CrossViViT v2). The grids
differ by design (see `docs/results/experiment_grid_design.md`) so the numbers
are not comparable; the old file is retained as a historical record only.
