# Phase 8 — Full Batch + Packaging (final) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** Take the benchmark from "dev-config, bounded-image, validated" to **publication-ready**: full image materialization, full-config benchmark numbers (the publishable E8/E9), and ESSD packaging (manifest/audit, README, figure-ready selection).

**State entering Phase 8 (master `abe0508`):** Phases 1–7 done. Framework + 6 baselines + optical-flow + multimodal all merged. Experiments E1–E9 + viz system merged (dev-config, bounded 200-frame images). E9 thesis demonstrated on synthetic. **All dev numbers are honest; Phase 8 produces the publishable full-config numbers + full data.**

**Three work streams** (T1 compute-heavy, T2 needs T1, T3 safe to start now):

## Stream A — Full image materialization (compute-heavy, background)
- **Folsom**: rebuild sky zarr UNBOUNDED — decode all 775,917 frames from the 49 GB tar → JPEG-q90 RGB @256 → ~10 GiB. (`scripts/rebuild_sky_zarr.py --dataset folsom --limit 0`; runs as a long background job.)
- **SKIPPD**: full uint8@64 (~5.4 GiB) — unbounded.
- **SunLake**: satellite already full-ish (bounded context); confirm/extend if needed.
- ⚠️ **Output/cache → D:** (C: was full earlier; `--out D:/...`, pip/tmp on D:). Resume-safe (builders are `image_limit`-bound; lift to 0/None).
- **Needs your GO** to launch (multi-hour CPU). Staged in `scripts/` ready to fire.

## Stream B — Full-config benchmark (GPU/compute-heavy; needs Stream A)
- Re-run **E8 + E9 (and E1–E9)** at FULL model configs on FULL data, all datasets × 3 horizons → the publishable skill table + the E9 crossover on real data (the ESSD headline).
- Deep models (LSTM/PatchTST) full epochs; **CrossViViT 145M training = a real GPU-budget decision (spec §15 deferred it)** — your call: (a) run full 145M in Phase 8, or (b) keep the small variant + document full training as future work.
- Outputs replace the dev figures/CSVs with the publication set.

## Stream C — Packaging (safe, start now)
- `manifest.json` / `audit.jsonl` finalization per dataset (row/frame counts, build provenance, splits).
- **README** (ESSD framing: the benchmark, the 4 datasets, the governance>complexity thesis (E9), how to reproduce, the honest dev-vs-full-config distinction).
- **Figure-ready selection**: pick the ~6-8 paper figures from the E1–E9 set (E9 governance = Fig. 1; E1 geo, E8 benchmark heatmap, E7 defects, E4 optical-flow, E6 alignment as supporting); export print-quality (PDF + 300dpi PNG) via `viz/io`.
- `.gitignore` hygiene (figures/, data/standardized/, .superpowers/, .claude/); handle the loose working-tree files (方法论文.docx etc. — user decides).
- Reproducibility: a single `make benchmark` / `run_experiments.py --full` entrypoint doc.

## Suggested task order (SDD)
- **8-T1 (Stream C, safe now):** manifest/audit finalization + README skeleton + figure-selection script + .gitignore hygiene.
- **8-T2 (Stream A, YOUR GO):** full Folsom + SKIPPD materialization (background jobs) → verify per-dataset sizes + a smoke image-attach.
- **8-T3 (Stream B, needs T2 + your CrossViViT call):** full-config E8 + E9 (real data) → publishable tables/figures; E9 crossover on real data.
- **8-T4:** final whole-repo test pass + release tag + ESSD figure bundle.

## Definition of Done
- Full image packages materialized (Folsom ~10 GiB JPEG, SKIPPD ~5.4 GiB uint8) + verified.
- Publishable full-config E8 (baselines×datasets×horizons) + E9 (governance crossover on REAL data) figures/CSVs.
- README + manifest/audit + selected paper figures (print-quality) committed; reproducible single-command run.
- Final test suite green; release tagged.

## Notes / decisions pending (YOU)
1. **Launch full Folsom materialization now?** (multi-hour CPU background job; output on D:)
2. **CrossViViT 145M full training in Phase 8, or small-variant + document?** (GPU-budget)
3. Loose working-tree files (方法论文.docx/.txt, 数据集说明.txt, extensions.py) — keep / gitignore / move?
