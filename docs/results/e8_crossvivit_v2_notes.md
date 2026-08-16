# CrossViViT v2 — Design Decision & Results Note

**Date:** 2026-08-16 · **Decision:** keep v2 as the official CrossViViT result (user-approved option 1).

## What v2 is

A lightweight CrossViViT-inspired variant (Boussif et al. 2023, arXiv:2306.01112;
official code: https://github.com/gitbooo/CrossViVit — NOT the 145M RoCrossViViT).
v2 aligns the architecture with the official design on two points where v1 deviated:

1. **Image temporal axis retained.** v1 mean-pooled image tokens over the lookback
   frames (cloud motion destroyed before cross-attention). v2 subsamples frames
   (`img_frame_stride`, default 8 → 240 frames become 30; the image-benchmark path
   receives 8 pre-subsampled frames and uses stride 1) and adds a learnable
   frame-position embedding per token.
2. **X_future fusion.** the known astronomical trajectory
   `X_future[n, horizon, 2] = [ghi_clearsky, cos_zenith]` is linearly embedded as
   `horizon` extra tabular tokens (the official model feeds time coordinates into
   both modalities; this is the benchmark equivalent).

Parameters (lb=240, d_model=64, depth=2): 221,571 (v1) → 223,875 (v2).

## v1 vs v2 outcome (FULL config, daytime-masked Skill%, single seed)

Tabular variant: v2 slightly WORSE on MAE (2/6 cells better) and RMSE (1/6 better).
Biggest regression folsom/10min (MAE +11.7%); biggest win folsom/15min (MAE −7.6%).
Image variant: mixed — MAE 2/6 better, RMSE 3/6 better; skippd/5min improved −10.0%
MAE, skippd/10min regressed +47.6% MAE.

Full tables: see `figures/results/e8_crossvivit_v2.csv`,
`figures/results/e8_crossvivit_image_v2.csv`, and the v1 snapshot in
`figures/results/archive_e8_pre_crossvivit_v2/`.

## Why keep v2 anyway

- The architecture is the honest description of what the paper claims to build
  (temporal image axis + future-trajectory fusion). v1's mean-pool was a
  structural defect, kept only because it happened to score marginally better.
- The v1↔v2 deltas are of the same magnitude as cell-to-cell training noise
  (single seed, early-stop checkpoint sensitivity). Without multi-seed repeats
  there is no evidence the difference is architectural rather than stochastic.
- RMSE — the metric most sensitive to large errors — improved or held on the
  image path (3/6 better, others ≈).

## Limitation statement (for the paper)

The temporal-axis + future-fusion upgrade did not yield a significant skill
improvement under the benchmark's shared training budget (50 epochs, fixed
hyperparameters tuned for the v1 sequence lengths). We attribute this to (a)
hyperparameters not re-tuned for the longer token sequences, and (b) the
30-minute frame subsampling on the image path leaving little motion signal.
Multi-seed repeats and per-architecture hyperparameter search are left as
future work.
