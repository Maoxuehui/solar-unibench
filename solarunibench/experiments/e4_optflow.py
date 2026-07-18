"""E4 — Optical-flow extrapolation quality + uncertainty calibration (SunLake).

Spec §11 (E4). *Hypothesis: forward-splat-extrapolated future frames track the
real future within the predicted uncertainty envelope; quality decays with
lead-time K.*

The authoritative forward-splat operator
:func:`solarunibench.operators.optical_flow.extrapolate_with_uncertainty`
advances a context frame by ``k * flow`` and emits a per-pixel reliability map
(high = reliable, decaying with lead ``k`` and local flow divergence). SunLake
ships the operator's output precomputed in ``satellite_future.zarr`` as
``future`` (the extrapolation) and ``uncertainty_map`` (the reliability), both
keyed by consecutive context-pair index — so E4 scores that extrapolation
against the **real** future context frames ``context[t+1+K]``.

Three outputs, all routed through the shared :mod:`solarunibench.viz` system:

1. ``fig_optflow_sequence.png`` — context → extrapolated → truth → |error| →
   uncertainty (reliability) gallery for one representative pair/lead.
2. ``fig_optflow_psnr_decay.png`` — PSNR (dB) and SSIM of extrapolated vs real
   future across lead K = 1..steps_k (two panels, different scales).
3. ``fig_uncertainty_reliability.png`` — prediction-interval coverage vs nominal
   level (reliability diagram) + the predicted-vs-empirical reliability curve.

Reliability is scored **distribution-free**: pixels are split into a calibration
half (pairs) and an evaluation half. A per-reliability-bin error scale
``σ̂(u)`` is fit on the calibration half; the standardized residual quantile at
nominal level α gives the interval half-width, and empirical coverage is
measured on the evaluation half. Perfect calibration → the reliability curve
lies on the diagonal. The operator's heuristic reliability (a fixed exponential
lead decay ``exp(-αk)`` × divergence penalty) is expected to be a WEAK predictor
of the actual PSNR decay — E4 reports that honestly (an ECE far from zero is a
finding, not a failure).

Pure function contract: ``run(loader, out_dir) -> {figures, metrics}`` reads
only standardized packages via ``loader`` and writes to ``out_dir``; it never
mutates the loader or the standardized data.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..viz.io import save_fig, save_metrics
from ..viz.style import PALETTE, set_style

__all__ = ["run", "psnr", "ssim", "coverage_curve"]


# ---------------------------------------------------------------------------
# Bounded dev configuration (full-config numbers come at Phase 8).
# ---------------------------------------------------------------------------

#: SunLake is the only standardized package with both the precomputed optical-
#: flow extrapolation AND real future context frames to compare against
#: (SKIPPD has sky images but no optical-flow field). See spec §4.4.
_DATASET: str = "sunlake"

#: Lead-time horizon (spec §4.5 default K = 12).
_STEPS_K: int = 12

#: Bounded number of consecutive context pairs to evaluate (dev speed cap; the
#: full cube is evaluated at Phase 8). The real SunLake cube has M-1 = 199 pairs.
_MAX_PAIRS_DEV: int = 48

#: SEVIRI channel names (context axis 1). E4 scores all 11; the gallery uses the
#: visible-channel 016 nm band as a perceptually-readable exemplar.
_CHANNELS: tuple[str, ...] = (
    "IR_016", "IR_039", "IR_087", "IR_097", "IR_108", "IR_120", "IR_134",
    "VIS006", "VIS008", "WV_062", "WV_073",
)
_GALLERY_CHANNEL: int = 0  # IR_016

#: Reliability diagram nominal coverage levels.
_NOMINAL_LEVELS: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)

#: Number of quantile bins for the reliability→error-scale calibration.
_REL_BINS: int = 10


# ---------------------------------------------------------------------------
# Image-quality primitives (pure numpy / cv2 — skimage is not a project dep)
# ---------------------------------------------------------------------------

def psnr(pred, truth, data_range: float = 1.0) -> float:
    """Peak SNR (dB) between two equally-shaped images.

    ``10 * log10(data_range**2 / MSE)``. Returns ``+inf`` when the frames are
    identical (MSE = 0) — the documented identical-frame guard.
    """
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if pred.shape != truth.shape:
        raise ValueError(f"psnr shape mismatch: {pred.shape} vs {truth.shape}")
    mse = float(np.mean((pred - truth) ** 2))
    if mse == 0.0:
        return float("inf")
    if not np.isfinite(mse) or data_range <= 0.0:
        return float("nan")
    return float(10.0 * np.log10((data_range ** 2) / mse))


def _gaussian(img: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """11x11 Gaussian blur (the standard SSIM window), float64."""
    return cv2.GaussianBlur(img, (11, 11), sigmaX=sigma, sigmaY=sigma)


def ssim(pred, truth, data_range: float = 1.0) -> float:
    """Global mean Structural Similarity (Wang et al. 2004) over a Gaussian window.

    Returns 1.0 for identical frames. Implemented in numpy + cv2 because
    scikit-image is not a project dependency. Matches the reference formula used
    by ``skimage.metrics.structural_similarity`` for a uniform Gaussian weight.
    """
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if pred.shape != truth.shape:
        raise ValueError(f"ssim shape mismatch: {pred.shape} vs {truth.shape}")
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = _gaussian(pred)
    mu_y = _gaussian(truth)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = _gaussian(pred * pred) - mu_x2
    sigma_y2 = _gaussian(truth * truth) - mu_y2
    sigma_xy = _gaussian(pred * truth) - mu_xy
    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = num / den
    # Interior only: the 11x11 Gaussian window is unreliable within 5px of the
    # border (matches skimage's default `crop_border`-equivalent behaviour for
    # small images; on 64x64 crops this keeps ~84% of the pixels).
    pad = 5
    if ssim_map.shape[0] > 2 * pad and ssim_map.shape[1] > 2 * pad:
        ssim_map = ssim_map[pad:-pad, pad:-pad]
    return float(np.mean(ssim_map))


# ---------------------------------------------------------------------------
# Reliability / prediction-interval coverage
# ---------------------------------------------------------------------------

def _fit_sigma_by_u(err_abs, u, n_bins: int):
    """Bin pixels by reliability ``u``; return ``(bin_centers, mean_|err|_per_bin)``.

    ``u`` is the operator's reliability (high = reliable = low expected error).
    The returned per-bin mean absolute error is the calibrated error scale
    ``σ̂(u)`` used to standardize residuals. Bins are equal-count quantile bins
    over ``u`` so sparse-reliability regions are still represented.
    """
    err_abs = np.asarray(err_abs, dtype=np.float64).ravel()
    u = np.asarray(u, dtype=np.float64).ravel()
    finite = np.isfinite(err_abs) & np.isfinite(u)
    err_abs = err_abs[finite]
    u = u[finite]
    if err_abs.size == 0:
        return np.array([]), np.array([])
    n_bins = max(2, min(n_bins, err_abs.size))
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(u, qs)
    # Deduplicate edges (very narrow reliability range → fewer effective bins).
    edges = np.unique(edges)
    if edges.size < 2:
        return np.array([float(u.mean())]), np.array([float(err_abs.mean())])
    centers, scales = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        last = lo == edges[-2]
        mask = (u >= lo) & (u <= hi) if last else (u >= lo) & (u < hi)
        if not mask.any():
            continue
        centers.append(float(0.5 * (lo + hi)))
        scales.append(float(err_abs[mask].mean()))
    centers = np.asarray(centers, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    # Ensure monotone-decreasing scale vs u (high u → low error): isotonic-style
    # floor so the interpolation is a well-defined reliability→error map. We do
    # NOT force monotonicity — reporting a non-monotone curve is itself a finding
    # (reliability uninformative). Keep raw bin means.
    return centers, scales


def coverage_curve(pred, truth, cal_u, eval_u, nominal_levels=_NOMINAL_LEVELS,
                   n_bins: int = _REL_BINS) -> dict:
    """Distribution-free prediction-interval coverage vs nominal level.

    Calibrates an error scale ``σ̂(u)`` on ``(pred, truth, cal_u)`` (mean |error|
    per reliability bin), then for each nominal level α takes the α-quantile of
    the calibration standardized residual ``|err| / σ̂(u)`` as the interval half-
    width and measures empirical coverage on ``(pred, truth, eval_u)``.

    Returns ``{nominal_levels, empirical_coverage, sigma_bin_centers,
    sigma_bin_scales}``. Perfect calibration → empirical == nominal.
    """
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    cal_u = np.asarray(cal_u, dtype=np.float64)
    eval_u = np.asarray(eval_u, dtype=np.float64)
    cal_err = np.abs(pred - truth)

    centers, scales = _fit_sigma_by_u(cal_err, cal_u, n_bins=n_bins)
    if centers.size == 0 or scales.size == 0:
        return {
            "nominal_levels": list(nominal_levels),
            "empirical_coverage": [float("nan")] * len(nominal_levels),
            "sigma_bin_centers": [],
            "sigma_bin_scales": [],
        }
    # σ̂(u): interpolate (constant-extrapolate at the reliability tails).
    sigma_cal = np.interp(cal_u.ravel(), centers, scales,
                          left=scales[0], right=scales[-1])
    sigma_eval = np.interp(eval_u.ravel(), centers, scales,
                           left=scales[0], right=scales[-1])
    # Guard zero-scale bins (a perfectly-reliable bin with all-zero error).
    sigma_cal = np.where(sigma_cal > 1e-12, sigma_cal, np.nan)
    sigma_eval = np.where(sigma_eval > 1e-12, sigma_eval, np.nan)

    z_cal = (cal_err.ravel() / sigma_cal)
    z_eval = (np.abs(pred - truth).ravel() / sigma_eval)
    z_cal = z_cal[np.isfinite(z_cal)]
    z_eval = z_eval[np.isfinite(z_eval)]

    emp = []
    for alpha in nominal_levels:
        if z_cal.size == 0 or z_eval.size == 0:
            emp.append(float("nan"))
            continue
        c_alpha = float(np.quantile(z_cal, alpha))
        emp.append(float(np.mean(z_eval <= c_alpha)))
    return {
        "nominal_levels": list(nominal_levels),
        "empirical_coverage": emp,
        "sigma_bin_centers": centers.tolist(),
        "sigma_bin_scales": scales.tolist(),
    }


# ---------------------------------------------------------------------------
# Core evaluation: extrapolated future vs real future, per lead + reliability
# ---------------------------------------------------------------------------

def _align_pred_truth(context, future, uncertainty, max_pairs):
    """Build lead-aligned ``(pred, truth)`` stacks + the reliability pixels.

    ``future[k-1, t]`` is the operator's extrapolation from ``context[t+1]`` at
    lead k; the real future at lead k from ``context[t+1]`` is ``context[t+1+k]``
    (valid while ``t+1+k < M``). Returns per-lead ``pred``/``truth`` stacks as
    ``(P_k, C, H, W)`` plus the per-lead reliability maps and the surviving pair
    indices (so reliability can be split cal/eval consistently across leads).
    """
    M, C, H, W = context.shape
    K = future.shape[0]
    pair_indices = list(range(min(max_pairs, M - 1)))
    per_lead: dict[int, dict] = {}
    for k in range(1, K + 1):
        seeds, preds, truths, uncs, pairs = [], [], [], [], []
        for t in pair_indices:
            if t + 1 + k > M - 1:
                continue
            seeds.append(context[t + 1])              # operator's seed frame
            preds.append(future[k - 1, t])            # extrapolation at lead k
            truths.append(context[t + 1 + k])         # REAL future at lead k
            uncs.append(uncertainty[k - 1, t])
            pairs.append(t)
        per_lead[k] = {
            "seed": np.stack(seeds) if seeds else np.empty((0, C, H, W)),
            "pred": np.stack(preds) if preds else np.empty((0, C, H, W)),
            "truth": np.stack(truths) if truths else np.empty((0, C, H, W)),
            "unc": np.stack(uncs) if uncs else np.empty((0, H, W)),
            "pairs": np.asarray(pairs, dtype=np.int64),
        }
    return per_lead


def _evaluate(context, future, uncertainty, *, max_pairs, nominal_levels):
    """Score the extrapolation against the real future across lead K.

    Returns ``{decay, reliability}``:

    * ``decay``: per-lead ``psnr_per_lead`` / ``ssim_per_lead`` arrays (length
      ``K``), plus first-lead scalar means.
    * ``reliability``: the distribution-free coverage curve + the per-bin
      predicted-vs-empirical reliability calibration, plus the Expected
      Calibration Error (mean |empirical − nominal|).
    """
    context = np.asarray(context)
    future = np.asarray(future)
    uncertainty = np.asarray(uncertainty)
    M, C, H, W = context.shape
    K = future.shape[0]
    per_lead = _align_pred_truth(context, future, uncertainty, max_pairs)

    psnr_per_lead = np.full(K, np.nan, dtype=np.float64)
    ssim_per_lead = np.full(K, np.nan, dtype=np.float64)
    for k in range(1, K + 1):
        pred = per_lead[k]["pred"]
        truth = per_lead[k]["truth"]
        if pred.shape[0] == 0:
            continue
        # Pooled PSNR across (pairs, channels, pixels) — one MSE per lead.
        diff = pred.astype(np.float64) - truth.astype(np.float64)
        mse = float(np.mean(diff ** 2))
        psnr_per_lead[k - 1] = float("inf") if mse == 0.0 else (
            float(10.0 * np.log10(1.0 / mse)) if mse > 0 else float("nan")
        )
        # SSIM: mean over (pair, channel) — the standard per-image SSIM average.
        ssims = [
            ssim(pred[i, c], truth[i, c], data_range=1.0)
            for i in range(pred.shape[0]) for c in range(C)
        ]
        ssim_per_lead[k - 1] = float(np.mean(ssims))

    # ---- Reliability: pool pixels across leads × pairs × channels ----------
    # Cal/eval split on PAIR index (first half calibrates σ̂(u), second half is
    # held out for empirical coverage) — no temporal leakage within a pair.
    pair_set = per_lead[1]["pairs"]  # pair indices that survive at K=1
    if pair_set.size >= 4:
        mid = pair_set.size // 2
        cal_pairs = set(pair_set[:mid].tolist())
        eval_pairs = set(pair_set[mid:].tolist())
    else:
        cal_pairs = eval_pairs = set(pair_set.tolist())

    cal_err_chunks, cal_u_chunks, eval_err_chunks, eval_u_chunks = [], [], [], []
    for k in range(1, K + 1):
        pred = per_lead[k]["pred"].astype(np.float64)
        truth = per_lead[k]["truth"].astype(np.float64)
        unc = per_lead[k]["unc"].astype(np.float64)
        pairs = per_lead[k]["pairs"]
        for i, t in enumerate(pairs):
            err = np.abs(pred[i] - truth[i])      # (C, H, W)
            u = np.broadcast_to(unc[i], err.shape)  # reliability shared across channels
            if t in cal_pairs:
                cal_err_chunks.append(err.ravel())
                cal_u_chunks.append(u.ravel())
            elif t in eval_pairs:
                eval_err_chunks.append(err.ravel())
                eval_u_chunks.append(u.ravel())
    cal_err = np.concatenate(cal_err_chunks) if cal_err_chunks else np.array([])
    cal_u = np.concatenate(cal_u_chunks) if cal_u_chunks else np.array([])
    eval_err = np.concatenate(eval_err_chunks) if eval_err_chunks else np.array([])
    eval_u = np.concatenate(eval_u_chunks) if eval_u_chunks else np.array([])

    # coverage_curve calibrates on (cal_err, cal_u); we feed the SAME pixel-pool
    # by using cal for fit and eval for measurement via a thin wrapper.
    centers, scales = _fit_sigma_by_u(cal_err, cal_u, n_bins=_REL_BINS)
    # Build a (pred, truth, cal_u, eval_u) pass through coverage_curve: it scores
    # |pred-truth| — but we need cal vs eval pixel sets. Use a direct computation
    # here (the helper above operates on a single pred/truth image pair).
    nominal = list(nominal_levels)
    emp = []
    if centers.size and scales.size and cal_err.size and eval_err.size:
        sigma_cal = np.interp(cal_u, centers, scales,
                              left=scales[0], right=scales[-1])
        sigma_eval = np.interp(eval_u, centers, scales,
                               left=scales[0], right=scales[-1])
        sigma_cal = np.where(sigma_cal > 1e-12, sigma_cal, np.nan)
        sigma_eval = np.where(sigma_eval > 1e-12, sigma_eval, np.nan)
        z_cal = cal_err / sigma_cal
        z_cal = z_cal[np.isfinite(z_cal)]
        z_eval = eval_err / sigma_eval
        z_eval = z_eval[np.isfinite(z_eval)]
        for alpha in nominal:
            if z_cal.size == 0 or z_eval.size == 0:
                emp.append(float("nan"))
                continue
            c_alpha = float(np.quantile(z_cal, alpha))
            emp.append(float(np.mean(z_eval <= c_alpha)))
    else:
        emp = [float("nan")] * len(nominal)

    nominal_arr = np.asarray(nominal, dtype=float)
    emp_arr = np.asarray(emp, dtype=float)
    finite = np.isfinite(emp_arr)
    ece = float(np.mean(np.abs(emp_arr[finite] - nominal_arr[finite]))) if finite.any() else float("nan")

    # Calibration curve: per reliability-bin predicted reliability (mean u) vs
    # empirical reliability (1 - mean|err|/global_mean|err|). Informative u →
    # monotone rising curve on the diagonal.
    cal_pred_rel: list[float] = []
    cal_emp_rel: list[float] = []
    if cal_err.size:
        global_err = float(cal_err.mean()) if cal_err.mean() > 0 else 1.0
        # Re-bin cal by u and report (mean u, 1 - mean|err|/global) per bin.
        edges = np.quantile(cal_u, np.linspace(0, 1, _REL_BINS + 1))
        edges = np.unique(edges)
        for j in range(len(edges) - 1):
            lo, hi = edges[j], edges[j + 1]
            last = j == len(edges) - 2
            mask = (cal_u >= lo) & ((cal_u <= hi) if last else (cal_u < hi))
            if not mask.any():
                continue
            cal_pred_rel.append(float(cal_u[mask].mean()))
            mean_err = float(cal_err[mask].mean())
            cal_emp_rel.append(float(1.0 - mean_err / global_err))

    decay = {
        "steps_k": int(K),
        "psnr_per_lead": [float(x) for x in psnr_per_lead],
        "ssim_per_lead": [float(x) for x in ssim_per_lead],
        "psnr_k1_mean": float(psnr_per_lead[0]) if K else float("nan"),
        "ssim_k1_mean": float(ssim_per_lead[0]) if K else float("nan"),
        "n_pairs_evaluated": int(per_lead[1]["pred"].shape[0]) if K else 0,
        "n_channels": int(C),
    }
    reliability = {
        "nominal_levels": nominal,
        "empirical_coverage": emp,
        "ece": ece,
        "calibration_predicted": cal_pred_rel,
        "calibration_empirical": cal_emp_rel,
        "sigma_bin_centers": centers.tolist(),
        "sigma_bin_scales": scales.tolist(),
        "n_cal_pixels": int(cal_u.size),
        "n_eval_pixels": int(eval_u.size),
    }
    return {"decay": decay, "reliability": reliability, "per_lead": per_lead}


# ---------------------------------------------------------------------------
# Figures (shared viz design system)
# ---------------------------------------------------------------------------

def _plot_sequence(per_lead, out_dir: Path) -> list[Path]:
    """``fig_optflow_sequence.png`` — context → extrapolated → truth → uncertainty.

    One representative pair at lead K = 1 (short-lead, where the extrapolation is
    expected to track truth most closely), gallery channel IR_016. The context
    panel is the operator's seed frame ``context[t+1]``; the truth panel is the
    REAL future ``context[t+1+K]``.
    """
    leads = [k for k in per_lead if per_lead[k]["pred"].shape[0] > 0]
    if not leads:
        return []
    k = 1 if 1 in leads else leads[0]
    pair_idx = 0
    seed = per_lead[k]["seed"][pair_idx, _GALLERY_CHANNEL]
    pred = per_lead[k]["pred"][pair_idx, _GALLERY_CHANNEL]
    truth = per_lead[k]["truth"][pair_idx, _GALLERY_CHANNEL]
    unc = per_lead[k]["unc"][pair_idx]

    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.2))
    titles = ("Context (seed)", f"Extrapolated (K={k})", "Real future",
              "Reliability (uncertainty)")
    imgs = (seed, pred, truth, unc)
    cmaps = ("inferno", "inferno", "inferno", "viridis")
    for ax, im, t, cm in zip(axes, imgs, titles, cmaps):
        m = ax.imshow(im, cmap=cm)
        ax.set_title(t, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"E4 — Optical-flow extrapolation vs real future "
        f"({_CHANNELS[_GALLERY_CHANNEL]}, pair #{pair_idx}, lead K={k})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save_fig(fig, "fig_optflow_sequence", out_dir)


def _plot_psnr_decay(decay: dict, out_dir: Path) -> list[Path]:
    """``fig_optflow_psnr_decay.png`` — PSNR (dB) and SSIM vs lead K.

    Two panels (different scales — PSNR in dB, SSIM in [0,1]) per the no-dual-
    axis rule. Identical-frame leads (+inf PSNR) are capped at a sane y-ceiling
    so the decay curve stays readable.
    """
    psnr = np.asarray(decay["psnr_per_lead"], dtype=float)
    ssim = np.asarray(decay["ssim_per_lead"], dtype=float)
    k = np.arange(1, psnr.size + 1)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    psnr_plot = np.where(np.isfinite(psnr), psnr, np.nan)
    ax_l.plot(k, psnr_plot, color=PALETTE[0], marker="o", markersize=6,
              linewidth=2.0, label="PSNR")
    if np.isfinite(psnr_plot).any():
        lo = float(np.nanmin(psnr_plot))
        hi = float(np.nanmax(psnr_plot))
        ax_l.set_ylim(min(lo - 2, hi - 4), hi + 2)
    ax_l.set_xlabel("Lead time K (steps)")
    ax_l.set_ylabel("PSNR (dB)")
    ax_l.set_title("Extrapolation quality vs lead time\nPSNR (higher = better)")
    ax_l.grid(True, alpha=0.3)

    ax_r.plot(k, ssim, color=PALETTE[1], marker="s", markersize=6,
              linewidth=2.0, label="SSIM")
    ax_r.set_xlabel("Lead time K (steps)")
    ax_r.set_ylabel("SSIM")
    if np.isfinite(ssim).any():
        lo, hi = float(np.nanmin(ssim)), float(np.nanmax(ssim))
        ax_r.set_ylim(min(0.0, lo - 0.05), min(1.0, hi + 0.05) + 1e-3)
    ax_r.set_title("Extrapolation quality vs lead time\nSSIM (1.0 = identical)")
    ax_r.grid(True, alpha=0.3)
    fig.suptitle("E4 — Optical-flow extrapolation quality decays with lead-time K",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_fig(fig, "fig_optflow_psnr_decay", out_dir)


def _plot_reliability(reliability: dict, out_dir: Path) -> list[Path]:
    """``fig_uncertainty_reliability.png`` — reliability diagram + calibration.

    Left: prediction-interval coverage vs nominal level (diagonal = perfect
    calibration). Right: per reliability-bin predicted reliability (mean u) vs
    empirical reliability (1 − normalized mean |error|). Above-diagonal =
    under-confident; below-diagonal = over-confident.
    """
    nominal = np.asarray(reliability["nominal_levels"], dtype=float)
    emp = np.asarray(reliability["empirical_coverage"], dtype=float)
    pred_rel = np.asarray(reliability["calibration_predicted"], dtype=float)
    emp_rel = np.asarray(reliability["calibration_empirical"], dtype=float)
    ece = reliability["ece"]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    # Reliability diagram.
    ax_l.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=1.2,
              label="Perfect calibration")
    finite = np.isfinite(emp)
    if finite.any():
        ax_l.plot(nominal[finite], emp[finite], color=PALETTE[4], marker="o",
                  markersize=7, linewidth=2.0, label="Empirical coverage")
    ax_l.set_xlim(0, 1); ax_l.set_ylim(0, 1)
    ax_l.set_xlabel("Nominal coverage level")
    ax_l.set_ylabel("Empirical coverage")
    ax_l.set_title(f"Prediction-interval reliability\nECE = {ece:.3f} "
                   f"(0 = perfectly calibrated)")
    ax_l.legend(loc="upper left", fontsize=9)
    ax_l.grid(True, alpha=0.3)
    # Calibration curve (predicted vs empirical reliability per u-bin).
    ax_r.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=1.2,
              label="Perfect reliability")
    if pred_rel.size:
        ax_r.plot(pred_rel, emp_rel, color=PALETTE[5], marker="s", markersize=7,
                  linewidth=2.0, label="Optical-flow reliability")
    ax_r.set_xlim(0, 1); ax_r.set_ylim(0, 1)
    ax_r.set_xlabel("Predicted reliability (mean u per bin)")
    ax_r.set_ylabel("Empirical reliability (1 − normalized mean |error|)")
    ax_r.set_title("Predicted vs empirical reliability\n(monotone rising = informative)")
    ax_r.legend(loc="lower right", fontsize=9)
    ax_r.grid(True, alpha=0.3)
    fig.suptitle("E4 — Uncertainty calibration of the optical-flow reliability map",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_fig(fig, "fig_uncertainty_reliability", out_dir)


def _save_results_csv(metrics: dict, out_dir: Path) -> Path:
    """Per-lead decay table (one row per lead K). Reliability numbers live in
    the returned metrics dict and the reliability figure, not this table."""
    decay = metrics["decay"]
    reliability = metrics["reliability"]
    # Attach the reliability ECE as a single summary column on every row so the
    # headline calibration number travels with the decay table on disk.
    ece = reliability["ece"]
    rows = []
    for j, k in enumerate(range(1, decay["steps_k"] + 1)):
        rows.append({
            "lead_k": k,
            "psnr_db": decay["psnr_per_lead"][j],
            "ssim": decay["ssim_per_lead"][j],
            "reliability_ece": ece,
        })
    df = pd.DataFrame(rows)
    return save_metrics(df, "results_e4_optflow", out_dir)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(loader, out_dir) -> dict:
    """Run experiment E4 and return ``{figures: [paths], metrics: {...}}``.

    Reads only the SunLake ``satellite_future.zarr`` via ``loader``; writes three
    figures + a results CSV under ``out_dir``. Pure: does not mutate ``loader``
    state or the standardized data. Datasets without a satellite extrapolation
    cube are Honest-skipped (reported, not crashed).
    """
    set_style("light")
    out_dir = Path(out_dir)

    context = loader.get_tensor_cube(_DATASET, "satellite_context")
    future = loader.get_tensor_cube(_DATASET, "satellite_future")
    unc = loader.get_tensor_cube(_DATASET, "satellite_uncertainty")
    if context is None or future is None or unc is None:
        return {
            "figures": [],
            "metrics": {
                "status": "skipped",
                "reason": (f"no satellite extrapolation cube for {_DATASET!r} "
                           "(context/future/uncertainty absent)"),
                "dataset": _DATASET,
            },
        }

    # Materialize a bounded slice (the real cube is small enough to read whole,
    # but cap pairs for dev speed; full cube at Phase 8).
    M = context.shape[0]
    m = min(M, _MAX_PAIRS_DEV + 1)
    ctx_np = np.asarray(context[:m], dtype=np.float32)
    fut_np = np.asarray(future[:, :m - 1], dtype=np.float32)
    unc_np = np.asarray(unc[:, :m - 1], dtype=np.float32)

    metrics = _evaluate(
        ctx_np, fut_np, unc_np,
        max_pairs=_MAX_PAIRS_DEV, nominal_levels=_NOMINAL_LEVELS,
    )
    metrics["status"] = "ran"
    metrics["dataset"] = _DATASET
    metrics["config"] = {
        "steps_k": _STEPS_K,
        "max_pairs_dev": _MAX_PAIRS_DEV,
        "channels": list(_CHANNELS),
        "gallery_channel": _CHANNELS[_GALLERY_CHANNEL],
        "nominal_levels": list(_NOMINAL_LEVELS),
        "operator": "extrapolate_with_uncertainty (forward-splat, incremental)",
        "truth": "real context frames context[t+1+K]",
        "dev_note": ("Bounded to the first "
                     f"{_MAX_PAIRS_DEV} context pairs; full cube @ Phase 8."),
    }

    figures: list[Path] = []
    figures += _plot_sequence(metrics["per_lead"], out_dir)
    figures += _plot_psnr_decay(metrics["decay"], out_dir)
    figures += _plot_reliability(metrics["reliability"], out_dir)
    _save_results_csv(metrics, out_dir)
    # Drop the heavy per-lead stacks from the returned metrics (pure summary).
    metrics.pop("per_lead", None)
    return {"figures": [str(p) for p in figures], "metrics": metrics}
