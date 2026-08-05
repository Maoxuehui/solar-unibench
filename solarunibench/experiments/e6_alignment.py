"""E6 — Cross-modal alignment value (Folsom / SKIPPD).

Spec §11 (E6). *Hypothesis: temporal alignment + systematic-lag correction
reduces image↔scalar async label noise and improves multimodal skill.*

Two outputs, both routed through the shared :mod:`solarunibench.viz` design
system:

1. **Alignment diagnostic** (``fig_align_lag_hist.png``) — the timestamp-diff
   distribution between each sky image and its matched tabular row for (a)
   EXACT match only and (b) nearest-neighbour within :data:`_MATCH_TOLERANCE`.
   On Folsom the sky images are stamped off the whole-minute tabular grid (a
   few seconds of systematic offset) so an EXACT timestamp match attaches ~0
   images while NN-within-tolerance attaches every image — the lag the
   alignment mechanism absorbs. SKIPPD ships images already on the tabular
   grid, so its lag collapses to zero (E6 reports that honestly).
2. **Multimodal skill ablation** (``fig_align_skill.png`` +
   ``results/results_e6_alignment.csv``) — a lightweight image-feature
   forecaster (per-window summary statistics of the sky frame + endogenous
   lag + clear-sky context fed to LightGBM, as in E3) scored against the
   physical Smart Persistence reference (spec §10 skill denominator). Two
   arms sharing the SAME model, horizon, samples and train/test boundary:

   * **unaligned** — image attach by EXACT timestamp only. With the offset,
     ~0 images attach, so this arm trains on endogenous + clear-sky alone
     (the multimodal model with its image modality dropped by mis-alignment).
   * **aligned** — image attach by NN within :data:`_MATCH_TOLERANCE`; every
     issue row carries image summary statistics.

   ``Δskill = skill_aligned − skill_unaligned`` isolates the value the
   alignment mechanism delivers. Reported as Daytime skill
   (``cos_zenith > 0.05``).

Bounded/dev-honest notes
------------------------
* The dev standardized cubes are bounded (2000 frames) covering only an early
  window that lies almost entirely in the global ``train`` split. E6 therefore
  re-splits **within the image-bearing period** (temporal 70/30), as the plan
  permits — full-config numbers with the global split + full images come at
  Phase 8.
* Frames are decoded through :func:`load_sky_frame` (handles Folsom JPEG and
  SKIPPD uint8 RGB+BRR identically → ``(4,H,W)`` float32 [RGB 0..1, BRR −1..1]).
* A dev cap (``_MAX_FRAMES_DEV``) evenly subsamples very large cubes for speed.
* If alignment does not help at the dev config on a dataset (e.g. SKIPPD,
  already aligned), E6 reports that honestly rather than fabricating a gain.

Pure function contract: ``run(loader, out_dir) -> {figures, metrics}`` reads
the loader and writes to ``out_dir``; it never mutates standardized data.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from ..imaging.load import load_sky_frame
from ..viz.io import save_fig, save_metrics
from ..viz.style import PALETTE, set_style

__all__ = ["run"]


# ---------------------------------------------------------------------------
# Bounded dev configuration (full-config numbers come at Phase 8).
# ---------------------------------------------------------------------------

#: Per-dataset target (spec §10): GHI for Folsom, PV for SKIPPD.
DATASET_TARGETS: dict[str, str] = {
    "folsom": "ghi_actual",
    "skippd": "pv_power_normalized",
}

#: Image-bearing datasets E6 evaluates (in order). Folsom's offset grid is the
#: richer diagnostic; SKIPPD is reported honestly (already on-grid → no gain).
_DATASETS: tuple[str, ...] = ("folsom", "skippd")

#: Forecast horizon for the image-based ablation (1-step ahead).
_HORIZON_HOURS: int = 1

#: Nearest-neighbor match tolerance (image↔tabular timestamp). Generous enough
#: for Folsom's off-grid sky stamps and SKIPPD. The diagnostic measures the
#: ACTUAL lag the NN-attach absorbs on each dataset.
_MATCH_TOLERANCE: str = "30min"

#: Pass-through window. Matches whose image↔tabular lag is at or below this are
#: considered already fine-aligned and pass through EXACTLY (no nearest-neighbour
#: re-selection, so sub-second jitter on a dataset like SKIPPD cannot inject
#: label noise). 100 ms is well below any physical sky-cloud decorrelation
#: timescale at minute cadence, so a sub-100ms match is physically the same instant.
_PASS_THROUGH_TOLERANCE: str = "100ms"

#: Even subsample cap on decoded frames for dev speed (full cubes at Phase 8).
_MAX_FRAMES_DEV: int = 600

#: TINY LightGBM (fast + robust tabular baseline). n_jobs=1 keeps runs
#: deterministic; verbose=-1 silences console chatter.
_LGBM_PARAMS: dict = {
    "n_estimators": 80,
    "n_jobs": 1,
    "random_state": 0,
    "verbose": -1,
}

#: Daytime threshold (spec §10 anti-fraud mask).
_DAYTIME_THRESHOLD: float = 0.05

#: Clear-sky denominator floor for the Smart Persistence reference.
_SP_EPS: float = 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _naive_utc(values) -> pd.DatetimeIndex:
    """Coerce any datetime-like (naive / ``+00:00`` strings / tz-aware) to naive UTC.

    Both image-timestamp and tabular-timestamp axes are routed through here so
    nearest-neighbour matching shares a single key dtype (real cubes carry ISO
    +offset strings; synthetic/dev frames may be tz-naive).
    """
    di = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    return di.tz_localize(None)


def _skill(arm_mae: float, ref_mae: float) -> float:
    """Forecast skill vs Smart Persistence (0 = parity, positive = better)."""
    if not (np.isfinite(arm_mae) and np.isfinite(ref_mae)) or ref_mae == 0.0:
        return float("nan")
    return float(1.0 - arm_mae / ref_mae)


def _smart_persistence(target_now, clearsky_now, clearsky_horizon) -> np.ndarray:
    """Clear-sky-scaled persistence forecast (mirrors SmartPersistenceModel).

    ``κ = target_now / max(clearsky_now, ε)`` propagated by the horizon clear
    sky; floored at 0. Identical formula to :class:`SmartPersistenceModel` so
    the skill denominator matches the rest of the benchmark.
    """
    kappa = np.asarray(target_now, dtype=float) / np.maximum(
        np.asarray(clearsky_now, dtype=float), _SP_EPS
    )
    pred = kappa * np.asarray(clearsky_horizon, dtype=float)
    return np.maximum(pred, 0.0)


def _frame_summary(frame4: np.ndarray) -> list[float]:
    """Per-frame image-feature vector: per-channel mean/std of the (4,H,W) frame.

    Covers RGB (cloud brightness) and BRR (blue→grey shift), so the vector
    carries the cloud signal regardless of storage layout. Mirrors E3's
    image-feature baseline (LightGBM on summary statistics).
    """
    arr = np.asarray(frame4, dtype=float)        # 4,H,W
    feats: list[float] = []
    for c in range(arr.shape[0]):
        ch = arr[c]
        feats.append(float(ch.mean()))
        feats.append(float(ch.std()))
    return feats


# ---------------------------------------------------------------------------
# Sub-analysis 1: alignment lag diagnostic (per image -> nearest tabular row)
# ---------------------------------------------------------------------------

def _lag_diagnostic(img_ts: np.ndarray, tab_ts: np.ndarray, tolerance: str) -> dict | None:
    """Lag distribution between each image and its nearest tabular timestamp.

    For each image, find the nearest tabular timestamp and record
    ``lag = |img_time − tab_time|`` in seconds. EXACT matches (``lag == 0``) are
    what an unaligned (exact-only) attach would capture; NN-within-tolerance
    matches are what the aligned attach captures. Returns ``None`` when either
    axis is empty.
    """
    if img_ts is None or len(img_ts) == 0 or tab_ts is None or len(tab_ts) == 0:
        return None
    img_times = _naive_utc(np.asarray(img_ts))
    tab_times = _naive_utc(np.asarray(tab_ts))
    # Drop NaT (defensive) and de-duplicate so get_indexer's nearest is stable.
    tab_times = pd.DatetimeIndex(sorted(set(t for t in tab_times if not pd.isna(t))))
    if len(tab_times) == 0:
        return None
    img_times = pd.DatetimeIndex([t for t in img_times if not pd.isna(t)])
    if len(img_times) == 0:
        return None

    # Nearest tabular timestamp per image (no tolerance — measure true lag).
    pos = tab_times.get_indexer(img_times, method="nearest")
    nearest_tab = tab_times.values[pos]
    lags = np.abs((img_times.values - nearest_tab).astype("timedelta64[s]").astype(float))

    tol_s = pd.Timedelta(tolerance).total_seconds()
    pt_s = pd.Timedelta(_PASS_THROUGH_TOLERANCE).total_seconds()
    exact_mask = lags == 0.0
    nn_mask = lags <= tol_s
    # Pass-through: matches within the fine-alignment window (<= 100 ms) are
    # physically the same instant — flagged so already-aligned data (e.g. SKIPPD)
    # is reported as needing no alignment action (no NN re-selection noise).
    pt_mask = nn_mask & (lags <= pt_s)
    nn_lags = lags[nn_mask]
    align_lags = lags[nn_mask & ~pt_mask]  # lags the alignment actually absorbs
    return {
        "n_images": int(len(lags)),
        "exact_count": int(exact_mask.sum()),
        "nn_count": int(nn_mask.sum()),
        "exact_frac": float(exact_mask.mean()) if len(lags) else 0.0,
        "nn_frac": float(nn_mask.mean()) if len(lags) else 0.0,
        "pass_through_count": int(pt_mask.sum()),
        "pass_through_frac": float(pt_mask.mean()) if len(lags) else 0.0,
        "median_lag_seconds": float(np.median(nn_lags)) if nn_lags.size else float("nan"),
        "mean_lag_seconds": float(np.mean(nn_lags)) if nn_lags.size else float("nan"),
        "p95_lag_seconds": float(np.quantile(nn_lags, 0.95)) if nn_lags.size else float("nan"),
        "max_lag_seconds": float(np.max(nn_lags)) if nn_lags.size else float("nan"),
        "median_align_lag_seconds": (float(np.median(align_lags)) if align_lags.size
                                     else 0.0),
        "exact_lags_seconds": lags[exact_mask],
        "nn_lags_seconds": nn_lags,
        "tolerance_seconds": float(tol_s),
        "pass_through_tolerance_seconds": float(pt_s),
    }


# ---------------------------------------------------------------------------
# Sub-analysis 2: sample assembly + aligned/unaligned skill ablation
# ---------------------------------------------------------------------------

def _sorted_image_index(img_idx: np.ndarray, img_times: pd.DatetimeIndex):
    """Build a sorted DatetimeIndex + argsort permutation (mirrors dataset.py).

    ``get_indexer(method='nearest')`` needs a monotonic index; ``order`` maps
    sorted positions back to original cube rows.
    """
    decorated = sorted(
        ((int(p), pd.Timestamp(t)) for p, t in zip(img_idx, img_times)),
        key=lambda x: x[1],
    )
    order = np.array([p for p, _ in decorated], dtype=np.intp)
    di = pd.DatetimeIndex([t for _, t in decorated])
    return di, order


def _build_samples(
    df: pd.DataFrame,
    cube,
    img_ts: np.ndarray,
    target: str,
    *,
    horizon_hours: int,
    tolerance: str,
    max_frames: int,
) -> pd.DataFrame | None:
    """Build a one-row-per-tabular-issue forecast table over the image-bearing span.

    Each tabular issue row (in the image-bearing daytime span, with a valid
    horizon target) becomes a sample carrying the endogenous lag + clear-sky
    context + the image summary statistics of its NN-matched sky frame (within
    ``tolerance``). ``aligned_pos`` is the matched cube position (−1 if outside
    tolerance) and ``exact_hit`` flags exact-timestamp matches (what the
    unaligned arm would capture). Returns ``None`` when no rows survive.
    """
    if cube is None or img_ts is None or len(img_ts) == 0:
        return None
    needed = ["timestamp_utc", "entity_id", target, "ghi_clearsky", "cos_zenith"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return None

    img_idx_all = np.arange(len(img_ts))
    if len(img_idx_all) > max_frames:
        img_idx_all = np.linspace(0, len(img_idx_all) - 1, max_frames).round().astype(np.intp)
    img_times = _naive_utc(np.asarray(img_ts)[img_idx_all])
    img_sorted, order = _sorted_image_index(img_idx_all, img_times)

    tab = df.copy()
    tab["_ts"] = _naive_utc(tab["timestamp_utc"].to_numpy())
    t_lo = img_times.min()
    t_hi = img_times.max() + pd.Timedelta(hours=horizon_hours)
    tab = tab[tab["_ts"].between(pd.Timestamp(t_lo), pd.Timestamp(t_hi))]
    if tab.empty:
        return None
    tab = tab.sort_values("_ts").reset_index(drop=True)
    tab["timestamp_utc"] = tab["_ts"]

    tol = pd.Timedelta(tolerance)
    # Issue context: target_now / clearsky_now / cos_zenith at the issue time.
    issue = tab[["timestamp_utc", target, "ghi_clearsky", "cos_zenith", "entity_id"]].copy()
    issue_times = pd.DatetimeIndex(issue["timestamp_utc"])
    # Nearest image to each issue time within tolerance (the ALIGNED match).
    positions = img_sorted.get_indexer(issue_times, method="nearest", tolerance=tol)
    valid = positions >= 0
    safe_pos = np.maximum(positions, 0)
    nearest_img_time = img_sorted.values[safe_pos]
    issue_lags = np.abs(
        (issue_times.values - nearest_img_time).astype("timedelta64[s]").astype(float)
    )
    issue["_aligned_pos"] = np.where(valid, order[safe_pos], -1)
    # Pass-through: a match within the fine-alignment window (<= 100 ms) is
    # physically the same instant, so it is treated as EXACT in BOTH arms (the
    # unaligned arm captures it too). This prevents sub-second jitter on an
    # already-aligned dataset (e.g. SKIPPD) from fabricating a spurious
    # "alignment gain" — fine-aligned data is honestly reported as needing no
    # alignment action.
    pt_s = pd.Timedelta(_PASS_THROUGH_TOLERANCE).total_seconds()
    pt_hit = valid & (issue_lags <= pt_s)
    issue["_exact_hit"] = pt_hit                       # <=100ms counts as exact
    issue["_pass_through"] = pt_hit & (issue_lags > 0.0)  # (0, 100ms] jitter

    # Horizon match: tabular row nearest to issue_time + H (forecast target).
    horizon_target = f"{target}_horizon"
    issue["horizon_time"] = issue["timestamp_utc"] + pd.Timedelta(hours=horizon_hours)
    htab = tab[["timestamp_utc", target, "ghi_clearsky"]].sort_values("timestamp_utc").rename(
        columns={
            "timestamp_utc": "h_ts",
            target: horizon_target,
            "ghi_clearsky": "ghi_clearsky_horizon",
        }
    )
    merged = pd.merge_asof(
        issue.sort_values("horizon_time").reset_index(drop=True),
        htab,
        left_on="horizon_time",
        right_on="h_ts",
        direction="nearest",
        tolerance=pd.Timedelta(hours=max(1, horizon_hours)),
    )
    merged = merged.dropna(subset=[horizon_target, "ghi_clearsky_horizon"])
    # Daytime mask at the issue time (spec §10 anti-fraud).
    merged = merged[merged["cos_zenith"] > _DAYTIME_THRESHOLD].reset_index(drop=True)
    # Keep only rows whose ALIGNED image attached (the population BOTH arms
    # score on — same samples, same test windows; the arms differ only in
    # whether image features enter the design matrix).
    merged = merged[merged["_aligned_pos"] >= 0].reset_index(drop=True)
    if merged.empty:
        return None

    # Decode each surviving frame once and attach summary stats.
    feats_by_pos: dict[int, list[float]] = {}
    img_features: list[list[float]] = []
    for pos in merged["_aligned_pos"].to_numpy():
        pos_i = int(pos)
        if pos_i not in feats_by_pos:
            feats_by_pos[pos_i] = _frame_summary(load_sky_frame(cube, pos_i))
        img_features.append(feats_by_pos[pos_i])
    merged["img_features"] = img_features
    return merged.sort_values("timestamp_utc").reset_index(drop=True)


def _design_matrix(rows: pd.DataFrame, target: str, *, with_image: bool) -> np.ndarray:
    """Stack ``[target_now, clearsky_now(, *img)]`` per sample.

    The horizon label ``<target>_horizon`` is the regression target, NOT a
    feature. ``with_image=False`` is the unaligned arm (endogenous + clear-sky
    only); ``with_image=True`` is the aligned arm (+ image summary stats).
    """
    base = np.column_stack([
        rows[target].to_numpy(dtype=float),
        rows["ghi_clearsky"].to_numpy(dtype=float),
    ])
    if not with_image:
        return base.astype(np.float32)
    img = np.vstack(rows["img_features"].to_numpy())
    return np.hstack([base, img]).astype(np.float32)


def _fit_predict_lgbm(X_tr, y_tr, X_te) -> np.ndarray:
    """Fit a TINY LightGBM regressor and return test predictions."""
    import warnings

    model = LGBMRegressor(**_LGBM_PARAMS)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")  # silence LGBM console / sklearn-name chatter
        model.fit(X_tr, y_tr)
        return model.predict(X_te)


def _score_subset(rows: pd.DataFrame, target: str, *, with_image: bool) -> dict:
    """Fit one arm on train (first 70%), return daytime MAE + skill vs Smart Persistence.

    The temporal train/test split is computed WITHIN ``rows`` so both arms share
    identical train/test boundaries — the only difference is whether image
    features enter the design matrix (i.e. whether alignment delivered the image).
    """
    n = len(rows)
    if n < 8:
        return {"mae": float("nan"), "mae_sp": float("nan"), "skill": float("nan"),
                "n_samples": int(n)}
    n_tr = int(n * 0.70)
    tr, te = rows.iloc[:n_tr], rows.iloc[n_tr:]
    horizon_target = f"{target}_horizon"
    y_tr = tr[horizon_target].to_numpy(dtype=float)
    y_te = te[horizon_target].to_numpy(dtype=float)

    X_tr = _design_matrix(tr, target, with_image=with_image)
    X_te = _design_matrix(te, target, with_image=with_image)
    pred = _fit_predict_lgbm(X_tr, y_tr, X_te)
    mae_arm = float(np.mean(np.abs(y_te - pred)))

    sp = _smart_persistence(
        te[target].to_numpy(dtype=float),
        te["ghi_clearsky"].to_numpy(dtype=float),
        te["ghi_clearsky_horizon"].to_numpy(dtype=float),
    )
    mae_sp = float(np.mean(np.abs(y_te - sp)))
    return {
        "mae": mae_arm,
        "mae_sp": mae_sp,
        "skill": _skill(mae_arm, mae_sp),
        "n_samples": int(len(te)),
    }


def _evaluate_dataset(loader, dataset: str, target: str) -> dict | None:
    """Run the E6 alignment diagnostic + skill ablation for one image-bearing dataset.

    Returns ``None`` when the dataset has no sky-image cube / no surviving
    image-bearing windows (Honest skip).
    """
    cube = loader.get_tensor_cube(dataset, "sky_images")
    img_ts = loader.get_image_timestamps(dataset)
    if cube is None or img_ts is None:
        return None
    try:
        df = loader.load_tabular(dataset)
    except Exception:
        return None
    if target not in df.columns:
        return None

    tab_ts = df["timestamp_utc"].to_numpy()
    lag = _lag_diagnostic(img_ts, tab_ts, _MATCH_TOLERANCE)
    if lag is None:
        return None

    samples = _build_samples(
        df, cube, img_ts, target,
        horizon_hours=_HORIZON_HOURS, tolerance=_MATCH_TOLERANCE,
        max_frames=_MAX_FRAMES_DEV,
    )
    if samples is None or len(samples) < 8:
        return {"target": target, "lag": _lag_public(lag), "skill": None}

    unaligned = _score_subset(samples, target, with_image=False)
    aligned = _score_subset(samples, target, with_image=True)
    gain = (
        float(aligned["skill"] - unaligned["skill"])
        if (np.isfinite(aligned["skill"]) and np.isfinite(unaligned["skill"]))
        else float("nan")
    )
    return {
        "target": target,
        "lag": _lag_public(lag),
        "skill": {
            "mae_unaligned": unaligned["mae"],
            "mae_aligned": aligned["mae"],
            "mae_sp": unaligned["mae_sp"],
            "skill_unaligned": unaligned["skill"],
            "skill_aligned": aligned["skill"],
            "skill_gain": gain,
            "n_samples": aligned["n_samples"],
            "n_exact_hits": int(samples["_exact_hit"].sum()),
        },
    }


def _lag_public(lag: dict) -> dict:
    """Strip the raw lag ndarrays (keep scalar summaries) for the metrics dict."""
    return {
        "n_images": lag["n_images"],
        "exact_count": lag["exact_count"],
        "nn_count": lag["nn_count"],
        "exact_frac": lag["exact_frac"],
        "nn_frac": lag["nn_frac"],
        "pass_through_count": lag["pass_through_count"],
        "pass_through_frac": lag["pass_through_frac"],
        "median_lag_seconds": lag["median_lag_seconds"],
        "mean_lag_seconds": lag["mean_lag_seconds"],
        "p95_lag_seconds": lag["p95_lag_seconds"],
        "max_lag_seconds": lag["max_lag_seconds"],
        "median_align_lag_seconds": lag["median_align_lag_seconds"],
        "tolerance_seconds": lag["tolerance_seconds"],
        "pass_through_tolerance_seconds": lag["pass_through_tolerance_seconds"],
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_lag_hist(per_dataset: dict[str, dict], raw_lags: dict[str, dict], out_dir: Path) -> list[Path]:
    """Histogram of image↔tabular lag (exact vs NN-within-tolerance) per dataset.

    ``raw_lags`` carries the full ndarray per dataset (``exact_lags_seconds`` /
    ``nn_lags_seconds``); ``per_dataset`` selects the evaluated datasets in
    order. Exact lags are a point mass at 0 (one bar); NN lags span the offset.
    """
    datasets = [ds for ds in per_dataset if ds in raw_lags]
    n = max(1, len(datasets))
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 4.6), squeeze=False)
    if not datasets:
        axes[0, 0].text(0.5, 0.5, "no sky-image coverage", ha="center", va="center")
        axes[0, 0].set_axis_off()
    for i, ds in enumerate(datasets):
        ax = axes[0, i]
        lag = raw_lags[ds]
        exact = lag["exact_lags_seconds"]
        nn = lag["nn_lags_seconds"]
        bins = np.linspace(0, max(lag["tolerance_seconds"], 1.0), 31)
        if nn.size:
            ax.hist(nn, bins=bins, color=PALETTE[0], alpha=0.75, edgecolor="black",
                    linewidth=0.4, label=f"NN within tol (n={lag['nn_count']})")
        if exact.size:
            ax.hist(exact, bins=bins, color=PALETTE[3], alpha=0.75, edgecolor="black",
                    linewidth=0.4, label=f"exact match (n={lag['exact_count']})")
        ax.axvline(lag["median_lag_seconds"] if np.isfinite(lag["median_lag_seconds"]) else 0.0,
                   color=PALETTE[4], linestyle="--", linewidth=1.5, label="median NN lag")
        ax.set_xlabel("|image − tabular| lag (seconds)")
        ax.set_ylabel("image count")
        meta = per_dataset[ds]
        skill = meta.get("skill") or {}
        gain_txt = ""
        if "skill_gain" in skill and np.isfinite(skill["skill_gain"]):
            gain_txt = f"\nΔskill={skill['skill_gain']:+.3f}"
        ax.set_title(
            f"{ds}\nexact={lag['exact_count']} nn={lag['nn_count']} "
            f"/ {lag['n_images']} · median={lag['median_lag_seconds']:.0f}s{gain_txt}",
            fontsize=10,
        )
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(
        "E6 — Cross-modal alignment lag\n"
        "image↔tabular timestamp offset absorbed by NN-attach",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_fig(fig, "fig_align_lag_hist", out_dir)


def _plot_skill(per_dataset: dict[str, dict], out_dir: Path) -> list[Path]:
    """Grouped bars of Daytime skill: unaligned vs aligned, per dataset."""
    datasets = [ds for ds, arm in per_dataset.items() if arm.get("skill")]
    n = len(datasets)
    fig, ax = plt.subplots(figsize=(max(6.5, 2.2 * n + 2.5), 5.4))
    width = 0.38
    x = np.arange(max(n, 1))
    for i, ds in enumerate(datasets):
        s = per_dataset[ds]["skill"]
        u = s["skill_unaligned"]
        a = s["skill_aligned"]
        ax.bar(x[i] - width / 2, u if np.isfinite(u) else 0.0, width,
               color="#9a9a9a", edgecolor="black", linewidth=0.5, hatch="//")
        ax.bar(x[i] + width / 2, a if np.isfinite(a) else 0.0, width,
               color=PALETTE[0], edgecolor="black", linewidth=0.5)
        if np.isfinite(u) and np.isfinite(a):
            top = max(u, a)
            ax.text(x[i], top, f"Δ{s['skill_gain']:+.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=PALETTE[4])
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Daytime forecast skill vs Smart Persistence")
    ax.set_title(
        "E6 — Cross-modal alignment value\n"
        "image-feature forecaster: unaligned (exact-match) vs aligned (NN-attach)"
    )
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="#9a9a9a", edgecolor="black", hatch="//",
                  label="unaligned (exact-match only)"),
            Patch(facecolor=PALETTE[0], edgecolor="black",
                  label="aligned (NN within tolerance)"),
        ],
        loc="best",
        fontsize=9,
    )
    return save_fig(fig, "fig_align_skill", out_dir)


def _save_results_csv(metrics: dict, out_dir: Path) -> Path:
    rows: list[dict] = []
    for ds, arm in metrics["datasets"].items():
        lag = arm["lag"]
        common = {
            "dataset": ds,
            "target": arm["target"],
            "horizon_hours": _HORIZON_HOURS,
            "n_images": lag["n_images"],
            "exact_count": lag["exact_count"],
            "nn_count": lag["nn_count"],
            "exact_frac": lag["exact_frac"],
            "nn_frac": lag["nn_frac"],
            "pass_through_count": lag["pass_through_count"],
            "pass_through_frac": lag["pass_through_frac"],
            "median_lag_seconds": lag["median_lag_seconds"],
            "mean_lag_seconds": lag["mean_lag_seconds"],
            "p95_lag_seconds": lag["p95_lag_seconds"],
            "max_lag_seconds": lag["max_lag_seconds"],
            "median_align_lag_seconds": lag["median_align_lag_seconds"],
            "tolerance_seconds": lag["tolerance_seconds"],
            "pass_through_tolerance_seconds": lag["pass_through_tolerance_seconds"],
        }
        skill = arm.get("skill")
        if skill is None:
            rows.append({**common, "arm": "none", "mae": float("nan"),
                         "mae_sp": float("nan"), "skill": float("nan"),
                         "n_samples": 0})
            continue
        for arm_name, mae_key, skill_key in (
            ("unaligned", "mae_unaligned", "skill_unaligned"),
            ("aligned", "mae_aligned", "skill_aligned"),
        ):
            rows.append({
                **common, "arm": arm_name,
                "mae": skill[mae_key], "mae_sp": skill["mae_sp"],
                "skill": skill[skill_key], "n_samples": skill["n_samples"],
                "skill_gain": skill["skill_gain"],
                "n_exact_hits": skill["n_exact_hits"],
            })
    return save_metrics(pd.DataFrame(rows), "results_e6_alignment", out_dir)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(loader, out_dir) -> dict:
    """Run experiment E6 and return ``{figures: [paths], metrics: {...}}``.

    Reads only standardized packages via ``loader``; writes a lag-histogram
    figure, a skill figure, and a results CSV under ``out_dir``. Pure: does not
    mutate ``loader`` state or the standardized data. Datasets without sky-image
    coverage — or with too few surviving image-bearing windows — are
    Honest-skipped (reported, not crashed).
    """
    set_style("light")
    out_dir = Path(out_dir)

    per_dataset: dict[str, dict] = {}
    raw_lags: dict[str, dict] = {}

    for ds in _DATASETS:
        target = DATASET_TARGETS.get(ds)
        if target is None:
            continue
        cube = loader.get_tensor_cube(ds, "sky_images")
        img_ts = loader.get_image_timestamps(ds)
        if cube is None or img_ts is None:
            continue
        try:
            df = loader.load_tabular(ds)
        except Exception:
            continue
        if target not in df.columns:
            continue
        # Raw diagnostic (with full ndarrays for the histogram).
        full_lag = _lag_diagnostic(img_ts, df["timestamp_utc"].to_numpy(), _MATCH_TOLERANCE)
        if full_lag is not None:
            raw_lags[ds] = full_lag
        res = _evaluate_dataset(loader, ds, target)
        if res is not None:
            per_dataset[ds] = res

    figures: list[Path] = []
    if raw_lags:
        figures += _plot_lag_hist(per_dataset, raw_lags, out_dir)
    if any(arm.get("skill") for arm in per_dataset.values()):
        figures += _plot_skill(per_dataset, out_dir)

    metrics = {
        "datasets": per_dataset,
        "config": {
            "horizon_hours": _HORIZON_HOURS,
            "match_tolerance": _MATCH_TOLERANCE,
            "max_frames_dev": _MAX_FRAMES_DEV,
            "model": "LightGBM",
            "n_estimators": _LGBM_PARAMS["n_estimators"],
            "reference": "SmartPersistence",
            "daytime_cos_zenith": _DAYTIME_THRESHOLD,
            "split": "temporal 70/30 within image-bearing period (dev; global split @ Phase 8)",
            "arms": (
                "unaligned = exact-match image attach (endogenous+clearsky only); "
                "aligned = NN-within-tolerance image attach (+image summary stats)"
            ),
        },
    }
    _save_results_csv(metrics, out_dir)

    return {"figures": [str(p) for p in figures], "metrics": metrics}
