"""E3 — BRR (Blue-to-Red Ratio) thin-cloud enhancement (Folsom / SKIPPD).

Spec §11 (E3). *Hypothesis: the BRR 4th channel amplifies thin-cloud edges and
improves image-based forecasting, especially on the thin-cloud subset.*

Two outputs, both routed through the shared :mod:`solarunibench.viz` design system:

1. **Qualitative gallery** (``fig_brr_gallery.png``) — clear / thin-cloud /
   overcast sky frames rendered as RGB vs BRR side-by-side. The BRR channel
   (``(B-R)/(B+R)``) collapses the bright blue-sky baseline and amplifies the
   blue→grey shift at cloud edges, exposing thin-cloud structure that is faint
   in RGB alone.
2. **Quantitative ablation** (``fig_brr_skill.png`` + ``results/results_e3_brr.csv``)
   — a lightweight image-feature forecaster (per-window summary statistics of
   the sky frame fed to LightGBM) scored against the physical Smart Persistence
   reference (spec §10 skill denominator). Two arms:

   * **RGB-only** — endogenous target lag + clear-sky context + RGB summary
     statistics (per-channel mean/std).
   * **RGB+BRR** — the same features + BRR summary statistics (mean / std /
     histogram entropy).

   The two arms share everything except the BRR features, so
   ``Δskill = skill_brr − skill_rgb`` isolates the marginal value of the BRR
   4th channel. Reported **overall** and on the **thin-cloud subset** (the
   middle third of samples by mean BRR — partial-cloud windows where the
   hypothesis predicts BRR helps most).

Bounded/dev-honest notes
------------------------
* The dev standardized cubes are bounded to 2000 frames covering only an early
  window (Folsom: 3 days; SKIPPD: ~2 months) that lies almost entirely in the
  global ``train`` split. E3 therefore re-splits **within the image-bearing
  period** (temporal 70/30), as the plan permits — full-config numbers with the
  global split + full images come at Phase 8.
* Frames are decoded through :func:`load_sky_frame` (handles Folsom JPEG and
  SKIPPD uint8 RGB+BRR identically → ``(4,H,W)`` float32 [RGB 0..1, BRR −1..1]).
  ``BRR`` lives in exactly one place (:func:`generate_brr_channel`); this module
  never re-implements it.
* A dev cap (``_MAX_FRAMES_DEV``) evenly subsamples very large cubes for speed.

Pure function contract: ``run(loader, out_dir) -> {figures, metrics}`` reads the
loader and writes to ``out_dir``; it never mutates standardized data.
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

#: Image-bearing datasets E3 evaluates (in order; gallery uses the first that
#: resolves a sky-image cube). Folsom's JPEG RGB @256 is the richer gallery.
_DATASETS: tuple[str, ...] = ("folsom", "skippd")

#: Forecast horizon for the image-based ablation (1-step ahead).
_HORIZON_HOURS: int = 1

#: Nearest-neighbor match tolerance (image↔tabular timestamp). Generous enough
#: for Folsom (second-cadence images vs hourly tabular) and SKIPPD.
_MATCH_TOLERANCE: str = "30min"

#: Even subssample cap on decoded frames for dev speed (full cubes at Phase 8).
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

#: Thin-cloud regime boundaries by sample mean-BRR quantile. ``thin`` is the
#: middle third (partial-cloud) — neither clear-sky (high BRR) nor overcast
#: (low BRR). Used for BOTH the gallery exemplars and the ablation subset so
#: the two outputs speak about the same population.
_REGIME_LO: float = 1.0 / 3.0
_REGIME_HI: float = 2.0 / 3.0


# ---------------------------------------------------------------------------
# Image summary statistics (the image-feature baseline)
# ---------------------------------------------------------------------------

def _naive_utc(values) -> pd.DatetimeIndex:
    """Coerce any datetime-like (naive / ``+00:00`` strings / tz-aware) to naive UTC.

    Both image-timestamp and tabular-timestamp axes are routed through here so
    :func:`pd.merge_asof` shares a single key dtype (real cubes are ISO+offset,
    synthetic/dev frames may be tz-naive).
    """
    di = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    return di.tz_localize(None)

def _entropy(chan: np.ndarray, bins: int = 16) -> float:
    """Shannon entropy of the BRR-channel histogram over [−1, 1]."""
    h, _ = np.histogram(chan, bins=bins, range=(-1.0, 1.0))
    total = h.sum()
    if total == 0:
        return 0.0
    p = h / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _frame_summary(frame4: np.ndarray) -> tuple[list[float], list[float], float, float]:
    """Per-frame summary statistics from a decoded ``(4,H,W)`` sky frame.

    Returns ``(rgb_features, brr_features, brr_mean, brr_std)`` where the
    feature lists are the tabular image-feature vector fed to LightGBM.

    * ``rgb_features`` = ``[R_mean, R_std, G_mean, G_std, B_mean, B_std]``.
    * ``brr_features`` = ``[BRR_mean, BRR_std, BRR_entropy]`` (the BRR-only lift).

    RGB global statistics are deliberately a weak proxy for localized thin
    clouds — that is the gap BRR is hypothesized to fill.
    """
    rgb = np.asarray(frame4[:3], dtype=float)      # 3,H,W in [0,1]
    brr = np.asarray(frame4[3], dtype=float)        # H,W in [-1,1]
    r, g, b = rgb[0], rgb[1], rgb[2]
    rgb_features = [
        float(r.mean()), float(r.std()),
        float(g.mean()), float(g.std()),
        float(b.mean()), float(b.std()),
    ]
    brr_mean = float(brr.mean())
    brr_std = float(brr.std())
    brr_features = [brr_mean, brr_std, _entropy(brr)]
    return rgb_features, brr_features, brr_mean, brr_std


# ---------------------------------------------------------------------------
# Sample assembly: per-image forecast windows over the image-bearing period
# ---------------------------------------------------------------------------

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
    """Build a one-row-per-image forecast table over the image-bearing period.

    Each image at time ``t`` is matched to its nearest tabular row (issue
    context: ``target_now`` / ``clearsky_now`` / ``cos_zenith``) and to the
    tabular row at ``t + horizon_hours`` (forecast target + horizon clear-sky).
    Rows that miss either match (outside tolerance), are non-daytime, or whose
    target/clear-sky is NaN are dropped (Honest NaN).

    Returns ``None`` when no image-bearing rows survive. The decoded image
    stats (RGB / BRR) are attached as columns. An even temporal subsample is
    applied when the cube exceeds ``max_frames`` (dev-config speed cap).
    """
    if cube is None or img_ts is None or len(img_ts) == 0:
        return None
    needed = ["timestamp_utc", "entity_id", target, "ghi_clearsky", "cos_zenith"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return None

    img_idx = np.arange(len(img_ts))
    # Even temporal subsample for dev speed (preserves the full time span).
    if len(img_idx) > max_frames:
        img_idx = np.linspace(0, len(img_idx) - 1, max_frames).round().astype(np.intp)

    # Normalize both axes to naive UTC so merge_asof keys share one dtype
    # (real cubes carry +00:00 ISO strings; synthetic/dev may be naive).
    img_times = _naive_utc(np.asarray(img_ts)[img_idx])
    img_table = pd.DataFrame({"img_pos": img_idx, "issue_time": img_times})

    # Restrict tabular to the image-bearing forecast span (issue .. issue+H).
    t_lo = img_times.min()
    t_hi = img_times.max() + pd.Timedelta(hours=horizon_hours)
    tab = df.copy()
    tab["_ts"] = _naive_utc(tab["timestamp_utc"].to_numpy())
    tab = tab[tab["_ts"].between(pd.Timestamp(t_lo), pd.Timestamp(t_hi))]
    if tab.empty:
        return None
    tab = tab.sort_values("_ts").reset_index(drop=True)
    tab["timestamp_utc"] = tab["_ts"]

    tol = pd.Timedelta(tolerance)
    # Issue match: image time -> nearest tabular row (within tolerance).
    merged = pd.merge_asof(
        img_table.sort_values("issue_time").reset_index(drop=True),
        tab[["timestamp_utc", target, "ghi_clearsky", "cos_zenith", "entity_id"]],
        left_on="issue_time",
        right_on="timestamp_utc",
        direction="nearest",
        tolerance=tol,
    )
    merged = merged.dropna(subset=[target, "ghi_clearsky", "cos_zenith"])
    if merged.empty:
        return None

    # Horizon match: tabular row nearest to issue_time + H (forecast target).
    horizon_target = f"{target}_horizon"
    horizon_clear = "ghi_clearsky_horizon"
    merged["horizon_time"] = merged["issue_time"] + pd.Timedelta(hours=horizon_hours)
    htab = tab[["timestamp_utc", target, "ghi_clearsky"]].sort_values("timestamp_utc").rename(
        columns={"timestamp_utc": "h_ts", target: horizon_target, "ghi_clearsky": horizon_clear}
    )
    merged = pd.merge_asof(
        merged.sort_values("horizon_time").reset_index(drop=True),
        htab,
        left_on="horizon_time",
        right_on="h_ts",
        direction="nearest",
        tolerance=pd.Timedelta(hours=max(1, horizon_hours)),
    )
    merged = merged.dropna(subset=[horizon_target, horizon_clear])
    # Daytime mask at the issue time (spec §10 anti-fraud).
    merged = merged[merged["cos_zenith"] > _DAYTIME_THRESHOLD].reset_index(drop=True)
    if merged.empty:
        return None

    # Decode each surviving frame once and attach summary stats.
    rgb_feats: list[list[float]] = []
    brr_feats: list[list[float]] = []
    brr_means: list[float] = []
    brr_stds: list[float] = []
    for pos in merged["img_pos"].to_numpy():
        frame = load_sky_frame(cube, int(pos))
        rgb, brr, bm, bs = _frame_summary(frame)
        rgb_feats.append(rgb)
        brr_feats.append(brr)
        brr_means.append(bm)
        brr_stds.append(bs)
    merged["rgb_features"] = rgb_feats
    merged["brr_features"] = brr_feats
    merged["brr_mean"] = brr_means
    merged["brr_std"] = brr_stds
    return merged.sort_values("issue_time").reset_index(drop=True)


def _regime(brr_mean: float, lo: float, hi: float) -> str:
    """Classify a frame by mean BRR: clear (high) / thin (mid) / overcast (low)."""
    if brr_mean >= hi:
        return "clear"
    if brr_mean <= lo:
        return "overcast"
    return "thin"


# ---------------------------------------------------------------------------
# Smart Persistence reference + LightGBM image arms
# ---------------------------------------------------------------------------

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


def _skill(arm_mae: float, ref_mae: float) -> float:
    """Forecast skill vs Smart Persistence (0 = parity, positive = better)."""
    if not (np.isfinite(arm_mae) and np.isfinite(ref_mae)) or ref_mae == 0.0:
        return float("nan")
    return float(1.0 - arm_mae / ref_mae)


def _design_matrix(rows: pd.DataFrame, target: str, *, with_brr: bool) -> np.ndarray:
    """Stack ``[target_now, clearsky_now, *rgb(, *brr)]`` per sample.

    ``target`` is the issue-time endogenous lag column (shared by both arms);
    the horizon label ``<target>_horizon`` is the regression target, NOT a
    feature. Passing ``target`` explicitly avoids ambiguous suffix inference
    (``ghi_clearsky_horizon`` also ends in ``_horizon``).
    """
    base = np.column_stack([
        rows[target].to_numpy(dtype=float),
        rows["ghi_clearsky"].to_numpy(dtype=float),
    ])
    rgb = np.vstack(rows["rgb_features"].to_numpy())
    parts = [base, rgb]
    if with_brr:
        parts.append(np.vstack(rows["brr_features"].to_numpy()))
    return np.hstack(parts).astype(np.float32)


def _fit_predict_lgbm(X_tr, y_tr, X_te) -> np.ndarray:
    """Fit a TINY LightGBM regressor and return test predictions."""
    import warnings

    model = LGBMRegressor(**_LGBM_PARAMS)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")  # silence LGBM console / sklearn-name chatter
        model.fit(X_tr, y_tr)
        return model.predict(X_te)


def _score_subset(rows: pd.DataFrame, target: str, *, with_brr: bool) -> dict:
    """Fit one arm on train, return daytime MAE + skill vs Smart Persistence.

    ``rows`` is already masked to the subset of interest (overall or thin-cloud)
    but the temporal train/test split is computed WITHIN ``rows`` (first 70%).
    """
    n = len(rows)
    if n < 8:
        return {"mae": float("nan"), "n_samples": int(n)}
    n_tr = int(n * 0.70)
    tr, te = rows.iloc[:n_tr], rows.iloc[n_tr:]
    horizon_target = f"{target}_horizon"
    y_tr = tr[horizon_target].to_numpy(dtype=float)
    y_te = te[horizon_target].to_numpy(dtype=float)

    X_tr = _design_matrix(tr, target, with_brr=with_brr)
    X_te = _design_matrix(te, target, with_brr=with_brr)
    pred = _fit_predict_lgbm(X_tr, y_tr, X_te)
    mae_arm = float(np.mean(np.abs(y_te - pred)))

    # Smart Persistence reference on the SAME test windows (skill denominator).
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
    """Run the E3 image-feature ablation for one image-bearing dataset.

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

    samples = _build_samples(
        df, cube, img_ts, target,
        horizon_hours=_HORIZON_HOURS, tolerance=_MATCH_TOLERANCE,
        max_frames=_MAX_FRAMES_DEV,
    )
    if samples is None or len(samples) < 8:
        return None

    # Regime thresholds from the evaluated sample's mean-BRR quantiles (the
    # population the ablation actually scores — gallery + subset share them).
    lo = float(np.quantile(samples["brr_mean"], _REGIME_LO))
    hi = float(np.quantile(samples["brr_mean"], _REGIME_HI))
    samples["regime"] = samples["brr_mean"].map(lambda m: _regime(m, lo, hi))
    thin = samples[samples["regime"] == "thin"].copy()

    out = {
        "target": target,
        "horizon_hours": _HORIZON_HOURS,
        "brr_mean_quantiles": {"p33": lo, "p66": hi},
        "overall": {},
        "thin_cloud": {},
    }
    # Overall ablation.
    rgb_all = _score_subset(samples, target, with_brr=False)
    brr_all = _score_subset(samples, target, with_brr=True)
    out["overall"] = {
        "mae_rgb": rgb_all["mae"], "mae_brr": brr_all["mae"], "mae_sp": rgb_all.get("mae_sp", float("nan")),
        "skill_rgb": rgb_all["skill"], "skill_brr": brr_all["skill"],
        "n_samples": rgb_all.get("n_samples", 0),
    }
    # Thin-cloud subset.
    if len(thin) >= 8:
        rgb_t = _score_subset(thin, target, with_brr=False)
        brr_t = _score_subset(thin, target, with_brr=True)
        out["thin_cloud"] = {
            "mae_rgb": rgb_t["mae"], "mae_brr": brr_t["mae"], "mae_sp": rgb_t.get("mae_sp", float("nan")),
            "skill_rgb": rgb_t["skill"], "skill_brr": brr_t["skill"],
            "n_samples": rgb_t.get("n_samples", 0),
        }
    else:
        out["thin_cloud"] = {
            "mae_rgb": float("nan"), "mae_brr": float("nan"), "mae_sp": float("nan"),
            "skill_rgb": float("nan"), "skill_brr": float("nan"),
            "n_samples": int(len(thin)),
        }
    return out


# ---------------------------------------------------------------------------
# Qualitative gallery: RGB vs BRR for clear / thin / overcast frames
# ---------------------------------------------------------------------------

def _pick_gallery_frames(cube, img_ts: np.ndarray, max_scan: int = 60) -> list[dict]:
    """Decode an even subsample of frames, classify, return 3 regime exemplars.

    Each exemplar dict carries the decoded ``(4,H,W)`` frame + its stats +
    regime label. Returns fewer than 3 when the cube is too small.
    """
    n = len(img_ts)
    if n == 0:
        return []
    scan_idx = (np.linspace(0, n - 1, min(max_scan, n)).round().astype(np.intp))
    decoded = []
    for pos in scan_idx:
        frame = load_sky_frame(cube, int(pos))
        _, _, bm, bs = _frame_summary(frame)
        decoded.append({"pos": int(pos), "frame": frame, "brr_mean": bm, "brr_std": bs})
    if not decoded:
        return []
    means = np.array([d["brr_mean"] for d in decoded])
    lo = float(np.quantile(means, _REGIME_LO)) if len(means) >= 3 else float(np.median(means))
    hi = float(np.quantile(means, _REGIME_HI)) if len(means) >= 3 else float(np.median(means))
    for d in decoded:
        d["regime"] = _regime(d["brr_mean"], lo, hi)

    # One exemplar per regime: the frame whose BRR mean is closest to the
    # regime's median BRR mean.
    exemplars: list[dict] = []
    for regime in ("clear", "thin", "overcast"):
        members = [d for d in decoded if d["regime"] == regime]
        if not members:
            continue
        target_median = float(np.median([d["brr_mean"] for d in members]))
        best = min(members, key=lambda d: abs(d["brr_mean"] - target_median))
        exemplars.append(best)
    return exemplars


def _plot_gallery(loader, dataset: str, out_dir: Path) -> list[Path]:
    """Render ``fig_brr_gallery.png`` (RGB vs BRR, 3 regime rows)."""
    cube = loader.get_tensor_cube(dataset, "sky_images")
    img_ts = loader.get_image_timestamps(dataset)
    exemplars = _pick_gallery_frames(cube, img_ts) if cube is not None and img_ts is not None else []
    n = max(1, len(exemplars))
    fig, axes = plt.subplots(n, 2, figsize=(7.0, 2.6 * n + 0.6), squeeze=False)
    if not exemplars:
        axes[0, 0].text(0.5, 0.5, "no sky-image frames available", ha="center", va="center")
        axes[0, 0].axis("off")
        axes[0, 1].axis("off")
    for i, ex in enumerate(exemplars):
        frame = ex["frame"]
        rgb = np.transpose(frame[:3], (1, 2, 0))            # H,W,3 in [0,1]
        brr = frame[3]                                       # H,W in [-1,1]
        axes[i, 0].imshow(np.clip(rgb, 0, 1))
        axes[i, 0].set_title(f"{ex['regime']} — RGB (frame #{ex['pos']})", fontsize=9)
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])
        im = axes[i, 1].imshow(brr, cmap="RdBu", vmin=-1.0, vmax=1.0)
        axes[i, 1].set_title(
            f"BRR  mean={ex['brr_mean']:+.2f}  std={ex['brr_std']:.2f}", fontsize=9
        )
        axes[i, 1].set_xticks([]); axes[i, 1].set_yticks([])
        fig.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)
    fig.suptitle(f"E3 — BRR thin-cloud enhancement ({dataset}): RGB vs BRR by regime", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_fig(fig, "fig_brr_gallery", out_dir)


# ---------------------------------------------------------------------------
# Skill figure + results CSV
# ---------------------------------------------------------------------------

def _plot_skill(per_dataset: dict[str, dict], out_dir: Path) -> list[Path]:
    """Grouped bars of Daytime skill: RGB vs RGB+BRR, overall + thin-cloud."""
    datasets = list(per_dataset.keys())
    n = len(datasets)
    fig, ax = plt.subplots(figsize=(max(6.5, 2.2 * n + 2.5), 5.4))
    width = 0.20
    x = np.arange(n)
    for i, ds in enumerate(datasets):
        arm = per_dataset[ds]
        ov, tv = arm["overall"], arm["thin_cloud"]
        for j, (subset, color, hatch) in enumerate((
            ("overall", PALETTE[0], ""),
            ("thin_cloud", PALETTE[3], "//"),
        )):
            s = arm[subset]
            rgb_sk = s["skill_rgb"]
            brr_sk = s["skill_brr"]
            base = x[i] + (j - 0.5) * 2 * width
            ax.bar(base - width / 2, rgb_sk if np.isfinite(rgb_sk) else 0.0, width,
                   color="#9a9a9a", edgecolor="black", linewidth=0.5, hatch=hatch)
            ax.bar(base + width / 2, brr_sk if np.isfinite(brr_sk) else 0.0, width,
                   color=color, edgecolor="black", linewidth=0.5, hatch=hatch)
            # Δskill annotation (the headline E3 number).
            if np.isfinite(rgb_sk) and np.isfinite(brr_sk):
                top = max(rgb_sk, brr_sk)
                ax.text(base + width / 2, top, f"Δ{brr_sk - rgb_sk:+.3f}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold",
                        color=PALETTE[4])
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ds}\n[overall | thin]" for ds in datasets])
    ax.set_ylabel("Daytime forecast skill vs Smart Persistence")
    ax.set_title("E3 — BRR thin-cloud enhancement\nimage-feature ablation: RGB-only vs RGB+BRR")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="#9a9a9a", edgecolor="black", label="RGB-only features"),
            Patch(facecolor=PALETTE[0], edgecolor="black", label="RGB+BRR (overall)"),
            Patch(facecolor=PALETTE[3], edgecolor="black", hatch="//", label="RGB+BRR (thin-cloud subset)"),
        ],
        loc="best",
        fontsize=8,
    )
    return save_fig(fig, "fig_brr_skill", out_dir)


def _save_results_csv(metrics: dict, out_dir: Path) -> Path:
    rows: list[dict] = []
    for ds, arm in metrics["datasets"].items():
        for subset in ("overall", "thin_cloud"):
            s = arm[subset]
            for arm_name, mae_key, skill_key in (
                ("rgb", "mae_rgb", "skill_rgb"),
                ("brr", "mae_brr", "skill_brr"),
            ):
                rows.append({
                    "dataset": ds,
                    "target": arm["target"],
                    "subset": subset,
                    "arm": arm_name,
                    "mae": s[mae_key],
                    "mae_sp": s["mae_sp"],
                    "skill": s[skill_key],
                    "n_samples": s["n_samples"],
                    "horizon_hours": arm["horizon_hours"],
                })
    return save_metrics(pd.DataFrame(rows), "results_e3_brr", out_dir)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(loader, out_dir) -> dict:
    """Run experiment E3 and return ``{figures: [paths], metrics: {...}}``.

    Reads only standardized packages via ``loader``; writes a gallery figure, a
    skill figure, and a results CSV under ``out_dir``. Pure: does not mutate
    ``loader`` state or the standardized data. Datasets without sky-image
    coverage are Honest-skipped (reported, not crashed).
    """
    set_style("light")
    out_dir = Path(out_dir)

    figures: list[Path] = []
    gallery_frames_meta: list[dict] = []
    gallery_dataset: str | None = None
    for ds in _DATASETS:
        cube = loader.get_tensor_cube(ds, "sky_images")
        img_ts = loader.get_image_timestamps(ds)
        if cube is not None and img_ts is not None:
            gallery_dataset = ds
            for ex in _pick_gallery_frames(cube, img_ts):
                gallery_frames_meta.append({
                    "dataset": ds, "regime": ex["regime"], "img_pos": ex["pos"],
                    "brr_mean": ex["brr_mean"], "brr_std": ex["brr_std"],
                })
            break

    if gallery_dataset is not None:
        figures += _plot_gallery(loader, gallery_dataset, out_dir)

    per_dataset: dict[str, dict] = {}
    for ds in _DATASETS:
        target = DATASET_TARGETS.get(ds)
        if target is None:
            continue
        res = _evaluate_dataset(loader, ds, target)
        if res is not None:
            per_dataset[ds] = res

    if per_dataset:
        figures += _plot_skill(per_dataset, out_dir)

    metrics = {
        "datasets": per_dataset,
        "gallery": {
            "dataset": gallery_dataset,
            "frames": gallery_frames_meta,
        },
        "config": {
            "horizon_hours": _HORIZON_HOURS,
            "match_tolerance": _MATCH_TOLERANCE,
            "max_frames_dev": _MAX_FRAMES_DEV,
            "model": "LightGBM",
            "n_estimators": _LGBM_PARAMS["n_estimators"],
            "reference": "SmartPersistence",
            "regime_quantiles": {"thin_lo": _REGIME_LO, "thin_hi": _REGIME_HI},
            "split": "temporal 70/30 within image-bearing period (dev; global split @ Phase 8)",
        },
    }
    if per_dataset:
        _save_results_csv(metrics, out_dir)

    return {"figures": [str(p) for p in figures], "metrics": metrics}
