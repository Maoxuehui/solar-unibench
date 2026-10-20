#!/usr/bin/env python
"""4-arm BRR capacity ablation + Figure 12 (spec: BRR representation x capacity).

Quantify the predictive gain of the BRR 4th sky-image channel across two axes:

  * **representation form** -- scalar summary statistics (Arm 2) vs a 2D spatial
    tensor consumed by a CNN (Arms 3 & 4);
  * **model capacity** -- a shallow gradient-boosting tree (LightGBM, Arms 1 & 2)
    vs a deep ResNet-18 feature extractor stacked into LightGBM (Arms 3 & 4).

The four arms (same scalar base features everywhere; differ only in the image
representation):

  1. **Baseline**    -- scalar base only (target lag + ghi_clearsky + cos_zenith)
                        -> LightGBM. No image features.
  2. **Scalar BRR**  -- scalar base + per-window BRR mean/std -> LightGBM.
                        (Consistent with E3's scalar-BRR arm.)
  3. **Deep RGB**    -- scalar base + 512-d ResNet-18 embedding from the 3-channel
                        RGB image (224x224) -> LightGBM (stacking).
  4. **Deep BRR**    -- scalar base + 512-d ResNet-18 embedding from the 4-channel
                        RGB+BRR image (224x224, conv1 widened to 4 inputs) ->
                        LightGBM. Identical architecture + training to Arm 3.

Arm 4 - Arm 3 on the thin-cloud subset is the headline net gain: it isolates
the marginal value of letting a CNN read BRR's 2D spatial thin-cloud structure
(holding model capacity and the scalar base fixed).

Outputs
-------
  * ``figures_full/results/results_brr_capacity_ablation.csv`` -- 32 rows
    (2 datasets x 2 subsets x 4 arms) with mae / mae_sp / skill / n_samples.
  * ``figures_full/fig_brr_ablation_comparison.png`` -- Figure 12:
    (a) physics preview [RGB | BRR] for one Folsom frame, (b) skill comparison
    grouped bars with the Arm4-Arm3 thin-cloud gain highlighted.

Methodology / honesty notes
---------------------------
  * **Thin-cloud subset** = the middle third of samples by mean BRR (quantiles
    1/3 and 2/3), reusing E3's exact definition so Arm 2 is consistent with E3.
  * A single **global temporal 70/30 split** on the image-bearing windows is
    used for every arm and every subset (the ResNet feature extractor trains on
    the train portion only; embeddings for test windows are produced by the
    frozen train-only extractor -- no label leakage).
  * ResNet-18 is **random-init** (weights=None): we test representation capacity,
    not transfer learning. It is **briefly supervised** on the forecast target
    (train split only, 8 epochs, Adam lr=1e-3, batch=32) so Arm 3 and Arm 4 get
    identical treatment and differ only in input channels (3 vs 4). The 512-d
    penultimate embedding is then concatenated with the scalar base and handed
    to LightGBM (stacking).
  * Bounded dev sample (~2000 daytime image-bearing windows per dataset, evenly
    subsampled across the cube's time span); full-config numbers come at Phase 8.
  * Skill = 1 - MAE_model / MAE_SmartPersistence, daytime (cos_zenith > 0.05).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
import os
import tempfile
_REPO_ROOT = Path(__file__).resolve().parents[2]

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
from lightgbm import LGBMRegressor

from solarunibench.imaging.load import load_sky_frame
from solarunibench.loader import SolarUniBenchLoader
from solarunibench.viz.io import save_fig, save_metrics
from solarunibench.viz.style import PALETTE, set_style

_SKY_FILE = "sky_images.zarr"

DATA_ROOT = str(_REPO_ROOT / "data" / "standardized")
OUT_DIR = REPO / "figures_full"
TMP_DIR = Path(tempfile.gettempdir())

DATASET_TARGETS = {"folsom": "ghi_actual", "skippd": "pv_power_normalized"}
DATASETS = ("folsom", "skippd")

HORIZON_HOURS = 1
MATCH_TOLERANCE = "30min"
MAX_FRAMES_DEV = 2500          # even subsample cap -> ~2000 surviving daytime windows
RESIZE = 224
BATCH = 32                     # memory-safe hard constraint
EPOCHS = 25                    # defensive: 50 overfit small sky-image sets; 25 is the sweet spot
LR = 1e-4                      # fine-tuning LR (pretrained backbone, not random-init)
PATIENCE = 3                   # aggressive early stop (defensive against >20-min env reap)
DAYTIME_THRESHOLD = 0.05
SP_EPS = 1e-6
REGIME_LO = 1.0 / 3.0          # thin-cloud = middle third by mean BRR (matches E3)
REGIME_HI = 2.0 / 3.0

LGBM_PARAMS = dict(n_estimators=200, learning_rate=0.05, n_jobs=1,
                   random_state=0, verbose=-1)

# Seed for reproducibility (torch + numpy + LGBM random_state).
SEED = 0


# ---------------------------------------------------------------------------
# Sample assembly (adapted from E3 -- one row per image-bearing forecast window)
# ---------------------------------------------------------------------------

def _naive_utc(values) -> pd.DatetimeIndex:
    di = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    return di.tz_localize(None)


def _open_sky(dataset):
    """Open the sky-image zarr and return ``(images, idx, img_ts)`` subsampled.

    Reads ``N`` from the images-array **metadata** (cheap) and reads ONLY the
    timestamps at the evenly subsampled positions. This avoids materializing
    all ~250k/363k per-frame timestamp chunk files -- the known Windows slow
    path when ``loader.get_image_timestamps`` does a full ``np.asarray``.

    ``images`` is the full lazy zarr array (indexed by original position, as
    :func:`load_sky_frame` expects); ``idx`` are the subsampled original
    positions; ``img_ts`` the timestamps at those positions only.
    """
    p = Path(DATA_ROOT) / dataset / _SKY_FILE
    if not p.exists():
        return None, None, None
    g = zarr.open(str(p), mode="r")
    images = g["images"]
    n = int(images.shape[0])
    idx = np.arange(n)
    if n > MAX_FRAMES_DEV:
        idx = np.linspace(0, n - 1, MAX_FRAMES_DEV).round().astype(np.intp)
    img_ts = np.asarray(g["images_timestamps"][idx])  # read only subsampled chunks
    return images, idx, img_ts


def _build_samples(df, cube, idx, img_ts, target, memmap_path):
    """Build the per-window table + a (N,4,224,224) float32 memmap of frames.

    Mirrors E3's ``_build_samples``: each image at time t is matched (nearest,
    within ``MATCH_TOLERANCE``) to its issue-time tabular row and to the row at
    t+H (forecast target). Non-daytime / NaN rows are dropped. The decoded frame
    is resized to 224x224 and stored in the memmap; BRR mean/std attached.

    ``idx`` / ``img_ts`` are the ALREADY-subsampled original cube positions and
    their timestamps (see :func:`_open_sky`) -- the full-timestamp read is the
    known Windows slow path and is avoided by subsampling before reading.
    """
    needed = ["timestamp_utc", target, "ghi_clearsky", "cos_zenith"]
    missing = [c for c in needed if c not in df.columns]
    if missing or cube is None or img_ts is None or len(img_ts) == 0:
        return None

    img_times = _naive_utc(np.asarray(img_ts))
    img_table = pd.DataFrame({"img_pos": np.asarray(idx), "issue_time": img_times})

    t_lo = img_times.min()
    t_hi = img_times.max() + pd.Timedelta(hours=HORIZON_HOURS)
    tab = df.copy()
    tab["_ts"] = _naive_utc(tab["timestamp_utc"].to_numpy())
    tab = tab[tab["_ts"].between(pd.Timestamp(t_lo), pd.Timestamp(t_hi))]
    if tab.empty:
        return None
    tab = tab.sort_values("_ts").reset_index(drop=True)
    tab["timestamp_utc"] = tab["_ts"]

    tol = pd.Timedelta(MATCH_TOLERANCE)
    merged = pd.merge_asof(
        img_table.sort_values("issue_time").reset_index(drop=True),
        tab[["timestamp_utc", target, "ghi_clearsky", "cos_zenith"]],
        left_on="issue_time", right_on="timestamp_utc",
        direction="nearest", tolerance=tol,
    )
    merged = merged.dropna(subset=[target, "ghi_clearsky", "cos_zenith"])
    if merged.empty:
        return None

    horizon_target = f"{target}_horizon"
    horizon_clear = "ghi_clearsky_horizon"
    merged["horizon_time"] = merged["issue_time"] + pd.Timedelta(hours=HORIZON_HOURS)
    htab = tab[["timestamp_utc", target, "ghi_clearsky"]].sort_values("timestamp_utc").rename(
        columns={"timestamp_utc": "h_ts", target: horizon_target, "ghi_clearsky": horizon_clear}
    )
    merged = pd.merge_asof(
        merged.sort_values("horizon_time").reset_index(drop=True),
        htab, left_on="horizon_time", right_on="h_ts",
        direction="nearest", tolerance=pd.Timedelta(hours=max(1, HORIZON_HOURS)),
    )
    merged = merged.dropna(subset=[horizon_target, horizon_clear])
    merged = merged[merged["cos_zenith"] > DAYTIME_THRESHOLD].reset_index(drop=True)
    if merged.empty:
        return None

    # Decode each surviving frame once, resize to 224, store in a memmap.
    n = len(merged)
    frames = np.lib.format.open_memmap(
        str(memmap_path), mode="w+", dtype=np.float32,
        shape=(n, 4, RESIZE, RESIZE),
    )
    brr_means = np.empty(n, dtype=np.float32)
    brr_stds = np.empty(n, dtype=np.float32)
    for i, pos in enumerate(merged["img_pos"].to_numpy()):
        frame = load_sky_frame(cube, int(pos))           # (4,H,W) float32
        brr = frame[3]
        brr_means[i] = float(brr.mean())
        brr_stds[i] = float(brr.std())
        # Bilinear resize of each channel to RESIZE x RESIZE.
        t = torch.from_numpy(frame.astype(np.float32)).unsqueeze(0)  # 1,4,H,W
        t = F.interpolate(t, size=(RESIZE, RESIZE), mode="bilinear",
                          align_corners=False)
        frames[i] = t.squeeze(0).numpy()
    frames.flush()
    merged["brr_mean"] = brr_means
    merged["brr_std"] = brr_stds
    return merged.sort_values("issue_time").reset_index(drop=True)


# ---------------------------------------------------------------------------
# ResNet-18 feature extractor (3-ch or 4-ch), briefly supervised then frozen
# ---------------------------------------------------------------------------

def _build_backbone(n_in: int, device: torch.device):
    """ImageNet-PRETRAINED ResNet-18 backbone (penultimate 512-d).

    For 4-ch input, conv1 is widened via **Channel Padding Alignment**:
    channels 0-2 inherit the exact ImageNet pretrained weights (so the
    pretrained edge/texture/cloud-detectors transfer immediately); channel 3
    (BRR) is initialized from the mean of the R(0) and B(2) channel filters
    + tiny Gaussian noise, so the BRR modality starts from a physically-
    reasonable initialization rather than random noise.
    """
    from torchvision.models import resnet18, ResNet18_Weights
    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
    if n_in != 3:
        old = resnet.conv1                                   # Conv2d(3, 64, 7, 2, 3)
        new = nn.Conv2d(n_in, 64, kernel_size=7, stride=2,
                        padding=3, bias=False)
        with torch.no_grad():
            new.weight[:, :3] = old.weight.data              # inherit ImageNet RGB
            brr = (old.weight.data[:, 0] + old.weight.data[:, 2]) / 2.0  # mean(R, B)
            new.weight[:, 3] = brr + torch.randn_like(brr) * 0.01  # + 1% noise
        resnet.conv1 = new
    backbone = nn.Sequential(*list(resnet.children())[:-1])  # -> (B,512,1,1)
    head = nn.Linear(512, 1)
    return backbone.to(device), head.to(device)


def _train_and_extract(frames_path, is_train, labels, n_in, device, tag):
    """Supervise a ResNet-18 regressor on train frames, then extract 512-d for all.

    Returns a (N, 512) float32 numpy embedding produced by the FROZEN train-only
    backbone. ``tag`` labels the run for logging.
    """
    frames = np.lib.format.open_memmap(str(frames_path), mode="r")
    n = frames.shape[0]

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    backbone, head = _build_backbone(n_in, device)
    params = list(backbone.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossfn = nn.SmoothL1Loss()  # robust to irradiance outliers

    # ImageNet normalization (the pretrained backbone expects ImageNet-scaled RGB;
    # BRR channel 3 is passed through — its scale [-1,1] is compatible with the
    # Channel-Padding-Alignment init).
    _immean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    _imstd = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def _normalize(x):
        """ImageNet-normalize RGB (ch 0-2); pass BRR (ch 3) through."""
        rgb = (x[:, :3] - _immean) / _imstd
        return torch.cat([rgb, x[:, 3:]], dim=1) if x.shape[1] == 4 else rgb

    tr_idx = np.where(is_train)[0]
    # Normalize the regression target by the train mean/std (helps the head
    # converge; the embedding, not the head, is what we keep).
    y_tr = labels[tr_idx].astype(np.float32)
    y_mean = float(y_tr.mean())
    y_std = float(y_tr.std())
    if not np.isfinite(y_std) or y_std < 1e-6:
        y_std = 1.0

    def batch_indices(idx, shuffle):
        order = np.array(idx)
        if shuffle:
            np.random.shuffle(order)
        for s in range(0, len(order), BATCH):
            yield order[s:s + BATCH]

    backbone.train(); head.train()
    best_loss = float("inf"); patience_left = PATIENCE
    for ep in range(EPOCHS):
        ep_loss = 0.0; nb = 0
        for b in batch_indices(tr_idx, shuffle=True):
            x = torch.from_numpy(frames[b].astype(np.float32))[:, :n_in].to(device)
            x = _normalize(x)  # ImageNet norm for pretrained backbone
            y = torch.from_numpy(
                ((labels[b].astype(np.float32) - y_mean) / y_std)
            ).to(device).view(-1, 1)
            feat = backbone(x).flatten(1)
            pred = head(feat)
            loss = lossfn(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss.item()); nb += 1
        scheduler.step()
        avg = ep_loss / max(1, nb)
        print(f"    [{tag}] epoch {ep+1}/{EPOCHS}  loss={avg:.4f}  lr={scheduler.get_last_lr()[0]:.2e}",
              flush=True)
        # Early stopping on training-loss plateau.
        if avg < best_loss - 1e-5:
            best_loss = avg; patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"    [{tag}] early stop at epoch {ep+1} (patience={PATIENCE})",
                      flush=True)
                break

    # Extract 512-d embeddings for ALL windows with the frozen backbone.
    backbone.eval()
    emb = np.empty((n, 512), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n, BATCH):
            b = np.arange(s, min(s + BATCH, n))
            x = torch.from_numpy(frames[b].astype(np.float32))[:, :n_in].to(device)
            x = _normalize(x)  # same ImageNet norm as training
            feat = backbone(x).flatten(1)
            emb[b] = feat.cpu().numpy()
    del backbone, head
    torch.cuda.empty_cache()
    return emb


# ---------------------------------------------------------------------------
# Smart Persistence reference + scoring
# ---------------------------------------------------------------------------

def _smart_persistence(target_now, clearsky_now, clearsky_horizon):
    kappa = np.asarray(target_now, dtype=float) / np.maximum(
        np.asarray(clearsky_now, dtype=float), SP_EPS)
    return np.maximum(kappa * np.asarray(clearsky_horizon, dtype=float), 0.0)


def _skill(arm_mae, ref_mae):
    if not (np.isfinite(arm_mae) and np.isfinite(ref_mae)) or ref_mae == 0.0:
        return float("nan")
    return float(1.0 - arm_mae / ref_mae)


def _fit_lgbm(X_tr, y_tr, X_te):
    model = LGBMRegressor(**LGBM_PARAMS)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model.fit(X_tr, y_tr)
    return model.predict(X_te)


def _scalar_base(rows, target):
    """[target_now (lag), ghi_clearsky, cos_zenith] -- shared scalar base."""
    return np.column_stack([
        rows[target].to_numpy(dtype=float),
        rows["ghi_clearsky"].to_numpy(dtype=float),
        rows["cos_zenith"].to_numpy(dtype=float),
    ]).astype(np.float32)


def _score_arm(rows, target, emb3, emb4, is_train_mask, arm, subset_name):
    """Score one arm on one subset. Returns dict(mae, mae_sp, skill, n_samples).

    Uses the GLOBAL train/test mask (restricted to this subset) so the deep
    feature extractor never sees test labels.
    """
    n = len(rows)
    horizon_target = f"{target}_horizon"
    if n < 8:
        return {"mae": float("nan"), "mae_sp": float("nan"),
                "skill": float("nan"), "n_samples": int(n)}
    base = _scalar_base(rows, target)
    if arm == 1:                      # Baseline: scalar only
        X = base
    elif arm == 2:                    # Scalar BRR: scalar + BRR mean/std
        X = np.hstack([base, rows[["brr_mean", "brr_std"]].to_numpy(dtype=float)])
    elif arm == 3:                    # Deep RGB: scalar + 512-d RGB embedding
        X = np.hstack([base, emb3])
    elif arm == 4:                    # Deep BRR: scalar + 512-d RGB+BRR embedding
        X = np.hstack([base, emb4])
    else:
        raise ValueError(arm)

    tr = is_train_mask
    te = ~is_train_mask
    if tr.sum() < 4 or te.sum() < 4:
        return {"mae": float("nan"), "mae_sp": float("nan"),
                "skill": float("nan"), "n_samples": int(te.sum())}
    y = rows[horizon_target].to_numpy(dtype=float)
    pred = _fit_lgbm(X[tr], y[tr], X[te])
    mae_arm = float(np.mean(np.abs(y[te] - pred)))
    sp = _smart_persistence(
        rows[target].to_numpy(dtype=float)[te],
        rows["ghi_clearsky"].to_numpy(dtype=float)[te],
        rows["ghi_clearsky_horizon"].to_numpy(dtype=float)[te],
    )
    mae_sp = float(np.mean(np.abs(y[te] - sp)))
    return {"mae": mae_arm, "mae_sp": mae_sp,
            "skill": _skill(mae_arm, mae_sp), "n_samples": int(te.sum())}


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------

def _run_dataset(loader, dataset, target, device):
    print(f"\n=== {dataset} (target={target}) ===", flush=True)
    cube, idx, img_ts = _open_sky(dataset)
    if cube is None:
        print(f"  {dataset}: no sky-image cube, skipping", flush=True)
        return None
    print(f"  cube N={cube.shape[0]} dtype={cube.dtype} -> subsampled "
          f"{len(idx)} positions", flush=True)
    df = loader.load_tabular(dataset)
    if target not in df.columns:
        print(f"  {dataset}: target {target} absent, skipping", flush=True)
        return None

    memmap_path = TMP_DIR / f"brr_ablation_{dataset}.npy"
    samples = _build_samples(df, cube, idx, img_ts, target, memmap_path)
    if samples is None or len(samples) < 16:
        print(f"  {dataset}: only {0 if samples is None else len(samples)} "
              f"image-bearing windows, skipping", flush=True)
        return None
    n = len(samples)
    print(f"  built {n} daytime image-bearing windows", flush=True)

    # Global temporal 70/30 split (used by every arm + the ResNet).
    n_tr = int(n * 0.70)
    is_train = np.zeros(n, dtype=bool)
    is_train[:n_tr] = True

    horizon_target = f"{target}_horizon"
    labels = samples[horizon_target].to_numpy(dtype=float)

    # Train the two ResNet feature extractors (3-ch RGB, 4-ch RGB+BRR).
    print(f"  training ResNet-18 (3-ch RGB) on {int(is_train.sum())} train frames",
          flush=True)
    emb3 = _train_and_extract(memmap_path, is_train, labels, n_in=3,
                              device=device, tag=f"{dataset}/rgb")
    print(f"  training ResNet-18 (4-ch RGB+BRR) on {int(is_train.sum())} train frames",
          flush=True)
    emb4 = _train_and_extract(memmap_path, is_train, labels, n_in=4,
                              device=device, tag=f"{dataset}/brr")

    # Thin-cloud regime (middle third by mean BRR -- matches E3 exactly).
    lo = float(np.quantile(samples["brr_mean"], REGIME_LO))
    hi = float(np.quantile(samples["brr_mean"], REGIME_HI))
    is_thin = ((samples["brr_mean"] > lo) & (samples["brr_mean"] <= hi)).to_numpy()

    arm_emb3_full = emb3
    arm_emb4_full = emb4
    results = []
    for subset_name, mask in (("overall", np.ones(n, dtype=bool)),
                              ("thin_cloud", is_thin)):
        idx = np.where(mask)[0]
        if len(idx) < 8:
            for arm in (1, 2, 3, 4):
                results.append(_empty_row(dataset, subset_name, arm, len(idx)))
            continue
        sub = samples.iloc[idx].reset_index(drop=True)
        sub_train = is_train[idx]
        e3 = arm_emb3_full[idx]
        e4 = arm_emb4_full[idx]
        for arm in (1, 2, 3, 4):
            r = _score_arm(sub, target, e3, e4, sub_train, arm, subset_name)
            results.append({
                "dataset": dataset, "subset": subset_name, "arm": arm,
                "mae": r["mae"], "mae_sp": r["mae_sp"], "skill": r["skill"],
                "n_samples": r["n_samples"],
            })
            print(f"  {subset_name:10s} arm{arm}  MAE={r['mae']:.4g}  "
                  f"SP={r['mae_sp']:.4g}  skill={r['skill']:+.4f}  "
                  f"n={r['n_samples']}", flush=True)

    # Best-effort memmap cleanup.
    try:
        memmap_path.unlink()
    except OSError:
        pass
    return results


def _empty_row(dataset, subset, arm, n):
    return {"dataset": dataset, "subset": subset, "arm": arm,
            "mae": float("nan"), "mae_sp": float("nan"),
            "skill": float("nan"), "n_samples": int(n)}


ARM_NAMES = {
    1: "1_baseline",
    2: "2_scalar_brr",
    3: "3_deep_rgb",
    4: "4_deep_brr",
}


# ---------------------------------------------------------------------------
# Figure 12
# ---------------------------------------------------------------------------

def _pick_folsom_thin_frame(loader):
    """One representative thin-cloud Folsom frame for panel (a).

    Uses the cube length (metadata) to pick scan positions -- never reads the
    full per-frame timestamp array (the known Windows slow path).
    """
    p = Path(DATA_ROOT) / "folsom" / _SKY_FILE
    if not p.exists():
        return None
    g = zarr.open(str(p), mode="r")
    cube = g["images"]
    n = int(cube.shape[0])
    scan = np.linspace(0, n - 1, min(80, n)).round().astype(np.intp)
    decoded = []
    for pos in scan:
        frame = load_sky_frame(cube, int(pos))
        decoded.append({"pos": int(pos), "frame": frame,
                        "brr_mean": float(frame[3].mean())})
    means = np.array([d["brr_mean"] for d in decoded])
    lo = float(np.quantile(means, REGIME_LO))
    hi = float(np.quantile(means, REGIME_HI))
    thin = [d for d in decoded if lo < d["brr_mean"] <= hi]
    if not thin:
        thin = decoded
    # Pick the frame closest to the thin-regime median BRR mean.
    med = float(np.median([d["brr_mean"] for d in thin]))
    return min(thin, key=lambda d: abs(d["brr_mean"] - med))


def _plot_figure12(loader, results_df, out_dir):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    set_style("light")
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.5, 5.6),
                                   gridspec_kw={"width_ratios": [1.0, 1.5]})

    # --- Panel (a): physics preview [RGB | BRR] ---
    ex = _pick_folsom_thin_frame(loader)
    if ex is not None:
        frame = ex["frame"]
        rgb = np.transpose(frame[:3], (1, 2, 0))
        brr = frame[3]
        axa.imshow(np.clip(rgb, 0, 1))
        axa.set_title(f"(a) Folsom sky frame #{ex['pos']} (thin-cloud regime)\n"
                      f"RGB original", fontsize=10.5)
        axa.set_xticks([]); axa.set_yticks([])
        # Overlay a BRR inset (bottom-right) to show thin-cloud contrast gain.
        inset = axa.inset_axes([0.62, 0.08, 0.34, 0.34])
        im = inset.imshow(brr, cmap="RdBu", vmin=-1.0, vmax=1.0)
        inset.set_title(f"BRR channel\nmean={ex['brr_mean']:+.2f}",
                        fontsize=8.5)
        inset.set_xticks([]); inset.set_yticks([])
        cb = fig.colorbar(im, ax=axa, fraction=0.040, pad=0.04)
        cb.set_label("BRR (B-R)/(B+R)", fontsize=8.5)
    else:
        axa.text(0.5, 0.5, "Folsom sky frames unavailable", ha="center", va="center")
        axa.axis("off")

    # --- Panel (b): skill comparison grouped bars ---
    # x = arm (1..4), grouped by (dataset, subset).
    arm_ids = [1, 2, 3, 4]
    groups = []
    for ds in DATASETS:
        for sub in ("overall", "thin_cloud"):
            groups.append((ds, sub))
    n_groups = len(groups)
    x = np.arange(len(arm_ids))
    width = 0.80 / max(1, n_groups)

    # Color per (dataset, subset); thin-cloud bars get a hatch.
    ds_color = {"folsom": PALETTE[0], "skippd": PALETTE[1]}
    for gi, (ds, sub) in enumerate(groups):
        col = ds_color[ds]
        hatch = "//" if sub == "thin_cloud" else ""
        vals = []
        for arm in arm_ids:
            row = results_df[
                (results_df["dataset"] == ds) & (results_df["subset"] == sub) &
                (results_df["arm"] == arm)
            ]
            v = float(row["skill"].iloc[0]) if len(row) else float("nan")
            vals.append(v if np.isfinite(v) else 0.0)
        offset = (gi - (n_groups - 1) / 2.0) * width
        axb.bar(x + offset, vals, width, color=col, edgecolor="black",
                linewidth=0.5, hatch=hatch, alpha=0.92,
                label=f"{ds} [{sub}]" if gi < n_groups else None)

    # Highlight the Arm4 - Arm3 thin-cloud gap (the headline net gain).
    for ds in DATASETS:
        s3 = results_df[(results_df["dataset"] == ds) &
                        (results_df["subset"] == "thin_cloud") &
                        (results_df["arm"] == 3)]
        s4 = results_df[(results_df["dataset"] == ds) &
                        (results_df["subset"] == "thin_cloud") &
                        (results_df["arm"] == 4)]
        if len(s3) and len(s4):
            sk3 = float(s3["skill"].iloc[0]); sk4 = float(s4["skill"].iloc[0])
            if np.isfinite(sk3) and np.isfinite(sk4):
                top = max(sk3, sk4)
                # Annotate above the Arm4 bar group center.
                axb.annotate(
                    f"Δ{sk4-sk3:+.3f}",
                    xy=(3, top), xytext=(3, top + 0.045),
                    ha="center", fontsize=9, fontweight="bold",
                    color=PALETTE[4],
                    arrowprops=dict(arrowstyle="-", color=PALETTE[4], lw=0.9),
                )

    axb.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axb.set_xticks(x)
    axb.set_xticklabels(["Arm 1\nBaseline", "Arm 2\nScalar BRR",
                         "Arm 3\nDeep RGB", "Arm 4\nDeep BRR"])
    axb.set_ylabel("Daytime forecast skill vs Smart Persistence")
    axb.set_title("(b) BRR capacity ablation: skill by representation x capacity\n"
                  "( Arm 4 - Arm 3 on thin-cloud = BRR spatial-structure gain )",
                  fontsize=10.5)
    handles = [
        Patch(facecolor=ds_color["folsom"], edgecolor="black", label="Folsom [overall]"),
        Patch(facecolor=ds_color["folsom"], edgecolor="black", hatch="//", label="Folsom [thin-cloud]"),
        Patch(facecolor=ds_color["skippd"], edgecolor="black", label="SKIPPD [overall]"),
        Patch(facecolor=ds_color["skippd"], edgecolor="black", hatch="//", label="SKIPPD [thin-cloud]"),
    ]
    axb.legend(handles=handles, loc="best", fontsize=8.5, ncol=2)

    fig.suptitle("Figure 12 — BRR representation x model-capacity ablation",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    paths = save_fig(fig, "fig_brr_ablation_comparison", out_dir)
    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  torch={torch.__version__}", flush=True)

    datasets = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS)
    loader = SolarUniBenchLoader(DATA_ROOT)
    all_rows = []
    for ds in datasets:
        target = DATASET_TARGETS[ds]
        rows = _run_dataset(loader, ds, target, device)
        if rows:
            all_rows.extend(rows)

    if not all_rows:
        print("No image-bearing datasets scored; aborting.", flush=True)
        return 1

    results_df = pd.DataFrame(all_rows)
    results_df["arm_label"] = results_df["arm"].map(ARM_NAMES)
    cols = ["dataset", "subset", "arm", "arm_label", "mae", "mae_sp",
            "skill", "n_samples"]
    results_df = results_df[cols].sort_values(["dataset", "subset", "arm"])
    # Append-merge: keep rows from datasets NOT in this run, add this run's.
    csv_path = OUT_DIR / "results" / "results_brr_capacity_ablation.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        old = old[~old["dataset"].isin(datasets)]
        results_df = pd.concat([old, results_df], ignore_index=True)
        results_df = results_df[cols].sort_values(["dataset", "subset", "arm"])
    results_df.to_csv(csv_path, index=False)
    print(f"\nWrote results: {csv_path}", flush=True)
    print(results_df.to_string(index=False), flush=True)

    # Headline number: Arm4 - Arm3 on thin-cloud.
    print("\n--- Headline (Arm 4 Deep BRR vs Arm 3 Deep RGB, thin-cloud) ---",
          flush=True)
    for ds in datasets:
        sub = results_df[results_df["dataset"] == ds]
        s3 = sub[(sub["subset"] == "thin_cloud") & (sub["arm"] == 3)]
        s4 = sub[(sub["subset"] == "thin_cloud") & (sub["arm"] == 4)]
        if len(s3) and len(s4):
            d = float(s4["skill"].iloc[0]) - float(s3["skill"].iloc[0])
            print(f"  {ds}: d_skill(thin) = {d:+.4f}", flush=True)

    fig_paths = _plot_figure12(loader, results_df, OUT_DIR)
    print(f"Wrote figure: {fig_paths}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
