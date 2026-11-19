"""E9-contiguous — full-data governance crossover on the E8-aligned protocol.

Branch experiment (exp/e9-contiguous-rerun). Replaces the bounded/subsampled
E9 grid (scripts/experiments/e9_cross_dataset.py) with a strict re-run whose
data regime, splits, windowing and evaluation are byte-aligned with the E8
unified benchmark (solarunibench/benchmark/run_e8.py):

  * FULL DATA, NO SUBSAMPLING — every station, every year, native cadence.
  * Frozen splits reused verbatim — the ``split`` column already stamped in
    data/standardized/<ds>/time_series.parquet by scripts/build_splits.py
    (the same membership E8 loads). The raw arm inherits identical labels
    via an exact (entity_id, timestamp_utc) merge, so both arms share one
    split.
  * Windows: T_window = 12 h and H = 1 h in PHYSICAL time on the native
    axis — (folsom, skippd): L=720, H=60; (sunlake): L=24, H=2;
    (gefcom2014): L=12, H=1. Windowing stride follows E8's FULL map
    ({1 min: 30, 30 min: 1, 1 h: 1}; stride-1 at 1-min x L=720 would need
    ~10 TB of materialized windows) and EVERY window is additionally
    contiguity-checked: it survives only if its timestamp span equals
    (L + H - 1) * native_dt exactly (real data holes drop the window).
  * Two arms, one test set (E5 discipline):
      - RAW arm:      target-only past block (feature_cols=[]), anchor
        columns physically dropped from the raw frames -> X_future is None;
        NaN targets get per-entity ffill then 0 (trainability only).
      - GOVERNED arm: the standardized package with E8's exact feature set
        [cos_zenith, ghi_clearsky, clearness_index_aligned]
        (benchmark.features.add_clearness_aligned; E8 applies it to ALL
        datasets, including the PV clearness proxy).
      Both arms are scored on the SAME governed test windows: daytime mask
      cos_zenith > 0.05, Smart Persistence (clear-sky-scaled, through the
      same pipeline) as the skill reference.
  * Models: E8-FULL templates (run_e8._build_full_models) filtered to
    LightGBM, MLP, LSTM, iTransformer, PatchTST-CD (+ CrossViViT on the
    image datasets folsom/skippd). Deep models use E8-FULL early stopping
    (patience=7 on the internal last-15% chronological train holdout).
    CrossViViT runs via ImageWindowEngineer with image_stride = L // 8
    (8 frames per window — the same per-window image budget as E8); the
    RAW CrossViViT consumes RGB-only 3-channel tiles (in_channels=3).
  * Crossover margins (skill %):
      chi_mu   = max(LGBM, MLP).skill_cleaned - mean(deep_pool).skill_raw
      chi_best = max(LGBM, MLP).skill_cleaned - max(deep_pool).skill_raw
    deep_pool = {LSTM, iTransformer, PatchTST-CD} (+ CrossViViT where the
    image modality exists). The tabular-only 3-model camp is also exported
    (chi_mu_3 / chi_best_3) for continuity with the legacy figure.

Outputs:
  results/results_e9_contiguous.csv          one row per (dataset, arm, model)
  results/results_e9_contiguous_chi.csv     per-dataset margins (both camps)
  results/results_e9_contiguous_summary.md  full markdown matrix

Run (conda env solar_uni):
    python scripts/experiments/e9_contiguous.py                 # all 4
    python scripts/experiments/e9_contiguous.py sunlake gefcom2014
    python scripts/experiments/e9_contiguous.py --quick folsom  # dry-run
"""
from __future__ import annotations

import os

os.environ.setdefault("SUB_FULL_CONFIG", "1")

import copy
import sys
import tempfile
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from solarunibench.benchmark.config import DATASET_CONFIG
from solarunibench.benchmark.evaluator import evaluate_predictions
from solarunibench.benchmark.features import add_clearness_aligned
from solarunibench.benchmark.run_e8 import _build_full_models
from solarunibench.forecast.core import (
    EndogenousFeatureEngineer,
    PipelineConfig,
    SolarForecastPipeline,
)
from solarunibench.forecast.models.crossvivit_model import CrossViViTModel
from solarunibench.forecast.models.lightgbm_model import LightGBMModel
from solarunibench.forecast.models.lstm_model import LSTMModel
from solarunibench.forecast.models.mlp_model import MLPModel
from solarunibench.forecast.models.smart_persistence import SmartPersistenceModel
from solarunibench.forecast.scaling import PhysicalFeatureScaler
from solarunibench.loader import SolarUniBenchLoader

OUT_DIR = _REPO_ROOT / "results"
OUT_CELLS = OUT_DIR / "results_e9_contiguous.csv"
OUT_CHI = OUT_DIR / "results_e9_contiguous_chi.csv"
OUT_MD = OUT_DIR / "results_e9_contiguous_summary.md"

#: E8's full-data windowing stride by native sampling rate (run_e8.py:106).
FULL_STRIDE = {1: 30, 30: 1, 60: 1}

#: 12-h lookback / 1-h horizon in STEPS per dataset (cadence-scaled).
ROW_PARAMS = {
    "folsom": (720, 60),
    "skippd": (720, 60),
    "sunlake": (24, 2),
    "gefcom2014": (12, 1),
}

_E8_FEATURES = ["cos_zenith", "ghi_clearsky", "clearness_index_aligned"]
_ANCHOR_COLS = ("ghi_clearsky", "cos_zenith", "clearness_index",
                "clearness_index_aligned", "dni_clearsky", "dhi_clearsky")
_DAYTIME = 0.05


# ---------------------------------------------------------------------------
# Contiguity-checked feature engineering
# ---------------------------------------------------------------------------
def _contiguous_keep(df: pd.DataFrame, lookback: int, horizon: int,
                     stride: int, dt_seconds: float) -> np.ndarray:
    """Per-window contiguity mask replicating the engineer's window loop.

    A window survives iff its full row span [start, start+L+H-1] has an
    exact timestamp span of (L+H-1)*dt — no missing rows inside. Evaluated
    per entity on the (entity_id, timestamp_utc)-sorted frame, i.e. the
    identical iteration order EndogenousFeatureEngineer.create_features
    uses, so the mask aligns 1:1 with the engineered window arrays.
    """
    keeps: list[bool] = []
    span = (lookback + horizon - 1) * dt_seconds
    df_sorted = df.sort_values(["entity_id", "timestamp_utc"], kind="mergesort")
    for _, g in df_sorted.groupby("entity_id", sort=False):
        ts = pd.DatetimeIndex(g["timestamp_utc"]).astype("int64").to_numpy() / 1e9
        n = len(ts)
        last_start = n - lookback - horizon
        for start in range(0, last_start + 1, stride):
            end = start + lookback + horizon - 1
            keeps.append(abs((ts[end] - ts[start]) - span) < 0.5)
    return np.asarray(keeps, dtype=bool)


class ContiguousEngineer(EndogenousFeatureEngineer):
    """EndogenousFeatureEngineer + strict per-window continuity filter."""

    def __init__(self, *a, dt_seconds: float = 60.0, **kw):
        super().__init__(*a, **kw)
        self._dt = float(dt_seconds)

    def create_features(self, df, lookback, horizon, stride=1):
        out = super().create_features(df, lookback, horizon, stride)
        keep = _contiguous_keep(df, lookback, horizon, stride, self._dt)
        xp, y, t, xf, cz, img = out
        if keep.size and int(keep.sum()) != xp.shape[0]:
            if keep.sum() == 0:
                raise RuntimeError(
                    "contiguity filter dropped every window — check dt_seconds")
            print(f"    [contig] kept {int(keep.sum())}/{xp.shape[0]} windows",
                  flush=True)
            xp, y, t = xp[keep], y[keep], t[keep]
            xf = xf[keep] if xf is not None else None
            cz = cz[keep] if cz is not None else None
        return xp, y, t, xf, cz, img


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------
def _expand_night_grid(df: pd.DataFrame, target: str, ds: str,
                       with_anchors: bool) -> pd.DataFrame:
    """SKIPP'D night-grid reconstruction (physical determinism, not imputation).

    The standardized SKIPP'D table has NO night rows (upstream design), so a
    strictly contiguous 12-h/1-min window can never exist (max observed run
    759 rows < 781). This reindexes each entity to its full native 1-min
    grid and fills the ADDED night rows with physically determined values:
    PV target = 0 (a PV plant produces zero at night), and — for the
    governed arm — astronomy/clear-sky anchors recomputed analytically
    (operators.clearsky). Existing daylight rows are untouched, so the
    raw-vs-governed contrast on real observations is preserved exactly.
    """
    if ds != "skippd":
        return df
    from solarunibench.operators.clearsky import (
        clearsky_ineichen,
        compute_astronomy,
    )

    meta = pd.read_parquet(_REPO_ROOT / "data" / "standardized" / ds
                           / "metadata.parquet").set_index("entity_id")
    out = []
    for eid, g in df.groupby("entity_id", sort=False):
        g = g.set_index("timestamp_utc").sort_index()
        grid = pd.date_range(g.index.min(), g.index.max(), freq="1min")
        add_ts = grid.difference(g.index)
        g = g.reindex(grid)
        g["entity_id"] = eid
        night_add = pd.Index(add_ts)
        g.loc[night_add, target] = 0.0
        if with_anchors:
            m = meta.loc[eid]
            astro = compute_astronomy(night_add, float(m["latitude"]),
                                      float(m["longitude"]))
            cs = clearsky_ineichen(night_add, float(m["latitude"]),
                                   float(m["longitude"]),
                                   elevation=float(m.get("elevation_meters")
                                                   or 0.0))
            for col, val in (("cos_zenith", astro["cos_zenith"]),
                             ("solar_azimuth", astro["solar_azimuth"]),
                             ("ghi_clearsky", cs["ghi_clearsky"]),
                             ("dni_clearsky", cs["dni_clearsky"]),
                             ("dhi_clearsky", cs["dhi_clearsky"])):
                g.loc[night_add, col] = val.to_numpy()
            # PV clearness proxy rule (features.add_clearness_aligned):
            # clearsky == 0 (night) -> 0
            g.loc[night_add, "clearness_index_aligned"] = 0.0
        g = g.reset_index().rename(columns={"index": "timestamp_utc"})
        if "split" in df.columns:
            g["split"] = g["split"].ffill().bfill()
        out.append(g)
    res = pd.concat(out, ignore_index=True)
    n_added = len(res) - len(df)
    print(f"    [night-grid] {ds}: +{n_added} physically-determined night rows"
          f" (existing rows untouched)", flush=True)
    return res


def governed_frames(loader: SolarUniBenchLoader, ds: str, target: str):
    df_tr = add_clearness_aligned(loader.load_tabular(ds, "train"), target)
    df_te = add_clearness_aligned(loader.load_tabular(ds, "test"), target)
    return df_tr, df_te


def raw_frames(loader: SolarUniBenchLoader, ds: str, target: str):
    """Pre-governance tabular frames carrying the GOVERNED split membership.

    Raw = the adapter's raw=True tabular output (full data, anchors merged
    but quality audit skipped), joined 1:1 onto the governed split keys by
    (entity_id, timestamp_utc). Anchor columns are then dropped (target-only
    arm) and NaN targets get the minimal per-entity ffill/0 fill.
    """
    from solarunibench.pipeline.adapters.folsom import FolsomPipeline
    from solarunibench.pipeline.adapters.gefcom import GEFCom2014Pipeline
    from solarunibench.pipeline.adapters.skippd import SKIPPDPipeline
    from solarunibench.pipeline.adapters.sunlake import SunLakePipeline

    adapters = {"folsom": FolsomPipeline, "skippd": SKIPPDPipeline,
                "sunlake": SunLakePipeline, "gefcom2014": GEFCom2014Pipeline}
    subs = {"folsom": "Folsom_data", "skippd": "SKIPPD_all",
            "sunlake": "SunLake_data", "gefcom2014": "GEFCom2014_data"}
    raw_root = os.environ.get(
        "SOLAR_RAW_ROOT", "D:/monica/workspace/solar_benchmark_dataset")

    if ds == "sunlake":  # tabular-only: never touch the hub satellite pull
        SunLakePipeline.process_and_save_tensors = lambda self, df: None
    tmp = tempfile.mkdtemp(prefix=f"e9c_{ds}_", dir=tempfile.gettempdir())
    pipe = adapters[ds](raw_data_dir=f"{raw_root}/{subs[ds]}",
                        output_dir=tmp, dataset_name=f"{ds}_raw")
    pipe.run_pipeline(raw=True, limit=None)
    raw = pd.read_parquet(Path(tmp) / f"{ds}_raw" / "time_series.parquet")
    raw = raw.drop(columns=[c for c in _ANCHOR_COLS if c in raw.columns],
                   errors="ignore")
    raw = (raw.sort_values(["entity_id", "timestamp_utc"], kind="mergesort")
              .drop_duplicates(["entity_id", "timestamp_utc"]))
    print(f"    [raw-rebuild] rows={len(raw)}", flush=True)

    frames = {}
    for split in ("train", "test"):
        src = loader.load_tabular(ds, split)
        part = src[["entity_id", "timestamp_utc"]].merge(
            raw, on=["entity_id", "timestamp_utc"], how="inner")
        part["split"] = split
        part[target] = (part.sort_values("timestamp_utc")
                        .groupby("entity_id")[target]
                        .transform(lambda s: s.ffill().fillna(0.0)))
        part = part.dropna(subset=[target])
        frames[split] = part
        print(f"    [raw-{split}] rows={len(part)}", flush=True)
    return frames["train"], frames["test"]


# ---------------------------------------------------------------------------
# Tabular arms
# ---------------------------------------------------------------------------
def run_tabular_arm(ds, target, train, test_gov, model, lb, hz, st, dt,
                    governed: bool):
    feats = list(_E8_FEATURES) if governed else []
    if not governed:
        # The raw arm must see NO anchor columns anywhere: pandas concat
        # would column-union them back as all-NaN on the raw rows, silently
        # producing a fully-NaN X_future (and NaN-crashing sklearn models).
        # The skill reference is computed separately on the governed frame.
        test_gov = test_gov.drop(
            columns=[c for c in _ANCHOR_COLS if c in test_gov.columns])
    eng = ContiguousEngineer(target, feature_cols=feats, dt_seconds=dt)
    cfg = PipelineConfig(dataset=ds, target=target, lookback=lb, horizon=hz,
                         stride=st, feature_cols=feats)
    full = pd.concat([train.assign(split="train"),
                      test_gov.assign(split="test")], ignore_index=True)
    pipe = SolarForecastPipeline(cfg, eng, model, scaler=PhysicalFeatureScaler())
    return pipe.run(df_provider=lambda c: full)


# ---------------------------------------------------------------------------
# CrossViViT arm (image datasets) — ImageWindowEngineer + contiguity
# ---------------------------------------------------------------------------


def run_crossvivit_arm(loader, ds, target, train_frame, test_gov, lb, hz,
                       st, dt, governed: bool):
    """Fit/predict CrossViViT on contiguous+image-covered windows; returns
    (y_true, pred, cz, sp_pred) all restricted to the SAME kept windows."""
    from solarunibench.benchmark.crossvivit_image import (
        ImageWindowEngineer,
        _resolve_sky_cube,
    )

    cube, img_ts = _resolve_sky_cube(loader, ds,
                                     str(_REPO_ROOT / "data" / "standardized"))
    if cube is None or img_ts is None:
        raise RuntimeError(f"no sky cube resolved for {ds}")
    channels = 4 if governed else 3
    feats = list(_E8_FEATURES) if governed else []
    # Raw arm: the TEST frame must not expose anchor columns either — the
    # column-presence asymmetry (raw train has none, governed test has all)
    # would leak them into the engineered test block (Fp mismatch vs train,
    # and physics features in a target-only arm). SP reference below uses a
    # separately built GOVERNED test with the full feature block on the
    # identical windows (same rows/stride/contiguity/kept-mask).
    test_frame_model = test_gov if governed else test_gov.drop(
        columns=[c for c in _ANCHOR_COLS if c in test_gov.columns])
    # E8's exact image regime: 8 frames at 30-step spacing over the LAST
    # ~3.5 h of the lookback (img_tail_frames). Camera uptime is duty-cycled
    # in multi-hour blocks, so a 12-h full-lookback image coverage is
    # structurally impossible on Folsom; the tail regime keeps the image
    # protocol byte-aligned with E8 while the tabular block stays 12 h.
    eng = ImageWindowEngineer(target, cube, img_ts, img_size=32,
                              image_stride=30, img_tail_frames=8,
                              feature_cols=feats)
    eng_gov = ImageWindowEngineer(
        target, cube, img_ts, img_size=32, image_stride=30,
        img_tail_frames=8, feature_cols=list(_E8_FEATURES)) \
        if not governed else eng

    def _build(fe, df, fit: bool):
        out = fe.create_features(df, lb, hz, st)
        keep = _contiguous_keep(df, lb, hz, st, dt) & out["kept"]
        if keep.sum() == 0:
            raise RuntimeError("no contiguous image-covered windows")
        return {k: (v[keep] if hasattr(v, "shape") and v.shape[:1]
                    == out["X_past"].shape[:1] else v)
                for k, v in out.items() if k != "kept"}, int(keep.sum())

    tr, _ = _build(eng, train_frame, fit=True)
    te, _ = _build(eng, test_frame_model, fit=False)
    te_gov, _ = _build(eng_gov, test_gov, fit=False)
    # Window identity must match across the model/governed-SP builds.
    assert len(te["times"]) == len(te_gov["times"]), \
        "image-arm window mismatch between model and SP builds"
    assert (te["times"] == te_gov["times"]).all()
    if channels == 3:
        tr["images"] = tr["images"][:, :, :3]
        te["images"] = te["images"][:, :, :3]

    # SP reference on the SAME windows, governed features (clear-sky
    # scaled) — the raw arm's model never sees these columns.
    sp = SmartPersistenceModel()
    sp.fit(tr["X_past"], tr["Y"])  # fit is a no-op for SP
    sp_pred = sp.predict(te_gov["X_past"], te_gov["times"],
                         te_gov["X_future"])

    # Raw-arm model contract: no future-anchor tokens at fit OR predict
    # (fit-time X_future is None because the raw train frame has no anchors).
    model_xf_tr = tr["X_future"] if governed else None
    model_xf_te = te["X_future"] if governed else None
    # E8 image-variant model preset VERBATIM (run_crossvivit_image FULL:
    # framework defaults d64/h4/depth2, patch 4, img 32/patch 4, 50 epochs,
    # lr 1e-3, wd 1e-4, batch 64 + accum 4, patience 7 / val 0.15).
    params = dict(d_model=64, n_heads=4, depth=2, patch_len=4,
                  img_size=32, img_patch=4, epochs=50, batch_size=64,
                  accum_steps=4, lr=1e-3, weight_decay=1e-4, dropout=0.1,
                  patience=7, val_fraction=0.15, device="cuda",
                  in_channels=channels, img_frame_stride=1)
    model = CrossViViTModel(**params)
    model.fit(tr["X_past"], tr["Y"], tr["times"], model_xf_tr,
              images=tr["images"])
    pred = model.predict(te["X_past"], te["times"], model_xf_te,
                         images=te["images"])
    # Daytime mask from the GOVERNED build (identical windows, asserted
    # above) — the raw-arm test frame has no cos_zenith column.
    return te["Y"], pred, te_gov["cos_zenith_issue"], sp_pred


# ---------------------------------------------------------------------------
# Model pool
# ---------------------------------------------------------------------------
def _model_pool(quick: bool):
    if not quick:
        keep = {"LightGBM", "MLP", "LSTM", "iTransformer", "PatchTST-CD",
                "CrossViViT"}
        return [m for m in _build_full_models("cuda") if m.name in keep]
    dev = "cuda"
    common = dict(epochs=2, patience=2, val_fraction=0.15, device=dev)
    return [
        LightGBMModel(params={"n_estimators": 30, "verbose": -1,
                              "n_jobs": -1, "random_state": 0}),
        MLPModel(hidden_layer_sizes=(32,), max_iter=20, random_state=0),
        LSTMModel(batch_size=64, **common),
        _pool_itrans(common),
        _pool_patchtst_cd(common),
        CrossViViTModel(d_model=32, n_heads=4, depth=1, patch_len=4,
                        img_size=64, img_patch=8, epochs=1, batch_size=16,
                        device=dev, in_channels=4, img_frame_stride=1),
    ]


def _pool_itrans(common):
    from solarunibench.forecast.models.itransformer_model import (
        iTransformerModel,
    )

    return iTransformerModel(batch_size=64, **common)


def _pool_patchtst_cd(common):
    from solarunibench.forecast.models.patchtst_model import PatchTSTModel

    return PatchTSTModel(variant="patchtst_cd", name="PatchTST-CD",
                         batch_size=64, **common)


_POOL_NAMES = ["LightGBM", "MLP", "LSTM", "iTransformer", "PatchTST-CD",
               "CrossViViT"]


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
def run_dataset(ds: str, loader: SolarUniBenchLoader, quick: bool) -> list[dict]:
    cfg = DATASET_CONFIG[ds]
    target = cfg.target
    lb, hz = ROW_PARAMS[ds]
    st = FULL_STRIDE[cfg.sampling_mins]
    dt = cfg.sampling_mins * 60.0
    if quick:
        st = lb + hz  # non-overlapping minimal windows for the dry-run
    print(f"\n=== {ds}: target={target} dt={cfg.sampling_mins}min "
          f"L={lb} H={hz} stride={st} ===", flush=True)

    gov_tr, gov_te = governed_frames(loader, ds, target)
    raw_tr, raw_te = raw_frames(loader, ds, target)
    # SKIPP'D: reconstruct the physically-determined night rows on the full
    # native grid so strict 12-h contiguity is achievable (see helper).
    gov_tr = _expand_night_grid(gov_tr, target, ds, with_anchors=True)
    gov_te = _expand_night_grid(gov_te, target, ds, with_anchors=True)
    raw_tr = _expand_night_grid(raw_tr, target, ds, with_anchors=False)
    raw_te = _expand_night_grid(raw_te, target, ds, with_anchors=False)
    if quick:
        gov_tr = gov_tr.groupby("entity_id", group_keys=False).tail(3000)
        gov_te = gov_te.groupby("entity_id", group_keys=False).tail(1200)
        raw_tr = raw_tr.groupby("entity_id", group_keys=False).tail(3000)

    # Shared governed reference (E8 features) — skill denominator.
    t0 = time.time()
    res_sp = run_tabular_arm(ds, target, gov_tr, gov_te,
                             SmartPersistenceModel(), lb, hz, st, dt, True)
    eng_gov = ContiguousEngineer(target, feature_cols=list(_E8_FEATURES),
                                  dt_seconds=dt)
    _, _, _, _, cz_gov, _ = eng_gov.create_features(gov_te, lb, hz, st)
    assert len(cz_gov) == len(res_sp["y_true"]), "SP window mismatch"
    print(f"  [SP] windows={len(res_sp['y_true'])} ({time.time()-t0:.0f}s)",
          flush=True)
    met_sp = evaluate_predictions(res_sp["y_true"], res_sp["pred"], cz_gov,
                                  res_sp["pred"], threshold=_DAYTIME)

    rows = [_row(ds, "cleaned", "SmartPersistence", met_sp)]
    for arm, tr in (("raw", raw_tr), ("cleaned", gov_tr)):
        for m in _model_pool(quick):
            if m.name == "CrossViViT" and not cfg.image_dataset:
                continue
            t0 = time.time()
            try:
                if m.name == "CrossViViT":
                    y, pred, cz, sp_pred = run_crossvivit_arm(
                        loader, ds, target, tr, gov_te, lb, hz, st, dt,
                        governed=(arm == "cleaned"))
                    met = evaluate_predictions(y, pred, cz, sp_pred,
                                               threshold=_DAYTIME)
                else:
                    model = copy.deepcopy(m)
                    res = run_tabular_arm(ds, target, tr, gov_te, model,
                                          lb, hz, st, dt,
                                          governed=(arm == "cleaned"))
                    assert len(res["y_true"]) == len(res_sp["y_true"]), \
                        f"window mismatch {m.name}/{arm}"
                    met = evaluate_predictions(res_sp["y_true"], res["pred"],
                                               cz_gov, res_sp["pred"],
                                               threshold=_DAYTIME)
            except Exception as exc:                      # noqa: BLE001
                print(f"  [{arm}/{m.name}] FAILED: {exc!r}", flush=True)
                rows.append(_row(ds, arm, m.name, None))
                continue
            rows.append(_row(ds, arm, m.name, met))
            print(f"  [{arm}/{m.name}] MAE={met['MAE']:.5f} "
                  f"SkillMAE={met['Skill_MAE_pct']:+.1f}% "
                  f"({time.time()-t0:.0f}s)", flush=True)
            _free_gpu()
    return rows


def _free_gpu():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _row(ds, arm, model, met):
    return {
        "dataset": ds, "arm": arm, "model": model,
        "MAE": None if met is None else met["MAE"],
        "RMSE": None if met is None else met["RMSE"],
        "Skill_MAE_pct": None if met is None else met["Skill_MAE_pct"],
        "Skill_RMSE_pct": None if met is None else met["Skill_RMSE_pct"],
        "n_daytime": 0 if met is None else met["n_daytime"],
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def compute_chi(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for ds, g in df.groupby("dataset"):
        def sk(model, arm):
            r = g[(g.model == model) & (g.arm == arm)]
            return float(r["Skill_MAE_pct"].iloc[0]) if len(r) else np.nan

        micro = max(sk("LightGBM", "cleaned"), sk("MLP", "cleaned"))
        deep3 = [sk(m, "raw") for m in ("LSTM", "iTransformer", "PatchTST-CD")]
        cx = sk("CrossViViT", "raw")
        deep_full = deep3 + ([cx] if np.isfinite(cx) else [])
        out.append({
            "dataset": ds,
            "micro_skill": micro,
            "chi_mu": micro - float(np.nanmean(deep_full)),
            "chi_best": micro - float(np.nanmax(deep_full)),
            "chi_mu_3": micro - float(np.nanmean(deep3)),
            "chi_best_3": micro - float(np.nanmax(deep3)),
            "deep_pool": "LSTM/iTr/PatchTST-CD"
                         + ("+CrossViViT" if len(deep_full) > 3 else ""),
        })
    return pd.DataFrame(out)


def write_markdown(df: pd.DataFrame, chi: pd.DataFrame) -> None:
    lines = [
        "# E9 contiguous rerun — full matrix", "",
        "Protocol: frozen E8 splits (test_start + val_ratio, stamped in the",
        "standardized packages); L/H per dataset = 12 h / 1 h physical",
        "(folsom/skippd 720/60, sunlake 24/2, gefcom 12/1 steps); windowing",
        "stride per E8 FULL map {1 min: 30, 30 min: 1, 1 h: 1}; strict",
        "per-window contiguity (timestamp span == (L+H-1)*dt); daytime mask",
        "cos_zenith > 0.05; Smart Persistence reference on the governed test.",
        "Arms: raw = target-only (anchors dropped, ffill/0 NaN policy);",
        "governed = E8 features [cos_zenith, ghi_clearsky,",
        "clearness_index_aligned]. CrossViViT: image datasets only, 8 frames",
        "per window (image_stride = L//8), raw arm = RGB 3-channel.", "",
    ]
    for ds, g in df.groupby("dataset"):
        lines += [f"## {ds}", "",
                  "| arm | model | MAE | RMSE | Skill_MAE_% | Skill_RMSE_%"
                  " | n_day |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in g.sort_values(["arm", "model"]).iterrows():
            if r.MAE is None or (isinstance(r.MAE, float) and np.isnan(r.MAE)):
                lines.append(f"| {r.arm} | {r.model} | n/a | n/a | n/a | n/a"
                             f" | 0 |")
            else:
                lines.append(
                    f"| {r.arm} | {r.model} | {r.MAE:.5f} | {r.RMSE:.5f} | "
                    f"{r.Skill_MAE_pct:+.1f} | {r.Skill_RMSE_pct:+.1f} | "
                    f"{int(r.n_daytime)} |")
        lines.append("")
    lines += ["## Crossover margins (skill %)", "",
              "| dataset | micro | chi_mu | chi_best | chi_mu_3 | chi_best_3"
              " | pool |",
              "|---|---|---|---|---|---|---|"]
    for _, r in chi.iterrows():
        lines.append(
            f"| {r.dataset} | {r.micro_skill:+.1f} | {r.chi_mu:+.1f} | "
            f"{r.chi_best:+.1f} | {r.chi_mu_3:+.1f} | {r.chi_best_3:+.1f} | "
            f"{r.deep_pool} |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    quick = "--quick" in sys.argv
    datasets = argv or ["gefcom2014", "sunlake", "skippd", "folsom"]
    loader = SolarUniBenchLoader(str(_REPO_ROOT / "data" / "standardized"))
    all_rows: list[dict] = []
    for ds in datasets:
        all_rows.extend(run_dataset(ds, loader, quick))

    df = pd.DataFrame(all_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_CELLS.exists() and not quick:
        old = pd.read_csv(OUT_CELLS)
        old = old[~old["dataset"].isin(datasets)]
        df = pd.concat([old, df], ignore_index=True)
    out_path = OUT_CELLS if not quick else OUT_DIR / "results_e9_contiguous_quick.csv"
    df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path} ({len(df)} rows)")
    if not quick:
        chi = compute_chi(df)
        chi.to_csv(OUT_CHI, index=False)
        write_markdown(df, chi)
        print(f"wrote {OUT_CHI} + {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
