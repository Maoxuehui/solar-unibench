"""Convergence diagnostic: is 50 epochs enough for the torch baselines?

Fits PatchTST / iTransformer / DLinear on a fixed cell (GEFCom h=6, where models
have signal) at epoch counts [10, 25, 50, 100, 200] and prints train+val MAE +
fit time. If val MAE keeps falling past 50, the FULL config is undertrained and
the epoch count should rise; if it plateaus by 50, 50 is sufficient.

Run: python scripts/convergence_check.py
"""
from __future__ import annotations
import sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solarunibench.loader import SolarUniBenchLoader
from solarunibench.forecast.core import EndogenousFeatureEngineer
from solarunibench.forecast.models import PatchTSTModel, iTransformerModel, DLinearModel
from solarunibench.experiments._config import MAX_ROWS_PER_ENTITY
from solarunibench.experiments.e8_benchmark import _cap_rows_per_entity, _usable_features


def _mae(p, y):
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(y))))


def build(dataset, target, horizon=6, lookback=12, stride=12):
    from solarunibench.forecast.scaling import PhysicalFeatureScaler
    loader = SolarUniBenchLoader("D:/monica/workspace/solar-unibench/data/standardized")
    df = loader.load_tabular(dataset)
    df = df.dropna(subset=[target]).reset_index(drop=True)
    df = _cap_rows_per_entity(df, MAX_ROWS_PER_ENTITY)
    feats = _usable_features(df)
    # Faithful to E8: split temporally, fit PhysicalFeatureScaler on TRAIN ONLY,
    # transform both, then build windows (zero leakage).
    df = df.sort_values("timestamp_utc", kind="mergesort").reset_index(drop=True)
    n = len(df)
    ntr = int(n * 0.70)
    train_df, val_df = df.iloc[:ntr], df.iloc[ntr:]
    scaler = PhysicalFeatureScaler().fit(train_df)
    train_df = scaler.transform(train_df)
    val_df = scaler.transform(val_df)
    eng = EndogenousFeatureEngineer(target, feature_cols=feats)
    Xtr, Ytr, *_ = eng.create_features(train_df, lookback, horizon, stride)
    Xva, Yva, *_ = eng.create_features(val_df, lookback, horizon, stride)
    return Xtr, Ytr, Xva, Yva


def run(label, factory, epochs_list, Xtr, Ytr, Xva, Yva):
    print(f"\n{label}:", flush=True)
    base = None
    for ep in epochs_list:
        m = factory(ep)
        t0 = time.time()
        m.fit(Xtr, Ytr)
        dt = time.time() - t0
        tr = _mae(m.predict(Xtr), Ytr)
        va = _mae(m.predict(Xva), Yva)
        if base is None:
            base = va
        print(f"  epochs={ep:>4}: train MAE={tr:.4f}  val MAE={va:.4f}  "
              f"(delta_vs_first_ep {'baseline' if ep==epochs_list[0] else f'{va-base:+.4f}'})  "
              f"[{dt:.1f}s]", flush=True)


def main():
    Xtr, Ytr, Xva, Yva = build("gefcom2014", "pv_power_normalized", horizon=6)
    print(f"GEFCom h=6: train={len(Ytr)} val={len(Yva)} windows, Fp={Xtr.shape[2]}",
          flush=True)
    eps = [10, 25, 50, 100, 200]
    run("PatchTST (CI, d=128)",
        lambda ep: PatchTSTModel(d_model=128, n_heads=8, patch_len=4, stride=2, depth=3,
                                 lr=1e-4, epochs=ep, batch_size=256, device="cuda"),
        eps, Xtr, Ytr, Xva, Yva)
    run("iTransformer (d=128)",
        lambda ep: iTransformerModel(d_model=128, n_heads=8, depth=3, lr=1e-3,
                                     epochs=ep, batch_size=256, device="cuda"),
        eps, Xtr, Ytr, Xva, Yva)
    run("DLinear (linear baseline — should converge fast)",
        lambda ep: DLinearModel(kernel_size=25, lr=1e-3, epochs=ep, batch_size=256,
                                device="cuda"),
        [10, 25, 50], Xtr, Ytr, Xva, Yva)


if __name__ == "__main__":
    main()
