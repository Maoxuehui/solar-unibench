"""SolarUniBenchDataset: sliding-window torch Dataset over standardized packages, modality auto-switch."""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .loader import SolarUniBenchLoader

DEFAULT_FEATURES = ["ghi_clearsky", "cos_zenith", "clearness_index"]


class SolarUniBenchDataset(Dataset):
    def __init__(self, loader: SolarUniBenchLoader, datasets, split="train",
                 target="pv_power_normalized", feature_cols=None, lookback=4, horizon=2,
                 stride=1, return_images=False):
        self.loader = loader
        self.target = target
        self.lookback = lookback
        self.horizon = horizon
        self.stride = stride
        self.return_images = return_images
        self.feat_cols = list(feature_cols) if feature_cols else list(DEFAULT_FEATURES)
        # all feature columns used: target lag first, then feat_cols
        self._all_feat = [target] + [c for c in self.feat_cols if c != target]
        self._samples = []  # (dataset, entity_id, df_index_slice_start)
        self._frames = {}   # dataset -> {entity_id -> (timestamps ndarray, feat ndarray, target ndarray)}
        self._img_index = {}  # dataset -> {iso_ts -> image_row} or None
        self._img_cube = {}
        for ds in datasets:
            df = loader.load_tabular(ds, split=split)
            if df.empty:
                continue
            ts_col = "timestamp_utc"
            img_ts = loader.get_image_timestamps(ds) if return_images else None
            cube = loader.get_tensor_cube(ds, "sky_images") if return_images else None
            self._img_cube[ds] = cube
            self._img_index[ds] = ({str(t): i for i, t in enumerate(np.asarray(img_ts))}
                                   if img_ts is not None else None)
            fmap = {}
            for eid, g in df.groupby("entity_id", sort=True):
                g = g.sort_values(ts_col).reset_index(drop=True)
                feat = g[self._all_feat].to_numpy(dtype=np.float32) if all(
                    c in g.columns for c in self._all_feat) else None
                tgt = g[self.target].to_numpy(dtype=np.float32) if self.target in g.columns else None
                tss = g[ts_col].dt.strftime("%Y-%m-%dT%H:%M:%S%z").to_numpy()
                fmap[eid] = (tss, feat, tgt)
                n = len(g)
                for start in range(0, n - lookback - horizon + 1, stride):
                    self._samples.append((ds, eid, start))
            self._frames[ds] = fmap

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, i):
        ds, eid, start = self._samples[i]
        tss, feat, tgt = self._frames[ds][eid]
        lb = slice(start, start + self.lookback)
        hb = slice(start + self.lookback, start + self.lookback + self.horizon)
        sample = {
            "features": torch.from_numpy(feat[lb]),
            "target": torch.from_numpy(tgt[hb]),
            "dataset": ds,
            "image": None,
        }
        if self.return_images and self._img_index[ds] is not None:
            cube = self._img_cube[ds]
            idx = self._img_index[ds]
            rows = [idx.get(str(t)) for t in tss[lb]]
            if all(r is not None for r in rows) and len(rows) == self.lookback:
                sample["image"] = torch.from_numpy(np.stack([cube[r] for r in rows]).astype(np.float32))
        return sample
