"""SolarUniBenchDataset: sliding-window torch Dataset over standardized packages, modality auto-switch."""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .loader import SolarUniBenchLoader

DEFAULT_FEATURES = ["ghi_clearsky", "cos_zenith", "clearness_index"]
# Canonical timestamp string used to key the image index AND to format the TS
# column. Both sides are routed through pd.Timestamp(...).strftime(TS_FMT) so a
# match is robust to the adapter's "+00:00" (colon) ISO form vs strftime's
# "+0000" (no colon) form, datetime64, or string inputs.
TS_FMT = "%Y-%m-%dT%H:%M:%S%z"


class SolarUniBenchDataset(Dataset):
    def __init__(self, loader: SolarUniBenchLoader, datasets, split="train",
                 target="pv_power_normalized", feature_cols=None, lookback=4, horizon=2,
                 stride=1, return_images=False, match_tolerance=None):
        self.loader = loader
        self.target = target
        self.lookback = lookback
        self.horizon = horizon
        self.stride = stride
        self.return_images = return_images
        # None = exact-timestamp match only (backward compatible). When set, an
        # exact miss falls back to the nearest image timestamp within this
        # tolerance (Folsom: second-cadence images vs hourly tabular). Routed
        # through pd.Timedelta so callers may pass a string ("30min") too.
        self._match_tolerance = (pd.Timedelta(match_tolerance)
                                 if match_tolerance is not None else None)
        self.feat_cols = list(feature_cols) if feature_cols else list(DEFAULT_FEATURES)
        # all feature columns used: target lag first, then feat_cols
        self._all_feat = [target] + [c for c in self.feat_cols if c != target]
        self._samples = []  # (dataset, entity_id, df_index_slice_start)
        self._frames = {}   # dataset -> {entity_id -> (timestamps ndarray, feat ndarray, target ndarray)}
        self._img_index = {}  # dataset -> {iso_ts -> image_row} or None
        self._img_cube = {}
        # dataset -> sorted pd.DatetimeIndex of image timestamps (built ONCE per
        # dataset in __init__, not per-sample) for vectorized nearest-neighbor.
        self._img_ts_sorted = {}
        for ds in datasets:
            df = loader.load_tabular(ds, split=split)
            if df.empty:
                continue
            missing = [c for c in self._all_feat if c not in df.columns]
            if missing:
                raise ValueError(
                    f"[{ds}] cannot window: missing required column(s) {missing}. "
                    f"target='{self.target}', features={self._all_feat}.")
            ts_col = "timestamp_utc"
            img_ts = loader.get_image_timestamps(ds) if return_images else None
            cube = loader.get_tensor_cube(ds, "sky_images") if return_images else None
            self._img_cube[ds] = cube
            self._img_index[ds] = (
                {pd.Timestamp(str(t)).strftime(TS_FMT): i for i, t in enumerate(np.asarray(img_ts))}
                if img_ts is not None else None)
            # Build the sorted DatetimeIndex once for the nearest-neighbor fallback.
            # ``get_indexer(method='nearest')`` needs a monotonic index, so we sort
            # (stable) and keep the argsort permutation to map back to cube rows.
            # Built ONCE per dataset here, not per-sample.
            if img_ts is not None and self._match_tolerance is not None:
                decorated = sorted(
                    ((i, pd.Timestamp(str(t))) for i, t in enumerate(np.asarray(img_ts))),
                    key=lambda x: x[1])
                order = np.array([i for i, _ in decorated], dtype=np.intp)
                di = pd.DatetimeIndex([t for _, t in decorated])
                self._img_ts_sorted[ds] = (di, order)
            else:
                self._img_ts_sorted[ds] = None
            fmap = {}
            for eid, g in df.groupby("entity_id", sort=True):
                g = g.sort_values(ts_col).reset_index(drop=True)
                feat = g[self._all_feat].to_numpy(dtype=np.float32)
                tgt = g[self.target].to_numpy(dtype=np.float32)
                tss = g[ts_col].dt.strftime(TS_FMT).to_numpy()
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
        if (self.return_images and self._img_index[ds] is not None
                and self._img_cube[ds] is not None):
            cube = self._img_cube[ds]
            idx = self._img_index[ds]
            rows = [idx.get(str(t)) for t in tss[lb]]
            # Exact match preferred. Only on a miss, and only when a tolerance is
            # configured, fall back to the nearest image timestamp within tol.
            if not (all(r is not None for r in rows) and len(rows) == self.lookback):
                rows = self._nearest_rows(ds, tss[lb])
            if rows is not None and len(rows) == self.lookback:
                from .imaging.load import load_sky_frame
                dec = np.stack([load_sky_frame(cube, r) for r in rows])
                sample["image"] = torch.from_numpy(dec.astype(np.float32))
        return sample

    def _nearest_rows(self, ds, lb_tss):
        """Vectorized nearest-neighbor image indices for ``lb_tss`` within tolerance.

        Returns a list of int cube positions (one per lookback timestamp) or None
        if tolerance is unset, no sorted index exists, or any timestamp's nearest
        image falls outside the tolerance. Uses ``DatetimeIndex.get_indexer`` so
        the whole lookback window is resolved in one vectorized call; results are
        remapped through the argsort permutation to original cube rows.
        """
        tol = self._match_tolerance
        packed = self._img_ts_sorted.get(ds)
        if tol is None or packed is None or len(lb_tss) == 0:
            return None
        sorted_idx, order = packed
        targets = pd.DatetimeIndex([pd.Timestamp(str(t)) for t in lb_tss])
        positions = sorted_idx.get_indexer(targets, method="nearest", tolerance=tol)
        if (positions < 0).any():
            return None
        return list(order[positions])
