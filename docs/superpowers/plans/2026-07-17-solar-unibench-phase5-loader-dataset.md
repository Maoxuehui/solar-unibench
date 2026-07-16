# Solar-UniBench Phase 5: Loader + PyTorch Dataset — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the downstream `SolarUniBenchLoader` (read standardized parquet + lazy zarr, filter by frozen split) and a `SolarUniBenchDataset` (torch Dataset) that windows the tabular series into (lookback features → horizon target) samples with per-dataset modality auto-switch (attach sky images where present), so a DataLoader can stream any dataset uniformly.

**Architecture:** The loader is a thin, dependency-light reader over the Phase 1–4 standardized packages (`metadata.parquet` + `time_series.parquet`[+`split`] + optional `sky_images.zarr`/`satellite_future.zarr`). The Dataset precomputes sliding-window sample indices per (dataset, entity) and yields tabular feature/target tensors plus, best-effort, the sky image(s) aligned by timestamp (via the `images_timestamps` index). Missing modalities → `None` (auto-switch).

**Tech Stack:** Python 3.10 (`solar_uni`), pandas, pyarrow, zarr, torch. Windows 11 / Git Bash.

## Global Constraints

- **Env:** conda env **`solar_uni`** (3.10.20) via `conda run -n solar_uni ...`. torch (cu124) installed. **Never use multi-line `python -c` with `conda run`** — write scripts to files.
- **Branch:** `feat/phase5-loader-dataset` (created from master). Commit per task.
- **TDD:** synthetic-fixture unit tests (build a tiny standardized package via the existing adapters in `tmp_path`, then load/iterate); real-data iteration is `RUN_SLOW`-gated.
- **Read by name:** the loader reads fixed contract column names and fixed zarr array names (`images`, `images_timestamps`, `context`, `future`, `uncertainty_map`); it does not hardcode dataset internals.
- **Honest absence → modality switch:** if `sky_images.zarr` (or `satellite_future.zarr`) is absent for a dataset, `get_tensor_cube` returns `None` and the Dataset yields `image=None`. No fabrication.
- **Splits:** `load_tabular(dataset, split)` filters on the `split` column stamped by `build_splits` (train/val/test). If `split` is None or the column is absent, return all rows.
- **UTC:** all joins/alignments are on UTC timestamps.

---

## File Structure

```
solar-unibench/
├── solarunibench/loader.py            # Task 1 — SolarUniBenchLoader
├── solarunibench/dataset.py           # Task 2 — SolarUniBenchDataset
└── tests/
    ├── test_loader.py                 # Task 1
    ├── test_dataset.py                # Task 2
    └── test_loader_integration.py     # Task 3 (slow)
```

---

### Task 1: SolarUniBenchLoader

**Files:**
- Create: `solarunibench/loader.py`, `tests/test_loader.py`

**Interfaces:**
- Produces: `SolarUniBenchLoader(data_root)` with:
  - `load_tabular(dataset: str, split: str | None = None) -> pd.DataFrame` — merges `metadata.parquet` + `time_series.parquet` on `entity_id`, filters by the `split` column when given; returns the merged frame (contract cols + `nwp_*`/`obs_*` extras + `split` if present).
  - `get_tensor_cube(dataset: str, kind: str) -> "zarr.Array | None"` — `kind ∈ {"sky_images","satellite_context"}`; lazy-opens `<dataset>/sky_images.zarr["images"]` or `<dataset>/satellite_future.zarr["context"]`; returns `None` if the file is absent.
  - `get_image_timestamps(dataset: str) -> np.ndarray | None` — reads `sky_images.zarr["images_timestamps"]` (for alignment) or `None`.

- [ ] **Step 1: Write the failing test `tests/test_loader.py`**

```python
import numpy as np
import pandas as pd
import zarr
from pathlib import Path
from solarunibench.loader import SolarUniBenchLoader


def _build_pkg(root: Path, dataset="toy", n=8, with_split=True, with_sky=False):
    d = root / dataset
    d.mkdir(parents=True)
    times = pd.date_range("2020-06-01 06:00", periods=n, freq="1h", tz="UTC")
    meta = pd.DataFrame([{"entity_id": "E1", "dataset_source": dataset, "latitude": 38.0,
                          "longitude": -121.0, "elevation_meters": 50.0, "local_timezone": "UTC",
                          "surface_albedo": 0.2, "capacity_kw": np.nan, "slope": np.nan,
                          "azimuth": np.nan, "geo_inferred": False}])
    meta.to_parquet(d / "metadata.parquet", index=False)
    ts = pd.DataFrame({"timestamp_utc": times, "entity_id": "E1",
                       "ghi_actual": np.linspace(0, 800, n), "dni_actual": np.nan, "dhi_actual": np.nan,
                       "pv_power_normalized": np.nan, "ghi_clearsky": np.linspace(0, 900, n),
                       "dni_clearsky": np.nan, "dhi_clearsky": np.nan, "cos_zenith": np.linspace(0, 0.9, n),
                       "solar_azimuth": 90.0, "clearness_index": np.linspace(0, 0.9, n), "quality_flag": 0})
    if with_split:
        ts["split"] = ["train"] * (n - 2) + ["val", "test"]
    ts.to_parquet(d / "time_series.parquet", index=False)
    if with_sky:
        store = zarr.DirectoryStore(str(d / "sky_images.zarr"))
        g = zarr.group(store=store, overwrite=True)
        g.array("images", data=np.zeros((n, 4, 256, 256), np.float32), chunks=(1, 4, 256, 256), overwrite=True)
        g.array("images_timestamps", data=np.array(times.strftime("%Y-%m-%dT%H:%M:%S%z")), chunks=(1,), overwrite=True)


def test_load_tabular_merges_metadata_and_filters_split(tmp_path):
    _build_pkg(tmp_path, with_split=True)
    ld = SolarUniBenchLoader(str(tmp_path))
    allrows = ld.load_tabular("toy")
    assert len(allrows) == 8
    assert "latitude" in allrows.columns  # metadata merged in
    train = ld.load_tabular("toy", split="train")
    assert (train["split"] == "train").all() and len(train) == 6
    test = ld.load_tabular("toy", split="test")
    assert len(test) == 1


def test_get_tensor_cube_absent_returns_none(tmp_path):
    _build_pkg(tmp_path, with_sky=False)
    ld = SolarUniBenchLoader(str(tmp_path))
    assert ld.get_tensor_cube("toy", "sky_images") is None
    assert ld.get_image_timestamps("toy") is None


def test_get_tensor_cube_present(tmp_path):
    _build_pkg(tmp_path, with_sky=True)
    ld = SolarUniBenchLoader(str(tmp_path))
    cube = ld.get_tensor_cube("toy", "sky_images")
    assert cube is not None and cube.shape == (8, 4, 256, 256)
    ts = ld.get_image_timestamps("toy")
    assert ts is not None and len(ts) == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n solar_uni python -m pytest tests/test_loader.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `solarunibench/loader.py`**

```python
"""SolarUniBenchLoader: read standardized parquet + lazy zarr, filter by frozen split."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import zarr

_SKY = "sky_images.zarr"
_SAT = "satellite_future.zarr"


class SolarUniBenchLoader:
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)

    def _path(self, dataset: str) -> Path:
        return self.data_root / dataset

    def load_tabular(self, dataset: str, split: str | None = None) -> pd.DataFrame:
        d = self._path(dataset)
        meta = pd.read_parquet(d / "metadata.parquet")
        ts = pd.read_parquet(d / "time_series.parquet")
        df = ts.merge(meta, on="entity_id", how="left", suffixes=("", "_meta"))
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)
        return df

    def _open(self, dataset: str, file: str):
        p = self._path(dataset) / file
        if not p.exists():
            return None
        return zarr.open(str(p), mode="r")

    def get_tensor_cube(self, dataset: str, kind: str):
        if kind == "sky_images":
            g = self._open(dataset, _SKY)
            return g["images"] if g is not None else None
        if kind == "satellite_context":
            g = self._open(dataset, _SAT)
            return g["context"] if g is not None else None
        raise ValueError(f"unknown tensor kind: {kind}")

    def get_image_timestamps(self, dataset: str):
        g = self._open(dataset, _SKY)
        if g is None or "images_timestamps" not in g:
            return None
        return np.asarray(g["images_timestamps"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n solar_uni python -m pytest tests/test_loader.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add solarunibench/loader.py tests/test_loader.py
git commit -m "feat(loader): SolarUniBenchLoader (tabular + lazy zarr, split filter)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: SolarUniBenchDataset

**Files:**
- Create: `solarunibench/dataset.py`, `tests/test_dataset.py`

**Interfaces:**
- Consumes: `SolarUniBenchLoader`.
- Produces: `SolarUniBenchDataset(torch.utils.data.Dataset)` with:
  - `__init__(self, loader, datasets, split="train", target="pv_power_normalized", feature_cols=None, lookback=4, horizon=2, stride=1, return_images=False)`.
  - `feature_cols` default: `["ghi_clearsky", "cos_zenith", "clearness_index"]` (present in all datasets) plus the `target` is prepended as a lag feature.
  - `__len__` = total sliding-window samples across all (dataset, entity).
  - `__getitem__(i) -> dict`: `{"features": (lookback, F) float32, "target": (horizon,) float32, "dataset": str, "image": (lookback, 4, 256, 256) float32 | None}`. Image attached (best-effort, exact-timestamp match via `images_timestamps`) only when `return_images` and the dataset has `sky_images.zarr`; else `None`.

- [ ] **Step 1: Write the failing test `tests/test_dataset.py`**

```python
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from solarunibench.loader import SolarUniBenchLoader
from solarunibench.dataset import SolarUniBenchDataset
from tests.test_loader import _build_pkg


def test_dataset_windows_tabular_and_counts(tmp_path):
    _build_pkg(tmp_path, dataset="toy", n=10, with_split=True)
    ld = SolarUniBenchLoader(str(tmp_path))
    ds = SolarUniBenchDataset(ld, ["toy"], split="train", target="ghi_actual",
                              feature_cols=["ghi_clearsky", "cos_zenith"], lookback=3, horizon=2, stride=1)
    # train split has 8 rows; samples = 8 - lookback - horizon + 1 = 4
    assert len(ds) == 4
    s = ds[0]
    assert isinstance(s["features"], torch.Tensor) and s["features"].shape == (3, 3)  # 2 feats + target lag
    assert s["target"].shape == (2,)
    assert s["dataset"] == "toy"
    assert s["image"] is None  # no sky_images


def test_dataset_attaches_image_when_present(tmp_path):
    _build_pkg(tmp_path, dataset="toy", n=10, with_split=True, with_sky=True)
    ld = SolarUniBenchLoader(str(tmp_path))
    ds = SolarUniBenchDataset(ld, ["toy"], split="train", target="ghi_actual",
                              feature_cols=["ghi_clearsky"], lookback=3, horizon=2, return_images=True)
    s = ds[0]
    assert s["image"] is not None and s["image"].shape == (3, 4, 256, 256)


def test_dataset_mixes_multiple_datasets(tmp_path):
    _build_pkg(tmp_path, dataset="a", n=10)
    _build_pkg(tmp_path, dataset="b", n=10)
    ld = SolarUniBenchLoader(str(tmp_path))
    ds = SolarUniBenchDataset(ld, ["a", "b"], split=None, target="ghi_actual",
                              feature_cols=["ghi_clearsky"], lookback=3, horizon=2)
    assert len(ds) == 2 * (10 - 3 - 2 + 1)
    seen = {ds[i]["dataset"] for i in range(len(ds))}
    assert seen == {"a", "b"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n solar_uni python -m pytest tests/test_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `solarunibench/dataset.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n solar_uni python -m pytest tests/test_dataset.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add solarunibench/dataset.py tests/test_dataset.py
git commit -m "feat(dataset): SolarUniBenchDataset (windowing + modality auto-switch)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Loader/Dataset integration on real standardized packages

**Files:**
- Create: `tests/test_loader_integration.py` (RUN_SLOW-gated)

**Interfaces:**
- Builds small REAL standardized packages (via the adapters + `build_splits`), then exercises the loader + Dataset end-to-end.

- [ ] **Step 1: Write `tests/test_loader_integration.py`**

```python
"""Real-data integration: build small standardized packages, load + iterate the Dataset. Slow."""
import os
import subprocess
import sys
import pytest
import pandas as pd
from pathlib import Path

DATA = os.environ.get("SOLAR_UNIBENCH_DATA_ROOT", "D:/monica/workspace/solar_benchmark_dataset")
slow = pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="set RUN_SLOW=1")


@slow
def test_gefcom_load_and_iterate(tmp_path):
    from solarunibench.pipeline.adapters.gefcom import GEFCom2014Pipeline
    from solarunibench.splits import build
    from solarunibench.loader import SolarUniBenchLoader
    from solarunibench.dataset import SolarUniBenchDataset
    import yaml
    GEFCom2014Pipeline(raw_data_dir=f"{DATA}/GEFCom2014_data", output_dir=str(tmp_path),
                       dataset_name="gefcom2014").run_pipeline(limit=2000)
    cfg = {"strategy": "temporal", "test_start": "2014-06-01", "val_ratio": 0.1}
    build(tmp_path / "gefcom2014", cfg, "1.0")
    ld = SolarUniBenchLoader(str(tmp_path))
    df = ld.load_tabular("gefcom2014", split="train")
    assert len(df) > 0 and (df["split"] == "train").all()
    ds = SolarUniBenchDataset(ld, ["gefcom2014"], split="train", target="pv_power_normalized",
                              lookback=4, horizon=2)
    assert len(ds) > 0
    s = ds[0]
    assert s["features"].shape[0] == 4 and s["target"].shape[0] == 2
    assert s["image"] is None  # GEFCom has no sky images


@slow
def test_skippd_dataset_with_sky_images(tmp_path):
    from solarunibench.pipeline.adapters.skippd import SKIPPDPipeline
    from solarunibench.loader import SolarUniBenchLoader
    from solarunibench.dataset import SolarUniBenchDataset
    SKIPPDPipeline(raw_data_dir=f"{DATA}/SKIPPD_all", output_dir=str(tmp_path),
                   dataset_name="skippd", config={"image_limit": 80}).run_pipeline(limit=2000)
    ld = SolarUniBenchLoader(str(tmp_path))
    # sky_images.zarr should exist (image_limit>0)
    assert ld.get_tensor_cube("skippd", "sky_images") is not None
    ds = SolarUniBenchDataset(ld, ["skippd"], split=None, target="pv_power_normalized",
                              lookback=3, horizon=2, return_images=True)
    assert len(ds) > 0
    n_with_img = sum(1 for i in range(min(len(ds), 50)) if ds[i]["image"] is not None)
    assert n_with_img > 0  # at least some samples attach sky images
```

- [ ] **Step 2: Verify** — `conda run -n solar_uni python -m pytest -q` (fast green, slow skipped). Then `RUN_SLOW=1 conda run -n solar_uni python -m pytest tests/test_loader_integration.py -v` → both PASS. Report sample counts + whether SKIPPD images attached.

- [ ] **Step 3: Commit**

```bash
git add tests/test_loader_integration.py
git commit -m "test(loader): real-data integration (GEFCom tabular + SKIPPD sky images)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Definition of Done (Phase 5)

- `conda run -n solar_uni python -m pytest -q` → all fast unit tests green.
- `SolarUniBenchLoader.load_tabular(ds, split)` merges metadata + filters split; `get_tensor_cube` returns `None` for absent modalities.
- `SolarUniBenchDataset` windows tabular into `(lookback,F)` features + `(horizon,)` target, mixes multiple datasets, and attaches sky images (best-effort) where present.
- `RUN_SLOW=1` integration: GEFCom load+iterate (tabular, no image) and SKIPPD iterate with sky images both pass.

## Notes / out of scope
- Satellite (`satellite_future.zarr`) cube access is exposed via `get_tensor_cube(ds, "satellite_context")` but not yet wired into the Dataset's `__getitem__` (Phase 6 forecast models can consume it). SunLake satellite image attachment is a Phase 6+ concern.
- Cross-target mixing (PV vs GHI datasets in one batch) is not unified — instantiate the Dataset per target type. A normalized multi-target variant is a Phase 6+ decision.
- The DataLoader collate needs a custom collate_fn if `image` can be `None` for some samples (default collate handles dicts of tensors + `None` is fine if all-or-none per dataset; mixing within a batch needs a custom collate). Note for Phase 6.
