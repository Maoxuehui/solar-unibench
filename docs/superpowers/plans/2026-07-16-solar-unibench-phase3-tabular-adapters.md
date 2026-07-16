# Solar-UniBench Phase 3: Tabular Adapters (Folsom, SKIPPD, SunLake) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the three remaining tabular adapters (Folsom, SKIPPD, SunLake) on the existing `BaseSolarPipeline`, each producing a contract-conformant `metadata.parquet` + `time_series.parquet`, and wire them into the `build_dataset` CLI. SunLake's tabular data is pulled from the Activeloop hub first.

**Architecture:** Identical to the Phase 1+2 GEFCom adapter pattern: each adapter subclasses `BaseSolarPipeline`, implements `load_and_standardize_raw() -> (df_meta, df_ts)` (parse raw → UTC timestamps → contract columns), and the inherited `run_pipeline()` adds clear-sky anchors + quality audit + writes parquet. **No geo-recovery needed** — Folsom/SKIPPD/SunLake have known public coordinates (`geo_inferred=False`), unlike GEFCom.

**Tech Stack:** Python 3.10 (`solar_uni` env), pandas, pyarrow, pvlib (via base), deeplake (SunLake hub pull). Windows 11 / Git Bash.

## Global Constraints

- **Env:** conda env **`solar_uni`** (Python 3.10.20); all commands via `conda run -n solar_uni ...`. **Never use multi-line `python -c` with `conda run`** (conda rejects newlines in args) — write scripts to files instead. `deeplake` 3.9.52 is installed (needed for SunLake).
- **Branch:** `feat/phase3-tabular-adapters` (already created from master). Commit per task.
- **TDD:** synthetic-fixture unit tests; real-data runs are `RUN_SLOW`-gated integration tests, separate from `pytest`.
- **UTC at ingest:** every timestamp normalized to `timestamp_utc` before leaving an adapter. Folsom `timeStamp` = **UTC-naive** (tz_localize UTC). SKIPPD `time` = **US/Pacific tz-aware** (tz_convert UTC). SunLake `time_utc` = **UTC** already.
- **Honest NaN:** Folsom/SunLake have **no PV** (`pv_power_normalized` NaN); SKIPPD has **no ground irradiance** (`ghi_actual` NaN; ERA5 `ssrd` is NWP → `nwp_*`). The Phase 1+2 quality-audit guard ensures missing modalities stay NaN.
- **Naming:** contract columns from `contract.py`; dataset NWP columns namespaced `nwp_*` (extras allowed by the validator).
- **Coords are known** (geo_inferred=False): Folsom 38.67/−121.15/elev 100/tz America/Los_Angeles; SKIPPD 37.42/−122.17/tz America/Los_Angeles, capacity_kw=30.1; SunLake per-station (from `tensor_info.json`, see Task 3).
- **SunLake data:** pull tabular PCCI (GHI) only from `hub://crossvivit/SunLake` (public, read-only) — NOT the satellite imagery (that's Phase 4).

---

## File Structure

```
solar-unibench/
├── solarunibench/pipeline/adapters/
│   ├── folsom.py          # Task 1
│   ├── skippd.py          # Task 2
│   └── sunlake.py         # Task 4
├── solarunibench/dataset_meta.py   # Task 1 — known per-dataset station metadata (coords/capacity)
├── scripts/
│   ├── pull_sunlake_tabular.py     # Task 3 — hub → local parquet
│   └── build_dataset.py            # Task 5 — extend ADAPTERS + RAW_SUBDIRS
└── tests/
    ├── test_folsom_adapter.py      # Task 1
    ├── test_skippd_adapter.py      # Task 2
    ├── test_sunlake_pull.py        # Task 3
    ├── test_sunlake_adapter.py     # Task 4
    ├── test_folsom_contract.py     # Task 6
    ├── test_skippd_contract.py     # Task 6
    ├── test_sunlake_contract.py    # Task 6
    ├── test_folsom_integration.py  # Task 6 (slow)
    ├── test_skippd_integration.py  # Task 6 (slow)
    └── test_sunlake_integration.py # Task 6 (slow)
```

---

### Task 1: Folsom adapter + shared dataset metadata

**Files:**
- Create: `solarunibench/dataset_meta.py`, `solarunibench/pipeline/adapters/folsom.py`, `tests/test_folsom_adapter.py`

**Interfaces:**
- Consumes: `BaseSolarPipeline` (Phase 1+2), `contract`.
- Produces: `FolsomPipeline(BaseSolarPipeline)`; `dataset_meta.FOLSOM` dict.

- [ ] **Step 1: Write `solarunibench/dataset_meta.py`** (shared known-metadata, used by all three adapters)

```python
"""Known per-dataset station metadata (coords are public; geo_inferred=False)."""

FOLSOM = {
    "entity_id": "FOLSOM_STN",
    "dataset_source": "folsom",
    "latitude": 38.67,
    "longitude": -121.15,
    "elevation_meters": 100.0,
    "local_timezone": "America/Los_Angeles",
    "surface_albedo": 0.2,
    "capacity_kw": float("nan"),  # Folsom is irradiance-only
    "geo_inferred": False,
}

SKIPP_CAPACITY_KW = 30.1  # SKIPPD residential array rated power (pv peaks ~29.6 raw)
SKIPP = {
    "entity_id": "SKIPPD_STFORD",
    "dataset_source": "skippd",
    "latitude": 37.42,
    "longitude": -122.17,
    "elevation_meters": 30.0,
    "local_timezone": "America/Los_Angeles",
    "surface_albedo": 0.2,
    "capacity_kw": SKIPP_CAPACITY_KW,
    "geo_inferred": False,
}
```

- [ ] **Step 2: Write the failing test `tests/test_folsom_adapter.py`**

```python
import pandas as pd
from pathlib import Path
from solarunibench.pipeline.adapters.folsom import FolsomPipeline


def _write_folsom_fixture(root: Path):
    d = root / "Folsom_data"
    d.mkdir(parents=True)
    ts = pd.date_range("2014-06-01 14:00", periods=6, freq="1min")  # UTC-naive
    stamp = ts.strftime("%Y-%m-%d %H:%M:%S")
    pd.DataFrame({"timeStamp": stamp, "ghi": [0, 100, 600, 900, 300, 0],
                  "dni": [0, 50, 500, 800, 200, 0], "dhi": [0, 60, 120, 150, 100, 0]}
                 ).to_csv(d / "Folsom_irradiance.csv", index=False)
    pd.DataFrame({"timeStamp": stamp, "air_temp": [10, 12, 15, 18, 16, 11],
                  "relhum": [60, 58, 55, 50, 52, 59], "press": [1010, 1010, 1011, 1011, 1010, 1010],
                  "windsp": [1.5, 1.8, 2.0, 2.2, 1.9, 1.6], "winddir": [10, 20, 30, 40, 30, 20],
                  "max_windsp": [3, 3, 4, 4, 3, 3], "precipitation": [0, 0, 0, 0, 0, 0]}
                 ).to_csv(d / "Folsom_weather.csv", index=False)
    # NAM: line 1 is a comment, header on line 2
    with open(d / "Folsom_NAM_lat38.6_lon-121.2.csv", "w") as f:
        f.write("12Z NAM forecast for Folsom Lat=38.6 Lon=-121.2\n")
        f.write("reftime,valtime,dwsw,cloud_cover,precipitation,pressure,wind-u,wind-v,temperature,rel_humidity\n")
        for s in stamp:
            f.write(f"2014-06-01 12:00:00,{s},500.0,20.0,0.0,100674.0,-1.8,-0.5,293.0,43.0\n")


def test_folsom_loads_one_entity_utc_and_namespaces_nwp(tmp_path):
    _write_folsom_fixture(tmp_path)
    p = FolsomPipeline(raw_data_dir=str(tmp_path / "Folsom_data"),
                       output_dir=str(tmp_path / "out"), dataset_name="folsom")
    df_meta, df_ts = p.load_and_standardize_raw()
    assert df_meta["entity_id"].nunique() == 1
    assert df_meta["geo_inferred"].iloc[0] is False
    assert df_ts["timestamp_utc"].dt.tz is not None  # UTC-aware
    assert {"ghi_actual", "dni_actual", "dhi_actual"} <= set(df_ts.columns)
    assert "pv_power_normalized" in df_ts.columns and df_ts["pv_power_normalized"].isna().all()
    assert any(c.startswith("nwp_") for c in df_ts.columns)  # NAM namespaced
```

- [ ] **Step 3: Run test to verify it fails**

Run: `conda run -n solar_uni python -m pytest tests/test_folsom_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 4: Write `solarunibench/pipeline/adapters/folsom.py`**

```python
"""Folsom adapter — ground irradiance (GHI/DNI/DHI) + weather + NAM NWP. Known coords."""
from __future__ import annotations
from pathlib import Path
import glob
import pandas as pd
from ..base import BaseSolarPipeline
from ...dataset_meta import FOLSOM


class FolsomPipeline(BaseSolarPipeline):
    def load_and_standardize_raw(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        root = Path(self.raw_data_dir)

        irr = pd.read_csv(root / "Folsom_irradiance.csv")
        irr["timestamp_utc"] = pd.to_datetime(irr["timeStamp"], format="%Y-%m-%d %H:%M:%S").dt.tz_localize("UTC")
        irr = irr.rename(columns={"ghi": "ghi_actual", "dni": "dni_actual", "dhi": "dhi_actual"})

        # weather (optional join)
        wpath = root / "Folsom_weather.csv"
        if wpath.exists():
            w = pd.read_csv(wpath)
            w["timestamp_utc"] = pd.to_datetime(w["timeStamp"], format="%Y-%m-%d %H:%M:%S").dt.tz_localize("UTC")
            w = w.rename(columns={"air_temp": "nwp_air_temp", "relhum": "nwp_relhum", "press": "nwp_press",
                                  "windsp": "nwp_windsp", "winddir": "nwp_winddir"})
            irr = irr.merge(w[["timestamp_utc", "nwp_air_temp", "nwp_relhum", "nwp_press",
                               "nwp_windsp", "nwp_winddir"]], on="timestamp_utc", how="left")

        # NAM NWP: line 1 is a comment -> skiprows=1; temperature in K, pressure in Pa (kept as-is, namespaced)
        nam_files = sorted(glob.glob(str(root / "Folsom_NAM_*.csv")))
        if nam_files:
            nam = pd.read_csv(nam_files[0], skiprows=1)  # nearest node only
            nam["timestamp_utc"] = pd.to_datetime(nam["valtime"], format="%Y-%m-%d %H:%M:%S").dt.tz_localize("UTC")
            keep = {c: f"nwp_{c}" for c in
                    ["dwsw", "cloud_cover", "precipitation", "pressure", "wind-u", "wind-v", "temperature", "rel_humidity"]}
            nam = nam.rename(columns=keep)
            irr = irr.merge(nam[["timestamp_utc"] + list(keep.values())], on="timestamp_utc", how="left")

        df_ts = irr[["timestamp_utc", "ghi_actual", "dni_actual", "dhi_actual"] +
                    [c for c in irr.columns if c.startswith("nwp_")]].copy()
        df_ts["entity_id"] = FOLSOM["entity_id"]
        df_meta = pd.DataFrame([{k: v for k, v in FOLSOM.items() if k != "capacity_kw"} | {
            "capacity_kw": FOLSOM["capacity_kw"], "slope": float("nan"), "azimuth": float("nan")}])
        return df_meta, df_ts

    def process_and_save_tensors(self, df_ts: pd.DataFrame) -> None:
        return None  # sky/satellite imagery is Phase 4
```

- [ ] **Step 5: Run test to verify it passes**

Run: `conda run -n solar_uni python -m pytest tests/test_folsom_adapter.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add solarunibench/dataset_meta.py solarunibench/pipeline/adapters/folsom.py tests/test_folsom_adapter.py
git commit -m "feat(adapter): Folsom tabular adapter + shared dataset metadata

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: SKIPPD adapter

**Files:**
- Create: `solarunibench/pipeline/adapters/skippd.py`, `tests/test_skippd_adapter.py`

**Interfaces:**
- Consumes: `BaseSolarPipeline`, `dataset_meta.SKIPP`/`SKIPP_CAPACITY_KW`.
- Produces: `SKIPPDPipeline(BaseSolarPipeline)`.

- [ ] **Step 1: Write the failing test `tests/test_skippd_adapter.py`**

```python
import pandas as pd
from pathlib import Path
from solarunibench.pipeline.adapters.skippd import SKIPPDPipeline


def _write_skippd_fixture(root: Path):
    base = root / "SKIPPD_all"
    for split in ("train", "test"):
        for cfg in ("labels", "default"):
            (base / cfg / split).mkdir(parents=True)
    pacific = pd.date_range("2017-06-01 12:00", periods=5, freq="1min", tz="US/Pacific")
    # labels: raw pv (peak ~29.6)
    pd.DataFrame({"time": pacific, "pv": [0.0, 10.0, 25.0, 29.0, 5.0]}
                 ).to_parquet(base / "labels" / "train" / "0000.parquet")
    pd.DataFrame({"time": pacific, "pv": [0.0, 10.0, 25.0, 29.0, 5.0]}
                 ).to_parquet(base / "labels" / "test" / "0000.parquet")
    # ERA5: join on valid_time (hourly)
    era5_t = pd.date_range("2017-06-01 12:00", periods=2, freq="1h", tz="US/Pacific")
    pd.DataFrame({"valid_time": era5_t, "ssrd_nn1": [5e6, 5.5e6], "t2m_nn1": [290.0, 291.0]}
                 ).to_parquet(base / "ERA5" / "train" / "0000.parquet")
    pd.DataFrame({"valid_time": era5_t, "ssrd_nn1": [5e6, 5.5e6], "t2m_nn1": [290.0, 291.0]}
                 ).to_parquet(base / "ERA5" / "test" / "0000.parquet")


def test_skippd_normalizes_pv_and_converts_pacific_to_utc(tmp_path):
    _write_skippd_fixture(tmp_path)
    p = SKIPPDPipeline(raw_data_dir=str(tmp_path / "SKIPPD_all"),
                       output_dir=str(tmp_path / "out"), dataset_name="skippd")
    df_meta, df_ts = p.load_and_standardize_raw()
    assert df_ts["timestamp_utc"].dt.tz is not None
    # pv normalized to [0,1] by capacity 30.1
    assert df_ts["pv_power_normalized"].max() <= 1.0 + 1e-6
    assert df_ts["pv_power_normalized"].min() >= 0.0
    # ghi_actual absent (Honest NaN) — SKIPPD has no ground irradiance
    assert "ghi_actual" in df_ts.columns and df_ts["ghi_actual"].isna().all()
    # ERA5 namespaced
    assert "nwp_ssrd_nn1" in df_ts.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n solar_uni python -m pytest tests/test_skippd_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `solarunibench/pipeline/adapters/skippd.py`**

```python
"""SKIPPD adapter — PV power (normalized) + ERA5/GFS NWP. Pacific time -> UTC. Known coords."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from ..base import BaseSolarPipeline
from ...dataset_meta import SKIPP, SKIPP_CAPACITY_KW


def _read_split(root: Path, cfg: str) -> pd.DataFrame:
    parts = []
    for split in ("train", "test"):
        d = root / cfg / split
        if not d.exists():
            continue
        for f in sorted(d.glob("*.parquet")):
            parts.append(pd.read_parquet(f))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


class SKIPPDPipeline(BaseSolarPipeline):
    def load_and_standardize_raw(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        root = Path(self.raw_data_dir)

        labels = _read_split(root, "labels")
        labels["timestamp_utc"] = pd.to_datetime(labels["time"]).dt.tz_convert("UTC")
        labels = labels.rename(columns={"pv": "pv_power_normalized"})
        labels["pv_power_normalized"] = (labels["pv_power_normalized"] / SKIPP_CAPACITY_KW).clip(0.0, 1.0)
        df_ts = labels[["timestamp_utc", "pv_power_normalized"]].copy()

        # ERA5 (join on valid_time) and GFS (join on time_forecast) -> namespace nwp_*
        for cfg, key in (("ERA5", "valid_time"), ("GFS", "time_forecast")):
            nwp = _read_split(root, cfg)
            if nwp.empty:
                continue
            nwp["timestamp_utc"] = pd.to_datetime(nwp[key]).dt.tz_convert("UTC")
            feat = [c for c in nwp.columns if c not in (key, "timestamp_utc", "latitude", "longitude")]
            nwp = nwp.rename(columns={c: f"nwp_{c}" for c in feat})
            df_ts = df_ts.merge(nwp[["timestamp_utc"] + [f"nwp_{c}" for c in feat]],
                                on="timestamp_utc", how="left")

        df_ts["entity_id"] = SKIPP["entity_id"]
        df_meta = pd.DataFrame([{k: v for k, v in SKIPP.items() if k != "capacity_kw"} | {
            "capacity_kw": SKIPP["capacity_kw"], "slope": float("nan"), "azimuth": float("nan")}])
        return df_meta, df_ts

    def process_and_save_tensors(self, df_ts: pd.DataFrame) -> None:
        return None  # sky images / satellite are Phase 4
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n solar_uni python -m pytest tests/test_skippd_adapter.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add solarunibench/pipeline/adapters/skippd.py tests/test_skippd_adapter.py
git commit -m "feat(adapter): SKIPPD tabular adapter (PV-normalized + ERA5/GFS NWP)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: SunLake tabular pull from Activeloop hub

**Files:**
- Create: `scripts/pull_sunlake_tabular.py`, `tests/test_sunlake_pull.py`
- Output: `solar_benchmark_dataset/SunLake_data/sunlake_tabular.parquet`

**Interfaces:**
- Produces: `find_channel_indices(channels) -> dict` (pure, unit-tested); `pull_sunlake_tabular(hub_path, out_path, stations, years) -> dict`; writes `sunlake_tabular.parquet` with columns `entity_id, timestamp_utc, ghi_actual, dni_actual, dhi_actual`.

- [ ] **Step 1: Write the failing test `tests/test_sunlake_pull.py`** (pure logic only — no hub)

```python
from scripts.pull_sunlake_tabular import find_channel_indices


def test_find_channel_indices_cab():
    ch = ["GHI", "RH [%]", "T2 [°C]", "PoPoPoPo [hPa]", "DIF [W/m**2]", "DIR [W/m**2]", "ghi", "dni", "dhi"]
    idx = find_channel_indices(ch)
    assert idx["ghi_actual"] == 0   # uppercase GHI = ground truth
    assert idx["dhi_actual"] == 4   # DIF
    assert idx["dni_actual"] == 5   # DIR


def test_find_channel_indices_missing_returns_none():
    idx = find_channel_indices(["RH [%]", "T2 [°C]"])  # no GHI/DIF/DIR
    assert idx["ghi_actual"] is None
    assert idx["dhi_actual"] is None
    assert idx["dni_actual"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n solar_uni python -m pytest tests/test_sunlake_pull.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `scripts/pull_sunlake_tabular.py`**

```python
"""Pull SunLake tabular PCCI (GHI/DIF/DIR) from the Activeloop hub -> local parquet.

Phase 3 (tabular) only: satellite context/opt_flow imagery is deferred to Phase 4.
Hub: hub://crossvivit/SunLake (public, read-only).
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HUB = "hub://crossvivit/SunLake"
STATIONS = ["CAB", "CNR", "IZA", "PAL", "PAY", "TAM"]  # skip IZA_bis (augmented duplicate)
YEARS = [str(y) for y in range(2008, 2023)]


def find_channel_indices(channels: list[str]) -> dict:
    """Map contract columns to channel indices in a station's channel list.

    Ground-truth GHI is uppercase 'GHI'; DIF->dhi, DIR->dni. Returns None if absent.
    """
    def _find(name):
        for i, c in enumerate(channels):
            if c == name:
                return i
        return None
    return {"ghi_actual": _find("GHI"), "dhi_actual": _find("DIF [W/m**2]"),
            "dni_actual": _find("DIR [W/m**2]")}


def pull_sunlake_tabular(hub_path: str, out_path: str, stations=STATIONS, years=YEARS) -> dict:
    import deeplake  # imported lazily so unit tests don't require it
    import json
    ds = deeplake.load(hub_path)
    root = Path("D:/monica/workspace/solar_benchmark_dataset/SunLake_data")
    rows = []
    meta_rows = []
    for year in years:
        for st in stations:
            pfx = f"{year}_nonhrv/PCCI_20082022_{st}"
            if f"{pfx}/data" not in ds.tensors:
                continue
            info = json.loads((root / pfx / "data" / "tensor_info.json").read_text())
            channels = info.get("timeseries_channels", [])
            coords = info.get("coordinates", [float("nan"), float("nan")])
            elev = info.get("elevation", float("nan"))
            idx = find_channel_indices(channels)
            data = ds[f"{pfx}/data"][0].numpy()      # (n_timesteps, n_channels)
            times = ds[f"{pfx}/time_utc"][0].numpy()  # (n_timesteps,) datetime64[ns]
            n = data.shape[0]
            eid = f"SUNLAKE_{st}"
            for i in range(n):
                rows.append({
                    "entity_id": eid,
                    "timestamp_utc": pd.Timestamp(times[i]).tz_localize("UTC"),
                    "ghi_actual": float(data[i, idx["ghi_actual"]]) if idx["ghi_actual"] is not None else float("nan"),
                    "dhi_actual": float(data[i, idx["dhi_actual"]]) if idx["dhi_actual"] is not None else float("nan"),
                    "dni_actual": float(data[i, idx["dni_actual"]]) if idx["dni_actual"] is not None else float("nan"),
                })
            meta_rows.append({"entity_id": eid, "station": st, "latitude": coords[0],
                              "longitude": coords[1], "elevation_meters": elev})
    df = pd.DataFrame(rows).drop_duplicates(subset=["entity_id", "timestamp_utc"]).sort_values(
        ["entity_id", "timestamp_utc"]).reset_index(drop=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    # station metadata sidecar
    pd.DataFrame(meta_rows).drop_duplicates("entity_id").to_parquet(
        str(Path(out_path).with_name("sunlake_stations.parquet")), index=False)
    return {"rows": len(df), "stations": df["entity_id"].nunique(),
            "years": sorted(df["timestamp_utc"].dt.year.unique().tolist())}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="D:/monica/workspace/solar_benchmark_dataset/SunLake_data/sunlake_tabular.parquet")
    args = ap.parse_args()
    print(pull_sunlake_tabular(HUB, args.out))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n solar_uni python -m pytest tests/test_sunlake_pull.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/pull_sunlake_tabular.py tests/test_sunlake_pull.py
git commit -m "feat(sunlake): tabular PCCI pull from Activeloop hub

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: SunLake adapter

**Files:**
- Create: `solarunibench/pipeline/adapters/sunlake.py`, `tests/test_sunlake_adapter.py`

**Interfaces:**
- Consumes: `BaseSolarPipeline`, the pulled `sunlake_tabular.parquet` + `sunlake_stations.parquet` (Task 3).
- Produces: `SunLakePipeline(BaseSolarPipeline)`.

- [ ] **Step 1: Write the failing test `tests/test_sunlake_adapter.py`**

```python
import pandas as pd
from pathlib import Path
from solarunibench.pipeline.adapters.sunlake import SunLakePipeline


def _write_sunlake_fixture(root: Path):
    root.mkdir(parents=True)
    ts = pd.date_range("2018-06-01 00:00", periods=5, freq="30min", tz="UTC")
    pd.DataFrame({"entity_id": ["SUNLAKE_CAB"] * 5, "timestamp_utc": ts,
                  "ghi_actual": [0.0, 50.0, 600.0, 800.0, 20.0],
                  "dhi_actual": [0.0, 40.0, 120.0, 150.0, 30.0],
                  "dni_actual": [0.0, 30.0, 500.0, 700.0, 10.0]}
                 ).to_parquet(root / "sunlake_tabular.parquet", index=False)
    pd.DataFrame([{"entity_id": "SUNLAKE_CAB", "station": "CAB", "latitude": 51.968,
                   "longitude": 4.928, "elevation_meters": 0.0}]
                ).to_parquet(root / "sunlake_stations.parquet", index=False)


def test_sunlake_loads_stations_and_ghi(tmp_path):
    _write_sunlake_fixture(tmp_path)
    p = SunLakePipeline(raw_data_dir=str(tmp_path), output_dir=str(tmp_path / "out"), dataset_name="sunlake")
    df_meta, df_ts = p.load_and_standardize_raw()
    assert "SUNLAKE_CAB" in set(df_meta["entity_id"])
    assert df_meta["geo_inferred"].iloc[0] is False
    assert df_ts["timestamp_utc"].dt.tz is not None
    assert {"ghi_actual", "dni_actual", "dhi_actual"} <= set(df_ts.columns)
    assert df_ts["pv_power_normalized"].isna().all()  # no PV
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n solar_uni python -m pytest tests/test_sunlake_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `solarunibench/pipeline/adapters/sunlake.py`**

```python
"""SunLake adapter — BSRN ground GHI/DIF/DIR for 6 stations. Reads pulled tabular parquet."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from ..base import BaseSolarPipeline


class SunLakePipeline(BaseSolarPipeline):
    def load_and_standardize_raw(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        root = Path(self.raw_data_dir)
        df_ts = pd.read_parquet(root / "sunlake_tabular.parquet")
        df_ts["timestamp_utc"] = pd.to_datetime(df_ts["timestamp_utc"], utc=True)
        stations = pd.read_parquet(root / "sunlake_stations.parquet")
        df_meta = pd.DataFrame([{
            "entity_id": r["entity_id"], "dataset_source": "sunlake",
            "latitude": r["latitude"], "longitude": r["longitude"],
            "elevation_meters": r["elevation_meters"], "local_timezone": "UTC",
            "surface_albedo": 0.2, "capacity_kw": float("nan"),
            "slope": float("nan"), "azimuth": float("nan"), "geo_inferred": False,
        } for _, r in stations.iterrows()])
        return df_meta, df_ts

    def process_and_save_tensors(self, df_ts: pd.DataFrame) -> None:
        return None  # satellite context/opt_flow is Phase 4
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n solar_uni python -m pytest tests/test_sunlake_adapter.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add solarunibench/pipeline/adapters/sunlake.py tests/test_sunlake_adapter.py
git commit -m "feat(adapter): SunLake tabular adapter

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Wire all three adapters into `build_dataset.py`

**Files:**
- Modify: `scripts/build_dataset.py`

**Interfaces:**
- Produces: `ADAPTERS = {gefcom2014, folsom, skippd, sunlake}` and a `RAW_SUBDIRS` map.

- [ ] **Step 1: Read the current `scripts/build_dataset.py`** to see the existing `ADAPTERS` dict and `RAW_SUBDIRS` inline map.

Use the `Read` tool on `D:/monica/workspace/solar-unibench/scripts/build_dataset.py`.

- [ ] **Step 2: Modify `scripts/build_dataset.py`** — replace the `ADAPTERS` dict and the inline `{"gefcom2014": "GEFCom2014_data"}` raw-subdir map. New top of file (after the `sys.path.insert`):

```python
from solarunibench.pipeline.adapters.gefcom import GEFCom2014Pipeline
from solarunibench.pipeline.adapters.folsom import FolsomPipeline
from solarunibench.pipeline.adapters.skippd import SKIPPDPipeline
from solarunibench.pipeline.adapters.sunlake import SunLakePipeline

ADAPTERS = {
    "gefcom2014": GEFCom2014Pipeline,
    "folsom": FolsomPipeline,
    "skippd": SKIPPDPipeline,
    "sunlake": SunLakePipeline,
}
RAW_SUBDIRS = {
    "gefcom2014": "GEFCom2014_data",
    "folsom": "Folsom_data",
    "skippd": "SKIPPD_all",
    "sunlake": "SunLake_data",
}
```

And in `main`, replace the inline subdir lookup with: `raw_dir = str(Path(data_root) / RAW_SUBDIRS[args.dataset])`, and change `choices=sorted(ADAPTERS)` to `choices=sorted(ADAPTERS)` (unchanged, now 4 entries).

- [ ] **Step 3: Verify** the CLI dispatches all four (no new test file needed — extend the existing `tests/test_build_dataset_cli.py` is optional; the per-adapter integration tests in Task 6 cover the CLI path). Smoke check:

Run: `conda run -n solar_uni python scripts/build_dataset.py --help`
Expected: `--dataset` choices include `folsom, gefcom2014, skippd, sunlake`.

- [ ] **Step 4: Commit**

```bash
git add scripts/build_dataset.py
git commit -m "feat(cli): dispatch folsom/skippd/sunlake adapters from build_dataset

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Contract conformance + real-data integration tests (all three)

**Files:**
- Create: `tests/test_folsom_contract.py`, `tests/test_skippd_contract.py`, `tests/test_sunlake_contract.py`, `tests/test_folsom_integration.py`, `tests/test_skippd_integration.py`, `tests/test_sunlake_integration.py`

- [ ] **Step 1: Contract conformance tests** (one per dataset; mirror `test_gefcom_contract.py`). Each builds a fixture package via the adapter's `run_pipeline()` then asserts `validate_metadata`/`validate_time_series` pass + Honest-NaN for the absent modality. Example for Folsom (`tests/test_folsom_contract.py`):

```python
import pandas as pd
from solarunibench.contract import validate_metadata, validate_time_series
from solarunibench.pipeline.adapters.folsom import FolsomPipeline
from tests.test_folsom_adapter import _write_folsom_fixture


def test_folsom_package_satisfies_contract(tmp_path):
    _write_folsom_fixture(tmp_path)
    FolsomPipeline(raw_data_dir=str(tmp_path / "Folsom_data"),
                   output_dir=str(tmp_path / "out"), dataset_name="folsom").run_pipeline()
    out = tmp_path / "out" / "folsom"
    meta = pd.read_parquet(out / "metadata.parquet")
    ts = pd.read_parquet(out / "time_series.parquet")
    validate_metadata(meta)
    validate_time_series(ts)
    assert meta["geo_inferred"].iloc[0] is False
    assert ts["pv_power_normalized"].isna().all()  # Folsom = irradiance only
```

Write analogous `test_skippd_contract.py` (assert `ts["ghi_actual"].isna().all()`) and `test_sunlake_contract.py` (assert `pv_power_normalized` all NaN), importing the respective `_write_*_fixture` helpers.

- [ ] **Step 2: Slow real-data integration tests** (RUN_SLOW-gated). Each runs the adapter + `build_splits` on the REAL data and asserts row counts / split presence. Example `tests/test_folsom_integration.py`:

```python
import os
import pytest
import pandas as pd
from solarunibench.pipeline.adapters.folsom import FolsomPipeline

DATA_ROOT = os.environ.get("SOLAR_UNIBENCH_DATA_ROOT", "D:/monica/workspace/solar_benchmark_dataset")
slow = pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="set RUN_SLOW=1")


@slow
def test_real_folsom_builds_and_splits(tmp_path):
    p = FolsomPipeline(raw_data_dir=f"{DATA_ROOT}/Folsom_data",
                       output_dir=str(tmp_path), dataset_name="folsom")
    m = p.run_pipeline()
    assert m["n_entities"] == 1
    ts = pd.read_parquet(tmp_path / "folsom" / "time_series.parquet")
    assert len(ts) > 100_000  # ~1.5M rows real
    assert "ghi_clearsky" in ts.columns and ts["ghi_clearsky"].notna().any()
```

Write analogous `test_skippd_integration.py` (assert `n_entities == 1`, pv normalized in [0,1], ERA5 cols present) and `test_sunlake_integration.py` (requires the Task-3 pull to have run; assert `n_entities == 6`, GHI present). For SunLake, the integration test should first call `pull_sunlake_tabular` if `sunlake_tabular.parquet` is absent — or skip with a clear message if the pull hasn't been done.

- [ ] **Step 3: Update `configs/splits.yaml`** — change the `skippd` entry from `strategy: predefined` to a leak-free temporal split:
```yaml
skippd:
  strategy: temporal
  test_start: "2019-01-01"
  val_ratio: 0.1
```
(SKIPPD's official HF train/test temporally overlap; a temporal cutoff is rigorous and `make_splits` already implements it.)

- [ ] **Step 4: Run + verify** — `conda run -n solar_uni python -m pytest -q` (fast tests green, slow skipped). Then run each slow test with `RUN_SLOW=1` (Folsom first — largest; SKIPPD; SunLake after its pull).

- [ ] **Step 5: Commit**

```bash
git add tests/test_folsom_contract.py tests/test_skippd_contract.py tests/test_sunlake_contract.py \
        tests/test_folsom_integration.py tests/test_skippd_integration.py tests/test_sunlake_integration.py \
        configs/splits.yaml
git commit -m "test(adapters): contract conformance + real-data integration (folsom/skippd/sunlake)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Definition of Done (Phase 3)

- `conda run -n solar_uni python -m pytest -q` → all fast unit tests green.
- `python scripts/pull_sunlake_tabular.py` writes `sunlake_tabular.parquet` (6 stations).
- `python scripts/build_dataset.py --dataset {folsom,skippd,sunlake}` each writes a contract-conformant standardized package.
- `python scripts/build_splits.py --dataset {folsom,skippd,sunlake}` stamps train/val/test (folsom 2016 / sunlake 2021 temporal; skippd changed to temporal `test_start: 2019-01-01` in Task 6 Step 3).
- `RUN_SLOW=1` integration tests pass for folsom + skippd (sunlake after pull).

## Note: SKIPPD split strategy (decided)
`configs/splits.yaml` currently sets SKIPPD `strategy: predefined`, but SKIPPD's official HF train/test split **temporally overlaps** (both span 2017–2019 — it tags specific days, not a clean cutoff), which is a leakage risk for a benchmark. For Phase 3 we use a **leak-free temporal split** instead: Task 6 Step 3 updates `configs/splits.yaml` to `skippd: {strategy: temporal, test_start: "2019-01-01", val_ratio: 0.1}`. (`make_splits` already implements `temporal`; no `predefined` work needed.) If the exact official split is later required, `predefined` can be implemented then — out of scope for Phase 3.

## Next phases
- **Phase 4:** zarr/image layer — Folsom sky images (BRR), SKIPPD sky images, SunLake/SKIPPD satellite optical-flow + uncertainty (requires full SunLake imagery pull).
- **Phase 5:** loader + PyTorch Dataset.
- **Phase 6–8:** forecast framework + baselines, experiments, full batch.
