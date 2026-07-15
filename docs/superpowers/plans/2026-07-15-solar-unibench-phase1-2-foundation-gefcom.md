# Solar-UniBench Phase 1+2: Foundation + GEFCom Vertical Spike — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the project skeleton, data contract, and physical operators, then prove the architecture end-to-end by building the GEFCom2014 (Solar-track) adapter that produces a standardized `metadata.parquet` + `time_series.parquet` package — including clear-sky geo-recovery validated by the TSR method.

**Architecture:** An OOP "operator assembly line": a `BaseSolarPipeline` abstract class holds all physics (clear-sky, quality audit, reverse-geolocation, TSR validation); per-dataset adapters only parse raw → standardized frames. A single `DataContract` (`contract.py`) is the source of truth every adapter must satisfy. Operators are ported from the existing, already-prototyped `solar_benchmark_dataset/GEFCom2014_data/geo_location.py`.

**Tech Stack:** Python 3.10 (existing `solar_uni` env), pandas, pyarrow (parquet), pvlib (clear-sky/solar position), numpy/scipy, pytest (TDD), tqdm. Windows 11 / Git Bash.

## Global Constraints

- **Environment:** the existing conda env **`solar_uni`** (Python 3.10.20) is this project's venv — run everything inside it via `conda run -n solar_uni`. numpy/pandas/pyarrow are already present; the remaining core deps (pvlib, scipy, scikit-learn, pytest, tqdm, pyyaml, matplotlib) are installed into it. torch/zarr/opencv/xarray/datasets come in later phase plans. Do NOT use base Anaconda.
- **Branch:** work on existing branch `feat/solar-unibench-build`. Commit after every task (frequent commits).
- **TDD strictly:** every unit test runs on **synthetic fixtures** (no 69 GB of real data in the test suite). Real-data runs are smoke tests, separate from `pytest`.
- **UTC at ingest:** every timestamp is parsed then normalized to `timestamp_utc` (UTC) before leaving an adapter. GEFCom `TIMESTAMP` (`YYYYMMDD HH:MM`) is already UTC per the competition spec.
- **Naming:** canonical columns/dtypes are defined once in `solarunibench/contract.py` and imported everywhere — never re-declared.
- **Paths:** use `pathlib.Path` and forward slashes inside Python (Windows-safe). Raw data root is the sibling dir `D:\monica\workspace\solar_benchmark_dataset` (configurable via `--data-root`).
- **GEFCom = Solar track only** (ignore Load/Wind/Price). Geo **recovery = clear-sky method**; **correctness validated by the TSR method**.
- **Idempotency:** `run_pipeline()` is resumable/idempotent and emits a `manifest.json` + appends to `audit.jsonl`. It also snapshots a `--raw` (pre-governance) frame for the later E9 ablation.
- **No placeholders:** every code step contains complete, runnable code.

---

## File Structure (created/modified in this plan)

```
solar-unibench/
├── requirements-core.txt                 # Task 1
├── pyproject.toml                        # Task 1
├── pytest.ini                            # Task 1
├── configs/paths.yaml                    # Task 1 (data-root, output-root)
├── solarunibench/__init__.py             # Task 1
├── solarunibench/contract.py             # Task 2 — DataContract (single source of truth)
├── solarunibench/pipeline/__init__.py    # Task 1
├── solarunibench/pipeline/base.py        # Task 8 — BaseSolarPipeline
├── solarunibench/pipeline/adapters/__init__.py           # Task 1
├── solarunibench/pipeline/adapters/gefcom.py             # Task 9
├── solarunibench/operators/__init__.py                   # Task 1
├── solarunibench/operators/solar_geometry.py             # Task 3
├── solarunibench/operators/clearsky.py                   # Task 4
├── solarunibench/operators/reverse_geolocate.py          # Task 5
├── solarunibench/operators/tsr_validation.py             # Task 6
├── solarunibench/operators/quality.py                    # Task 7
├── solarunibench/splits.py                               # Task 13
├── configs/splits.yaml                                   # Task 13
├── scripts/build_dataset.py                              # Task 10
├── scripts/build_splits.py                               # Task 13
└── tests/
    ├── __init__.py                                       # Task 1
    ├── conftest.py                                       # Task 1 (synthetic fixtures factory)
    ├── test_contract.py                                  # Task 2
    ├── test_solar_geometry.py                            # Task 3
    ├── test_clearsky.py                                  # Task 4
    ├── test_reverse_geolocate.py                         # Task 5
    ├── test_tsr_validation.py                            # Task 6
    ├── test_quality.py                                   # Task 7
    ├── test_base_pipeline.py                             # Task 8
    ├── test_gefcom_adapter.py                            # Task 9
    ├── test_build_dataset_cli.py                         # Task 10
    ├── test_gefcom_contract.py                           # Task 11
    ├── test_splits.py                                    # Task 13
    ├── test_build_splits_cli.py                          # Task 13
    └── test_gefcom_geo_integration.py                    # Task 12
```

---

### Task 1: Environment + project scaffold

**Files:**
- Create: `requirements-core.txt`, `pyproject.toml`, `pytest.ini`, `configs/paths.yaml`, `solarunibench/__init__.py`, `solarunibench/pipeline/__init__.py`, `solarunibench/pipeline/adapters/__init__.py`, `solarunibench/operators/__init__.py`, `tests/__init__.py`, `tests/conftest.py`

**Interfaces:**
- Produces: an importable `solarunibench` package and a `tests/synthetic_gefcom()` fixture factory (used by Tasks 2, 8, 9). Signature: `synthetic_gefcom(n_hours=48, n_zones=3, seed=0) -> tuple[pd.DataFrame, pd.DataFrame]` returning `(df_meta, df_ts)` with contract-like columns.

- [ ] **Step 1: Verify the `solar_uni` env has core deps**

The project env `solar_uni` already exists (Python 3.10.20) with numpy/pandas/pyarrow. Ensure the remaining core deps are installed (run from Git Bash):
```bash
conda run -n solar_uni pip install pvlib scipy scikit-learn pytest tqdm pyyaml matplotlib
```
Expected: installs complete; `conda run -n solar_uni python -c "import pandas, pvlib, pyarrow, scipy, sklearn, pytest; print('ok')"` prints `ok`.

- [ ] **Step 2: Write `requirements-core.txt`**

```
numpy>=1.26
pandas>=2.2
pyarrow>=15.0
pvlib>=0.10
scipy>=1.11
scikit-learn>=1.4
pytest>=8.0
tqdm>=4.66
pyyaml>=6.0
matplotlib>=3.8
```

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[project]
name = "solarunibench"
version = "0.1.0"
description = "Solar-UniBench: multi-modal standardized solar forecasting benchmark + pipeline"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
include = ["solarunibench*"]
```

- [ ] **Step 4: Write `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -q
```

- [ ] **Step 5: Write `configs/paths.yaml`**

```yaml
data_root: D:/monica/workspace/solar_benchmark_dataset
output_root: D:/monica/workspace/solar-unibench/data/standardized
```

- [ ] **Step 6: Create package `__init__.py` files**

`solarunibench/__init__.py`:
```python
"""Solar-UniBench: unified multi-modal solar forecasting benchmark."""
__version__ = "0.1.0"
```
`solarunibench/pipeline/__init__.py`, `solarunibench/pipeline/adapters/__init__.py`, `solarunibench/operators/__init__.py`, `tests/__init__.py`: each an empty file with a one-line module docstring.

- [ ] **Step 7: Write the synthetic fixture factory `tests/conftest.py`**

```python
"""Shared synthetic fixtures. No real data lives in the test suite."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def synthetic_gefcom(n_hours: int = 48, n_zones: int = 3, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (df_meta, df_ts) mimicking the GEFCom contract for unit tests.

    df_ts columns: timestamp_utc, entity_id, ghi_actual, pv_power_normalized, ghi_clearsky.
    df_meta columns: entity_id, dataset_source, latitude, longitude, elevation_meters,
                     local_timezone, surface_albedo, capacity_kw, slope, azimuth, geo_inferred.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2013-06-01 06:00", tz="UTC")
    times = pd.date_range(start, periods=n_hours, freq="1h", tz="UTC")
    # A toy clear-sky-ish bell curve over daytime hours
    hod = times.hour.to_numpy()
    bell = np.clip(np.sin(np.pi * (hod - 6) / 13.0), 0.0, None)
    meta_rows, ts_rows = [], []
    for z in range(1, n_zones + 1):
        eid = f"GEFCOM_ZONE_{z}"
        meta_rows.append({
            "entity_id": eid, "dataset_source": "gefcom2014",
            "latitude": -30.0 - z, "longitude": 140.0 + z, "elevation_meters": 50.0,
            "local_timezone": "Australia/Brisbane", "surface_albedo": 0.2,
            "capacity_kw": float("nan"), "slope": float("nan"), "azimuth": float("nan"),
            "geo_inferred": True,
        })
        ghi = bell * 900.0 + rng.normal(0, 10, n_hours)
        pv = np.clip(bell * 0.9 + rng.normal(0, 0.02, n_hours), 0, 1)
        for i, t in enumerate(times):
            ts_rows.append({
                "timestamp_utc": t, "entity_id": eid,
                "ghi_actual": float(ghi[i]), "pv_power_normalized": float(pv[i]),
                "ghi_clearsky": float(bell[i] * 950.0),
            })
    df_meta = pd.DataFrame(meta_rows)
    df_ts = pd.DataFrame(ts_rows)
    return df_meta, df_ts
```

- [ ] **Step 8: Verify the scaffold imports and pytest collects**

Run:
```bash
conda run -n solar_uni python -c "import solarunibench; print(solarunibench.__version__)"
conda run -n solar_uni python -m pytest
```
Expected: prints `0.1.0`; pytest reports `no tests ran` (0 passed, 0 failed) — collection works.

- [ ] **Step 9: Commit**

```bash
git add requirements-core.txt pyproject.toml pytest.ini configs/ solarunibench/ tests/
git commit -m "chore: scaffold project, conda env, synthetic test fixtures"
```

---

### Task 2: Data contract (`contract.py`)

**Files:**
- Create: `solarunibench/contract.py`
- Test: `tests/test_contract.py`

**Interfaces:**
- Consumes: pandas DataFrames produced by adapters.
- Produces: module-level `META_COLUMNS`, `TS_COLUMNS`, `META_DTYPES`, `TS_DTYPES` (dicts), and two functions:
  - `validate_metadata(df: pd.DataFrame) -> None` — raises `ContractError` on violations.
  - `validate_time_series(df: pd.DataFrame) -> None` — raises `ContractError` on violations.
  - Exception class `ContractError(ValueError)`.

- [ ] **Step 1: Write the failing test `tests/test_contract.py`**

```python
import pandas as pd
import numpy as np
import pytest
from solarunibench.contract import (
    validate_metadata, validate_time_series, ContractError, META_COLUMNS, TS_COLUMNS,
)


def _good_meta():
    return pd.DataFrame([{
        "entity_id": "Z1", "dataset_source": "gefcom2014", "latitude": -30.0,
        "longitude": 140.0, "elevation_meters": 50.0, "local_timezone": "UTC",
        "surface_albedo": 0.2, "capacity_kw": np.nan, "slope": np.nan, "azimuth": np.nan,
        "geo_inferred": True,
    }])


def _good_ts():
    return pd.DataFrame([{
        "timestamp_utc": pd.Timestamp("2013-06-01 06:00", tz="UTC"), "entity_id": "Z1",
        "ghi_actual": 500.0, "dni_actual": np.nan, "dhi_actual": np.nan,
        "pv_power_normalized": 0.5, "ghi_clearsky": 600.0, "dni_clearsky": np.nan,
        "dhi_clearsky": np.nan, "cos_zenith": 0.5, "solar_azimuth": 90.0,
        "clearness_index": 0.83, "quality_flag": 0,
    }])


def test_good_frames_pass():
    validate_metadata(_good_meta())
    validate_time_series(_good_ts())


def test_missing_meta_column_raises():
    df = _good_meta().drop(columns=["latitude"])
    with pytest.raises(ContractError):
        validate_metadata(df)


def test_missing_ts_column_raises():
    df = _good_ts().drop(columns=["clearness_index"])
    with pytest.raises(ContractError):
        validate_time_series(df)


def test_quality_flag_domain():
    df = _good_ts()
    df.loc[0, "quality_flag"] = 9
    with pytest.raises(ContractError):
        validate_time_series(df)


def test_contract_constants_present():
    assert "entity_id" in META_COLUMNS
    assert "clearness_index" in TS_COLUMNS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n solar_uni python -m pytest tests/test_contract.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'solarunibench.contract'`.

- [ ] **Step 3: Write `solarunibench/contract.py`**

```python
"""Data contract: the single source of truth for standardized packages."""
import pandas as pd


class ContractError(ValueError):
    """Raised when a frame violates the Solar-UniBench data contract."""


META_COLUMNS = [
    "entity_id", "dataset_source", "latitude", "longitude", "elevation_meters",
    "local_timezone", "surface_albedo", "capacity_kw", "slope", "azimuth", "geo_inferred",
]

TS_COLUMNS = [
    "timestamp_utc", "entity_id",
    "ghi_actual", "dni_actual", "dhi_actual", "pv_power_normalized",
    "ghi_clearsky", "dni_clearsky", "dhi_clearsky",
    "cos_zenith", "solar_azimuth", "clearness_index", "quality_flag",
]

META_DTYPES = {
    "entity_id": "string", "dataset_source": "string", "latitude": "float32",
    "longitude": "float32", "elevation_meters": "float32", "local_timezone": "string",
    "surface_albedo": "float32", "capacity_kw": "float32", "slope": "float32",
    "azimuth": "float32", "geo_inferred": "bool",
}

TS_DTYPES = {
    "timestamp_utc": "datetime64[ns, UTC]", "entity_id": "string",
    "ghi_actual": "float32", "dni_actual": "float32", "dhi_actual": "float32",
    "pv_power_normalized": "float32", "ghi_clearsky": "float32", "dni_clearsky": "float32",
    "dhi_clearsky": "float32", "cos_zenith": "float32", "solar_azimuth": "float32",
    "clearness_index": "float32", "quality_flag": "int8",
}

VALID_QUALITY_FLAGS = {0, 1, 2, 3}


def validate_metadata(df: pd.DataFrame) -> None:
    missing = [c for c in META_COLUMNS if c not in df.columns]
    if missing:
        raise ContractError(f"metadata missing columns: {missing}")


def validate_time_series(df: pd.DataFrame) -> None:
    missing = [c for c in TS_COLUMNS if c not in df.columns]
    if missing:
        raise ContractError(f"time_series missing columns: {missing}")
    if "quality_flag" in df.columns:
        bad = set(df["quality_flag"].dropna().unique()) - VALID_QUALITY_FLAGS
        if bad:
            raise ContractError(f"invalid quality_flag values: {bad}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n solar_uni python -m pytest tests/test_contract.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add solarunibench/contract.py tests/test_contract.py
git commit -m "feat(contract): data contract + validators"
```

---

### Task 3: Solar-geometry helpers (`operators/solar_geometry.py`)

Port the pure-math solar-geometry functions from `GEFCom2014_data/geo_location.py` (lines 30–121). These are used by both the clear-sky recovery and the TSR validation.

**Files:**
- Create: `solarunibench/operators/solar_geometry.py`
- Test: `tests/test_solar_geometry.py`

**Interfaces:**
- Produces: `SOLAR_CONSTANT`, `day_of_year(times)`, `solar_declination(doy)`, `eccentricity_correction(doy)`, `sunset_hour_angle(lat_rad, declination)`, `toa_daily_irradiation(doy, lat_deg)`, `solar_noon_to_longitude(peak_utc_hour)`.

- [ ] **Step 1: Write the failing test `tests/test_solar_geometry.py`**

```python
import numpy as np
import pandas as pd
from solarunibench.operators import solar_geometry as sg


def test_declination_summer_solstice():
    # DOY 172 ~ June solstice -> declination ~ +23.45 deg
    dec = sg.solar_declination(np.array([172.0]))
    assert abs(np.rad2deg(dec[0]) - 23.45) < 0.6


def test_eccentricity_in_expected_range():
    dr = sg.eccentricity_correction(np.arange(1, 366, dtype=float))
    assert dr.min() > 0.96 and dr.max() < 1.04


def test_sunset_hour_angle_equator():
    # At the equator, year-round sunset hour angle ~ pi/2 (12h day)
    omega = sg.sunset_hour_angle(np.deg2rad(0.0), sg.solar_declination(np.array([172.0])))
    assert abs(omega[0] - np.pi / 2) < 0.05


def test_toa_daily_irradiation_monotonic_in_latitude():
    doy = np.array([172.0])
    h_eq = sg.toa_daily_irradiation(doy, 0.0)[0]
    h_mid = sg.toa_daily_irradiation(doy, 45.0)[0]
    # June solstice: equator receives less daily TOA than 45N (long day)
    assert h_mid > h_eq


def test_solar_noon_to_longitude():
    assert abs(sg.solar_noon_to_longitude(12.0) - 0.0) < 1e-9
    assert abs(sg.solar_noon_to_longitude(10.0) - 30.0) < 1e-9  # east
    assert abs(sg.solar_noon_to_longitude(14.0) - (-30.0)) < 1e-9  # west
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n solar_uni python -m pytest tests/test_solar_geometry.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `solarunibench/operators/solar_geometry.py`**

```python
"""Pure solar-geometry helpers (no external deps). Ported from geo_location.py."""
import numpy as np
import pandas as pd

SOLAR_CONSTANT = 1367.0  # W/m^2


def day_of_year(times: pd.DatetimeIndex) -> np.ndarray:
    return times.dayofyear.values.astype(float)


def solar_declination(doy: np.ndarray) -> np.ndarray:
    """Cooper equation: declination (radians)."""
    return np.deg2rad(23.45) * np.sin(2 * np.pi * (284 + doy) / 365.0)


def eccentricity_correction(doy: np.ndarray) -> np.ndarray:
    return 1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)


def sunset_hour_angle(lat_rad: float, declination: np.ndarray) -> np.ndarray:
    cos_omega_s = -np.tan(lat_rad) * np.tan(declination)
    cos_omega_s = np.clip(cos_omega_s, -1.0, 1.0)
    return np.arccos(cos_omega_s)


def toa_daily_irradiation(doy: np.ndarray, lat_deg: float) -> np.ndarray:
    """Daily top-of-atmosphere irradiation (J/m^2). Pure geometry (no atmosphere)."""
    lat_rad = np.deg2rad(lat_deg)
    delta = solar_declination(doy)
    dr = eccentricity_correction(doy)
    omega_s = sunset_hour_angle(lat_rad, delta)
    term = omega_s * np.sin(lat_rad) * np.sin(delta) + np.cos(lat_rad) * np.cos(delta) * np.sin(omega_s)
    return (24.0 / np.pi) * SOLAR_CONSTANT * dr * term * 3600.0


def solar_noon_to_longitude(peak_utc_hour: float) -> float:
    """Longitude (deg, east positive) from the UTC hour of solar noon."""
    return (12.0 - peak_utc_hour) * 15.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n solar_uni python -m pytest tests/test_solar_geometry.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add solarunibench/operators/solar_geometry.py tests/test_solar_geometry.py
git commit -m "feat(operators): solar-geometry helpers"
```

---

### Task 4: Clear-sky + astronomy operator (`operators/clearsky.py`)

pvlib-based clear-sky and solar position (the production path), plus a dependency-free simplified Ineichen used by the grid-search recovery (Task 5) so recovery doesn't call pvlib 20k+ times.

**Files:**
- Create: `solarunibench/operators/clearsky.py`
- Test: `tests/test_clearsky.py`

**Interfaces:**
- Produces:
  - `compute_astronomy(times: pd.DatetimeIndex, lat: float, lon: float) -> pd.DataFrame` with columns `cos_zenith, solar_azimuth`.
  - `clearsky_ineichen(times, lat, lon, elevation=0.0) -> pd.DataFrame` with `ghi_clearsky, dni_clearsky, dhi_clearsky` (pvlib Ineichen, **fixed** from the methodology's broken `appointment_time=` call).
  - `clearsky_ghi_simple(times, lat, lon, turbidity=2.0) -> np.ndarray` (simplified, ported from `_clearsky_ghi`).

- [ ] **Step 1: Write the failing test `tests/test_clearsky.py`**

```python
import numpy as np
import pandas as pd
from solarunibench.operators import clearsky


def _summer_noon_times():
    # 2013-06-01 12:00 local at a site ~ lon=0 -> 12:00 UTC is solar noon
    return pd.date_range("2013-06-01 10:00", periods=5, freq="1h", tz="UTC")


def test_cos_zenith_bounded_and_night_low():
    times = _summer_noon_times()
    astro = clearsky.compute_astronomy(times, lat=45.0, lon=0.0)
    assert ((astro["cos_zenith"] >= 0) & (astro["cos_zenith"] <= 1)).all()


def test_clearsky_ineichen_night_near_zero():
    # Midnight UTC at lon=0 -> night -> GHI ~ 0
    night = pd.date_range("2013-06-01 00:00", periods=3, freq="1h", tz="UTC")
    cs = clearsky.clearsky_ineichen(night, lat=45.0, lon=0.0)
    assert (cs["ghi_clearsky"] < 1.0).all()


def test_clearsky_simple_matches_ineichen_shape_daytime():
    times = _summer_noon_times()
    simple = clearsky.clearsky_ghi_simple(times, lat=45.0, lon=0.0)
    cs = clearsky.clearsky_ineichen(times, lat=45.0, lon=0.0)
    # Both peak around local noon (lon=0 -> 12 UTC); daytime values positive & ordered similarly
    assert (simple[2:4] > 0).all()
    assert np.corrcoef(simple, cs["ghi_clearsky"].to_numpy())[0, 1] > 0.95
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n solar_uni python -m pytest tests/test_clearsky.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `solarunibench/operators/clearsky.py`**

```python
"""Clear-sky + solar-position operators (pvlib) and a simplified Ineichen for grid search."""
import numpy as np
import pandas as pd
import pvlib
from . import solar_geometry as sg


def compute_astronomy(times: pd.DatetimeIndex, lat: float, lon: float) -> pd.DataFrame:
    solpos = pvlib.solarposition.get_solarposition(times, lat, lon)
    cos_zenith = np.maximum(0.0, np.cos(np.radians(solpos["zenith"].values))).astype(np.float32)
    az = solpos["azimuth"].values.astype(np.float32)
    return pd.DataFrame({"cos_zenith": cos_zenith, "solar_azimuth": az}, index=times)


def clearsky_ineichen(times: pd.DatetimeIndex, lat: float, lon: float,
                      elevation: float = 0.0) -> pd.DataFrame:
    """pvlib Ineichen clear-sky. NOTE: correct API (methodology's appointment_time= was wrong)."""
    solpos = pvlib.solarposition.get_solarposition(times, lat, lon, altitude=elevation)
    app_zen = solpos["apparent_zenith"]
    try:
        # Linke turbidity climatology (needs internet on first use; falls back below).
        lt = pvlib.clearsky.lookup_linke_turbidity(times, lat, lon)
    except Exception:
        lt = 3.0  # global clean-atmosphere default
    cs = pvlib.clearsky.ineichen(app_zen, airmass_abs=None, linke_turbidity=lt,
                                 altitude=elevation, interp_turbidity=False)
    return pd.DataFrame({
        "ghi_clearsky": cs["ghi"].astype(np.float32),
        "dni_clearsky": cs["dni"].astype(np.float32),
        "dhi_clearsky": cs["dhi"].astype(np.float32),
    }, index=times)


def clearsky_ghi_simple(times: pd.DatetimeIndex, lat_deg: float, lon_deg: float,
                        turbidity: float = 2.0) -> np.ndarray:
    """Dependency-light Ineichen GHI (ported from geo_location._clearsky_ghi). For grid search only."""
    doy = sg.day_of_year(times)
    delta = sg.solar_declination(doy)
    dr = sg.eccentricity_correction(doy)
    utc_hours = times.hour.to_numpy() + times.minute.to_numpy() / 60.0
    local_solar_time = utc_hours + lon_deg / 15.0
    hour_angle = np.deg2rad(15.0 * (local_solar_time - 12.0))
    lat_rad = np.deg2rad(lat_deg)
    cos_zenith = np.sin(lat_rad) * np.sin(delta) + np.cos(lat_rad) * np.cos(delta) * np.cos(hour_angle)
    cos_zenith = np.clip(cos_zenith, 0.0, 1.0)
    zenith = np.rad2deg(np.arccos(cos_zenith))
    am = 1.0 / (cos_zenith + 0.50572 * (96.07995 - zenith) ** -1.6364)
    am = np.clip(am, 0.0, 40.0)
    fh1 = np.exp(-lat_deg / 8000.0)
    fh2 = np.exp(-lat_deg / 8000.0)
    cg1 = 5.09e-5 * (turbidity ** 2) + 3.16e-3 * turbidity + 0.079
    cg2 = 1.03e-7 * (turbidity ** 3) - 3.73e-5 * (turbidity ** 2) + 5.45e-3 * turbidity + 0.109
    i0 = sg.SOLAR_CONSTANT * dr
    ghi = cg1 * i0 * cos_zenith * np.exp(-cg2 * am * (fh1 + fh2 * turbidity))
    return np.maximum(ghi, 0.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n solar_uni python -m pytest tests/test_clearsky.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add solarunibench/operators/clearsky.py tests/test_clearsky.py
git commit -m "feat(operators): clear-sky + solar-position (pvlib + simple)"
```

---

### Task 5: Reverse-geolocation (clear-sky recovery) (`operators/reverse_geolocate.py`)

Grid-search the lat/lon minimizing shape residual between observed `POWER` (normalized PV) and the simplified clear-sky GHI — ported from `geo_location.infer_geo_from_clearsky`. This is the **recovery** method.

**Files:**
- Create: `solarunibench/operators/reverse_geolocate.py`
- Test: `tests/test_reverse_geolocate.py`

**Interfaces:**
- Consumes: `solar_geometry`, `clearsky.clearsky_ghi_simple`.
- Produces: `reverse_geolocate(times: pd.DatetimeIndex, power: np.ndarray, lat_grid=None, lon_grid=None) -> dict` with keys `latitude, longitude, details`.

- [ ] **Step 1: Write the failing test `tests/test_reverse_geolocate.py`**

```python
import numpy as np
import pandas as pd
from solarunibench.operators import clearsky, reverse_geolocate


def test_recovers_known_site_within_tolerance():
    true_lat, true_lon = 35.0, -120.0
    times = pd.date_range("2013-03-01", periods=24 * 60, freq="1h", tz="UTC")
    # Synthesize a POWER shape from the simple clear-sky model at the true site.
    shape = clearsky.clearsky_ghi_simple(times, true_lat, true_lon)
    power = shape / (shape.max() + 1e-9)  # normalized like GEFCom POWER
    res = reverse_geolocate.reverse_geolocate(times, power,
                                              lat_grid=np.arange(20, 51, 2.0),
                                              lon_grid=np.arange(-140, -100, 2.0))
    assert abs(res["latitude"] - true_lat) <= 4.0
    assert abs(res["longitude"] - true_lon) <= 6.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n solar_uni python -m pytest tests/test_reverse_geolocate.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `solarunibench/operators/reverse_geolocate.py`**

```python
"""Clear-sky geo-recovery: grid-search lat/lon by POWER-vs-clearsky shape match."""
import numpy as np
import pandas as pd
from .clearsky import clearsky_ghi_simple


def reverse_geolocate(times: pd.DatetimeIndex, power: np.ndarray,
                      lat_grid: np.ndarray = None, lon_grid: np.ndarray = None) -> dict:
    """Recover (lat, lon) by matching normalized POWER to the clear-sky GHI shape.

    MSE is computed on peak-normalized daytime values (POWER is unit-less, so only
    shape information is used). This is the geo *recovery* method.
    """
    power = np.asarray(power, dtype=float)
    if lat_grid is None:
        lat_grid = np.arange(-60, 61, 2.0)
    if lon_grid is None:
        lon_grid = np.arange(-180, 181, 2.0)

    daytime = power > 0.01
    best_lat, best_lon, best_mse = None, None, np.inf
    for lat in lat_grid:
        for lon in lon_grid:
            cs = clearsky_ghi_simple(times, lat, lon)
            mask = daytime & (cs > 1.0)
            if mask.sum() < 100:
                continue
            scale = power[mask].max() / (cs[mask].max() + 1e-9)
            mse = float(np.mean((power[mask] - cs[mask] * scale) ** 2))
            if mse < best_mse:
                best_mse, best_lat, best_lon = mse, float(lat), float(lon)
    return {
        "latitude": best_lat, "longitude": best_lon,
        "details": {"best_mse": best_mse, "method": "clearsky_shape_match"},
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n solar_uni python -m pytest tests/test_reverse_geolocate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add solarunibench/operators/reverse_geolocate.py tests/test_reverse_geolocate.py
git commit -m "feat(operators): clear-sky geo-recovery"
```

---

### Task 6: TSR validation (`operators/tsr_validation.py`)

Given **recovered** coords and observed `VAR178 (TSR)`, validate correctness: latitude via R² of daily TSR vs theoretical TOA at the recovered lat; longitude via solar-noon UTC hour vs recovered lon. Ported from `geo_location` TSR/TOA logic.

**Files:**
- Create: `solarunibench/operators/tsr_validation.py`
- Test: `tests/test_tsr_validation.py`

**Interfaces:**
- Consumes: `solar_geometry.toa_daily_irradiation`, `solar_geometry.solar_noon_to_longitude`.
- Produces: `validate_with_tsr(times, tsr_hourly, lat, lon) -> dict` with keys `r2_latitude, rmse_latitude, longitude_check_hour, longitude_implied, details`.

- [ ] **Step 1: Write the failing test `tests/test_tsr_validation.py`**

```python
import numpy as np
import pandas as pd
from solarunibench.operators import solar_geometry as sg, tsr_validation as tv


def test_tsr_validation_high_r2_for_correct_lat():
    true_lat = 40.0
    times = pd.date_range("2013-01-01", periods=24 * 365, freq="1h", tz="UTC")
    # Build an hourly TSR whose daily sum follows theoretical TOA at true_lat (scaled).
    doy = times.dayofyear.to_numpy()
    daily_toa = sg.toa_daily_irradiation(doy.astype(float), true_lat)
    # spread daily total uniformly-ish across day hours (toy, monotonic in TOA is enough)
    tsr = np.where(daily_toa > 0, daily_toa / 12.0, 0.0)
    res = tv.validate_with_tsr(times, tsr, lat=true_lat, lon=0.0)
    assert res["r2_latitude"] > 0.90


def test_tsr_validation_low_r2_for_wrong_lat():
    true_lat = 40.0
    times = pd.date_range("2013-01-01", periods=24 * 365, freq="1h", tz="UTC")
    doy = times.dayofyear.to_numpy()
    daily_toa = sg.toa_daily_irradiation(doy.astype(float), true_lat)
    tsr = np.where(daily_toa > 0, daily_toa / 12.0, 0.0)
    res = tv.validate_with_tsr(times, tsr, lat=10.0, lon=0.0)  # wrong latitude
    assert res["r2_latitude"] < 0.90
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n solar_uni python -m pytest tests/test_tsr_validation.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `solarunibench/operators/tsr_validation.py`**

```python
"""TSR (VAR178) validation of recovered coordinates. TOA radiation is pure geometry."""
import numpy as np
import pandas as pd
from . import solar_geometry as sg


def _r_squared(obs: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(pred) & (pred > 1e-6)
    if mask.sum() < 5:
        return float("nan")
    o, p = obs[mask], pred[mask]
    scale = np.median(o / p)
    ss_res = np.sum((o - p * scale) ** 2)
    ss_tot = np.sum((o - o.mean()) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def validate_with_tsr(times: pd.DatetimeIndex, tsr_hourly: np.ndarray,
                      lat: float, lon: float) -> dict:
    """Validate recovered (lat, lon) against observed top-of-atmosphere radiation.

    Latitude: R^2/RMSE of daily TSR vs theoretical TOA daily irradiation at `lat`.
    Longitude: solar-noon UTC hour implied by TSR's diurnal peak vs `lon`.
    """
    s = pd.Series(np.asarray(tsr_hourly, dtype=float), index=times)
    daily = s.resample("1D").sum()
    doys = daily.index.dayofyear.to_numpy().astype(float)
    obs = daily.to_numpy()
    theory = sg.toa_daily_irradiation(doys, lat)
    valid = theory > 1e-3
    r2 = _r_squared(obs[valid], theory[valid])
    rmse = float(np.sqrt(np.nanmean((obs[valid] - theory[valid]) ** 2))) if valid.sum() else float("nan")

    # Longitude check via diurnal peak hour of positive TSR increments
    diff = s.diff()
    daytime = diff[diff > 0]
    peak_hour = float(daytime.groupby(daytime.index.hour).mean().idxmax()) if len(daytime) else 12.0
    lon_implied = sg.solar_noon_to_longitude(peak_hour)
    return {
        "r2_latitude": r2, "rmse_latitude": rmse,
        "longitude_check_hour": peak_hour, "longitude_implied": lon_implied,
        "details": {"recovered_longitude": lon, "n_days": int(valid.sum())},
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n solar_uni python -m pytest tests/test_tsr_validation.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add solarunibench/operators/tsr_validation.py tests/test_tsr_validation.py
git commit -m "feat(operators): TSR-based validation of recovered coordinates"
```

---

### Task 7: Quality audit operator (`operators/quality.py`)

Night-zeroing, `clearness_index`, `quality_flag`, physical clamp.

**Files:**
- Create: `solarunibench/operators/quality.py`
- Test: `tests/test_quality.py`

**Interfaces:**
- Produces: `apply_quality_audit(df_ts: pd.DataFrame, capacity: float = 1.0) -> pd.DataFrame` that consumes `ghi_actual, ghi_clearsky, cos_zenith, pv_power_normalized` and adds/overwrites `clearness_index, quality_flag`, zeroing night values.

- [ ] **Step 1: Write the failing test `tests/test_quality.py`**

```python
import numpy as np
import pandas as pd
from solarunibench.operators import quality


def _df():
    return pd.DataFrame({
        "cos_zenith": [0.0, 0.5, 0.5, 0.5],
        "ghi_actual": [10.0, 600.0, 1500.0, 400.0],   # night, good, >clearsky(outlier), good
        "ghi_clearsky": [0.0, 800.0, 800.0, 800.0],
        "pv_power_normalized": [0.05, 0.5, 1.2, 0.4],
    })


def test_night_zeroed_and_flagged():
    out = quality.apply_quality_audit(_df(), capacity=1.0)
    assert out.loc[0, "ghi_actual"] == 0.0
    assert out.loc[0, "pv_power_normalized"] == 0.0
    assert out.loc[0, "quality_flag"] == 3  # night


def test_clearness_index_bounded():
    out = quality.apply_quality_audit(_df(), capacity=1.0)
    assert out["clearness_index"].between(0.0, 1.5).all()


def test_pv_clamped_to_capacity():
    out = quality.apply_quality_audit(_df(), capacity=1.0)
    assert out["pv_power_normalized"].max() <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n solar_uni python -m pytest tests/test_quality.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `solarunibench/operators/quality.py`**

```python
"""Quality audit: night-zeroing, clearness index, quality flag, physical clamp."""
import numpy as np
import pandas as pd


def apply_quality_audit(df_ts: pd.DataFrame, capacity: float = 1.0) -> pd.DataFrame:
    df = df_ts.copy()
    eps = 1e-6
    df["clearness_index"] = np.clip(
        df.get("ghi_actual", np.nan) / (df.get("ghi_clearsky", np.nan) + eps), 0.0, 1.5
    ).astype("float32")

    df["quality_flag"] = 0  # good
    night = df["cos_zenith"] <= 0
    df.loc[night, ["ghi_actual", "pv_power_normalized"]] = 0.0
    df.loc[night, "quality_flag"] = 3  # night

    # curtailment / clipping: PV pinned at capacity across daytime
    if "pv_power_normalized" in df:
        df["pv_power_normalized"] = np.clip(df["pv_power_normalized"], 0.0, capacity)
    # physical clamp on irradiance (no negatives)
    for c in ("ghi_actual", "dni_actual", "dhi_actual"):
        if c in df:
            df[c] = df[c].clip(lower=0.0)
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n solar_uni python -m pytest tests/test_quality.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add solarunibench/operators/quality.py tests/test_quality.py
git commit -m "feat(operators): quality audit + clearness index"
```

---

### Task 8: `BaseSolarPipeline` (`pipeline/base.py`)

Abstract base wiring operators into a tabular `run_pipeline()`. Emits `metadata.parquet`, `time_series.parquet`, `manifest.json`, appends `audit.jsonl`. Supports a `raw=True` snapshot (pre-governance frame) for the later E9 ablation.

**Files:**
- Create: `solarunibench/pipeline/base.py`
- Test: `tests/test_base_pipeline.py`

**Interfaces:**
- Consumes: `contract`, `operators.clearsky`, `operators.quality`.
- Produces: abstract class `BaseSolarPipeline` with:
  - `__init__(self, raw_data_dir, output_dir, dataset_name, config=None)`
  - abstract `load_and_standardize_raw(self) -> tuple[pd.DataFrame, pd.DataFrame]`
  - `process_and_save_tensors(self, df_ts) -> None` (default no-op; overridden in later phases)
  - `run_pipeline(self, raw=False, limit=None) -> dict` — orchestrates and writes artifacts.

- [ ] **Step 1: Write the failing test `tests/test_base_pipeline.py`**

```python
import pandas as pd, numpy as np, json
from pathlib import Path
from solarunibench.pipeline.base import BaseSolarPipeline


class _ToyAdapter(BaseSolarPipeline):
    def load_and_standardize_raw(self):
        from tests.conftest import synthetic_gefcom
        return synthetic_gefcom(n_hours=24, n_zones=1)


def test_run_pipeline_writes_conformant_artifacts(tmp_path):
    p = _ToyAdapter(raw_data_dir=".", output_dir=str(tmp_path), dataset_name="toy")
    manifest = p.run_pipeline()
    out = Path(tmp_path) / "toy"
    assert (out / "metadata.parquet").exists()
    assert (out / "time_series.parquet").exists()
    assert (out / "manifest.json").exists()
    ts = pd.read_parquet(out / "time_series.parquet")
    for col in ["timestamp_utc", "entity_id", "ghi_clearsky", "cos_zenith",
                "clearness_index", "quality_flag"]:
        assert col in ts.columns
    assert manifest["dataset"] == "toy"


def test_raw_snapshot_is_pre_governance(tmp_path):
    p = _ToyAdapter(raw_data_dir=".", output_dir=str(tmp_path), dataset_name="toy")
    p.run_pipeline(raw=True)
    out = Path(tmp_path) / "toy"
    ts = pd.read_parquet(out / "time_series.parquet")
    # raw frame = standardized but BEFORE quality audit: clearness_index/quality_flag are NA.
    assert ts["quality_flag"].isna().all()
    assert ts["clearness_index"].isna().all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n solar_uni python -m pytest tests/test_base_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `solarunibench/pipeline/base.py`**

```python
"""BaseSolarPipeline: the operator assembly line. Physics in base, parsing in adapters."""
from __future__ import annotations
import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from ..contract import validate_metadata, validate_time_series, TS_COLUMNS
from ..operators.clearsky import compute_astronomy, clearsky_ineichen
from ..operators.quality import apply_quality_audit


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class BaseSolarPipeline(ABC):
    def __init__(self, raw_data_dir: str, output_dir: str, dataset_name: str, config: dict | None = None):
        self.raw_data_dir = raw_data_dir
        self.output_dir = Path(output_dir) / dataset_name
        self.dataset_name = dataset_name
        self.config = config or {}
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def load_and_standardize_raw(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (df_meta, df_ts) in standardized frames (UTC timestamps, contract-ish)."""

    def process_and_save_tensors(self, df_ts: pd.DataFrame) -> None:
        """Override in adapters with image/satellite modalities. Default: no-op."""
        return None

    def _compute_clearsky_and_astronomy(self, df_meta: pd.DataFrame, df_ts: pd.DataFrame) -> pd.DataFrame:
        blocks = []
        for _, m in df_meta.iterrows():
            sub = df_ts[df_ts["entity_id"] == m["entity_id"]]
            if sub.empty:
                continue
            times = pd.DatetimeIndex(sub["timestamp_utc"])
            elev = m.get("elevation_meters", 0.0)
            elev = 0.0 if pd.isna(elev) else float(elev)
            astro = compute_astronomy(times, m["latitude"], m["longitude"])
            cs = clearsky_ineichen(times, m["latitude"], m["longitude"], elevation=elev)
            # Index by the tz-aware timestamps so reset_index yields a tz-aware column
            # (avoids the object-dtype merge hazard from .to_numpy()).
            block = pd.DataFrame({
                "entity_id": m["entity_id"],
                "cos_zenith": astro["cos_zenith"].to_numpy(),
                "solar_azimuth": astro["solar_azimuth"].to_numpy(),
                "ghi_clearsky": cs["ghi_clearsky"].to_numpy(),
                "dni_clearsky": cs["dni_clearsky"].to_numpy(),
                "dhi_clearsky": cs["dhi_clearsky"].to_numpy(),
            }, index=times)
            blocks.append(block.reset_index().rename(columns={"index": "timestamp_utc"}))
        return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()

    def run_pipeline(self, raw: bool = False, limit: int | None = None) -> dict:
        df_meta, df_ts = self.load_and_standardize_raw()
        if limit:
            df_ts = df_ts.groupby("entity_id", group_keys=False).head(limit)

        # Merge anchors
        anchors = self._compute_clearsky_and_astronomy(df_meta, df_ts)
        df = df_ts.merge(anchors, on=["timestamp_utc", "entity_id"], how="left")

        if raw:
            # E9 control arm: standardized frame BEFORE governance.
            df_out = df.copy()
            for missing in set(TS_COLUMNS) - set(df_out.columns):
                df_out[missing] = pd.NA
        else:
            capacity = float(df_meta["capacity_kw"].dropna().max()) if df_meta["capacity_kw"].notna().any() else 1.0
            df_out = apply_quality_audit(df, capacity=capacity)
            for missing in set(TS_COLUMNS) - set(df_out.columns):
                df_out[missing] = pd.NA

        validate_metadata(df_meta)
        validate_time_series(df_out)
        self._write(df_meta, df_out)
        if not raw:
            self.process_and_save_tensors(df_out)

        manifest = {"dataset": self.dataset_name, "raw": raw,
                    "n_entities": int(df_meta["entity_id"].nunique()),
                    "n_rows": int(len(df_out)), "built_at": _utc_now_iso()}
        (self.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        self._audit(f"run_pipeline raw={raw} rows={len(df_out)}")
        return manifest

    def _write(self, df_meta: pd.DataFrame, df_ts: pd.DataFrame) -> None:
        df_meta.to_parquet(self.output_dir / "metadata.parquet", index=False)
        df_ts.to_parquet(self.output_dir / "time_series.parquet", index=False)

    def _audit(self, message: str) -> None:
        with open(self.output_dir.parent / "audit.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"dataset": self.dataset_name, "ts": _utc_now_iso(), "msg": message}) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n solar_uni python -m pytest tests/test_base_pipeline.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add solarunibench/pipeline/base.py tests/test_base_pipeline.py
git commit -m "feat(pipeline): BaseSolarPipeline tabular assembly line"
```

---

### Task 9: GEFCom adapter (`pipeline/adapters/gefcom.py`)

Parse the **Solar track only**: merge `train{n}.csv`/`predictors{n}.csv`/`Solution to Task 15.csv` across Tasks 1–15, parse `TIMESTAMP` (`YYYYMMDD HH:MM`, UTC), recover geo per zone (clear-sky), validate via TSR, build `df_meta` (3 zones) + `df_ts`.

**Files:**
- Create: `solarunibench/pipeline/adapters/gefcom.py`
- Test: `tests/test_gefcom_adapter.py`

**Interfaces:**
- Consumes: `BaseSolarPipeline`, `reverse_geolocate`, `tsr_validation`, contract.
- Produces: class `GEFCom2014Pipeline(BaseSolarPipeline)` whose `load_and_standardize_raw()` returns contract-conformant `(df_meta, df_ts)` with `pv_power_normalized`, NWP columns namespaced `nwp_VAR<id>`, and recovered/inferred geo.

- [ ] **Step 1: Write the failing test `tests/test_gefcom_adapter.py`**

This test uses a tiny synthetic on-disk GEFCom-like fixture (no real data):

```python
import pandas as pd, numpy as np
from pathlib import Path
from solarunibench.pipeline.adapters.gefcom import GEFCom2014Pipeline


def _write_fixture(root: Path):
    solar = root / "GEFCom2014_data" / "Solar" / "Task 15"
    solar.mkdir(parents=True)
    times = pd.date_range("2013-06-01 06:00", periods=12, freq="1h")
    stamp = times.strftime("%Y%m%d %H:%M")
    rows_p, rows_t = [], []
    for z in [1, 2, 3]:
        for s in stamp:
            power = float(np.clip(np.random.rand(), 0, 1))
            rows_t.append({"ZONEID": z, "TIMESTAMP": s, "POWER": power})
            rows_p.append({"ZONEID": z, "TIMESTAMP": s, "VAR178": 1e6, "VAR169": 5e5,
                           "VAR78": 0, "VAR79": 0, "VAR134": 1e5, "VAR157": 50, "VAR164": 0.5,
                           "VAR165": 1, "VAR166": 1, "VAR167": 290, "VAR175": 3e5, "VAR228": 0,
                           "POWER": power})
    pd.DataFrame(rows_t).to_csv(solar / "train15.csv", index=False)
    pd.DataFrame(rows_p).to_csv(solar / "predictors15.csv", index=False)


def test_gefcom_loads_three_zones_and_parses_timestamps(tmp_path):
    _write_fixture(tmp_path)
    p = GEFCom2014Pipeline(raw_data_dir=str(tmp_path / "GEFCom2014_data"),
                           output_dir=str(tmp_path / "out"), dataset_name="gefcom2014")
    df_meta, df_ts = p.load_and_standardize_raw()
    assert df_meta["entity_id"].nunique() == 3
    assert {"pv_power_normalized", "timestamp_utc", "entity_id"} <= set(df_ts.columns)
    assert df_ts["timestamp_utc"].dt.tz is not None  # UTC-aware
    assert any(c.startswith("nwp_") for c in df_ts.columns)  # NWP namespaced
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n solar_uni python -m pytest tests/test_gefcom_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `solarunibench/pipeline/adapters/gefcom.py`**

```python
"""GEFCom2014 adapter — Solar track only. Clear-sky geo recovery, TSR-validated."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from ..base import BaseSolarPipeline
from ...operators.reverse_geolocate import reverse_geolocate
from ...operators.tsr_validation import validate_with_tsr

NWP_VARS = ["VAR78", "VAR79", "VAR134", "VAR157", "VAR164", "VAR165", "VAR166",
            "VAR167", "VAR169", "VAR175", "VAR178", "VAR228"]
TASKS = list(range(1, 16))


class GEFCom2014Pipeline(BaseSolarPipeline):
    def load_and_standardize_raw(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        solar = Path(self.raw_data_dir) / "Solar"
        frames = []
        for t in TASKS:
            tdir = solar / f"Task {t}"
            train = tdir / f"train{t}.csv"
            pred = tdir / f"predictors{t}.csv"
            if not train.exists() and not pred.exists():
                continue
            parts = []
            if train.exists():
                parts.append(pd.read_csv(train))
            if pred.exists():
                pp = pd.read_csv(pred)
                if "POWER" in pp.columns and train.exists():
                    pp = pp.drop(columns=["POWER"])
                parts.append(pp)
            df = parts[0] if len(parts) == 1 else parts[0].merge(parts[1], on=["ZONEID", "TIMESTAMP"], how="outer")
            frames.append(df)
        # Append the held-out solution target rows (POWER truth) for Task 15
        sol = solar / "Solution to Task 15" / "Solution to Task 15.csv"
        if sol.exists():
            frames.append(pd.read_csv(sol))
        raw = pd.concat(frames, ignore_index=True)
        raw = raw.drop_duplicates(subset=["ZONEID", "TIMESTAMP"]).sort_values(["ZONEID", "TIMESTAMP"])

        raw["timestamp_utc"] = pd.to_datetime(raw["TIMESTAMP"], format="%Y%m%d %H:%M", utc=True)
        raw = raw.rename(columns={"ZONEID": "zone", "POWER": "pv_power_normalized"})

        meta_rows, ts_rows = [], []
        for z, g in raw.groupby("zone"):
            eid = f"GEFCOM_ZONE_{int(z)}"
            times = g["timestamp_utc"]
            power = g["pv_power_normalized"].to_numpy(dtype=float)
            rec = reverse_geolocate(times, power)
            lat, lon = rec["latitude"], rec["longitude"]
            tsr = g["VAR178"].to_numpy(dtype=float) if "VAR178" in g else np.full(len(g), np.nan)
            val = validate_with_tsr(times, tsr, lat, lon) if np.isfinite(tsr).any() else {}
            meta_rows.append({
                "entity_id": eid, "dataset_source": "gefcom2014",
                "latitude": lat, "longitude": lon, "elevation_meters": float("nan"),
                "local_timezone": "UTC", "surface_albedo": 0.2,
                "capacity_kw": float("nan"), "slope": float("nan"), "azimuth": float("nan"),
                "geo_inferred": True,
            })
            for _, row in g.iterrows():
                rec_row = {"timestamp_utc": row["timestamp_utc"], "entity_id": eid,
                           "pv_power_normalized": float(row["pv_power_normalized"])}
                for v in NWP_VARS:
                    if v in g.columns:
                        rec_row[f"nwp_{v}"] = float(row[v])
                ts_rows.append(rec_row)
            # stash validation on meta via attribute for the audit/manifest step
            self._last_validation = getattr(self, "_last_validation", {})
            self._last_validation[eid] = {"recovery": rec, "tsr_validation": val}

        df_meta = pd.DataFrame(meta_rows)
        df_ts = pd.DataFrame(ts_rows)
        return df_meta, df_ts

    def process_and_save_tensors(self, df_ts: pd.DataFrame) -> None:
        return None  # GEFCom has no image/satellite modalities
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n solar_uni python -m pytest tests/test_gefcom_adapter.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add solarunibench/pipeline/adapters/gefcom.py tests/test_gefcom_adapter.py
git commit -m "feat(adapter): GEFCom2014 Solar-track adapter with geo recovery"
```

---

### Task 10: `build_dataset.py` CLI

**Files:**
- Create: `scripts/build_dataset.py`
- Test: `tests/test_build_dataset_cli.py`

**Interfaces:**
- Produces: a CLI `python scripts/build_dataset.py --dataset gefcom2014 [--raw] [--limit N] [--data-root DIR] [--output-root DIR]` that runs the adapter and prints the manifest.

- [ ] **Step 1: Write the failing test `tests/test_build_dataset_cli.py`**

```python
import subprocess, sys
from pathlib import Path
import pandas as pd


def test_cli_builds_gefcom_from_fixture(tmp_path):
    # reuse the same fixture writer as the adapter test
    from tests.test_gefcom_adapter import _write_fixture
    _write_fixture(tmp_path)
    out = tmp_path / "out"
    cmd = [sys.executable, "scripts/build_dataset.py", "--dataset", "gefcom2014",
           "--data-root", str(tmp_path / "GEFCom2014_data"), "--output-root", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (out / "gefcom2014" / "metadata.parquet").exists()
    assert (out / "gefcom2014" / "manifest.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n solar_uni python -m pytest tests/test_build_dataset_cli.py -v`
Expected: FAIL — `FileNotFoundError` (no `scripts/build_dataset.py`).

- [ ] **Step 3: Write `scripts/build_dataset.py`**

```python
"""CLI: build a standardized Solar-UniBench package for one dataset."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from solarunibench.pipeline.adapters.gefcom import GEFCom2014Pipeline

ADAPTERS = {"gefcom2014": GEFCom2014Pipeline}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build standardized Solar-UniBench package")
    ap.add_argument("--dataset", required=True, choices=sorted(ADAPTERS))
    ap.add_argument("--raw", action="store_true", help="emit pre-governance snapshot (E9 control)")
    ap.add_argument("--limit", type=int, default=None, help="rows per entity (debug)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--output-root", default=None)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "paths.yaml").read_text())
    data_root = args.data_root or cfg["data_root"]
    output_root = args.output_root or cfg["output_root"]

    adapter_cls = ADAPTERS[args.dataset]
    # dataset-specific raw subdir
    raw_dir = str(Path(data_root) / {"gefcom2014": "GEFCom2014_data"}.get(args.dataset, ""))
    pipe = adapter_cls(raw_data_dir=raw_dir, output_dir=output_root, dataset_name=args.dataset)
    manifest = pipe.run_pipeline(raw=args.raw, limit=args.limit)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n solar_uni python -m pytest tests/test_build_dataset_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_dataset.py tests/test_build_dataset_cli.py
git commit -m "feat(cli): build_dataset.py for standardized package generation"
```

---

### Task 11: Contract conformance test on the produced GEFCom package

**Files:**
- Test: `tests/test_gefcom_contract.py` (no new source — validates the full pipeline output against `contract.py`)

- [ ] **Step 1: Write the test `tests/test_gefcom_contract.py`**

```python
import pandas as pd
from solarunibench.contract import validate_metadata, validate_time_series
from solarunibench.pipeline.adapters.gefcom import GEFCom2014Pipeline


def _build(tmp_path):
    from tests.test_gefcom_adapter import _write_fixture
    _write_fixture(tmp_path)
    p = GEFCom2014Pipeline(raw_data_dir=str(tmp_path / "GEFCom2014_data"),
                           output_dir=str(tmp_path / "out"), dataset_name="gefcom2014")
    p.run_pipeline()
    return tmp_path / "out" / "gefcom2014"


def test_gefcom_package_satisfies_contract(tmp_path):
    out = _build(tmp_path)
    meta = pd.read_parquet(out / "metadata.parquet")
    ts = pd.read_parquet(out / "time_series.parquet")
    validate_metadata(meta)
    validate_time_series(ts)
    assert meta["dataset_source"].eq("gefcom2014").all()
    assert meta["geo_inferred"].all()  # all GEFCom coords are recovered
    assert ts["quality_flag"].isin([0, 1, 2, 3]).all()
```

- [ ] **Step 2: Run and verify it passes**

Run: `conda run -n solar_uni python -m pytest tests/test_gefcom_contract.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_gefcom_contract.py
git commit -m "test(gefcom): contract-conformance on produced package"
```

---

### Task 12: Geo-recovery + TSR validation integration on real GEFCom data

This is a **smoke/integration test on real data** (not part of the fast unit suite). It confirms recovery returns coords for all 3 zones and TSR validation runs.

**Files:**
- Test: `tests/test_gefcom_geo_integration.py` (marked `slow`, skipped unless `--runslow` or env var set)

- [ ] **Step 1: Write the integration test `tests/test_gefcom_geo_integration.py`**

```python
"""Real-data integration: geo recovery + TSR validation on GEFCom Solar. Slow."""
import os
import pytest
import pandas as pd
from solarunibench.pipeline.adapters.gefcom import GEFCom2014Pipeline

DATA_ROOT = os.environ.get("SOLAR_UNIBENCH_DATA_ROOT", "D:/monica/workspace/solar_benchmark_dataset")
slow = pytest.mark.skipif(not os.environ.get("RUN_SLOW"),
                          reason="set RUN_SLOW=1 to run real-data integration tests")


@slow
def test_real_gefcom_recovers_three_zones_and_validates_tsr(tmp_path):
    p = GEFCom2014Pipeline(raw_data_dir=f"{DATA_ROOT}/GEFCom2014_data",
                           output_dir=str(tmp_path), dataset_name="gefcom2014")
    df_meta, _ = p.load_and_standardize_raw()
    assert df_meta["entity_id"].nunique() == 3
    assert df_meta["latitude"].notna().all() and df_meta["longitude"].notna().all()
    val = getattr(p, "_last_validation", {})
    for eid, v in val.items():
        assert "tsr_validation" in v and "r2_latitude" in v["tsr_validation"]
```

- [ ] **Step 2: Verify it is collected (skipped by default) and runs when enabled**

Run:
```bash
conda run -n solar_uni python -m pytest tests/test_gefcom_geo_integration.py -v   # SKIPPED
RUN_SLOW=1 conda run -n solar_uni python -m pytest tests/test_gefcom_geo_integration.py -v   # runs on real data
```
Expected: first SKIPPED; second runs and PASSes (coords for 3 zones; TSR R² reported in `_last_validation`).

- [ ] **Step 3: Commit**

```bash
git add tests/test_gefcom_geo_integration.py
git commit -m "test(gefcom): real-data geo-recovery + TSR validation integration (slow)"
```

---

### Task 13: Frozen train/val/test splits for GEFCom (`splits.py` + `build_splits.py`)

Stamp a frozen, leak-free `split` column into GEFCom's `time_series.parquet`: `test` = the competition's held-out future month (`timestamp >= 2014-06-01`, i.e. Solution to Task 15), the remainder split chronologically into train/val. Also writes `splits_manifest.json` (cutoffs + `splits_version`). Generalizes to the other datasets via `configs/splits.yaml` (temporal strategy now; SKIPPD's `predefined` strategy is implemented in Phase 3).

**Files:**
- Create: `solarunibench/splits.py`, `scripts/build_splits.py`, `configs/splits.yaml`
- Test: `tests/test_splits.py`, `tests/test_build_splits_cli.py`

**Interfaces:**
- Consumes: `time_series.parquet` produced by Tasks 9–10.
- Produces: `make_splits(df_ts, cfg) -> pd.DataFrame` (adds a `split` column); `build(dataset_dir, cfg, version) -> manifest` (stamps the `split` column back into `time_series.parquet` in place + writes `splits_manifest.json`); a CLI driving it.

- [ ] **Step 1: Write `configs/splits.yaml`**

```yaml
splits_version: "1.0"
gefcom2014:
  strategy: temporal
  test_start: "2014-06-01"   # Solution to Task 15 (held-out future month)
  val_ratio: 0.1             # of the pre-test remainder
folsom:
  strategy: temporal
  test_start: "2016-01-01"
  val_ratio: 0.1
sunlake:
  strategy: temporal
  test_start: "2021-01-01"
  val_ratio: 0.1
skippd:
  strategy: predefined       # reuse official HF train/test; carve val from train (Phase 3)
  val_ratio: 0.1
spatial:                     # cross-domain leave-one-dataset-out (benchmark-level)
  folds:
    - {train: [gefcom2014, folsom, skippd], test: [sunlake]}
    - {train: [gefcom2014, folsom, sunlake], test: [skippd]}
    - {train: [gefcom2014, skippd, sunlake], test: [folsom]}
    - {train: [folsom, skippd, sunlake], test: [gefcom2014]}
```

- [ ] **Step 2: Write the failing test `tests/test_splits.py`**

```python
import pandas as pd
from solarunibench.splits import make_splits


def _ts():
    # GEFCom-like: 2013-01-01 .. 2014-06-30 hourly, one zone
    times = pd.date_range("2013-01-01", "2014-06-30 23:00", freq="1h", tz="UTC")
    return pd.DataFrame({"timestamp_utc": times, "entity_id": "GEFCOM_ZONE_1"})


def test_test_set_is_the_held_out_future_month():
    cfg = {"strategy": "temporal", "test_start": "2014-06-01", "val_ratio": 0.1}
    out = make_splits(_ts(), cfg)
    test = out[out["split"] == "test"]
    assert test["timestamp_utc"].min() >= pd.Timestamp("2014-06-01", tz="UTC")
    assert test["timestamp_utc"].max() <= pd.Timestamp("2014-06-30 23:00", tz="UTC")


def test_no_leakage_train_and_val_before_test():
    cfg = {"strategy": "temporal", "test_start": "2014-06-01", "val_ratio": 0.1}
    out = make_splits(_ts(), cfg)
    train_max = out[out["split"] == "train"]["timestamp_utc"].max()
    val_max = out[out["split"] == "val"]["timestamp_utc"].max()
    test_min = out[out["split"] == "test"]["timestamp_utc"].min()
    assert train_max < test_min
    assert val_max < test_min


def test_every_row_labeled():
    cfg = {"strategy": "temporal", "test_start": "2014-06-01", "val_ratio": 0.1}
    out = make_splits(_ts(), cfg)
    assert set(out["split"].unique()) <= {"train", "val", "test"}
    assert out["split"].notna().all()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `conda run -n solar_uni python -m pytest tests/test_splits.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 4: Write `solarunibench/splits.py`**

```python
"""Frozen train/val/test splits. Per-entity chronological; dataset-specific test cutoff."""
import json
from pathlib import Path
import pandas as pd


def make_splits(df_ts: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return df_ts with an added `split` column in {train, val, test}, per entity.

    Temporal strategy: rows with timestamp >= test_start are `test`; the most recent
    `val_ratio` fraction of the remainder is `val`; the rest is `train` — leak-free.
    """
    strategy = cfg["strategy"]
    if strategy != "temporal":
        raise NotImplementedError(f"split strategy '{strategy}' is implemented in a later phase")
    out = df_ts.copy()
    labels = pd.Series("train", index=out.index, dtype="object")
    test_start = pd.Timestamp(cfg["test_start"], tz="UTC")
    val_ratio = float(cfg.get("val_ratio", 0.1))
    for _, idx in out.groupby("entity_id").groups.items():
        sub = out.loc[idx].sort_values("timestamp_utc")
        is_test = sub["timestamp_utc"] >= test_start
        rest = sub[~is_test]
        n_val = max(1, int(len(rest) * val_ratio))
        val_idx = rest.index[-n_val:]
        labels.loc[sub.index[is_test]] = "test"
        labels.loc[val_idx] = "val"
    out["split"] = labels.values
    return out


def build(dataset_dir, cfg: dict, splits_version: str) -> dict:
    """Stamp the `split` column into time_series.parquet (in place) + write manifest."""
    dataset_dir = Path(dataset_dir)
    ts_path = dataset_dir / "time_series.parquet"
    df_ts = pd.read_parquet(ts_path)
    splits = make_splits(df_ts, cfg)
    # write the full table back WITH the new split column (single source of truth)
    splits.to_parquet(ts_path, index=False)
    manifest = {"dataset": dataset_dir.name, "splits_version": splits_version,
                "strategy": cfg.get("strategy"),
                "counts": {k: int(v) for k, v in splits["split"].value_counts().to_dict().items()}}
    (dataset_dir / "splits_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
```

- [ ] **Step 5: Run test to verify it passes**

Run: `conda run -n solar_uni python -m pytest tests/test_splits.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Write `scripts/build_splits.py` + `tests/test_build_splits_cli.py`**

`scripts/build_splits.py`:
```python
"""CLI: build frozen splits.parquet for a dataset from configs/splits.yaml."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from solarunibench.splits import build


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build frozen train/val/test splits")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output-root", default=None)
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    cfg_all = yaml.safe_load((root / "configs" / "splits.yaml").read_text())
    version = cfg_all.pop("splits_version", "1.0")
    paths = yaml.safe_load((root / "configs" / "paths.yaml").read_text())
    out_root = Path(args.output_root or paths["output_root"])
    manifest = build(out_root / args.dataset, cfg_all[args.dataset], version)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
```

`tests/test_build_splits_cli.py`:
```python
import subprocess
import sys
import pandas as pd


def test_cli_writes_gefcom_splits(tmp_path):
    from tests.test_gefcom_adapter import _write_fixture
    from solarunibench.pipeline.adapters.gefcom import GEFCom2014Pipeline
    _write_fixture(tmp_path)
    GEFCom2014Pipeline(raw_data_dir=str(tmp_path / "GEFCom2014_data"),
                       output_dir=str(tmp_path / "out"), dataset_name="gefcom2014").run_pipeline()
    out = tmp_path / "out"
    cmd = [sys.executable, "scripts/build_splits.py", "--dataset", "gefcom2014",
           "--output-root", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    ts = pd.read_parquet(out / "gefcom2014" / "time_series.parquet")
    assert "split" in ts.columns
    assert set(ts["split"].unique()) <= {"train", "val", "test"}
    assert (out / "gefcom2014" / "splits_manifest.json").exists()
```

Note: the GEFCom fixture's short time range predates `test_start`, so the CLI test only asserts the split domain and the manifest — the true future-month cutoff is exercised by the real-data run in Task 12.

- [ ] **Step 7: Run + commit**

```bash
conda run -n solar_uni python -m pytest tests/test_build_splits_cli.py tests/test_splits.py -v
git add solarunibench/splits.py scripts/build_splits.py configs/splits.yaml tests/test_splits.py tests/test_build_splits_cli.py
git commit -m "feat(splits): frozen train/val/test splits + CLI"
```

---

## Definition of Done (Phase 1+2)

- `conda run -n solar_uni python -m pytest` → all fast unit tests green.
- `RUN_SLOW=1 ... pytest tests/test_gefcom_geo_integration.py` → recovers 3 zones, TSR-validated.
- `python scripts/build_dataset.py --dataset gefcom2014` writes a contract-conformant `data/standardized/gefcom2014/{metadata,time_series}.parquet` + `manifest.json`.
- `python scripts/build_splits.py --dataset gefcom2014` stamps a `split` column (train/val/test) into `time_series.parquet` + writes `splits_manifest.json`.
- All 13 tasks committed on `feat/solar-unibench-build`.

## Next phases (separate plans, each self-contained)

- **Phase 3:** tabular adapters for Folsom, SKIPPD, SunLake (Tables 1–4) + SKIPPD download.
- **Phase 4:** zarr/image layer (sky images BRR; satellite optical-flow + uncertainty) for Folsom/SKIPPD/SunLake.
- **Phase 5:** loader + splits + PyTorch Dataset.
- **Phase 6:** forecast framework + baseline suite (LightGBM, MLP, LSTM, PatchTST Transformer, CrossViViT) + metrics.
- **Phase 7:** experiments & visualizations E1–E9 (incl. governance-vs-complexity ablation).
- **Phase 8:** full batch execution + README/figure-ready outputs.
