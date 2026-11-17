# Solar-UniBench Curation & Onboarding Specification (`SKILL.md`)

## Overview

This document defines the standardized data schemas, metadata requirements, and deterministic execution workflow for **Solar-UniBench v1.0**. It serves as a structured reference for both researchers and code-generation assistants to onboard new solar observation archives into the analysis-ready Parquet/Zarr format without altering the underlying physics algorithms.

All curation steps invoke fixed, deterministic physical operators (`solarunibench/operators/` and their paper-facing seams in `solarunibench/curation/`) governed by explicit physical thresholds — workflow orchestration only determines execution scheduling and schema validation, never numerical values.

---

## 1. Onboarding New Archives (Adapters & Metadata)

The 11 onboarding sites of v1.0 (1 Folsom + 1 SKIPP'D + 6 SunLake BSRN + 3 GEFCom2014 zones) are integrated through **per-archive adapter classes** that subclass the abstract base — there is no YAML-driven site plugin mechanism. To incorporate a new observing station or raw archive without modifying the core physics engine, provide the following components following the repository's existing layout:

1. **Format Adapter (`solarunibench/pipeline/adapters/<dataset>.py`, subclassing `BaseSolarPipeline` from `solarunibench/pipeline/base.py`):** implements the single abstract method
   `load_and_standardize_raw() -> (df_meta, df_ts)`, mapping the raw archive's file layout, column names, and physical units into the canonical staging schema:
   - `timestamp_utc`: tz-aware `datetime64[ns, UTC]`; naive local timestamps must be localized with the site's `local_timezone` and converted to UTC before returning.
   - `ghi_actual` / `dni_actual` / `dhi_actual` (W m⁻²) and `pv_power_normalized` (0–1, plant-normalized): modalities the source genuinely carries; absent modalities are returned as all-`NA` columns — **never fabricated zeros** (the "Honest NaN" rule enforced by `apply_quality_audit`).
   - Sky/satellite imagery: adapters additionally override `process_and_save_tensors(df_ts)` to write chunked Zarr stores next to the Parquet outputs (conventions in §3, step 4).
   The base class then computes and **owns** the anchor columns (`cos_zenith`, `solar_azimuth`, `ghi_clearsky`, `dni_clearsky`, `dhi_clearsky`) via `compute_astronomy` / `clearsky_ineichen` (`solarunibench/operators/clearsky.py`); any same-named columns returned by the adapter are dropped before the merge so the physics values always win.

2. **Metadata Configuration (in-adapter Python `dict`, not YAML/JSON):** site-level constants are declared as rows of the `df_meta` DataFrame returned by the adapter, using exactly these contract keys (`solarunibench/contract.py`, `META_COLUMNS`):

   | key | type | unit / domain | required |
   |---|---|---|---|
   | `entity_id` | string | unique station/zone id | yes |
   | `dataset_source` | string | archive slug, e.g. `folsom` | yes |
   | `latitude`, `longitude` | float32 | WGS-84 degrees | yes |
   | `elevation_meters` | float32 | m a.s.l. (drives the Ineichen clear-sky model; defaults 0.0 if NaN) | yes |
   | `local_timezone` | string | IANA name, e.g. `America/Los_Angeles` | yes |
   | `surface_albedo` | float32 | 0–1 (Folsom default 0.2) | yes |
   | `capacity_kw` | float32 | kW<sub>peak</sub>; NaN when unknown | yes (may be NaN) |
   | `slope`, `azimuth` | float32 | degrees; NaN for horizontal/unknown | yes (may be NaN) |
   | `geo_inferred` | bool | `True` when lat/lon came from the Eq.-(1) inversion instead of the source | yes |

Minimal example — the GEFCom2014 adapter's row construction (`solarunibench/pipeline/adapters/gefcom.py`), where coordinates were recovered by the clear-sky inversion rather than published:

```python
{"entity_id": eid, "dataset_source": "gefcom2014",
 "latitude": _nn(lat), "longitude": _nn(lon),
 "local_timezone": "UTC", "surface_albedo": 0.2,
 "capacity_kw": float("nan"), "slope": float("nan"),
 "azimuth": float("nan"),
 "geo_inferred": True}
```

---

## 2. Deterministic Physics Operators

The curation pipeline executes the following fixed operators in canonical order:

| Operator / Function | Paper Ref. | Deterministic Physical Action & Thresholds |
|---|---|---|
| `reverse_geolocate(times, power, lat_grid, lon_grid)` — `solarunibench/operators/reverse_geolocate.py` (agent seam: `tool_recover_geolocation` in `solarunibench/agent/tools.py`) | Eq. (1) | Grid-searches lat/lon (default 2° global grid) by minimizing the MSE between the daytime-normalized target shape and the analytic clear-sky curve `clearsky_ineichen`; daytime mask `power > 0.01`, requires ≥ 100 matched samples per grid point. Timezone is derived from the recovered longitude (fixed UTC offset = round(lon/15)). |
| `apply_quality_audit(df_ts, capacity)` — `solarunibench/operators/quality.py` (agent seam: `tool_apply_physical_qc`) | Eq. (2)–(3) | (i) `clearness_index = ghi_actual / (ghi_clearsky + 1e-6)` clipped to **[0, 1.5]**; (ii) nocturnal mask `cos_zenith <= 0` → `ghi_actual`/`pv_power_normalized` zeroed, `clearness_index = 0`, `quality_flag = 3` (night); (iii) PV clipped to `[0, capacity]` (curtailment); (iv) irradiance columns clipped `lower=0` (no negative flux). `quality_flag` coding: `0` good, `1` suspect (reserved), `2` clipped/curtailed (reserved), `3` night — the contract validator (`validate_time_series`) enforces membership in `{0,1,2,3}`. Honest-NaN: a modality that is entirely NA keeps its NA (no fabricated night zeros). |
| `nrbr_map(rgb)` and `optical_flow_reliability(frames, max_pairs=4)` — `solarunibench/curation/feature_extraction.py` (agent seam: `tool_extract_physics_features`) | Eq. (4)–(5) | **NRBR** = `(B − R)/(B + R + 1e-6)` per pixel, averaged over the stack; cloud-consistent pixels are `NRBR > 0.1`. **Optical flow**: Farneback parameters `pyr_scale=0.5, levels=3, winsize=21, iterations=3, poly_n=5, poly_sigma=1.2` on consecutive grayscale pairs (≤ `max_pairs`); the reliability map is the last pair's flow magnitude normalized by its mean, so coherent advection ≈ 1 and noise ≈ 0. Defaults mirror `config/default_curation.yaml`. |

---

## 3. Step-by-Step Execution & Quality-Control Procedures

1. **Schema & Metadata Validation:** `run_pipeline()` first calls `load_and_standardize_raw()`; `validate_metadata` (`solarunibench/contract.py`) raises `ContractError` on any missing `META_COLUMNS` key. Sites lacking published coordinates are recovered **before** adapter return via the Eq.-(1) inversion (`reverse_geolocate` / `recover_geo_clearsky_global`) and flagged `geo_inferred=True` (see the GEFCom2014 adapter).
2. **Physical Boundary Conditioning:** the base merges the computed anchor columns, then applies `apply_quality_audit(df, capacity=max(df_meta.capacity_kw))` — kt clamp, nocturnal zeroing + `quality_flag=3`, PV curtailment clip, non-negative irradiance clip (§2 row 2). The `raw=True` invocation skips this pass and emits the pre-governance frame (the E9 control arm).
3. **Visual Feature Extraction (Conditional):** only when the archive has imagery; the adapter's `process_and_save_tensors` writes Zarr stores, and Eq.-(4)/(5) features are computed on demand by the agent tools / experiments (never materialized into the package). Folsom imagery is stored as JPEG blobs (`sky_images_<year>.zarr`: `images (N,) object VLenBytes` + `images_timestamps (N,) <U25`); SKIPP'D as `(N, 4, 64, 64)` float tensors (RGB + BRR channel); SunLake satellite as `satellite_future.zarr` with `context (200, 11, 64, 64)`, `future (12, 199, 11, 64, 64)`, `uncertainty_map (12, 199, 64, 64)`.
4. **Canonical Export & Post-QC Assertions:** each dataset writes `data/standardized/<dataset>/` containing `metadata.parquet`, `time_series.parquet` (all `TS_COLUMNS` present; NA-filled if absent), `manifest.json` (`dataset`, `raw`, `n_entities`, `n_rows`, `built_at` UTC ISO), `audit.jsonl`, and the optional Zarr stores; `validate_time_series` additionally rejects duplicate `(timestamp_utc, entity_id)` keys and out-of-domain `quality_flag` values before any write. `scripts/build_splits.py` then stamps the frozen `split` column (`train`/`val`/`test`) from `config/splits.yaml`.

---

## 4. Command-Line Execution

Run the deterministic batch curation pipeline or invoke individual verified operators via the CLI entry points:

```bash
# Full batch curation of one archive (physics + QC + contract write):
python scripts/build_dataset.py --dataset folsom [--raw] [--limit N] [--data-root DIR] [--output-root DIR]
#   alias: scripts/run_curation.py takes the identical arguments.

# One-minute onboarding demo on the shipped 1-day sample
# (stages data/sample/raw, runs the REAL pipeline, compares to the reference):
python scripts/build_dataset.py --sample

# Deterministic step-dispatch over the verified operators
# (geolocation / physical QC / visual features + canonical Parquet export):
python scripts/run_agent_curation.py [--instruction "..."] [--export out.parquet]
#   default instruction runs all three steps on the sample; a selective
#   instruction (e.g. "recover the site coordinates only") dispatches a subset.
```
