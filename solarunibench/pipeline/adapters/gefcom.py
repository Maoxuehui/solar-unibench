"""GEFCom2014 adapter — Solar track only.

Geo-recovery is performed with three methods and cross-validated against TSR
(top-of-atmosphere radiation, pure geometry):

* **Primary** -- ``recover_geo_tsr`` (Method 1, analytical TSR/VAR178). Its lat/lon
  populate ``df_meta`` (``geo_inferred=True``).
* **Cross-check** -- ``recover_geo_pvlib`` (Method 5, pvlib clear-sky grid search
  on POWER).
* **Baseline** -- ``reverse_geolocate`` (Method 3, simplified clear-sky match),
  retained for comparison.

``validate_with_tsr`` reports the R^2 of the primary coords against observed TSR.
All three recoveries + the TSR validation are stashed on ``self._last_validation``.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from ..base import BaseSolarPipeline
from ...operators.reverse_geolocate import reverse_geolocate
from ...operators.geo_recover import recover_geo_tsr, recover_geo_pvlib
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

        self._last_validation = {}
        meta_rows, ts_rows = [], []
        for z, g in raw.groupby("zone"):
            eid = f"GEFCOM_ZONE_{int(z)}"
            g = g.sort_values("timestamp_utc")
            times = pd.DatetimeIndex(g["timestamp_utc"])
            power = g["pv_power_normalized"].to_numpy(dtype=float)
            tsr = g["VAR178"].to_numpy(dtype=float) if "VAR178" in g else np.full(len(g), np.nan)

            # --- Primary recovery: TSR (VAR178) analytical ---
            tsr_rec = recover_geo_tsr(times, tsr) if np.isfinite(tsr).any() else {
                "latitude": None, "longitude": None, "details": {"method": "TSR (none: VAR178 missing)"}}
            lat, lon = tsr_rec["latitude"], tsr_rec["longitude"]

            # --- TSR validation of the primary coords (geometry self-consistency) ---
            if lat is not None and lon is not None and np.isfinite(tsr).any():
                val = validate_with_tsr(times, tsr, lat, lon)
            else:
                val = {}

            # --- Independent cross-check: pvlib clear-sky grid search on POWER ---
            pvlib_rec = recover_geo_pvlib(times, power)

            # --- Baseline: simplified clear-sky shape match (Method 3) ---
            simple_rec = reverse_geolocate(times, power)

            def _nn(x):
                return float("nan") if x is None else float(x)

            meta_rows.append({
                "entity_id": eid, "dataset_source": "gefcom2014",
                "latitude": _nn(lat), "longitude": _nn(lon),
                "elevation_meters": float("nan"),
                "local_timezone": "UTC", "surface_albedo": 0.2,
                "capacity_kw": float("nan"), "slope": float("nan"), "azimuth": float("nan"),
                "geo_inferred": True,
            })
            nwp_present = [v for v in NWP_VARS if v in g.columns]
            for _, row in g.iterrows():
                rec_row = {"timestamp_utc": row["timestamp_utc"], "entity_id": eid,
                           "pv_power_normalized": float(row["pv_power_normalized"])}
                for v in nwp_present:
                    rec_row[f"nwp_{v}"] = float(row[v])
                ts_rows.append(rec_row)

            self._last_validation[eid] = {
                "tsr_recovery": tsr_rec,
                "pvlib_recovery": pvlib_rec,
                "simplified_baseline": simple_rec,
                "tsr_validation": val,
            }

            # One-line summary per zone.
            pv = pvlib_rec.get("latitude"), pvlib_rec.get("longitude")
            sb = simple_rec.get("latitude"), simple_rec.get("longitude")
            r2 = val.get("r2_latitude", float("nan"))
            print(f"[GEFCom {eid}] TSR=({_nn(lat):.2f},{_nn(lon):.2f}) "
                  f"pvlib=({_nn(pv[0]):.2f},{_nn(pv[1]):.2f}) "
                  f"simple=({_nn(sb[0]):.2f},{_nn(sb[1]):.2f}) tsr_r2_lat={r2:.3f}")

        df_meta = pd.DataFrame(meta_rows)
        df_ts = pd.DataFrame(ts_rows)
        return df_meta, df_ts

    def process_and_save_tensors(self, df_ts: pd.DataFrame) -> None:
        return None  # GEFCom has no image/satellite modalities
