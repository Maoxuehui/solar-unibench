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
            times = pd.DatetimeIndex(g["timestamp_utc"])
            power = g["pv_power_normalized"].to_numpy(dtype=float)
            rec = reverse_geolocate(times, power)
            lat, lon = rec["latitude"], rec["longitude"]
            tsr = g["VAR178"].to_numpy(dtype=float) if "VAR178" in g else np.full(len(g), np.nan)
            if lat is not None and lon is not None and np.isfinite(tsr).any():
                val = validate_with_tsr(times, tsr, lat, lon)
            else:
                val = {}
            meta_rows.append({
                "entity_id": eid, "dataset_source": "gefcom2014",
                "latitude": float("nan") if lat is None else float(lat),
                "longitude": float("nan") if lon is None else float(lon),
                "elevation_meters": float("nan"),
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
