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

        # weather (optional join): Folsom_weather.csv are concurrent GROUND
        # measurements, not forecasts -> namespace obs_* (per spec §4.2, nwp_* is
        # reserved for forecasts). Only Folsom_NAM_*.csv is NWP (kept as nwp_*).
        wpath = root / "Folsom_weather.csv"
        if wpath.exists():
            w = pd.read_csv(wpath)
            w["timestamp_utc"] = pd.to_datetime(w["timeStamp"], format="%Y-%m-%d %H:%M:%S").dt.tz_localize("UTC")
            w = w.rename(columns={"air_temp": "obs_air_temp", "relhum": "obs_relhum", "press": "obs_press",
                                  "windsp": "obs_windsp", "winddir": "obs_winddir"})
            irr = irr.merge(w[["timestamp_utc", "obs_air_temp", "obs_relhum", "obs_press",
                               "obs_windsp", "obs_winddir"]], on="timestamp_utc", how="left")

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
                    [c for c in irr.columns if c.startswith(("nwp_", "obs_"))]].copy()
        df_ts["entity_id"] = FOLSOM["entity_id"]
        # Honest NaN: Folsom is irradiance-only (no PV sensor) — column is present but entirely null.
        df_ts["pv_power_normalized"] = pd.NA
        df_meta = pd.DataFrame([{k: v for k, v in FOLSOM.items() if k != "capacity_kw"} | {
            "capacity_kw": FOLSOM["capacity_kw"], "slope": float("nan"), "azimuth": float("nan")}])
        # Hold geo_inferred as a Python bool (object dtype) so its identity (`is False`)
        # survives the DataFrame round-trip — numpy bool_ fails that check.
        df_meta["geo_inferred"] = df_meta["geo_inferred"].astype(object)
        return df_meta, df_ts

    def process_and_save_tensors(self, df_ts: pd.DataFrame) -> None:
        return None  # sky/satellite imagery is Phase 4
