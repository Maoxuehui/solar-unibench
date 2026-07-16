"""GEFCom2014 adapter — Solar track only.

Geo-recovery uses two **complementary** methods, selected by data availability:

* **Default -- ``recover_geo_clearsky_global``** (Method 5 global, POWER only): pvlib
  Ineichen clear-sky GHI on a global grid, gamma peak-normalized MSE. Returns the
  top-N candidates and flags N/S mirror pairs (the inherent hemisphere ambiguity of
  clear-sky MSE on cloudy PV). Always run (POWER is always present); it is the *only*
  option when VAR178 is missing.
* **Disambiguator -- ``recover_geo_tsr``** (Method 1, VAR178 analytical): precise,
  no cloud interference, no gamma assumption. Run when ``VAR178`` is present. Its
  coord populates ``df_meta`` and is used to **disambiguate** the clear-sky mirror
  candidates (pick the hemisphere).

``reverse_geolocate`` (Method 3 simplified) is retained as a cheap baseline for the
E1 comparison table. ``validate_with_tsr`` reports the R^2 of the chosen coords
against observed TSR. All recoveries + disambiguation + TSR validation are stashed on
``self._last_validation``.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from ..base import BaseSolarPipeline
from ...operators.reverse_geolocate import reverse_geolocate
from ...operators.geo_recover import recover_geo_tsr, recover_geo_clearsky_global
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
            has_var178 = bool(np.isfinite(tsr).any())

            # --- DEFAULT recovery: clear-sky global (POWER only) ---
            cs_rec = recover_geo_clearsky_global(times, power)
            cs_lat, cs_lon = cs_rec.get("latitude"), cs_rec.get("longitude")
            mirror_pairs = cs_rec.get("details", {}).get("mirror_pairs", [])
            has_mirror = bool(mirror_pairs)

            # --- DISAMBIGUATOR: TSR (VAR178) analytical, when available ---
            if has_var178:
                tsr_rec = recover_geo_tsr(times, tsr)
            else:
                tsr_rec = {"latitude": None, "longitude": None,
                           "details": {"method": "TSR (none: VAR178 missing)"}}
            tsr_lat, tsr_lon = tsr_rec.get("latitude"), tsr_rec.get("longitude")

            # --- Coord selection: TSR (precise) when available, else clear-sky best ---
            # TSR also disambiguates the clear-sky N/S mirror candidates (picks hemisphere).
            if tsr_lat is not None and tsr_lon is not None:
                lat, lon = tsr_lat, tsr_lon
                coord_source = "tsr"
            else:
                lat, lon = cs_lat, cs_lon  # may be None on tiny/empty fixtures
                coord_source = "clearsky_global"

            # --- TSR validation of the chosen coords (geometry self-consistency) ---
            if lat is not None and lon is not None and has_var178:
                val = validate_with_tsr(times, tsr, lat, lon)
            else:
                val = {}

            # --- Cheap Method-3 baseline (E1 comparison table) ---
            simple_rec = reverse_geolocate(times, power)

            # --- Hemisphere agreement between clear-sky best and TSR ---
            def _hemi(x):
                return None if x is None else ("north" if float(x) >= 0 else "south")
            cs_hemi = _hemi(cs_lat)
            tsr_hemi = _hemi(tsr_lat)
            if cs_hemi is not None and tsr_hemi is not None:
                agree = (cs_hemi == tsr_hemi)
            else:
                agree = None
            mirror_resolved = has_mirror and tsr_hemi is not None

            def _nn(x):
                return float("nan") if x is None else float(x)

            meta_rows.append({
                "entity_id": eid, "dataset_source": "gefcom2014",
                "latitude": _nn(lat), "longitude": _nn(lon),
                "elevation_meters": float("nan"),
                "local_timezone": "UTC", "surface_albedo": 0.2,
                "capacity_kw": float("nan"), "slope": float("nan"), "azimuth": float("nan"),
                "geo_inferred": True,
                "geo_inferred_method": coord_source,
                "clearsky_mirror_ambiguous": has_mirror and coord_source != "tsr",
            })
            nwp_present = [v for v in NWP_VARS if v in g.columns]
            for _, row in g.iterrows():
                rec_row = {"timestamp_utc": row["timestamp_utc"], "entity_id": eid,
                           "pv_power_normalized": float(row["pv_power_normalized"])}
                for v in nwp_present:
                    rec_row[f"nwp_{v}"] = float(row[v])
                ts_rows.append(rec_row)

            self._last_validation[eid] = {
                "clearsky_global": cs_rec,
                "tsr_recovery": tsr_rec,
                "simplified_baseline": simple_rec,
                "tsr_validation": val,
                "disambiguation": {
                    "coord_source": coord_source,
                    "clearsky_best_hemisphere": cs_hemi,
                    "tsr_hemisphere": tsr_hemi,
                    "hemispheres_agree": agree,
                    "mirror_pair_count": len(mirror_pairs),
                    "mirror_resolved_by_tsr": mirror_resolved,
                    "mirror_pairs": mirror_pairs,
                },
            }

            # One-line per-zone summary: clear-sky best, mirror flag, TSR coord, agreement.
            agree_str = "NA" if agree is None else ("Y" if agree else "N")
            mir_str = "Y" if has_mirror else "N"
            print(f"[GEFCom {eid}] clearsky=({_nn(cs_lat):.1f},{_nn(cs_lon):.1f}) "
                  f"mirror={mir_str} "
                  f"tsr=({_nn(tsr_lat):.1f},{_nn(tsr_lon):.1f}) "
                  f"agree={agree_str} -> meta=({_nn(lat):.1f},{_nn(lon):.1f})[{coord_source}]")

        df_meta = pd.DataFrame(meta_rows)
        df_ts = pd.DataFrame(ts_rows)
        return df_meta, df_ts

    def process_and_save_tensors(self, df_ts: pd.DataFrame) -> None:
        return None  # GEFCom has no image/satellite modalities
