"""SKIPPD adapter — PV power (normalized) + ERA5/GFS NWP. Pacific time -> UTC. Known coords."""
from __future__ import annotations
from pathlib import Path
import numpy as np
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
        labels["entity_id"] = SKIPP["entity_id"]
        # _read_split concatenates train+test with no dedup; collapse duplicate
        # (timestamp_utc, entity_id) keys BEFORE the NWP merges so the joins don't
        # fan out and the time_series key stays unique (contract now enforces this).
        labels = labels.drop_duplicates(subset=["timestamp_utc", "entity_id"])
        df_ts = labels[["timestamp_utc", "entity_id", "pv_power_normalized"]].copy()

        # ERA5 (join on valid_time) and GFS (join on time_forecast) -> namespace nwp_*
        for cfg, key in (("ERA5", "valid_time"), ("GFS", "time_forecast")):
            nwp = _read_split(root, cfg)
            if nwp.empty:
                continue
            if cfg == "ERA5":
                # valid_time is a tz-aware scalar (US/Pacific) -> convert to UTC.
                nwp["timestamp_utc"] = pd.to_datetime(nwp[key]).dt.tz_convert("UTC")
            else:  # GFS: time_forecast is an object-dtype column of 2-element
                # datetime64[us] ndarrays — first element is the valid forecast time
                # (UTC-naive), second is a sentinel/placeholder (epoch+1us) to discard.
                # pd.to_datetime on the raw arrays raises TypeError, so extract [0].
                first = nwp[key].apply(
                    lambda a: a[0] if hasattr(a, "__len__") and not isinstance(a, str) else a
                )
                nwp["timestamp_utc"] = pd.to_datetime(first, utc=False).dt.tz_localize("UTC")
            # _read_split concatenated train+test with no dedup, so collapse duplicate
            # timestamp_utc rows before merging — otherwise the join fans out and the
            # time_series key becomes non-unique (contract now enforces uniqueness).
            nwp = nwp.drop_duplicates(subset=["timestamp_utc"])
            feat = [c for c in nwp.columns if c not in (key, "timestamp_utc", "latitude", "longitude")]
            nwp = nwp.rename(columns={c: f"nwp_{c}" for c in feat})
            df_ts = df_ts.merge(nwp[["timestamp_utc"] + [f"nwp_{c}" for c in feat]],
                                on="timestamp_utc", how="left")

        # entity_id was assigned to the labels frame above (pre-dedup) and carried
        # through the NWP merges unchanged.
        # Honest NaN: SKIPPD has no ground irradiance sensor — column is present but
        # entirely null, mirroring Folsom's PV-less convention. The T1-hardened
        # apply_quality_audit guards this with `df[c].notna().any()` so it stays NaN
        # through the pipeline (no fabricated night zeros). Emitted as float NaN
        # (not pd.NA) so the clearness-index division in the audit stays numeric and
        # `.astype("float32")` succeeds — an object-dtype all-NA column would crash it.
        df_ts["ghi_actual"] = np.nan
        df_meta = pd.DataFrame([{k: v for k, v in SKIPP.items() if k != "capacity_kw"} | {
            "capacity_kw": SKIPP["capacity_kw"], "slope": float("nan"), "azimuth": float("nan")}])
        # Hold geo_inferred as a Python bool (object dtype) so its identity (`is False`)
        # survives the DataFrame round-trip — numpy bool_ fails that check (matches Folsom).
        df_meta["geo_inferred"] = df_meta["geo_inferred"].astype(object)
        return df_meta, df_ts

    def process_and_save_tensors(self, df_ts: pd.DataFrame) -> None:
        return None  # sky images / satellite are Phase 4
