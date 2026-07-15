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

# Physics/anchor columns the base computes and owns. A standardized df_ts may
# already carry one of these (e.g. a toy ghi_clearsky); the base's value must
# win, so they are dropped from df_ts before the anchor merge to avoid pandas
# suffixing them to _x/_y and leaving the contract column all-NA.
_ANCHOR_COLS = ["cos_zenith", "solar_azimuth", "ghi_clearsky", "dni_clearsky", "dhi_clearsky"]


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
        if limit is not None:
            df_ts = df_ts.groupby("entity_id", group_keys=False).head(limit)

        # The base owns the physics/anchor columns: drop any that the standardized
        # df_ts already carries so the base's computed values win and no _x/_y
        # suffix collision corrupts the contract columns (e.g. ghi_clearsky).
        overlap = [c for c in _ANCHOR_COLS if c in df_ts.columns]
        if overlap:
            df_ts = df_ts.drop(columns=overlap)

        # Merge anchors
        anchors = self._compute_clearsky_and_astronomy(df_meta, df_ts)
        if anchors.empty:
            # Degenerate: no anchor rows produced — skip the merge; the missing
            # columns are NA-filled below so the contract still validates.
            df = df_ts
        else:
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
