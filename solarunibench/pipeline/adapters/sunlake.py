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
        # Honest NaN: SunLake is irradiance-only (no PV sensor) — column is present but
        # entirely null, mirroring Folsom's PV-less convention. The T1-hardened
        # apply_quality_audit guards this with `df[c].notna().any()` so it stays NaN
        # through the pipeline (no fabricated night zeros).
        df_ts["pv_power_normalized"] = pd.NA
        stations = pd.read_parquet(root / "sunlake_stations.parquet")
        df_meta = pd.DataFrame([{
            "entity_id": r["entity_id"], "dataset_source": "sunlake",
            "latitude": r["latitude"], "longitude": r["longitude"],
            "elevation_meters": r["elevation_meters"], "local_timezone": "UTC",
            "surface_albedo": 0.2, "capacity_kw": float("nan"),
            "slope": float("nan"), "azimuth": float("nan"), "geo_inferred": False,
        } for _, r in stations.iterrows()])
        # Hold geo_inferred as a Python bool (object dtype) so its identity (`is False`)
        # survives the DataFrame round-trip — numpy bool_ fails that check (matches Folsom).
        df_meta["geo_inferred"] = df_meta["geo_inferred"].astype(object)
        return df_meta, df_ts

    def process_and_save_tensors(self, df_ts: pd.DataFrame) -> None:
        """Build the satellite future zarr (optical-flow extrapolation) from the
        SunLake hub. Bounded by ``self.config.get("image_limit", 200)`` frames.

        Imagery is pulled on-demand from ``hub://crossvivit/SunLake`` per build
        (not materialized locally) using SunLake's precomputed ``ctx_opt_flow``.
        """
        from ...imaging.sunlake_satellite import build_sunlake_satellite_zarr
        limit_frames = self.config.get("image_limit", 200)
        build_sunlake_satellite_zarr(
            "hub://crossvivit/SunLake",
            self.output_dir,
            station="CAB",
            year="2008",
            limit_frames=limit_frames,
            steps_k=12,
        )
