"""Pull SunLake tabular PCCI (GHI/DIF/DIR) from the Activeloop hub -> local parquet.

Phase 3 (tabular) only: satellite context/opt_flow imagery is deferred to Phase 4.
Hub: hub://crossvivit/SunLake (public, read-only).
"""
from __future__ import annotations
import sys
from pathlib import Path

import os

_REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = Path(os.environ.get(
    "SOLAR_RAW_ROOT",
    "D:/monica/workspace/solar_benchmark_dataset"))
import pandas as pd
_REPO_ROOT = Path(__file__).resolve().parents[2]

ROOT = Path(__file__).resolve().parents[2]
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
    root = RAW_ROOT / "SunLake_data"
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
    ap.add_argument("--out", default=str(RAW_ROOT / "SunLake_data" / "sunlake_tabular.parquet"))
    args = ap.parse_args()
    print(pull_sunlake_tabular(HUB, args.out))
