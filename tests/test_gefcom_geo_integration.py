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

    # Surface recovered coords + TSR R^2 for the task report (non-load-bearing).
    print("\n=== GEFCom real-data geo recovery + TSR validation ===")
    for _, row in df_meta.sort_values("entity_id").iterrows():
        eid = row["entity_id"]
        tsr = val.get(eid, {}).get("tsr_validation", {})
        print(f"{eid}: lat={row['latitude']:.4f}, lon={row['longitude']:.4f}, "
              f"tsr_r2_lat={tsr.get('r2_latitude', float('nan')):.4f}, "
              f"lon_implied={tsr.get('longitude_implied', float('nan')):.4f}")
