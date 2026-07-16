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

    # The three GEFCom solar zones are in Australia (southern hemisphere, APAC
    # east longitude). The TSR primary recovery should now reflect that, and the
    # TSR-vs-TOA geometry validation should be positive.
    assert (df_meta["latitude"] < 0).all(), "expected southern hemisphere (Australia)"
    assert (df_meta["longitude"] > 60).all(), "expected eastern longitude (Asia-Pacific/Australia)"
    r2s = [val[eid]["tsr_validation"]["r2_latitude"] for eid in val]
    assert all(r > 0 for r in r2s), f"expected positive TSR R^2, got {r2s}"

    # Surface recovered coords + all methods for the task report (non-load-bearing).
    print("\n=== GEFCom real-data geo recovery (all methods) ===")
    for _, row in df_meta.sort_values("entity_id").iterrows():
        eid = row["entity_id"]
        v = val.get(eid, {})
        tsr_r = v.get("tsr_recovery", {})
        pv_r = v.get("pvlib_recovery", {})
        sb_r = v.get("simplified_baseline", {})
        tv = v.get("tsr_validation", {})
        print(f"{eid}:")
        print(f"  TSR   (primary): lat={tsr_r.get('latitude')}, lon={tsr_r.get('longitude')}, "
              f"tsr_r2_lat={tv.get('r2_latitude', float('nan')):.4f}, "
              f"lon_implied={tv.get('longitude_implied', float('nan')):.4f}")
        print(f"  pvlib (xcheck) : lat={pv_r.get('latitude')}, lon={pv_r.get('longitude')}")
        print(f"  simple(base)   : lat={sb_r.get('latitude')}, lon={sb_r.get('longitude')}")
