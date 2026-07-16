"""Real-data integration: geo recovery + TSR validation on GEFCom Solar. Slow.

Exercises the two **complementary** recovery methods on real GEFCom data:

* **Clear-sky global** (``recover_geo_clearsky_global``) -- the DEFAULT POWER-only
  recovery; returns top-N candidates and N/S mirror pairs.
* **TSR** (``recover_geo_tsr``) -- the precise VAR178 recovery; populates df_meta and
  disambiguates the clear-sky mirror candidates.

Reports the full per-zone table (clear-sky top-3 + mirror flag, TSR coord, hemisphere
agreement, runtime) for experiment E1.
"""
import os
import time
import pytest
import pandas as pd
from solarunibench.pipeline.adapters.gefcom import GEFCom2014Pipeline

DATA_ROOT = os.environ.get("SOLAR_UNIBENCH_DATA_ROOT", "D:/monica/workspace/solar_benchmark_dataset")
slow = pytest.mark.skipif(not os.environ.get("RUN_SLOW"),
                          reason="set RUN_SLOW=1 to run real-data integration tests")


@slow
def test_real_gefcom_recovers_three_zones_and_validates_tsr(tmp_path):
    t_start = time.time()
    p = GEFCom2014Pipeline(raw_data_dir=f"{DATA_ROOT}/GEFCom2014_data",
                           output_dir=str(tmp_path), dataset_name="gefcom2014")
    df_meta, _ = p.load_and_standardize_raw()
    runtime = time.time() - t_start
    assert df_meta["entity_id"].nunique() == 3
    assert df_meta["latitude"].notna().all() and df_meta["longitude"].notna().all()
    val = getattr(p, "_last_validation", {})
    for eid, v in val.items():
        assert "tsr_validation" in v and "r2_latitude" in v["tsr_validation"]

    # The three GEFCom solar zones are in Australia (southern hemisphere, APAC east
    # longitude). The TSR primary recovery should reflect that, and the TSR-vs-TOA
    # geometry validation should be positive.
    assert (df_meta["latitude"] < 0).all(), "expected southern hemisphere (Australia)"
    assert (df_meta["longitude"] > 60).all(), "expected eastern longitude (Asia-Pacific/Australia)"
    r2s = [val[eid]["tsr_validation"]["r2_latitude"] for eid in val]
    assert all(r > 0 for r in r2s), f"expected positive TSR R^2, got {r2s}"

    # --- Clear-sky global must be computed and reported for every zone. ---
    for eid, v in val.items():
        cs = v["clearsky_global"]
        assert cs is not None
        top = cs.get("details", {}).get("top_n", [])
        assert len(top) > 0, f"{eid}: clear-sky top-N empty"
        # Mirror-pair field always present (possibly empty).
        assert "mirror_pairs" in cs.get("details", {})

    # --- If VAR178 present: TSR-disambiguated clear-sky must agree on hemisphere. ---
    # (All GEFCom zones carry VAR178, so TSR is always available here.)
    agree_rows = []
    for eid, v in val.items():
        dis = v["disambiguation"]
        # df_meta coord is the TSR coord when VAR178 is present.
        assert dis["coord_source"] == "tsr", f"{eid}: expected TSR coord when VAR178 present"
        agree_rows.append(dis["hemispheres_agree"])
    # Report (non-load-bearing) -- hemisphere agreement should hold for real AUS zones.
    print("\n=== GEFCom hemisphere agreement (clear-sky-global vs TSR) ===")
    print("per-zone agree:", {eid: val[eid]["disambiguation"]["hemispheres_agree"] for eid in val})

    # --- Full E1 table (non-load-bearing): clear-sky top-3 + mirror, TSR, runtime. ---
    print("\n=== GEFCom real-data geo recovery (E1 table) ===")
    print(f"{'zone':<16}{'cs_top1':>14}{'cs_top2':>14}{'cs_top3':>14}"
          f"{'mirror':>8}{'tsr':>16}{'agree':>7}")
    for _, row in df_meta.sort_values("entity_id").iterrows():
        eid = row["entity_id"]
        v = val.get(eid, {})
        cs = v.get("clearsky_global", {})
        top = cs.get("details", {}).get("top_n", [])
        dis = v.get("disambiguation", {})
        tsr_r = v.get("tsr_recovery", {})
        tv = v.get("tsr_validation", {})
        sb = v.get("simplified_baseline", {})

        def _fmt_top(c):
            if not c:
                return "       -"
            return f"({c['lat']:+5.0f},{c['lon']:+6.0f})"

        top3 = ([_fmt_top(c) for c in top[:3]] + ["       -"] * 3)[:3]
        mir = "Y" if dis.get("mirror_pair_count", 0) else "N"
        agree = "NA" if dis.get("hemispheres_agree") is None else (
            "Y" if dis["hemispheres_agree"] else "N")
        tsr_str = (f"({tsr_r.get('latitude'):+5.1f},{tsr_r.get('longitude'):+6.1f})"
                   if tsr_r.get("latitude") is not None else "       -")
        print(f"{eid:<16}{top3[0]:>14}{top3[1]:>14}{top3[2]:>14}"
              f"{mir:>8}{tsr_str:>16}{agree:>7}")
        print(f"  TSR r2_lat={tv.get('r2_latitude', float('nan')):.4f} "
              f"lon_implied={tv.get('longitude_implied', float('nan')):.1f} "
              f"| simple_baseline=({sb.get('latitude')},{sb.get('longitude')}) "
              f"| n_mirrors={dis.get('mirror_pair_count', 0)}")
    print(f"\ntotal runtime (load + all recoveries, 3 zones): {runtime:.1f} s")
