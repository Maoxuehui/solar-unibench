import pandas as pd, numpy as np
from pathlib import Path
from solarunibench.pipeline.adapters.gefcom import GEFCom2014Pipeline, NWP_VARS


def _nwp_row(z, s, power=None):
    """One NWP row with all 12 vars; optionally carrying a POWER column."""
    row = {"ZONEID": z, "TIMESTAMP": s,
           "VAR78": 0.01, "VAR79": 0.02, "VAR134": 9.5e4, "VAR157": 60.0,
           "VAR164": 0.5, "VAR165": 1.0, "VAR166": 1.0, "VAR167": 290.0,
           "VAR169": 5e5, "VAR175": 3e5, "VAR178": 1.2e6, "VAR228": 0.0}
    if power is not None:
        row["POWER"] = power
    return row


def _write_fixture(root: Path):
    solar = root / "GEFCom2014_data" / "Solar" / "Task 15"
    solar.mkdir(parents=True)
    times = pd.date_range("2013-06-01 06:00", periods=12, freq="1h")
    stamp = times.strftime("%Y%m%d %H:%M")
    rng = np.random.default_rng(0)  # deterministic shared fixture
    rows_p, rows_t = [], []
    for z in [1, 2, 3]:
        for s in stamp:
            power = float(np.clip(rng.random(), 0, 1))
            rows_t.append({"ZONEID": z, "TIMESTAMP": s, "POWER": power})
            rows_p.append({"ZONEID": z, "TIMESTAMP": s, "VAR178": 1e6, "VAR169": 5e5,
                           "VAR78": 0, "VAR79": 0, "VAR134": 1e5, "VAR157": 50, "VAR164": 0.5,
                           "VAR165": 1, "VAR166": 1, "VAR167": 290, "VAR175": 3e5, "VAR228": 0,
                           "POWER": power})
    pd.DataFrame(rows_t).to_csv(solar / "train15.csv", index=False)
    pd.DataFrame(rows_p).to_csv(solar / "predictors15.csv", index=False)


def _write_two_task_fixture(root: Path):
    """Mimic the real GEFCom layout that exposes the Solution-loss bug.

    Task 1: short train history (train1 + predictors1, no POWER in predictors).
    Task 2: train2 (train period POWER) + predictors2 (NWP for train AND blind
            period, POWER column only on train-period rows -- as in predictors15.csv).
    Solution to Task 15: ground-truth POWER for the blind-period timestamps
            (the rows that used to be lost to drop_duplicates keep="first").
    """
    solar = root / "GEFCom2014_data" / "Solar"
    (solar / "Task 1").mkdir(parents=True)
    (solar / "Task 2").mkdir(parents=True)
    (solar / "Solution to Task 15").mkdir(parents=True)

    train_times = pd.date_range("2013-06-01 06:00", periods=18, freq="1h")
    blind_times = pd.date_range("2013-07-01 06:00", periods=12, freq="1h")  # daytime-ish
    rng = np.random.default_rng(7)

    # --- Task 1: earlier train block (will be absorbed into train2 superset) ---
    t1_times = train_times[:6]
    t1_rows, p1_rows = [], []
    for z in [1, 2, 3]:
        for s in t1_times:
            p = float(np.clip(rng.random(), 0.05, 0.95))
            t1_rows.append({"ZONEID": z, "TIMESTAMP": s.strftime("%Y%m%d %H:%M"), "POWER": p})
            p1_rows.append(_nwp_row(z, s.strftime("%Y%m%d %H:%M")))
    pd.DataFrame(t1_rows).to_csv(solar / "Task 1" / "train1.csv", index=False)
    pd.DataFrame(p1_rows).to_csv(solar / "Task 1" / "predictors1.csv", index=False)

    # --- Task 2: train (full train period) + predictors (train AND blind, POWER on train rows) ---
    t2_train_rows, p2_rows = [], []
    for z in [1, 2, 3]:
        for s in train_times:
            p = float(np.clip(rng.random(), 0.05, 0.95))
            t2_train_rows.append({"ZONEID": z, "TIMESTAMP": s.strftime("%Y%m%d %H:%M"), "POWER": p})
            p2_rows.append(_nwp_row(z, s.strftime("%Y%m%d %H:%M"), power=p))
        # blind-period predictors rows: NWP present, POWER marked NaN (forecast/blind period)
        for s in blind_times:
            p2_rows.append(_nwp_row(z, s.strftime("%Y%m%d %H:%M"), power=np.nan))
    pd.DataFrame(t2_train_rows).to_csv(solar / "Task 2" / "train2.csv", index=False)
    pd.DataFrame(p2_rows).to_csv(solar / "Task 2" / "predictors2.csv", index=False)

    # --- Solution: ground-truth POWER for the blind period (this is what gets lost in the bug) ---
    sol_rows = []
    for z in [1, 2, 3]:
        for s in blind_times:
            # Distinctive value (>0.5) so we can assert it survived the merge.
            p = 0.7 + 0.1 * (z - 1)
            sol_rows.append({"ZONEID": z, "TIMESTAMP": s.strftime("%Y%m%d %H:%M"), "POWER": p})
    pd.DataFrame(sol_rows).to_csv(solar / "Solution to Task 15" / "Solution to Task 15.csv", index=False)
    return train_times, blind_times


def test_gefcom_loads_three_zones_and_parses_timestamps(tmp_path):
    _write_fixture(tmp_path)
    p = GEFCom2014Pipeline(raw_data_dir=str(tmp_path / "GEFCom2014_data"),
                           output_dir=str(tmp_path / "out"), dataset_name="gefcom2014")
    df_meta, df_ts = p.load_and_standardize_raw()
    assert df_meta["entity_id"].nunique() == 3
    assert {"pv_power_normalized", "timestamp_utc", "entity_id"} <= set(df_ts.columns)
    assert df_ts["timestamp_utc"].dt.tz is not None  # UTC-aware
    assert any(c.startswith("nwp_") for c in df_ts.columns)  # NWP namespaced


def test_gefcom_solution_power_survives_in_blind_period(tmp_path):
    """Regression: 2014-06 blind-period POWER from `Solution to Task 15` must NOT
    be dropped by the train/predictors concat + drop_duplicates(keep="first").

    A blind-period timestamp appears in `predictors2` (NWP, POWER=NaN) AND in
    `Solution` (real POWER). The standardized row must carry Solution's POWER.
    """
    train_times, blind_times = _write_two_task_fixture(tmp_path)
    p = GEFCom2014Pipeline(raw_data_dir=str(tmp_path / "GEFCom2014_data"),
                           output_dir=str(tmp_path / "out"), dataset_name="gefcom2014")
    df_meta, df_ts = p.load_and_standardize_raw()

    # 3 zones preserved
    assert df_meta["entity_id"].nunique() == 3
    # All 12 NWP vars namespaced
    assert {f"nwp_{v}" for v in NWP_VARS} <= set(df_ts.columns)

    df_ts["ts_naive"] = df_ts["timestamp_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
    # Pick one blind timestamp and zone; Solution POWER for (zone=2, blind) is 0.8
    z2 = df_ts[df_ts["entity_id"] == "GEFCOM_ZONE_2"]
    blind_ts = pd.Timestamp("2013-07-01 07:00")
    row = z2[z2["ts_naive"] == blind_ts]
    assert len(row) == 1, f"expected exactly one standardized row at {blind_ts}, got {len(row)}"
    pw = float(row["pv_power_normalized"].iloc[0])
    assert abs(pw - 0.8) < 1e-6, (
        f"Solution POWER for blind-period row was lost / overwritten: got {pw}, expected ~0.8"
    )
    # NWP must still be present on that same blind row (joined from predictors2)
    assert pd.notna(row["nwp_VAR178"].iloc[0]) and pd.notna(row["nwp_VAR169"].iloc[0])

    # Across ALL blind timestamps & zones, POWER must be the Solution value (non-NaN, >0.5)
    blind_mask = df_ts["ts_naive"].isin(blind_times)
    blind_pw = df_ts.loc[blind_mask, "pv_power_normalized"]
    assert blind_pw.notna().all(), "some blind-period rows lost their Solution POWER"
    assert float(blind_pw.min()) >= 0.7, f"blind POWER min {float(blind_pw.min())} < 0.7 (Solution lost)"


def test_gefcom_temporal_splits_have_nonzero_daytime_power(tmp_path):
    """After a chronological split, each of train/val/test must retain non-zero
    daytime POWER (max > 0.5). This guards against the Solution-loss bug leaking
    into the test split (2014-06 blind period)."""
    from solarunibench.splits import make_splits

    _write_two_task_fixture(tmp_path)
    p = GEFCom2014Pipeline(raw_data_dir=str(tmp_path / "GEFCom2014_data"),
                           output_dir=str(tmp_path / "out"), dataset_name="gefcom2014")
    df_meta, df_ts = p.load_and_standardize_raw()

    # test_start sits between the train block (2013-06) and the blind block (2013-07)
    splits = make_splits(df_ts, {"strategy": "temporal", "test_start": "2013-07-01", "val_ratio": 0.3})
    for label in ["train", "val", "test"]:
        sub = splits[splits["split"] == label]
        assert len(sub) > 0, f"split {label} is empty"
        mx = float(sub["pv_power_normalized"].max())
        assert mx > 0.5, f"split {label} daytime POWER max {mx} <= 0.5 (likely Solution POWER lost)"
