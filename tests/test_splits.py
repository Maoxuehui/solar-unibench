import pandas as pd
from solarunibench.splits import make_splits


def _ts():
    # GEFCom-like: 2013-01-01 .. 2014-06-30 hourly, one zone
    times = pd.date_range("2013-01-01", "2014-06-30 23:00", freq="1h", tz="UTC")
    return pd.DataFrame({"timestamp_utc": times, "entity_id": "GEFCOM_ZONE_1"})


def test_test_set_is_the_held_out_future_month():
    cfg = {"strategy": "temporal", "test_start": "2014-06-01", "val_ratio": 0.1}
    out = make_splits(_ts(), cfg)
    test = out[out["split"] == "test"]
    assert test["timestamp_utc"].min() >= pd.Timestamp("2014-06-01", tz="UTC")
    assert test["timestamp_utc"].max() <= pd.Timestamp("2014-06-30 23:00", tz="UTC")


def test_no_leakage_train_and_val_before_test():
    cfg = {"strategy": "temporal", "test_start": "2014-06-01", "val_ratio": 0.1}
    out = make_splits(_ts(), cfg)
    train_max = out[out["split"] == "train"]["timestamp_utc"].max()
    val_max = out[out["split"] == "val"]["timestamp_utc"].max()
    test_min = out[out["split"] == "test"]["timestamp_utc"].min()
    assert train_max < test_min
    assert val_max < test_min


def test_every_row_labeled():
    cfg = {"strategy": "temporal", "test_start": "2014-06-01", "val_ratio": 0.1}
    out = make_splits(_ts(), cfg)
    assert set(out["split"].unique()) <= {"train", "val", "test"}
    assert out["split"].notna().all()
