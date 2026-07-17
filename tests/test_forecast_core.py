import numpy as np, pandas as pd, pytest
from solarunibench.forecast.core import (PipelineConfig, EndogenousFeatureEngineer,
                                          SolarForecastPipeline, ForecastModel)

def _toy_df(n=60, target="pv_power_normalized"):
    t = pd.date_range("2020-06-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"timestamp_utc": t, "entity_id": "E1",
        target: np.sin(np.arange(n)/6)**2 + 0.1,
        "ghi_clearsky": np.linspace(0, 800, n), "cos_zenith": np.linspace(0, 0.9, n),
        "clearness_index": np.linspace(0, 0.9, n), "split": ["train"]*(n-12) + ["val"]*6 + ["test"]*6})

def test_endogenous_windows_shapes():
    fe = EndogenousFeatureEngineer(target="pv_power_normalized",
        feature_cols=["ghi_clearsky", "cos_zenith", "clearness_index"])
    Xp, Y, T, Xf, cz_issue = fe.create_features(_toy_df(), lookback=6, horizon=3, stride=1)
    assert Xp.ndim == 3 and Xp.shape[1] == 6 and Y.shape[1] == 3
    assert Xf is not None and Xf.shape[1] == 3          # future clearsky+cos_zenith
    assert len(T) == Xp.shape[0] == Y.shape[0]
    # 5th return: issue-time cos_zenith per sample (length n), since cos_zenith
    # is a column in the toy df.
    assert cz_issue is not None
    assert cz_issue.shape == (Xp.shape[0],)


def test_endogenous_features_cos_zenith_issue_none_when_absent():
    """Backward-compat: when cos_zenith is NOT a column, 5th return is None."""
    df = _toy_df().drop(columns=["cos_zenith"])
    fe = EndogenousFeatureEngineer(target="pv_power_normalized")
    _, _, _, _, cz_issue = fe.create_features(df, lookback=6, horizon=3, stride=1)
    assert cz_issue is None

class _Const(ForecastModel):
    def fit(self, Xp, Y, times=None, Xf=None): pass
    def predict(self, Xp, times=None, Xf=None): return np.zeros((len(Xp), 3), np.float32)
    @property
    def name(self): return "const"

def test_pipeline_runs_end_to_end():
    cfg = PipelineConfig(dataset="toy", target="pv_power_normalized", lookback=6, horizon=3)
    res = SolarForecastPipeline(cfg, EndogenousFeatureEngineer(target="pv_power_normalized",
        feature_cols=["ghi_clearsky","cos_zenith","clearness_index"]), _Const()).run(df_provider=lambda c: _toy_df())
    assert res["pred"].shape[0] == res["y_true"].shape[0] and res["pred"].shape[1] == 3


def test_pipeline_emits_daytime_metrics_when_cos_zenith_present():
    """When cos_zenith is available, pipeline.run adds daytime_* keys to result."""
    # Larger test split so at least one window survives (lb=6, hz=3 → need ≥9
    # test rows); cos_zenith rises monotonically so test rows are all daytime.
    n = 60
    t = pd.date_range("2020-06-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({"timestamp_utc": t, "entity_id": "E1",
        "pv_power_normalized": np.sin(np.arange(n) / 6) ** 2 + 0.1,
        "ghi_clearsky": np.linspace(0, 800, n),
        "cos_zenith": np.linspace(0, 0.9, n),
        "clearness_index": np.linspace(0, 0.9, n),
        "split": ["train"] * 30 + ["val"] * 6 + ["test"] * 24})
    cfg = PipelineConfig(dataset="toy", target="pv_power_normalized", lookback=6, horizon=3)
    res = SolarForecastPipeline(cfg, EndogenousFeatureEngineer(target="pv_power_normalized",
        feature_cols=["ghi_clearsky", "cos_zenith", "clearness_index"]),
        _Const()).run(df_provider=lambda c: df)
    # Some test windows survived and produced daytime metrics.
    assert res["pred"].shape[0] > 0
    assert "daytime" in res
    dm = res["daytime"]
    for key in ("Daytime_MAE", "Daytime_RMSE", "n_daytime", "n_total"):
        assert key in dm, f"missing {key}"
    assert dm["n_total"] == res["pred"].shape[0]
    assert 0 <= dm["n_daytime"] <= dm["n_total"]


def test_pipeline_omits_daytime_metrics_when_cos_zenith_absent():
    """Backward-compat: no cos_zenith column → no 'daytime' key in result."""
    df = _toy_df().drop(columns=["cos_zenith"])
    cfg = PipelineConfig(dataset="toy", target="pv_power_normalized", lookback=6, horizon=3)
    res = SolarForecastPipeline(cfg, EndogenousFeatureEngineer(target="pv_power_normalized",
        feature_cols=["ghi_clearsky"]), _Const()).run(df_provider=lambda c: df)
    assert "daytime" not in res


def test_target_missing_raises_value_error():
    """Honest NaN: a missing target column must raise, not fabricate data."""
    fe = EndogenousFeatureEngineer(target="pv_power_normalized")
    bad = _toy_df().drop(columns=["pv_power_normalized"])
    with pytest.raises(ValueError):
        fe.create_features(bad, lookback=6, horizon=3, stride=1)
