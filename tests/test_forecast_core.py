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
    Xp, Y, T, Xf = fe.create_features(_toy_df(), lookback=6, horizon=3, stride=1)
    assert Xp.ndim == 3 and Xp.shape[1] == 6 and Y.shape[1] == 3
    assert Xf is not None and Xf.shape[1] == 3          # future clearsky+cos_zenith
    assert len(T) == Xp.shape[0] == Y.shape[0]

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


def test_target_missing_raises_value_error():
    """Honest NaN: a missing target column must raise, not fabricate data."""
    fe = EndogenousFeatureEngineer(target="pv_power_normalized")
    bad = _toy_df().drop(columns=["pv_power_normalized"])
    with pytest.raises(ValueError):
        fe.create_features(bad, lookback=6, horizon=3, stride=1)
