import numpy as np
from solarunibench.forecast.metrics import mae, rmse, mape, skill_vs_reference, pinball_loss, crps

def test_point_metrics():
    y = np.array([[1.0, 2.0],[3.0,4.0]]); p = np.array([[1.5,2.0],[2.0,5.0]])
    assert abs(mae(y,p) - 0.625) < 1e-6
    assert rmse(y,p) > 0

def test_skill_zero_when_equal():
    y = np.array([[1.0,2.0]]); p = np.array([[1.0,2.0]])
    assert abs(skill_vs_reference(y, p, p)) < 1e-6      # same preds -> 0 skill

def test_pinball_crps_shapes():
    y = np.array([[1.0,2.0]]); q = np.array([[0.9,1.8],[1.1,2.2]])  # 2 quantiles
    assert pinball_loss(y, q, quantiles=[0.1,0.9]) >= 0
    assert crps(y, q, quantiles=[0.1,0.9]) >= 0
