import numpy as np
from solarunibench.forecast.metrics import (
    mae,
    rmse,
    mape,
    skill_vs_reference,
    pinball_loss,
    crps,
    daytime_metrics,
)

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


# ---------------------------------------------------------------------------
# Daytime-window evaluator (cos_zenith > 0.05 anti-fraud filter)
# ---------------------------------------------------------------------------
# Solar-forecasting eval discipline: score only DAYTIME samples so that
# nighttime rows (target ≈ 0, prediction ≈ 0 → residual ≈ 0) cannot
# fraudulently deflate MAE/RMSE. ``daytime_metrics`` masks per-sample using
# the issue-time cos_zenith; the mask broadcasts over the full horizon.

def test_daytime_metrics_filters_nighttime_and_lifts_mae():
    """Half nighttime (cos_zenith<0.05) → filtered. Daytime_MAE > all-sample MAE.

    Anti-fraud headline: with zero nighttime residuals (y=pred=0), including
    night rows deflates the all-sample MAE. The daytime-filtered MAE must be
    LARGER because only daytime samples (real signal, real error) are counted.
    """
    # 8 samples, single-step horizon. Day/night alternate.
    y_true = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    y_pred = np.array([0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0])
    cos_zenith = np.array([0.5, 0.01, 0.5, 0.01, 0.5, 0.01, 0.5, 0.01])

    result = daytime_metrics(y_true, y_pred, cos_zenith)

    # 4 daytime samples counted, 4 nighttime filtered.
    assert result["n_daytime"] == 4
    assert result["n_total"] == 8
    # Daytime MAE = mean(|1.0 - 0.5| over 4 day samples) = 0.5.
    assert abs(result["Daytime_MAE"] - 0.5) < 1e-6
    assert result["Daytime_RMSE"] > 0
    # All-sample MAE = (4×0.5 + 4×0.0) / 8 = 0.25 — fraudulently low.
    all_mae = mae(y_true, y_pred)
    assert abs(all_mae - 0.25) < 1e-6
    # Headline anti-fraud assertion: filter lifts MAE above the deflated value.
    assert result["Daytime_MAE"] > all_mae
    # No reference given → skill key omitted.
    assert "Daytime_skill" not in result


def test_daytime_metrics_broadcasts_mask_over_horizon():
    """cos_zenith is per-sample (n,); y/p are (n, H) → ALL H steps of a daytime
    sample are scored; nighttime samples drop entirely."""
    # 3 samples, 2-step horizon.
    y = np.array([[1.0, 2.0], [10.0, 20.0], [3.0, 4.0]])  # (3, 2)
    p = np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 4.0]])     # errs: 1,2,10,20,0,0
    cz = np.array([0.5, 0.5, 0.01])                         # samples 0,1 daytime
    result = daytime_metrics(y, p, cz)
    # 2 daytime samples → 4 scored elements with errs [1, 2, 10, 20].
    assert result["n_daytime"] == 2
    assert result["n_total"] == 3
    assert abs(result["Daytime_MAE"] - 8.25) < 1e-6


def test_daytime_metrics_skill_vs_reference():
    """Daytime_skill reuses skill_vs_reference on daytime-masked arrays.

    Sign convention: 0 = parity with reference, positive = better than ref.
    """
    y = np.array([1.0, 1.0, 1.0, 1.0])
    p = np.array([0.5, 0.5, 0.5, 0.5])  # |err| = 0.5 everywhere
    r = np.array([0.0, 0.0, 0.0, 0.0])  # |err| = 1.0 everywhere → ref worse
    cz = np.array([0.5, 0.01, 0.5, 0.01])  # 2 daytime, 2 night
    result = daytime_metrics(y, p, cz, reference=r)
    # Daytime MAE(pred)=0.5, MAE(ref)=1.0 → skill = 1 - 0.5/1.0 = +0.5.
    assert abs(result["Daytime_skill"] - 0.5) < 1e-6
    assert result["n_daytime"] == 2


def test_daytime_metrics_nan_safe():
    """NaNs in y/p at daytime rows are masked out (not counted, not poisoning)."""
    y = np.array([1.0, np.nan, 1.0, 0.0])
    p = np.array([0.5, 0.5, np.nan, 0.0])
    cz = np.array([0.5, 0.5, 0.5, 0.01])  # samples 0,1,2 daytime; 3 night
    result = daytime_metrics(y, p, cz)
    # 3 daytime samples selected; NaNs at idx 1 (y) and idx 2 (p) drop → only idx 0.
    assert result["n_daytime"] == 3
    assert abs(result["Daytime_MAE"] - 0.5) < 1e-6  # only |1.0 - 0.5| counts


def test_daytime_metrics_respects_custom_threshold():
    """threshold is the cos_zenith cutoff (default 0.05); raising it drops more."""
    y = np.array([1.0, 1.0, 1.0])
    p = np.array([0.0, 0.0, 0.0])
    cz = np.array([0.1, 0.4, 0.9])
    # Default threshold 0.05 → all 3 are daytime.
    r_default = daytime_metrics(y, p, cz)
    assert r_default["n_daytime"] == 3
    # Threshold 0.5 → only cz=0.9 counts.
    r_strict = daytime_metrics(y, p, cz, threshold=0.5)
    assert r_strict["n_daytime"] == 1
