"""Tests for ``solarunibench.benchmark.evaluator.evaluate_predictions``.

Daytime-masked MAE/RMSE + Forecast Skill% vs a persistence reference. The
daytime mask (``cos_zenith > threshold``) is reused from
:func:`solarunibench.forecast.metrics.daytime_metrics`; this evaluator adds the
signed, percentage Forecast-Skill layer (``Skill_MAE_pct`` / ``Skill_RMSE_pct``)
and the persistence-reference handling.

Sign convention (per the E8 spec): 0% = parity with persistence, positive =
better than persistence, negative = worse. Persistence metric == 0 → NaN
Skill (guarded divide-by-zero).
"""
import math

import numpy as np
import pytest

from solarunibench.benchmark.evaluator import evaluate_predictions


# ---------------------------------------------------------------------------
# Daytime mask: night rows (cos_zenith <= threshold) excluded
# ---------------------------------------------------------------------------

def test_night_rows_excluded_from_metrics():
    """cos_zenith <= threshold rows are dropped (n_daytime < n_total)."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([0.5, 0.5, 0.5, 0.5])
    persistence = np.array([0.0, 0.0, 0.0, 0.0])
    cos_zenith = np.array([0.5, 0.01, 0.5, 0.01])  # samples 0,2 daytime only

    result = evaluate_predictions(y_true, y_pred, cos_zenith, persistence)

    assert result["n_daytime"] == 2
    assert result["n_total"] == 4
    # Daytime subset: y_true=[1,3], y_pred=[0.5,0.5] → MAE = mean(0.5, 2.5) = 1.5.
    assert abs(result["MAE"] - 1.5) < 1e-9


def test_custom_threshold_tightens_daytime_window():
    """Raising the threshold drops borderline daylight rows."""
    y_true = np.array([1.0, 1.0, 1.0])
    y_pred = np.array([0.0, 0.0, 0.0])
    persistence = np.array([0.5, 0.5, 0.5])
    cos_zenith = np.array([0.1, 0.4, 0.9])

    loose = evaluate_predictions(y_true, y_pred, cos_zenith, persistence, threshold=0.05)
    strict = evaluate_predictions(y_true, y_pred, cos_zenith, persistence, threshold=0.5)

    assert loose["n_daytime"] == 3
    assert strict["n_daytime"] == 1


# ---------------------------------------------------------------------------
# Forecast Skill% sign and value
# ---------------------------------------------------------------------------

def test_skill_zero_when_model_equals_persistence():
    """Identical model and persistence predictions → 0% skill (parity)."""
    y_true = np.array([1.0, 2.0, 3.0])
    pred = np.array([0.5, 1.5, 2.5])
    cz = np.array([0.5, 0.5, 0.5])

    result = evaluate_predictions(y_true, pred, cz, pred)

    assert abs(result["Skill_MAE_pct"]) < 1e-9
    assert abs(result["Skill_RMSE_pct"]) < 1e-9


def test_skill_positive_when_model_better_than_persistence():
    """Model error < persistence error → positive Skill%."""
    y_true = np.array([1.0, 2.0, 3.0])
    persistence = np.array([0.0, 1.0, 2.0])  # |err| = 1.0 each
    model_pred = np.array([0.9, 1.9, 2.9])   # |err| = 0.1 each
    cz = np.array([0.5, 0.5, 0.5])

    result = evaluate_predictions(y_true, model_pred, cz, persistence)

    assert result["Skill_MAE_pct"] > 0
    # (1 - 0.1/1.0) * 100 = 90.0
    assert abs(result["Skill_MAE_pct"] - 90.0) < 1e-6
    assert abs(result["Skill_RMSE_pct"] - 90.0) < 1e-6


def test_skill_negative_when_model_worse_than_persistence():
    """Model error > persistence error → negative Skill%."""
    y_true = np.array([1.0, 2.0, 3.0])
    persistence = np.array([0.9, 1.9, 2.9])  # |err| = 0.1 each
    model_pred = np.array([0.0, 1.0, 2.0])   # |err| = 1.0 each
    cz = np.array([0.5, 0.5, 0.5])

    result = evaluate_predictions(y_true, model_pred, cz, persistence)

    assert result["Skill_MAE_pct"] < 0
    # (1 - 1.0/0.1) * 100 = -900.0
    assert abs(result["Skill_MAE_pct"] - (-900.0)) < 1e-6


def test_skill_is_percentage_scale():
    """Skill is reported in PERCENT (×100), not as a fraction in [-1, 1]."""
    y_true = np.array([1.0, 2.0])
    persistence = np.array([0.0, 1.0])  # MAE_ref = 1.0
    model_pred = np.array([0.5, 1.5])   # MAE_model = 0.5
    cz = np.array([0.5, 0.5])

    result = evaluate_predictions(y_true, model_pred, cz, persistence)

    # Skill fraction would be 0.5; percentage must be 50.0.
    assert abs(result["Skill_MAE_pct"] - 50.0) < 1e-9


# ---------------------------------------------------------------------------
# NaN-safety on zero persistence metric
# ---------------------------------------------------------------------------

def test_skill_nan_when_persistence_metric_is_zero():
    """Perfect persistence (MAE_persistence == 0) → Skill is NaN, not a crash."""
    y_true = np.array([1.0, 2.0, 3.0])
    persistence = y_true.copy()  # exact match → MAE_persistence = 0
    model_pred = np.array([0.9, 1.9, 2.9])
    cz = np.array([0.5, 0.5, 0.5])

    result = evaluate_predictions(y_true, model_pred, cz, persistence)

    assert math.isnan(result["Skill_MAE_pct"])
    assert math.isnan(result["Skill_RMSE_pct"])
    # MAE/RMSE of the model are still well-defined and finite.
    assert np.isfinite(result["MAE"])
    assert np.isfinite(result["RMSE"])


# ---------------------------------------------------------------------------
# Finite MAE / RMSE on a normal case
# ---------------------------------------------------------------------------

def test_mae_rmse_finite_and_nonneg():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    persistence = np.array([0.5, 1.5, 2.5, 3.5])
    cz = np.array([0.5, 0.5, 0.5, 0.5])

    result = evaluate_predictions(y_true, y_pred, cz, persistence)

    assert np.isfinite(result["MAE"])
    assert np.isfinite(result["RMSE"])
    assert result["MAE"] >= 0
    assert result["RMSE"] >= 0
    # RMSE >= MAE for the same residuals (Jensen's inequality).
    assert result["RMSE"] >= result["MAE"]


def test_return_keys_exact():
    """Return dict has exactly the spec'd keys."""
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([0.5, 1.5])
    persistence = np.array([0.0, 1.0])
    cz = np.array([0.5, 0.5])

    result = evaluate_predictions(y_true, y_pred, cz, persistence)

    assert set(result.keys()) == {
        "MAE", "RMSE", "Skill_MAE_pct", "Skill_RMSE_pct",
        "n_daytime", "n_total",
    }


def test_skill_matches_manual_formula_on_2d_horizon():
    """Mask broadcasts over a 2-D horizon; Skill matches the by-hand formula."""
    # 3 samples, 2-step horizon. Samples 0,1 daytime (cz>0.05), sample 2 night.
    y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    model_pred = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])  # |err| = 1 each
    persistence = np.array([[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]])  # |err| = 0.5 each
    cz = np.array([0.5, 0.5, 0.01])

    result = evaluate_predictions(y_true, model_pred, cz, persistence)

    # Daytime: 2 samples × 2 horizon = 4 elements; all |err_model|=1, |err_pers|=0.5.
    assert result["n_daytime"] == 2
    assert abs(result["MAE"] - 1.0) < 1e-9
    # Skill = (1 - 1.0/0.5) * 100 = -100
    assert abs(result["Skill_MAE_pct"] - (-100.0)) < 1e-6
