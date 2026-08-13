"""Daytime-masked evaluator with Forecast Skill% vs persistence.

The headline E8 evaluator: apply the anti-fraud daytime mask
(``cos_zenith > threshold``) so that nighttime rows — where both target and
prediction are ~0 and the residual is therefore ~0 — cannot deflate MAE/RMSE,
then compute Forecast Skill% of the model forecast against a persistence
reference on that SAME daytime subset.

Daytime masking reuses :func:`solarunibench.forecast.metrics.daytime_metrics`;
this module adds the **signed, percentage Forecast-Skill** layer for both MAE
and RMSE, plus the persistence-reference handling.

Sign convention
---------------
``Skill_MAE_pct = (1 - MAE_model / MAE_persistence) * 100`` and likewise for
RMSE. ``0%`` = parity with persistence, **positive = better than persistence**,
**negative = worse**. When the persistence metric is ``0`` (perfect persistence)
the corresponding Skill is ``NaN`` (guarded divide-by-zero) rather than ±inf.
"""
from __future__ import annotations

import numpy as np

from ..forecast.metrics import daytime_metrics

__all__ = ["evaluate_predictions"]


def _skill_pct(model_metric: float, ref_metric: float) -> float:
    """Signed Forecast Skill%: ``(1 - model_metric / ref_metric) * 100``.

    Sign: ``0`` = parity with the reference, positive = better, negative = worse.
    Returns ``NaN`` when either metric is non-finite or the reference metric is
    ``0`` (divide-by-zero guard).
    """
    if not np.isfinite(model_metric) or not np.isfinite(ref_metric):
        return float("nan")
    if ref_metric == 0.0:
        return float("nan")
    return float((1.0 - model_metric / ref_metric) * 100.0)


def evaluate_predictions(
    y_true,
    y_pred,
    cos_zenith,
    persistence_pred,
    threshold: float = 0.05,
) -> dict:
    """Daytime-masked MAE/RMSE + Forecast Skill% vs a persistence reference.

    Applies the daytime mask (``cos_zenith > threshold``) to ``y_true``,
    ``y_pred`` and ``persistence_pred`` — the SAME mask is used for the model and
    the persistence reference so the Skill comparison is apples-to-apples. MAE
    and RMSE are computed for both forecasts on the daytime subset, and Forecast
    Skill% is reported per metric: ``(1 - metric_model / metric_persistence) *
    100``. Sign: ``0%`` = parity with persistence, positive = better, negative =
    worse.

    Parameters
    ----------
    y_true, y_pred, persistence_pred : array-like, shape ``(n,)`` or ``(n, H)``
        Ground truth, model prediction, and persistence reference prediction.
    cos_zenith : array-like, shape ``(n,)`` or ``(n, H)``
        Issue-time solar-zenith cosine per sample. The daytime mask is per-sample
        (``cos_zenith > threshold``) and broadcasts over the horizon.
    threshold : float, default 0.05
        Day/night cutoff on ``cos_zenith``.

    Returns
    -------
    dict
        ``MAE``, ``RMSE`` (daytime-masked, model), ``Skill_MAE_pct``,
        ``Skill_RMSE_pct`` (signed Forecast Skill% vs persistence), plus
        ``n_daytime`` and ``n_total``. Skill% is ``NaN`` when the corresponding
        persistence metric is ``0`` or non-finite.
    """
    # Reuse daytime_metrics for the masked MAE/RMSE + per-sample mask broadcast.
    # Calling it twice (model and persistence as "prediction") yields both
    # forecasts' metrics on the identical daytime subset.
    model_res = daytime_metrics(y_true, y_pred, cos_zenith, threshold=threshold)
    ref_res = daytime_metrics(
        y_true, persistence_pred, cos_zenith, threshold=threshold
    )

    mae_model = model_res["Daytime_MAE"]
    rmse_model = model_res["Daytime_RMSE"]
    mae_ref = ref_res["Daytime_MAE"]
    rmse_ref = ref_res["Daytime_RMSE"]

    return {
        "MAE": mae_model,
        "RMSE": rmse_model,
        "Skill_MAE_pct": _skill_pct(mae_model, mae_ref),
        "Skill_RMSE_pct": _skill_pct(rmse_model, rmse_ref),
        "n_daytime": model_res["n_daytime"],
        "n_total": model_res["n_total"],
    }
