"""Metric seam: MAE / RMSE / Skill-vs-reference (+ pinball for quantiles).

``skill_vs_reference`` follows the paper's skill definition —
``1 - MAE_model / MAE_reference`` with Smart Persistence as the reference.
"""
from ..forecast.metrics import (
    mae,
    mape,
    pinball_loss,
    rmse,
    skill_vs_reference,
)

__all__ = ["mae", "rmse", "skill_vs_reference", "pinball_loss", "mape"]
