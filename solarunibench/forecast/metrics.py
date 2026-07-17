"""Forecast scoring metrics.

Pure-numpy point and probabilistic metrics used to score predictions. All point
metrics are NaN-safe (NaNs in ``y_true``/``pred`` are masked out, not counted);
MAPE additionally guards against zero denominators. ``skill_vs_reference``
follows the convention **0 = parity** with the reference forecast (e.g. Smart
Persistence) and **positive = better than reference**.

Shapes
------
- Point metrics (``mae``/``rmse``/``mape``/``skill_vs_reference``): ``y_true`` and
  ``pred`` of shape ``(n, horizon)`` or ``(n,)``.
- Quantile metrics (``pinball_loss``/``crps``): ``quantile_pred`` of shape
  ``(n, Q, horizon)`` paired with ``y_true`` of shape ``(n, horizon)``. For
  convenience, a 2-D ``quantile_pred`` of shape ``(Q, horizon)`` (single sample,
  leading axis omitted) or ``(n, Q)`` (single-step horizon omitted) is also
  accepted and reshaped automatically.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "mae",
    "rmse",
    "mape",
    "skill_vs_reference",
    "pinball_loss",
    "crps",
]


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    """Boolean mask of entries finite across all broadcasted arrays."""
    shape = np.broadcast_shapes(*[np.asarray(a).shape for a in arrays])
    mask = np.ones(shape, dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    return mask


def mae(y_true, pred) -> float:
    """Mean Absolute Error, ignoring NaNs in either array."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float)
    mask = _finite_mask(y, p)
    if not mask.any():
        return float("nan")
    return float(np.abs((y - p)[mask]).mean())


def rmse(y_true, pred) -> float:
    """Root Mean Squared Error, ignoring NaNs in either array."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float)
    mask = _finite_mask(y, p)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(((y - p)[mask] ** 2).mean()))


def mape(y_true, pred) -> float:
    """Mean Absolute Percentage Error (fraction, not %).

    NaN-safe and zero-denominator safe: entries where ``y_true`` is NaN, ``pred``
    is NaN, or ``y_true == 0`` are skipped.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float)
    mask = _finite_mask(y, p) & (np.abs(y) > 0.0)
    if not mask.any():
        return float("nan")
    # Guard zero denominators silently (masked entries are excluded anyway).
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.abs((y - p) / y)
    return float(ratio[mask].mean())


def skill_vs_reference(y_true, pred, ref) -> float:
    """Forecast skill score vs a reference forecast (e.g. Smart Persistence).

    Defined as ``1 - mae(y, pred) / mae(y, ref)``. **0 = parity** with the
    reference, **positive = better than reference**, **negative = worse**.
    """
    num = mae(y_true, pred)
    den = mae(y_true, ref)
    if not np.isfinite(num) or not np.isfinite(den):
        return float("nan")
    if den == 0.0:
        # Perfect reference; parity (0) if pred is also perfect, else +inf.
        return 0.0 if num == 0.0 else float("inf")
    return float(1.0 - num / den)


def _prepare_quantile(y_true, quantile_pred, quantiles):
    """Align ``y_true`` (n, H) and ``quantile_pred`` to (n, Q, H).

    Sorts quantile predictions by ``quantiles`` ascending so CRPS integration
    proceeds over consecutive quantile levels.
    """
    y = np.asarray(y_true, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2:
        raise ValueError(f"y_true must be 1D or 2D, got shape {y.shape}")

    taus = np.asarray(quantiles, dtype=float).ravel()
    if taus.size == 0:
        raise ValueError("quantiles must be a non-empty sequence")
    if np.any(taus < 0.0) or np.any(taus > 1.0):
        raise ValueError("quantiles must lie in [0, 1]")

    q = np.asarray(quantile_pred, dtype=float)
    Q = taus.size
    if q.ndim == 3:
        pass  # already (n, Q, H)
    elif q.ndim == 2:
        # Disambiguate (Q, H) [single sample] vs (n, Q) [single-step horizon].
        if q.shape == (Q, y.shape[1]):
            q = q[None, :, :]
        elif q.shape == (y.shape[0], Q):
            q = q[:, :, None]
        else:
            # Fall back: assume (Q, H) and broadcast across samples.
            q = q[None, :, :]
    else:
        raise ValueError(
            f"quantile_pred must be 2D or 3D, got shape {q.shape}"
        )

    n, H = y.shape
    if q.shape != (n, Q, H):
        raise ValueError(
            f"quantile_pred shape {q.shape} incompatible with y_true "
            f"{y.shape} and {Q} quantiles (expected {(n, Q, H)})"
        )

    # Sort quantiles ascending so consecutive pairs are well-defined for CRPS.
    order = np.argsort(taus, kind="stable")
    taus = taus[order]
    q = q[:, order, :]
    return y, q, taus


def _quantile_score(y, q_sorted, taus):
    """Pinball/quantile score per element: ρ_τ(y - x_τ).

    ``y`` is (n, H); ``q_sorted`` and ``taus`` are sorted along the Q axis with
    ``q_sorted`` shaped (n, Q, H). Returns (n, Q, H).
    """
    diff = y[:, None, :] - q_sorted                       # y - x_τ
    taus_b = taus.reshape(1, -1, 1)
    return np.maximum(taus_b * diff, (taus_b - 1.0) * diff)


def pinball_loss(y_true, quantile_pred, quantiles) -> float:
    """Mean pinball loss across all quantiles, samples and horizon steps.

    ``ρ_τ(u) = max(τ·u, (τ−1)·u)`` with ``u = y − x_τ``; always non-negative.
    """
    y, q, taus = _prepare_quantile(y_true, quantile_pred, quantiles)
    scores = _quantile_score(y, q, taus)                  # (n, Q, H)
    mask = np.isfinite(scores) & np.isfinite(y[:, None, :])
    if not mask.any():
        return float("nan")
    return float(scores[mask].mean())


def crps(y_true, quantile_pred, quantiles) -> float:
    """Continuous Ranked Probability Score via the quantile decomposition.

    Uses the identity ``CRPS = ∫₀¹ ρ_τ(y − x_τ) dτ`` and integrates it with the
    trapezoidal rule over consecutive quantile pairs. Always non-negative.
    """
    y, q, taus = _prepare_quantile(y_true, quantile_pred, quantiles)
    scores = _quantile_score(y, q, taus)                  # (n, Q, H), sorted on Q
    if scores.shape[1] < 2:
        # Single quantile: no pair to integrate over; fall back to the score.
        integrand = scores
    else:
        dtau = np.diff(taus)                              # (Q-1,)
        trapezoid = 0.5 * (scores[:, :-1, :] + scores[:, 1:, :])  # (n, Q-1, H)
        integrand = trapezoid * dtau.reshape(1, -1, 1)
    mask = np.isfinite(integrand) & np.isfinite(y[:, None, :])
    if not mask.any():
        return float("nan")
    return float(integrand[mask].mean())
