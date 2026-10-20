"""Agent-facing Curation Tool Functions (Agentic Curation tool layer).

The paper-facing physics lives in :mod:`solarunibench.curation`
(Eq. 1/2-3/4-5). These ``tool_*`` wrappers give each operator a stable,
LLM-readable contract (typed args, documented returns, honest skip notes)
so any tool-use framework can drive them; see ``SKILL.md`` for the
execution policy and ``solarunibench/agent/__init__.py`` for a LangChain
adapter snippet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..curation.alignment import recover_geolocation
from ..curation.feature_extraction import nrbr_map, optical_flow_reliability
from ..curation.quality_audit import apply_physical_bounds

__all__ = [
    "tool_recover_geolocation",
    "tool_apply_physical_qc",
    "tool_extract_physics_features",
]


def tool_recover_geolocation(
    times: pd.DatetimeIndex,
    power: np.ndarray,
    lat_grid: np.ndarray | None = None,
    lon_grid: np.ndarray | None = None,
) -> dict:
    """Recover site (latitude, longitude, timezone) — Eq. (1).

    Grid-search clear-sky shape matching over the daytime portion of
    ``power`` (unit-less; only shape is used).

    Args:
        times: UTC timestamps aligned with ``power`` (>= 1 day recommended).
        power: 1-D target series (normalized PV or GHI).
        lat_grid / lon_grid: optional search grids (default 2-deg global).

    Returns:
        dict: ``latitude``, ``longitude``, ``timezone`` (``UTC+XX``),
        ``details`` (best MSE, method), and a validation note.
    """
    out = recover_geolocation(times, power, lat_grid=lat_grid,
                              lon_grid=lon_grid)
    out.setdefault("note", "recovered via clear-sky shape match; validate "
                           "against metadata when available (paper E1)")
    return out


def tool_apply_physical_qc(df_ts: pd.DataFrame,
                           capacity: float = 1.0) -> pd.DataFrame:
    """Physical-bounds governance pass — Eq. (2)-(3).

    Nocturnal zeroing (cos_zenith <= 0), clearness clamp k_t in [0, 1.5],
    quality flags and PV capacity clipping. Honest-NaN: absent modalities
    stay absent. Returns a NEW frame (input untouched).
    """
    return apply_physical_bounds(df_ts, capacity=capacity)


def tool_extract_physics_features(image_tensor: np.ndarray,
                                  max_flow_pairs: int = 4) -> dict:
    """NRBR + optical-flow reliability from an RGB stack — Eq. (4)-(5).

    Args:
        image_tensor: ``(T, H, W, 3)`` RGB stack or a single ``(H, W, 3)``
            frame; grayscale input is rejected honestly.
        max_flow_pairs: cap on consecutive frame pairs for flow.

    Returns:
        dict: ``nrbr`` (H, W) map, ``nrbr_cloud_fraction``,
        ``flow`` (u, v of the last pair), ``reliability`` (H, W),
        ``flow_mean_magnitude`` per-pair summary.
    """
    t = np.asarray(image_tensor)
    single = t.ndim == 3
    if single:
        t = t[None, ...]
    if t.ndim != 4 or t.shape[-1] < 3:
        raise ValueError("image_tensor must be (T, H, W, 3) RGB (got "
                         f"{np.shape(image_tensor)})")
    nrbr = nrbr_map(t)
    flow = optical_flow_reliability(t, max_pairs=max_flow_pairs)
    return {
        "nrbr": nrbr,
        "nrbr_cloud_fraction": float((nrbr > 0.1).mean()),
        "flow": flow["flow"],
        "reliability": flow["reliability"],
        "flow_mean_magnitude": flow["flow_mean_magnitude"],
    }
