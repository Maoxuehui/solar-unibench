"""Agent-facing tool layer for the curation pipeline (paper "Agentic Skill").

Each ``tool_*`` function is a thin, documented wrapper around the REAL
physics operators in :mod:`solarunibench.operators` — the same code the
batch pipeline uses, exposed with a stable, agent-readable contract
(clear docstrings, plain-python types, honest return payloads) so an LLM
agent (LangChain / AutoGEN / custom ReAct) can call them via tool-use.

Equation references follow the paper:
  Eq. (1)     clear-sky shape-match geolocation inversion
  Eq. (2)-(3) clearness-index clamp  k_t in [0, 1.5] and nocturnal zeroing
  Eq. (4)-(5) NRBR color ratio and optical-flow reliability
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..operators.brr import generate_brr_channel
from ..operators.quality import apply_quality_audit
from ..operators.reverse_geolocate import reverse_geolocate

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
    """Recover site (latitude, longitude) from a power/GHI time series — Eq. (1).

    Grid-searches lat/lon by matching the normalized daytime power shape to
    an analytic clear-sky curve; only SHAPE information is used, so any
    unit-less series (normalized PV or GHI) works.

    Args:
        times: UTC timestamps aligned with ``power`` (>= 1 day recommended).
        power: 1-D array of daytime-bearing values (W/m^2 or 0-1 normalized).
        lat_grid / lon_grid: optional search grids (default 2-deg global).

    Returns:
        dict with ``latitude``, ``longitude`` (None if no viable match),
        ``timezone`` (fixed-UTC offset of solar noon at the recovered site)
        and ``details`` (best MSE, method name) for the agent's observation.
    """
    out = reverse_geolocate(times, np.asarray(power, dtype=float),
                            lat_grid=lat_grid, lon_grid=lon_grid)
    if out["latitude"] is None:
        return {**out, "timezone": None,
                "note": "no grid point matched; need more daytime samples"}
    lon = out["longitude"]
    tz = int(round(lon / 15.0))
    out["timezone"] = f"UTC{tz:+03d}"
    out["note"] = ("recovered via clear-sky shape match; validate against "
                   "metadata when available (paper E1 blind test)")
    return out


def tool_apply_physical_qc(
    df_ts: pd.DataFrame, capacity: float = 1.0
) -> pd.DataFrame:
    """Apply the physical-bounds governance pass — Eq. (2)-(3).

    Wraps :func:`solarunibench.operators.quality.apply_quality_audit`:
    * nocturnal masking — target values are zeroed where cos(zenith) <= 0;
    * clearness clamp — k_t = GHI / GHI_clearsky clipped to [0, 1.5];
    * per-row ``quality_flag`` (0 good / 1 suspect) and capacity clipping
      for PV series.

    Honest-NaN policy: modalities that are entirely absent stay absent —
    the tool never fabricates zeros for a sensor that does not exist.

    Args:
        df_ts: standardized-shaped frame with at least ``cos_zenith`` and a
            target column (``ghi_actual`` and/or ``pv_power_normalized``);
            ``ghi_clearsky`` enables the k_t column.
        capacity: nameplate cap for PV clipping (normalized units).

    Returns:
        A NEW DataFrame with ``clearness_index`` and ``quality_flag``
        columns added/updated; the input is not mutated.
    """
    return apply_quality_audit(df_ts, capacity=capacity)


def tool_extract_physics_features(
    image_tensor: np.ndarray, max_flow_pairs: int = 4
) -> dict:
    """Compute NRBR + optical-flow reliability from sky/satellite frames — Eq. (4)-(5).

    * NRBR (Eq. 4): the normalized blue-to-red ratio ``(B - R) / (B + R)`` —
      thin-cloud / haze contrast channel (positive = cloud-consistent).
    * Flow reliability (Eq. 5): Farneback motion between consecutive frames;
      the reliability map is the per-pixel flow magnitude normalized by the
      frame-pair mean, so coherent cloud motion reads ~1 and noise reads ~0.

    Args:
        image_tensor: ``(T, H, W, 3)`` RGB stack (uint8/float) or a single
            ``(H, W, 3)`` frame. Grayscale input is rejected honestly.
        max_flow_pairs: cap on consecutive pairs evaluated (last T-1 kept).

    Returns:
        dict with ``nrbr`` (H, W) mean map, ``nrbr_cloud_fraction``,
        ``flow`` (u, v of the last pair), ``reliability`` (H, W) map and
        ``flow_mean_magnitude`` per-pair summary.
    """
    import cv2

    t = np.asarray(image_tensor)
    if t.ndim == 3:
        t = t[None, ...]
    if t.ndim != 4 or t.shape[-1] < 3:
        raise ValueError("image_tensor must be (T, H, W, 3) RGB "
                         f"(got shape {np.shape(image_tensor)})")
    rgb = t[..., :3].astype(np.float32)

    r, b = rgb[..., 0], rgb[..., 2]
    nrbr = (b - r) / (b + r + 1e-6)                      # Eq. (4)
    cloud = nrbr > 0.1
    h, w = nrbr.shape[-2:]

    def _to_gray(frame: np.ndarray) -> np.ndarray:
        scaled = (np.clip(frame, 0, 1) * 255) if frame.max() <= 1.0 else frame
        return cv2.cvtColor(scaled.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    gray = [_to_gray(f) for f in rgb]
    mag_hist, u_last, v_last = [], np.zeros((h, w), np.float32), np.zeros(
        (h, w), np.float32)
    for a, bfr in zip(gray[:-1], gray[1:]):
        fl = cv2.calcOpticalFlowFarneback(a, bfr, None, 0.5, 3, 21, 3, 5,
                                          1.2, 0)
        u_last, v_last = fl[..., 0], fl[..., 1]
        mag_hist.append(float(np.hypot(u_last, v_last).mean()))
        if len(mag_hist) >= max_flow_pairs:
            break
    mag = np.hypot(u_last, v_last)                       # Eq. (5)
    denom = mag.mean() + 1e-9
    return {
        "nrbr": nrbr.mean(axis=0),
        "nrbr_cloud_fraction": float(cloud.mean()),
        "flow": (u_last, v_last),
        "reliability": mag / denom,
        "flow_mean_magnitude": mag_hist,
    }
