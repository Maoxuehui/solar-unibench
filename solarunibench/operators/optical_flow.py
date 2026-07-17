"""Optical-flow extrapolation + uncertainty (forward-splat + divergence decay).

Takes a PRECOMPUTED flow field (u, v) so the extrapolation is decoupled from how the
flow was obtained (SunLake ships ctx_opt_flow; synthetic tests pass a known flow).

The warp core is the authoritative forward-splat method (per
``docs/references/crossvivit_tscontext_dataset_ref.py`` ``warp_image``): each source
pixel at p carries its value to r = p + k*(u, v). Collisions are averaged by bilinear
weights; disocclusion holes are filled from the nearest covered pixel (or cv2.inpaint).
Because the flow and frame share the same grid here, no prev-grid resample step is
needed — the net advection is simply ``k * flow``.
"""
from functools import lru_cache

import numpy as np
import cv2


@lru_cache(maxsize=64)
def _meshgrid(h: int, w: int):
    """Cached float32 pixel grid. Returns (yy[H,W], xx[H,W]) with ``indexing="ij"``."""
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    return yy, xx


def _bilinear_forward_splat(values, dst_x, dst_y, valid_mask=None):
    """Bilinearly forward-splat source-grid ``values`` onto continuous destination coords.

    P0: a single ``np.bincount`` pass accumulates ALL four corner contributions
    (collision-averaged), matching the naive ``np.add.at`` reference exactly.

    Args:
        values: [H, W] source values.
        dst_x, dst_y: [H, W] continuous destination coords.
        valid_mask: optional [H, W] bool; False entries do not contribute.

    Returns:
        (output[H, W] collision-normalized (0 where uncovered), weight_sum[H, W]).
    """
    values = np.asarray(values, dtype=np.float32)
    dst_x = np.asarray(dst_x, dtype=np.float32)
    dst_y = np.asarray(dst_y, dtype=np.float32)
    h, w = values.shape
    valid_mask = (
        np.ones((h, w), dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    finite = (
        np.isfinite(values) & np.isfinite(dst_x) & np.isfinite(dst_y) & valid_mask
    )
    x0 = np.floor(dst_x).astype(np.int64)
    y0 = np.floor(dst_y).astype(np.int64)
    wx = dst_x - x0
    wy = dst_y - y0
    n = h * w
    flats, wlist, vlist = [], [], []
    for dx, dy, ww in [
        (0, 0, (1 - wx) * (1 - wy)),
        (1, 0, wx * (1 - wy)),
        (0, 1, (1 - wx) * wy),
        (1, 1, wx * wy),
    ]:
        xi = x0 + dx
        yi = y0 + dy
        ok = finite & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h) & (ww > 0)
        if not np.any(ok):
            continue
        flats.append(yi[ok] * w + xi[ok])
        wlist.append(ww[ok])
        vlist.append((values * ww)[ok])
    if not flats:
        return np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    flat = np.concatenate(flats)
    ws = np.concatenate(wlist)
    vs = np.concatenate(vlist)
    weight = np.bincount(flat, weights=ws, minlength=n).astype(np.float32)
    accum = np.bincount(flat, weights=vs, minlength=n).astype(np.float32)
    out = np.zeros(n, np.float32)
    covered = weight > 1e-6
    out[covered] = accum[covered] / weight[covered]
    return out.reshape(h, w), weight.reshape(h, w)


def forward_warp(frame, u, v, k=1, hole_fill="nearest", inpaint_radius=3):
    """Forward-splat ``frame`` by ``k * (u, v)`` (authoritative advection core).

    Each source pixel at p=(y,x) carries ``frame[p]`` to its forward destination
    r = p + k*(u, v). Collisions at a destination pixel are averaged by bilinear
    weights; disocclusion holes (uncovered destinations) are filled via ``hole_fill``:
      - ``"nearest"``: nearest covered pixel (cv2.distanceTransformWithLabels).
      - ``"inpaint"``: cv2.inpaint (Telea, radius ``inpaint_radius``).

    Net advection is ``k * flow`` (positive u -> content moves RIGHT), preserving the
    ``future[k-1] = frame + k*flow`` contract. Does NOT negate the flow; the flow and
    frame share the same grid, so no resample step is needed (unlike the
    prev-grid-aligned ``warp_image`` reference).
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if hole_fill not in ("nearest", "inpaint"):
        raise ValueError(f"hole_fill must be 'nearest' or 'inpaint', got {hole_fill!r}")
    frame = np.asarray(frame, dtype=np.float32)
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    h, w = frame.shape
    yy, xx = _meshgrid(h, w)

    dst_x = xx + k * u
    dst_y = yy + k * v
    valid = np.isfinite(dst_x) & np.isfinite(dst_y)
    predicted, weight = _bilinear_forward_splat(frame, dst_x, dst_y, valid_mask=valid)
    covered = weight > 1e-6
    if np.all(covered):
        return predicted
    if not np.any(covered):
        return frame.copy()
    hole_map = (~covered).astype(np.uint8)
    if hole_fill == "inpaint":
        return cv2.inpaint(
            predicted, hole_map * 255,
            inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA,
        ).astype(np.float32)
    _, labels = cv2.distanceTransformWithLabels(
        hole_map, cv2.DIST_L2, cv2.DIST_MASK_5, labelType=cv2.DIST_LABEL_PIXEL,
    )
    label_values = np.zeros(int(labels.max()) + 1, dtype=np.float32)
    label_values[labels[covered]] = predicted[covered]
    predicted[~covered] = label_values[labels[~covered]]
    return predicted


def extrapolate_with_uncertainty(flow_u: np.ndarray, flow_v: np.ndarray,
                                 frame_t1: np.ndarray, steps_k: int = 12, *,
                                 incremental: bool = True,
                                 hole_fill: str = "nearest",
                                 decay_tau: float = None):
    """Extrapolate `steps_k` future frames by forward-splat advection of frame_t1.

    Returns (future (steps_k,H,W) float32, uncertainty (steps_k,H,W) float32 in [0,1]).
    The second array is a reliability/confidence score: highest (=1) at short lead and
    low flow divergence, decaying as lead step grows or local divergence rises. Despite
    the parameter/contract name "uncertainty", high = reliable (per the methodology's
    "物理靠谱程度" convention). The returned name is kept as `uncertainty` for contract
    compatibility.

    Advection contract (preserved): ``future[k-1]`` = frame advected forward by
    ``k*flow`` (k=1 -> +u px, k=2 -> +2u). Two chaining modes (keyword-only flags):
      - ``incremental=True`` (default): chain ``forward_warp(current, u, v, 1)`` per
        step -> cumulative k*flow. Smaller per-step holes; step-by-step Lagrangian.
      - ``incremental=False`` (direct): one-shot ``forward_warp(frame, u, v, eff_k)``
        per step, where ``eff_k = k * exp(-k / decay_tau)`` if ``decay_tau`` is set
        else ``k``. ``decay_tau`` thus damps the effective advection at long leads
        (direct mode only; an E4 ablation knob).

    ``hole_fill`` is forwarded to ``forward_warp`` (``"nearest"`` | ``"inpaint"``).
    """
    u = np.asarray(flow_u, dtype=np.float32)
    v = np.asarray(flow_v, dtype=np.float32)
    frame = np.asarray(frame_t1, dtype=np.float32)

    div = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3) + cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3)
    divergence = np.abs(div)

    alpha, beta = 0.02, 0.05  # lead-time decay / divergence penalty (per methodology)
    futures, unc = [], []
    current = frame
    for k in range(1, steps_k + 1):
        if incremental:
            # +1*flow per step on the running extrapolate -> cumulative k*flow
            current = forward_warp(current, u, v, 1, hole_fill=hole_fill)
            fut = current
        else:
            eff_k = k * (float(np.exp(-k / decay_tau)) if decay_tau else 1.0)
            fut = forward_warp(frame, u, v, eff_k, hole_fill=hole_fill)
        futures.append(fut)
        u_map = np.exp(-alpha * k) * (1.0 - np.minimum(1.0, beta * k * divergence))
        unc.append(np.clip(u_map, 0.0, 1.0).astype(np.float32))
    return np.asarray(futures, dtype=np.float32), np.asarray(unc, dtype=np.float32)
