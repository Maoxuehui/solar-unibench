"""Optical-flow extrapolation + uncertainty (semi-Lagrangian warp + divergence decay).

Takes a PRECOMPUTED flow field (u, v) so the extrapolation is decoupled from how the
flow was obtained (SunLake ships ctx_opt_flow; synthetic tests pass a known flow).
"""
import numpy as np
import cv2


def extrapolate_with_uncertainty(flow_u: np.ndarray, flow_v: np.ndarray,
                                 frame_t1: np.ndarray, steps_k: int = 12):
    """Extrapolate `steps_k` future frames by repeated semi-Lagrangian warp of frame_t1.

    Returns (future (steps_k,H,W) float32, uncertainty (steps_k,H,W) float32 in [0,1]).
    The second array is a reliability/confidence score: highest (=1) at short lead and
    low flow divergence, decaying as lead step grows or local divergence rises. Despite
    the parameter/contract name "uncertainty", high = reliable (per the methodology's
    "物理靠谱程度" convention). The returned name is kept as `uncertainty` for contract
    compatibility.
    """
    u = np.asarray(flow_u, dtype=np.float32)
    v = np.asarray(flow_v, dtype=np.float32)
    frame = np.asarray(frame_t1, dtype=np.float32)
    h, w = frame.shape

    div = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3) + cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3)
    divergence = np.abs(div)

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    alpha, beta = 0.02, 0.05  # lead-time decay / divergence penalty (per methodology)
    futures, unc = [], []
    for k in range(1, steps_k + 1):
        map_x = (xx + k * u).astype(np.float32)
        map_y = (yy + k * v).astype(np.float32)
        warped = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        futures.append(warped)
        u_map = np.exp(-alpha * k) * (1.0 - np.minimum(1.0, beta * k * divergence))
        unc.append(np.clip(u_map, 0.0, 1.0).astype(np.float32))
    return np.asarray(futures, dtype=np.float32), np.asarray(unc, dtype=np.float32)
