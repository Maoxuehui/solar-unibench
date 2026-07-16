"""SunLake satellite context -> satellite_future.zarr (optical-flow extrapolation + uncertainty).

Uses SunLake's PRECOMPUTED ctx_opt_flow (vx/vy per SEVIRI channel). Bounded by limit_frames.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from ..operators.optical_flow import extrapolate_with_uncertainty
from ..operators.zarr_io import write_array

CTX_CHANNELS = ["IR_016", "IR_039", "IR_087", "IR_097", "IR_108", "IR_120", "IR_134",
                "VIS006", "VIS008", "WV_062", "WV_073"]


def extrapolate_context_series(context: np.ndarray, flow: np.ndarray,
                               channels=None, steps_k: int = 12):
    """context (M, C, H, W); flow (M, 2C, H, W) with [vx_c0, vy_c0, vx_c1, vy_c1, ...].
    Returns future (steps_k, M-1, C, H, W), uncertainty (steps_k, M-1, H, W).
    For each consecutive pair (t, t+1), warp frame t+1 by the flow at t+1 for K steps.
    """
    M, C, H, W = context.shape
    channels = list(range(C)) if channels is None else channels
    futures, uncs = [], []
    for t in range(M - 1):
        frame_next = context[t + 1]            # (C,H,W)
        f_t = flow[t + 1]                       # (2C, H, W); layout [vx0,vy0,vx1,vy1,...]
        # representative (shared) uncertainty from channel 0 (per spec: one map per timestep)
        vx0, vy0 = f_t[2 * channels[0]], f_t[2 * channels[0] + 1]
        _, unc_per_ch = extrapolate_with_uncertainty(vx0, vy0, frame_next[channels[0]], steps_k=steps_k)
        fut_per_ch = []
        for ci in channels:
            vx, vy = f_t[2 * ci], f_t[2 * ci + 1]
            f, _ = extrapolate_with_uncertainty(vx, vy, frame_next[ci], steps_k=steps_k)
            fut_per_ch.append(f)                # (K,H,W) per channel
        futures.append(np.stack(fut_per_ch, axis=1))  # (K, C, H, W)
        uncs.append(unc_per_ch)                       # (K, H, W)
    future = np.stack(futures, axis=1)   # (K, M-1, C, H, W)
    unc = np.stack(uncs, axis=1)         # (K, M-1, H, W)
    return future.astype(np.float32), unc.astype(np.float32)


def build_sunlake_satellite_zarr(hub_path: str, out_path, station="CAB", year="2008",
                                 limit_frames=200, steps_k=12) -> dict:
    import deeplake  # lazy
    ds = deeplake.load(hub_path)
    pfx = f"{year}_nonhrv"
    # SunLake hub `context/data` has shape (M, 11, 64, 64) where M = number of
    # timesteps (~7926 for CAB/2008). Indexing `[0]` returns a SINGLE frame
    # (11, 64, 64), not the series — read the whole tensor instead. When a bound
    # is requested, slice the hub tensor directly so we never materialize the
    # full ~7926-frame cube (deeplake supports python slicing on tensors).
    if limit_frames:
        ctx = ds[f"{pfx}/context/data"][:limit_frames].numpy().astype(np.float32)
        flow = ds[f"{pfx}/ctx_opt_flow/data"][:limit_frames].numpy().astype(np.float32)
    else:
        ctx = ds[f"{pfx}/context/data"].numpy().astype(np.float32)        # (M, 11, 64, 64)
        flow = ds[f"{pfx}/ctx_opt_flow/data"].numpy().astype(np.float32)  # (M, 22, 64, 64)
    future, unc = extrapolate_context_series(ctx, flow, channels=list(range(ctx.shape[1])),
                                             steps_k=steps_k)
    out_path = Path(out_path)
    write_array(out_path / "satellite_future.zarr", "context", ctx, chunks=(1, 11, 64, 64))
    write_array(out_path / "satellite_future.zarr", "future", future, chunks=(1, 1, 11, 64, 64))
    write_array(out_path / "satellite_future.zarr", "uncertainty_map", unc, chunks=(1, 1, 64, 64))
    return {"station": station, "year": year, "frames": int(ctx.shape[0]),
            "steps_k": steps_k, "out_path": str(out_path / "satellite_future.zarr")}
