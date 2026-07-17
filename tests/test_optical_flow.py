import os

import numpy as np
import pytest

from solarunibench.operators.optical_flow import extrapolate_with_uncertainty


def _shifted_pair(shift=2):
    # A bright square on a dark background. Between f0 and f1 it moves RIGHT by
    # `shift` px. The flow is FORWARD DISPLACEMENT: positive u = motion right
    # (this is SunLake's ctx_opt_flow convention).
    f0 = np.zeros((32, 32), np.float32); f0[10:20, 10:20] = 1.0
    f1 = np.zeros((32, 32), np.float32); f1[10:20, 10 + shift:20 + shift] = 1.0
    u = np.full((32, 32), float(shift), np.float32)  # forward: rightward motion
    v = np.zeros((32, 32), np.float32)
    return f0, f1, u, v


def test_extrapolation_shape_and_bounds():
    f0, f1, u, v = _shifted_pair()
    future, unc = extrapolate_with_uncertainty(u, v, f1, steps_k=4)
    assert future.shape == (4, 32, 32) and future.dtype == np.float32
    assert unc.shape == (4, 32, 32)
    assert np.all(unc >= 0.0) and np.all(unc <= 1.0)


def test_reliability_decays_with_lead_time():
    f0, f1, u, v = _shifted_pair()
    future, unc = extrapolate_with_uncertainty(u, v, f1, steps_k=5)
    # The returned score is a reliability/confidence (high = reliable): despite the
    # contract name "uncertainty", it is non-increasing with lead step.
    mean_by_step = unc.mean(axis=(1, 2))
    assert np.all(np.diff(mean_by_step) <= 1e-6)  # reliability is non-increasing with lead


def test_forward_flow_advects_right():
    # Physically-correct expectation: a FORWARD-displacement flow with positive u
    # (object moving right) must advect content to the RIGHT. f1's bright square
    # sits at rows 10:20, cols 12:22 (shift=2); after one step with u=+2 it must
    # land at rows 10:20, cols 14:24, and its previous left edge must be empty.
    f0, f1, u, v = _shifted_pair(shift=2)
    future, _ = extrapolate_with_uncertainty(u, v, f1, steps_k=2)
    # k=1: square advances right by 2 -> cols 14:24
    assert future[0][10:20, 14:24].max() > 0.9
    # old left edge (cols 12:14) must now be dark
    assert future[0][10:20, 12:14].max() < 0.1
    # k=2: square advances right by 4 -> cols 16:26
    assert future[1][10:20, 16:26].max() > 0.9


def test_negative_flow_advects_left():
    # Symmetry: negative u (leftward motion) must advect content LEFT.
    f1 = np.zeros((32, 32), np.float32); f1[10:20, 12:22] = 1.0
    u = np.full((32, 32), -2.0, np.float32)
    v = np.zeros((32, 32), np.float32)
    future, _ = extrapolate_with_uncertainty(u, v, f1, steps_k=1)
    # k=1: square shifts left by 2 -> cols 10:20
    assert future[0][10:20, 10:20].max() > 0.9
    # old right edge (cols 20:22) must now be dark
    assert future[0][10:20, 20:22].max() < 0.1


slow = pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="set RUN_SLOW=1")



@slow
def test_real_sunlake_warp_direction():
    """Real-data direction check on SunLake CAB/2008.

    The operator's warp (which uses map_x = xx - k*u) must correlate HIGHER with
    the real future frame than the inverted (+) sign would. Locks the
    forward-displacement convention (positive u = motion right) against
    regressions. Multi-channel: IR_108, VIS006, VIS008.
    """
    import cv2
    import deeplake

    ds = deeplake.load("hub://crossvivit/SunLake")
    ctx = ds["2008_nonhrv/context/data"][:8].numpy().astype(np.float32)
    flow = ds["2008_nonhrv/ctx_opt_flow/data"][:8].numpy().astype(np.float32)
    M, C, H, W = ctx.shape

    def _warp(frame, u, v, k, sign):
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        return cv2.remap(frame,
                         (xx + sign * k * u).astype(np.float32),
                         (yy + sign * k * v).astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def _pearson(a, b):
        a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
        a -= a.mean(); b -= b.mean()
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))

    corr_minus, corr_plus = [], []
    for ci in (4, 7, 8):  # IR_108, VIS006, VIS008
        for t in range(M - 2):
            real = ctx[t + 2, ci]
            vx = flow[t + 1, 2 * ci]; vy = flow[t + 1, 2 * ci + 1]
            base = ctx[t + 1, ci]
            corr_minus.append(_pearson(_warp(base, vx, vy, 1, -1), real))
            corr_plus.append(_pearson(_warp(base, vx, vy, 1, +1), real))
    m_minus = float(np.mean(corr_minus))
    m_plus = float(np.mean(corr_plus))
    # The operator uses the '-' sign, so '-' must win.
    assert m_minus > m_plus, (
        f"operator '-' sign should beat '+' on real SunLake: "
        f"-={m_minus:.4f} += {m_plus:.4f}"
    )
    print(f"\n[SunLake warp direction] -={m_minus:.4f} += {m_plus:.4f}")


# ---------------------------------------------------------------------------
# Phase 6b Task 1 — forward-splat revision (Step A: splat helper + meshgrid)
# ---------------------------------------------------------------------------

def test_forward_splat_uniform_destination():
    from solarunibench.operators.optical_flow import _bilinear_forward_splat
    # All source points land exactly on integer dst (no fractional) -> identity-ish
    h = w = 8
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    vals = (xx.astype(np.float32) + 1.0)  # value = col+1
    out, weight = _bilinear_forward_splat(vals, xx.astype(np.float32), yy.astype(np.float32))
    assert out.shape == (h, w) and weight.shape == (h, w)
    assert np.allclose(out, vals)            # integer dst -> exact copy
    assert np.all(weight > 0)                # fully covered


def test_forward_splat_matches_add_at():
    # P0 correctness: bincount path must equal the naive np.add.at accumulation.
    from solarunibench.operators.optical_flow import _bilinear_forward_splat
    rng = np.random.default_rng(0)
    h = w = 12
    vals = rng.random((h, w)).astype(np.float32)
    dst_x = rng.uniform(0, w - 1.0, (h, w)).astype(np.float32)
    dst_y = rng.uniform(0, h - 1.0, (h, w)).astype(np.float32)
    out, _ = _bilinear_forward_splat(vals, dst_x, dst_y)
    # naive reference
    x0 = np.floor(dst_x).astype(int); y0 = np.floor(dst_y).astype(int)
    ref = np.zeros((h, w), np.float32); wsum = np.zeros((h, w), np.float32)
    wx = dst_x - x0; wy = dst_y - y0
    for dx, dy, ww in [(0,0,(1-wx)*(1-wy)),(1,0,wx*(1-wy)),(0,1,(1-wx)*wy),(1,1,wx*wy)]:
        xi, yi = x0+dx, y0+dy
        ok = (xi>=0)&(xi<w)&(yi>=0)&(yi<h)&(ww>0)
        np.add.at(ref, (yi[ok], xi[ok]), (vals*ww)[ok])
        np.add.at(wsum, (yi[ok], xi[ok]), ww[ok])
    covered = wsum > 1e-6; ref[~covered] = 0
    np.putmask(ref, covered, ref/np.maximum(wsum,1e-12))
    assert np.allclose(out, ref, atol=1e-5)


def test_meshgrid_is_cached():
    # P1: _meshgrid is lru_cache-backed; same (h,w) returns identical objects.
    from solarunibench.operators.optical_flow import _meshgrid
    yy1, xx1 = _meshgrid(16, 24)
    yy2, xx2 = _meshgrid(16, 24)
    assert yy1 is yy2 and xx1 is xx2
    assert yy1.dtype == np.float32 and xx1.dtype == np.float32
    assert yy1.shape == (16, 24) and xx1.shape == (16, 24)


# ---------------------------------------------------------------------------
# Phase 6b Task 1 — forward-splat revision (Step B: forward_warp + extrapolate)
# ---------------------------------------------------------------------------

def test_forward_warp_advects_right_and_fills_disocclusion():
    # forward_warp(frame, u, v, steps=1) must net-advect by 1*flow (right for u>0)
    # and fill disocclusion holes (finite everywhere).
    from solarunibench.operators.optical_flow import forward_warp
    frame = np.zeros((32, 32), np.float32); frame[10:20, 10:18] = 1.0
    u = np.full((32, 32), 4.0, np.float32); v = np.zeros((32, 32), np.float32)
    out = forward_warp(frame, u, v, steps=1)
    assert out.shape == (32, 32) and out.dtype == np.float32
    # net advection +4: bright square lands at cols 14:22
    assert out[10:20, 14:22].max() > 0.9
    # old position (cols 10:14) is no longer bright
    assert out[10:20, 10:14].max() < 0.1
    # holes (left-edge disocclusion) are filled -> fully finite
    assert np.all(np.isfinite(out))


def test_forward_warp_steps_scales_displacement():
    # forward_warp(frame, u, v, steps=k) nets k*flow (one-shot / direct).
    from solarunibench.operators.optical_flow import forward_warp
    frame = np.zeros((32, 32), np.float32); frame[10:20, 8:16] = 1.0
    u = np.full((32, 32), 3.0, np.float32); v = np.zeros((32, 32), np.float32)
    out1 = forward_warp(frame, u, v, steps=1)   # +3 -> cols 11:19
    out2 = forward_warp(frame, u, v, steps=2)   # +6 -> cols 14:22
    assert out1[10:20, 11:19].max() > 0.9
    assert out2[10:20, 14:22].max() > 0.9


def test_forward_warp_negative_flow_advects_left():
    from solarunibench.operators.optical_flow import forward_warp
    frame = np.zeros((32, 32), np.float32); frame[10:20, 14:22] = 1.0
    u = np.full((32, 32), -2.0, np.float32); v = np.zeros((32, 32), np.float32)
    out = forward_warp(frame, u, v, steps=1)    # -2 -> cols 12:20
    assert out[10:20, 12:20].max() > 0.9
    assert np.all(np.isfinite(out))


def test_forward_warp_hole_fill_modes():
    # Both hole_fill modes must produce a finite, fully-covered output.
    from solarunibench.operators.optical_flow import forward_warp
    frame = np.zeros((32, 32), np.float32); frame[10:20, 10:18] = 1.0
    u = np.full((32, 32), 4.0, np.float32); v = np.zeros((32, 32), np.float32)
    out_nearest = forward_warp(frame, u, v, steps=1, hole_fill="nearest")
    out_inpaint = forward_warp(frame, u, v, steps=1, hole_fill="inpaint")
    assert np.all(np.isfinite(out_nearest)) and np.all(np.isfinite(out_inpaint))
    # square still advected to the same place under both fills
    assert out_nearest[10:20, 14:22].max() > 0.9
    assert out_inpaint[10:20, 14:22].max() > 0.9


def test_extrapolate_incremental_kflow():
    # incremental=True (default): future[k-1] = frame + k*flow (cumulative).
    from solarunibench.operators.optical_flow import extrapolate_with_uncertainty
    frame = np.zeros((32, 32), np.float32); frame[10:20, 10:20] = 1.0
    u = np.full((32, 32), 2.0, np.float32); v = np.zeros((32, 32), np.float32)
    future, unc = extrapolate_with_uncertainty(u, v, frame, steps_k=3)
    assert future.shape == (3, 32, 32)
    # k=1:+2 cols 12:22 | k=2:+4 cols 14:24 | k=3:+6 cols 16:26
    assert future[0][10:20, 12:22].max() > 0.9
    assert future[1][10:20, 14:24].max() > 0.9
    assert future[2][10:20, 16:26].max() > 0.9
    # reliability still non-increasing with lead
    assert np.all(np.diff(unc.mean(axis=(1, 2))) <= 1e-6)


def test_extrapolate_direct_mode_kflow():
    # incremental=False (direct): one-shot forward_warp(frame, u, v, k) per step.
    from solarunibench.operators.optical_flow import extrapolate_with_uncertainty
    frame = np.zeros((32, 32), np.float32); frame[10:20, 10:20] = 1.0
    u = np.full((32, 32), 2.0, np.float32); v = np.zeros((32, 32), np.float32)
    future, _ = extrapolate_with_uncertainty(u, v, frame, steps_k=3, incremental=False)
    # Same cumulative k*flow displacement as incremental.
    assert future[0][10:20, 12:22].max() > 0.9
    assert future[1][10:20, 14:24].max() > 0.9
    assert future[2][10:20, 16:26].max() > 0.9

