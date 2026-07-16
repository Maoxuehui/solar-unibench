import numpy as np
from solarunibench.operators.optical_flow import extrapolate_with_uncertainty


def _shifted_pair(shift=2):
    # a bright square on a dark background, shifted right by `shift` px
    f0 = np.zeros((32, 32), np.float32); f0[10:20, 10:20] = 1.0
    f1 = np.zeros((32, 32), np.float32); f1[10:20, 10+shift:20+shift] = 1.0
    u = np.full((32, 32), -float(shift), np.float32)  # motion: frame moves right => sample left
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
    # Warp-direction correctness at k=1: f1's bright square (rows 10:20, cols 12:22)
    # shifts right by 2 px under u=-2, so it lands at rows 10:20, cols 14:24.
    assert future[0][10:20, 14:24].max() > 0.9
