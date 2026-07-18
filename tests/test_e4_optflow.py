"""Tests for E4 — Optical-flow extrapolation quality + uncertainty calibration.

Spec §11 (E4). *Hypothesis: forward-splat-extrapolated future frames track the
real future within the predicted uncertainty envelope; quality decays with
lead-time K.*

Three outputs (all routed through the shared :mod:`solarunibench.viz` system):

* **Sequence gallery** (``fig_optflow_sequence.png``) — for one lead, context →
  extrapolated → truth side-by-side with the uncertainty (reliability) row.
* **Decay curve** (``fig_optflow_psnr_decay.png``) — PSNR / SSIM of extrapolated
  vs real future across lead K = 1..steps_k.
* **Uncertainty reliability** (``fig_uncertainty_reliability.png``) — prediction-
  interval coverage vs nominal level (a reliability diagram) + the calibration
  curve of predicted vs empirical reliability.

The fast tests use a SYNTHETIC context series whose frames translate by a KNOWN
constant per-step shift, so the real future at lead K is the seed translated by
K steps. The extrapolation is produced by the authoritative
:func:`extrapolate_with_uncertainty` (forward-splat) via the SunLake builder's
:func:`extrapolate_context_series` (the same path that built the real zarr), and
the builder-emitted future array is compared against the genuinely-translated
real context frames. A ``RUN_SLOW``-gated test runs E4 end-to-end on the real
SunLake ``satellite_future.zarr``.
"""
import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import shift as ndi_shift

from solarunibench.experiments import e4_optflow
from solarunibench.imaging.sunlake_satellite import extrapolate_context_series

# Keep pytest output pristine.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

slow = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW"),
    reason="set RUN_SLOW=1 to run the real-data E4 integration test",
)

_REAL_DATA_ROOT = "D:/monica/workspace/solar-unibench/data/standardized"


# ---------------------------------------------------------------------------
# Image-quality primitives (PSNR / SSIM) — unit tests on tiny arrays
# ---------------------------------------------------------------------------

def test_psnr_identical_frames_is_inf():
    img = np.zeros((16, 16), dtype=np.float32)
    img[4:8, 4:8] = 1.0
    assert e4_optflow.psnr(img, img) == float("inf")


def test_psnr_known_value():
    # MSE = 1e-4, data_range = 1 → PSNR = 10*log10(1/1e-4) = 40 dB.
    truth = np.zeros((100, 100), dtype=np.float32)
    pred = truth + 0.01
    assert e4_optflow.psnr(pred, truth, data_range=1.0) == pytest.approx(40.0, abs=0.05)


def test_ssim_identical_frames_is_one():
    rng = np.random.default_rng(0)
    img = rng.random((32, 32)).astype(np.float32)
    assert e4_optflow.ssim(img, img, data_range=1.0) == pytest.approx(1.0, abs=1e-6)


def test_ssim_decreases_with_noise():
    rng = np.random.default_rng(1)
    img = rng.random((64, 64)).astype(np.float32)
    noisy = np.clip(img + rng.normal(0, 0.2, img.shape), 0.0, 1.0).astype(np.float32)
    assert e4_optflow.ssim(noisy, img, data_range=1.0) < 0.9


# ---------------------------------------------------------------------------
# Synthetic SunLake-style context + flow (known translation)
# ---------------------------------------------------------------------------

def _smooth_frame(h: int = 32, w: int = 32, seed: int = 0) -> np.ndarray:
    """A smooth low-frequency frame (gaussian blob + ramp) in [0, 1]."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h / 2.0, w / 2.0
    blob = np.exp(-(((yy - cy) / 8.0) ** 2 + ((xx - cx) / 10.0) ** 2))
    ramp = xx / w + yy / h
    frame = 0.6 * blob + 0.3 * ramp + 0.05 * rng.random((h, w))
    return np.clip(frame, 0.0, 1.0).astype(np.float32)


def _build_synthetic_satellite(
    n_frames: int = 14,
    n_channels: int = 3,
    steps_k: int = 8,
    shift_u: float = 0.7,
    shift_v: float = 0.4,
    size: int = 32,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A bounded SunLake-shape satellite package with a KNOWN translation.

    Returns ``(context, flow, future, uncertainty)``:

    * ``context`` (M, C, H, W): the seed frame translated by ``t * (shift_v,
      shift_u)`` per step (scipy bilinear shift — independent ground truth).
    * ``flow`` (M, 2C, H, W): the constant per-step translation field
      ``[vx_c, vy_c]`` per channel, layout matching the SunLake hub.
    * ``future`` (K, M-1, C, H, W): ``extrapolate_context_series`` output — the
      authoritative forward-splat extrapolation from ``context[t+1]``.
    * ``uncertainty`` (K, M-1, H, W): the operator's reliability map.

    Because the context series genuinely translates by ``(shift_v, shift_u)``
    per step, the real future at lead K from ``context[t+1]`` is exactly
    ``context[t+1+K]`` — so the decay/reliability evaluation has a real target.
    """
    seed_frame = _smooth_frame(size, size, seed=seed)
    # Build the moving context series: context[t] = seed shifted by t*(dy,dx).
    context = np.stack(
        [ndi_shift(seed_frame, (t * shift_v, t * shift_u), order=1, mode="nearest")
         for t in range(n_frames)],
        axis=0,
    ).astype(np.float32)
    context = np.stack([context] * n_channels, axis=1)  # (M, C, H, W) — same pattern per channel

    h, w = size, size
    # Constant flow field: u = shift_u (content moves right), v = shift_v (down).
    u = np.full((h, w), shift_u, dtype=np.float32)
    v = np.full((h, w), shift_v, dtype=np.float32)
    # Hub layout: [vx_c0, vy_c0, vx_c1, vy_c1, ...] per timestep.
    flow = np.stack(
        [np.stack([u, v] * n_channels, axis=0) for _ in range(n_frames)],
        axis=0,
    ).astype(np.float32)  # (M, 2C, H, W)

    future, unc = extrapolate_context_series(
        context, flow, channels=list(range(n_channels)), steps_k=steps_k,
    )
    return context, flow, future, unc


class _FakeLoader:
    """Mimics :class:`SolarUniBenchLoader` for the satellite cube APIs.

    ``satellite[dataset]`` is a ``(context, future, uncertainty)`` triple or
    ``None`` (no satellite coverage → Honest-skip path).
    """

    def __init__(self, satellite, data_root: str = "/fake"):
        self._satellite = satellite
        self.data_root = data_root

    def _get(self, dataset, kind):
        entry = self._satellite.get(dataset)
        if entry is None:
            return None
        ctx, fut, unc = entry
        if kind == "satellite_context":
            return ctx
        if kind == "satellite_future":
            return fut
        if kind == "satellite_uncertainty":
            return unc
        raise ValueError(f"unknown tensor kind: {kind}")

    def get_tensor_cube(self, dataset, kind):
        return self._get(dataset, kind)


@pytest.fixture
def synth_loader() -> _FakeLoader:
    ctx, _flow, fut, unc = _build_synthetic_satellite()
    return _FakeLoader(satellite={"sunlake": (ctx, fut, unc)})


@pytest.fixture
def no_sat_loader() -> _FakeLoader:
    return _FakeLoader(satellite={"sunlake": None})


# ---------------------------------------------------------------------------
# Contract + figure tests for run(loader, out_dir)
# ---------------------------------------------------------------------------

def test_run_returns_figures_and_metrics_dict(synth_loader, tmp_path):
    res = e4_optflow.run(synth_loader, tmp_path)
    assert isinstance(res, dict)
    assert set(res) >= {"figures", "metrics"}
    for fig in ("fig_optflow_sequence.png", "fig_optflow_psnr_decay.png",
                "fig_uncertainty_reliability.png"):
        p = tmp_path / fig
        assert p.exists(), f"missing {fig}"
        assert p.stat().st_size > 0
    csvs = list((tmp_path / "results").glob("*.csv"))
    assert len(csvs) >= 1
    for fig in ("fig_optflow_sequence.png", "fig_optflow_psnr_decay.png",
                "fig_uncertainty_reliability.png"):
        assert any(fig in p for p in res["figures"])


def test_synthetic_extrapolation_matches_truth_at_short_lead(synth_loader, tmp_path):
    """Load-bearing hypothesis check on synthetic data.

    At lead K=1 the forward-splat extrapolation (one step of the same flow that
    generated the moving context series) must track the real next frame closely
    (high PSNR). This proves the evaluation wiring pairs extrapolation against
    the correct real-future target.
    """
    res = e4_optflow.run(synth_loader, tmp_path)
    decay = res["metrics"]["decay"]
    psnr_k1 = decay["psnr_per_lead"][0]
    assert np.isfinite(psnr_k1) and psnr_k1 > 25.0, (
        f"PSNR@K=1 = {psnr_k1:.2f} dB — extrapolation does not match truth at short lead"
    )


def test_quality_decays_with_lead_time(synth_loader, tmp_path):
    """The headline E4 decay assertion: PSNR at the final lead < PSNR at K=1."""
    res = e4_optflow.run(synth_loader, tmp_path)
    decay = res["metrics"]["decay"]
    psnr = np.asarray(decay["psnr_per_lead"], dtype=float)
    ssim = np.asarray(decay["ssim_per_lead"], dtype=float)
    # Monotone-ish decay: first-lead quality exceeds last-lead quality.
    assert psnr[0] > psnr[-1], (
        f"PSNR not decaying: K1={psnr[0]:.2f} vs Kmax={psnr[-1]:.2f}"
    )
    assert ssim[0] >= ssim[-1], (
        f"SSIM not decaying: K1={ssim[0]:.3f} vs Kmax={ssim[-1]:.3f}"
    )
    # Reported scalar summaries are finite + consistent.
    assert np.isfinite(decay["psnr_k1_mean"])
    assert np.isfinite(decay["ssim_k1_mean"])


def test_coverage_and_reliability_computed(synth_loader, tmp_path):
    """Reliability machinery: empirical coverage in [0,1], ECE finite."""
    res = e4_optflow.run(synth_loader, tmp_path)
    rel = res["metrics"]["reliability"]
    nominal = np.asarray(rel["nominal_levels"], dtype=float)
    coverage = np.asarray(rel["empirical_coverage"], dtype=float)
    assert len(nominal) == len(coverage) >= 2
    assert np.all(np.isfinite(coverage))
    assert np.all(coverage >= 0.0) and np.all(coverage <= 1.0)
    assert np.isfinite(rel["ece"]) and 0.0 <= rel["ece"] <= 1.0
    # Calibration curve (predicted vs empirical reliability per u-bin) reported.
    assert len(rel["calibration_predicted"]) == len(rel["calibration_empirical"])


def test_metrics_finite_and_csv_round_trips(synth_loader, tmp_path):
    res = e4_optflow.run(synth_loader, tmp_path)
    decay = res["metrics"]["decay"]
    for arr_name in ("psnr_per_lead", "ssim_per_lead"):
        arr = np.asarray(decay[arr_name], dtype=float)
        # Per-lead PSNR may legitimately be +inf for an identical frame; mask.
        finite = arr[np.isfinite(arr)]
        assert finite.size > 0 and np.all(finite >= 0), f"{arr_name}: bad values"
    # Results CSV round-trips with the expected per-lead schema.
    csv = list((tmp_path / "results").glob("*.csv"))[0]
    df = pd.read_csv(csv)
    assert {"lead_k", "psnr_db", "ssim"}.issubset(df.columns)
    assert (df["lead_k"] == np.arange(1, len(df) + 1)).all()


def test_run_does_not_mutate_inputs(synth_loader, tmp_path):
    ctx, fut, unc = synth_loader._satellite["sunlake"]
    before = (ctx.copy(), fut.copy(), unc.copy())
    e4_optflow.run(synth_loader, tmp_path)
    np.testing.assert_array_equal(before[0], synth_loader._satellite["sunlake"][0])
    np.testing.assert_array_equal(before[1], synth_loader._satellite["sunlake"][1])
    np.testing.assert_array_equal(before[2], synth_loader._satellite["sunlake"][2])


def test_no_satellite_honest_skip(no_sat_loader, tmp_path):
    """A dataset with no satellite extrapolation cube is Honest-skipped."""
    res = e4_optflow.run(no_sat_loader, tmp_path)
    assert isinstance(res, dict)
    assert res["metrics"]["status"] == "skipped"
    assert not res["figures"], "no figures expected when no satellite data exists"


# ---------------------------------------------------------------------------
# Direct calibration-machinery sanity check (well-calibrated → diagonal)
# ---------------------------------------------------------------------------

def test_reliability_diagonal_when_uncertainty_tracks_error():
    """When the predicted error scale exactly equals the true per-pixel error
    scale, empirical coverage of a Normal prediction interval must match the
    nominal level (within sampling noise) — i.e. the reliability curve lies on
    the diagonal. This validates the calibration math independent of the
    optical-flow operator."""
    rng = np.random.default_rng(7)
    H = W = 32
    # Truth + prediction: pred = truth + Normal(0, sigma_pixel) noise.
    truth = rng.random((H, W)).astype(np.float32)
    sigma_true = rng.uniform(0.01, 0.1, (H, W)).astype(np.float32)
    pred = truth + rng.normal(0, sigma_true).astype(np.float32)
    # Pretend the reliability map perfectly encodes sigma via u = 1 - sigma/max.
    # E4 converts u→sigma_hat via calibration; hand it the TRUE sigma map as
    # the calibration target by passing a reliability map that ranks it.
    rel_map = (1.0 - sigma_true / sigma_true.max()).astype(np.float32)
    # Evaluate coverage directly through the public metric helper.
    out = e4_optflow.coverage_curve(pred, truth, rel_map, rel_map,
                                    nominal_levels=(0.5, 0.8, 0.9, 0.95))
    emp = np.asarray(out["empirical_coverage"])
    nom = np.asarray(out["nominal_levels"])
    # Well-calibrated → empirical within ~0.06 of nominal on this sample size.
    assert np.all(np.abs(emp - nom) <= 0.06), (
        f"reliability machinery not diagonal when calibrated: emp={emp} nom={nom}"
    )


# ---------------------------------------------------------------------------
# Real-data RUN_SLOW integration (real SunLake satellite_future.zarr)
# ---------------------------------------------------------------------------

@slow
def test_run_real_sunlake(tmp_path):
    from solarunibench.loader import SolarUniBenchLoader

    loader = SolarUniBenchLoader(_REAL_DATA_ROOT)
    zarr_ctx = loader.get_tensor_cube("sunlake", "satellite_context")
    if zarr_ctx is None:
        pytest.skip("SunLake satellite_future.zarr not present")
    res = e4_optflow.run(loader, tmp_path)
    assert isinstance(res, dict)
    assert (tmp_path / "fig_optflow_psnr_decay.png").exists()
    assert (tmp_path / "fig_uncertainty_reliability.png").exists()
    assert list((tmp_path / "results").glob("*.csv"))
    decay = res["metrics"]["decay"]
    rel = res["metrics"]["reliability"]
    print("\n=== E4 real-data (SunLake optical-flow extrapolation) ===")
    psnr = np.asarray(decay["psnr_per_lead"], dtype=float)
    ssim = np.asarray(decay["ssim_per_lead"], dtype=float)
    print(f"  PSNR @K=1 = {psnr[0]:.2f} dB   SSIM @K=1 = {ssim[0]:.3f}")
    print(f"  PSNR @Kmax({len(psnr)}) = {psnr[-1]:.2f} dB   "
          f"SSIM @Kmax = {ssim[-1]:.3f}")
    print(f"  reliability ECE = {rel['ece']:.3f}  "
          f"(empirical @nominal=0.9: "
          f"{np.asarray(rel['empirical_coverage'])[np.searchsorted(rel['nominal_levels'], 0.9)]:.3f})")
