"""Tests for E6 — Cross-modal alignment value (Folsom / SKIPPD).

Spec §11 (E6). *Hypothesis: temporal alignment + systematic-lag correction
reduces image↔scalar async label noise and improves multimodal skill.* Two
outputs:

* **Alignment diagnostic** (``fig_align_lag_hist.png``) — the timestamp-diff
  distribution between each image and its matched tabular row for EXACT match
  only vs NN-within-tolerance attach (the lag NN-attach absorbs).
* **Multimodal skill ablation** (``fig_align_skill.png`` +
  ``results/results_e6_alignment.csv``) — an image-feature forecaster (LightGBM
  on image summary stats + endogenous/clear-sky) scored WITHOUT alignment
  (exact-match → no image windows → endogenous-only) vs WITH NN alignment
  (images attached). Both arms share model / horizon / samples; the only
  difference is whether alignment delivers the image modality.

The fast tests use a SYNTHETIC loader + image cube where the sky images are
OFFSET from the tabular timestamps (so exact-match attaches ~0 while NN within
tolerance attaches every image). The forecast target at the horizon is a clean
function of the cloud state encoded in the IMAGE at issue time, while the
endogenous lag reflects a STALE cloud state — so the aligned arm (image
attached) reaches lower error than the unaligned (endogenous-only) arm. A
``RUN_SLOW``-gated test runs E6 end-to-end on the real standardized packages.
"""
import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from solarunibench.experiments import e6_alignment

# LightGBM is quiet by default; keep pytest output pristine regardless.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

slow = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW"),
    reason="set RUN_SLOW=1 to run the real-data E6 integration test",
)

_REAL_DATA_ROOT = "D:/monica/workspace/solar-unibench/data/standardized"

# Synthetic offset between image and tabular timestamps (images lag the tabular
# grid). Chosen so EXACT-match attaches 0 images while NN within ``_TOL``
# attaches every image, and the lag histogram resolves at a clearly non-zero lag.
_OFFSET_SECONDS = 300          # 5 minutes
_TOL = "30min"


# ---------------------------------------------------------------------------
# Synthetic loader + image cube
# ---------------------------------------------------------------------------

class _FakeLoader:
    """Mimics :class:`SolarUniBenchLoader` for tabular + sky-image APIs.

    ``images[dataset]`` is a ``(cube, image_timestamps)`` pair or ``None`` (no
    sky-image coverage → Honest-skip path). The cube is a uint8 RGB array;
    :func:`load_sky_frame` derives the BRR channel on load.
    """

    def __init__(
        self,
        tabular: dict[str, pd.DataFrame],
        images: dict[str, tuple[np.ndarray, np.ndarray] | None],
        data_root: str = "/fake",
    ):
        self._tabular = tabular
        self._images = images
        self.data_root = data_root

    def load_tabular(self, dataset: str, split: str | None = None) -> pd.DataFrame:
        df = self._tabular[dataset].copy()
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)
        return df

    def get_tensor_cube(self, dataset: str, kind: str):
        if kind != "sky_images":
            raise ValueError(f"unknown tensor kind: {kind}")
        entry = self._images.get(dataset)
        return entry[0] if entry is not None else None

    def get_image_timestamps(self, dataset: str):
        entry = self._images.get(dataset)
        return entry[1] if entry is not None else None


def _make_sky_frame_global(cloud: float, rng: np.random.Generator, size: int = 32) -> np.ndarray:
    """One synthetic sky frame as uint8 RGB (3,H,W) whose GLOBAL stats encode cloud.

    Unlike E3's localized blob, the whole frame mixes blue-sky → grey by
    ``cloud`` so the per-channel mean is a clean linear function of ``cloud``
    (R_mean = 50 + 130*cloud, etc.). Image summary statistics therefore carry
    the cloud signal that the endogenous lag lacks.
    """
    bg = np.array([50.0, 80.0, 200.0])       # clear-sky blue
    grey = np.array([180.0, 180.0, 180.0])    # cloud grey
    mix = float(np.clip(cloud, 0.0, 1.0))
    rgb = bg[None, None, :] * (1 - mix) + grey[None, None, :] * mix
    rgb = rgb + rng.normal(0.0, 3.0, (size, size, 3))
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)        # H, W, 3
    return rgb.transpose(2, 0, 1)                        # 3, H, W


def _build_offset_image_dataset(
    name: str,
    target_col: str,
    n_images: int = 240,
    horizon_hours: int = 1,
    offset_seconds: int = _OFFSET_SECONDS,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Bounded synthetic frame + cube with images OFFSET from tabular timestamps.

    Tabular is hourly (``T[i] = start + i*1h``); image ``j`` is stamped at
    ``T[j] + offset_seconds`` so an EXACT timestamp match attaches ~0 images
    while a nearest-neighbour match within tolerance attaches every image. The
    forecast label at issue ``T[i]`` is a clean function of the cloud state
    ``s[i]`` encoded in image ``i`` (the aligned match), while the endogenous
    lag ``target_now`` reflects ``s[i-1]`` (stale) — so the image modality,
    unlocked only by alignment, carries unique forecast signal.
    """
    rng = np.random.default_rng(seed)
    n_rows = n_images + horizon_hours
    times = pd.date_range("2020-06-01 06:00", periods=n_rows, freq="1h", tz="UTC")
    naive_ts = times.tz_convert("UTC").tz_localize(None)

    # Latent cloud state s[i], INDEPENDENT over time so the stale endogenous
    # lag (s[i-1]) carries no information about the horizon target (f(s[i])).
    s = rng.uniform(0.0, 1.0, n_rows)

    baseline, beta = 400.0, 400.0
    clearsky = np.full(n_rows, 800.0)
    # Tabular target column y[i] = baseline + beta*s[i-1]  (reflects PREVIOUS
    # step's cloud — the systematic async between tabular and image modalities).
    y = np.empty(n_rows)
    y[0] = baseline
    y[1:] = baseline + beta * s[:-1]

    df = pd.DataFrame({
        "timestamp_utc": naive_ts,
        "entity_id": f"{name.upper()}_E0",
        target_col: y,
        "ghi_clearsky": clearsky,
        "cos_zenith": 0.55,                 # daytime for every row
        "clearness_index": 1.0 - 0.6 * s,
        "dataset_source": name,
    })
    n_tr = int(n_images * 0.70)
    df["split"] = ["train"] * n_tr + ["test"] * (n_rows - n_tr)

    # Cube: one frame per issue timestamp (frame j encodes s[j]); stamp each at
    # T[j] + offset so exact-match misses but NN-within-tol attaches.
    offset = pd.Timedelta(seconds=offset_seconds)
    issue_times = naive_ts[:n_images]
    img_times = issue_times + offset
    cube = np.stack([_make_sky_frame_global(s[i], rng) for i in range(n_images)], axis=0)
    img_ts = np.array([t.isoformat() for t in img_times])
    return df, cube, img_ts


@pytest.fixture
def fake_loader() -> _FakeLoader:
    df_f, cube_f, ts_f = _build_offset_image_dataset("folsom", "ghi_actual", seed=1)
    df_s, cube_s, ts_s = _build_offset_image_dataset("skippd", "pv_power_normalized", seed=2)
    return _FakeLoader(
        tabular={"folsom": df_f, "skippd": df_s},
        images={"folsom": (cube_f, ts_f), "skippd": (cube_s, ts_s)},
    )


@pytest.fixture
def no_image_loader() -> _FakeLoader:
    """A loader whose datasets have NO sky-image coverage (Honest-skip path)."""
    df, _, _ = _build_offset_image_dataset("gefcom2014", "pv_power_normalized", seed=3)
    return _FakeLoader(tabular={"gefcom2014": df}, images={"gefcom2014": None})


# ---------------------------------------------------------------------------
# Contract tests for run(loader, out_dir)
# ---------------------------------------------------------------------------

def test_run_returns_figures_and_metrics_dict(fake_loader, tmp_path):
    res = e6_alignment.run(fake_loader, tmp_path)
    assert isinstance(res, dict)
    assert set(res) >= {"figures", "metrics"}
    lag_hist = tmp_path / "fig_align_lag_hist.png"
    skill = tmp_path / "fig_align_skill.png"
    assert lag_hist.exists() and lag_hist.stat().st_size > 0
    assert skill.exists() and skill.stat().st_size > 0
    csvs = list((tmp_path / "results").glob("*.csv"))
    assert len(csvs) >= 1
    assert any("fig_align_lag_hist.png" in p for p in res["figures"])
    assert any("fig_align_skill.png" in p for p in res["figures"])


def test_lag_diagnostic_reflects_the_offset(fake_loader, tmp_path):
    """The load-bearing diagnostic check.

    Images are stamped ``offset_seconds`` off the tabular grid, so:
      * EXACT-match attaches ~0 images (exact_count ≈ 0);
      * NN within tolerance attaches every image (nn_count == n_images);
      * the NN lag distribution concentrates at ``offset_seconds``.
    """
    res = e6_alignment.run(fake_loader, tmp_path)
    assert res["metrics"]["datasets"], "no dataset produced a diagnostic"
    for ds, arm in res["metrics"]["datasets"].items():
        lag = arm["lag"]
        assert lag["n_images"] > 0, f"{ds}: no images"
        assert lag["exact_count"] == 0, f"{ds}: exact-match should attach 0 (got {lag['exact_count']})"
        assert lag["nn_count"] == lag["n_images"], (
            f"{ds}: NN should attach every image ({lag['nn_count']}/{lag['n_images']})")
        # The NN lag distribution resolves at the offset (5 min = 300 s).
        assert abs(lag["median_lag_seconds"] - _OFFSET_SECONDS) <= 30.0, (
            f"{ds}: median lag {lag['median_lag_seconds']}s != offset {_OFFSET_SECONDS}s")
        assert abs(lag["mean_lag_seconds"] - _OFFSET_SECONDS) <= 30.0


def test_aligned_arm_beats_unaligned(fake_loader, tmp_path):
    """The load-bearing hypothesis check on synthetic data.

    The image encodes the cloud state that drives the horizon target, while the
    endogenous lag is stale. With NN alignment the image attaches and the
    aligned arm must beat the unaligned (endogenous-only) arm on Daytime skill
    vs Smart Persistence. This proves alignment unlocks useful multimodal
    signal — the whole point of E6.
    """
    res = e6_alignment.run(fake_loader, tmp_path)
    for ds, arm in res["metrics"]["datasets"].items():
        sk = arm["skill"]
        assert np.isfinite(sk["skill_unaligned"]), f"{ds}: unaligned skill not finite"
        assert np.isfinite(sk["skill_aligned"]), f"{ds}: aligned skill not finite"
        assert sk["n_samples"] > 0, f"{ds}: no test samples"
        gain = sk["skill_aligned"] - sk["skill_unaligned"]
        assert gain > 0.0, (
            f"{ds}: aligned did not beat unaligned (Δ={gain:+.4f}); ablation broken")


def test_metrics_finite_and_reported(fake_loader, tmp_path):
    res = e6_alignment.run(fake_loader, tmp_path)
    for ds, arm in res["metrics"]["datasets"].items():
        sk = arm["skill"]
        for key in ("mae_unaligned", "mae_aligned", "mae_sp"):
            assert np.isfinite(sk[key]), f"{ds}/{key} not finite"
            assert sk[key] >= 0, f"{ds}/{key} negative"
        assert np.isfinite(arm["lag"]["median_lag_seconds"]), f"{ds}: median lag not finite"


def test_no_sky_images_honest_skip(no_image_loader, tmp_path):
    """A dataset with no sky-image coverage is Honest-skipped (no crash)."""
    res = e6_alignment.run(no_image_loader, tmp_path)
    assert isinstance(res, dict)
    assert res["metrics"]["datasets"] == {}
    assert not res["figures"], "no figures expected when no sky images exist"


def test_run_does_not_mutate_loader_data(fake_loader, tmp_path):
    before_df = {k: v.copy(deep=True) for k, v in fake_loader._tabular.items()}
    before_cube = {k: v[0].copy() for k, v in fake_loader._images.items() if v is not None}
    e6_alignment.run(fake_loader, tmp_path)
    for k, v in before_df.items():
        pd.testing.assert_frame_equal(v, fake_loader._tabular[k])
    for k, v in before_cube.items():
        np.testing.assert_array_equal(v, fake_loader._images[k][0])


def test_results_csv_round_trips(fake_loader, tmp_path):
    e6_alignment.run(fake_loader, tmp_path)
    csv = list((tmp_path / "results").glob("*.csv"))[0]
    df = pd.read_csv(csv)
    assert {"dataset", "arm", "mae", "skill", "n_samples"}.issubset(df.columns)
    for ds in ("folsom", "skippd"):
        sub = df[df["dataset"] == ds]
        assert (sub["arm"] == "unaligned").any()
        assert (sub["arm"] == "aligned").any()
    # The lag diagnostic is also recorded per dataset.
    assert "median_lag_seconds" in df.columns


# ---------------------------------------------------------------------------
# Real-data RUN_SLOW integration
# ---------------------------------------------------------------------------

@slow
def test_run_real_standardized_packages(tmp_path):
    from solarunibench.loader import SolarUniBenchLoader

    loader = SolarUniBenchLoader(_REAL_DATA_ROOT)
    res = e6_alignment.run(loader, tmp_path)
    assert isinstance(res, dict)
    assert (tmp_path / "fig_align_lag_hist.png").exists()
    assert list((tmp_path / "results").glob("*.csv"))
    print("\n=== E6 real-data headline (Daytime skill vs Smart Persistence) ===")
    for ds, arm in res["metrics"]["datasets"].items():
        lag = arm["lag"]
        sk = arm["skill"]
        print(
            f"  {ds:8s} lag : exact={lag['exact_count']}/{lag['n_images']} "
            f"nn={lag['nn_count']}/{lag['n_images']} "
            f"median_lag={lag['median_lag_seconds']:.1f}s"
        )
        print(
            f"  {ds:8s} skill: unaligned={sk['skill_unaligned']:+.4f} "
            f"aligned={sk['skill_aligned']:+.4f} "
            f"gain={sk['skill_aligned'] - sk['skill_unaligned']:+.4f} (n={sk['n_samples']})"
        )
