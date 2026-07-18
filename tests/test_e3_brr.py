"""Tests for E3 — BRR (Blue-to-Red Ratio) thin-cloud enhancement (Folsom / SKIPPD).

Spec §11 (E3). *Hypothesis: the BRR 4th channel amplifies thin-cloud edges and
improves image-based forecasting.* Two outputs:

* **Qualitative gallery** — clear / thin-cloud / overcast frames rendered as RGB
  vs BRR (BRR should expose thin-cloud structure invisible in RGB).
* **Quantitative ablation** — an image-feature baseline (per-window summary
  statistics of RGB vs RGB+BRR, fed to LightGBM) scored against Smart
  Persistence. Reported overall AND on the thin-cloud subset.

The fast tests use a SYNTHETIC loader + image cube (uint8 RGB, BRR derived on
load via :func:`load_sky_frame`) so they run in seconds. The synthetic frames
embed a localized "thin-cloud" blob whose BRR channel carries the cloud signal
while the RGB global statistics stay nearly constant — so RGB-only features
cannot predict the target but RGB+BRR can. A ``RUN_SLOW``-gated test runs E3
end-to-end on the real standardized packages.
"""
import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from solarunibench.experiments import e3_brr

# LightGBM is quiet by default; keep pytest output pristine regardless.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

slow = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW"),
    reason="set RUN_SLOW=1 to run the real-data E3 integration test",
)

_REAL_DATA_ROOT = "D:/monica/workspace/solar-unibench/data/standardized"


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


def _make_sky_frame(cloud_strength: float, rng: np.random.Generator, size: int = 32) -> np.ndarray:
    """One synthetic sky frame as uint8 RGB (3,H,W).

    A localized gaussian "cloud blob" mixes the blue-sky background toward grey
    by ``cloud_strength``. ``cloud_strength=0`` → clear blue (BRR→+0.6);
    intermediate → thin cloud (BRR dips at the blob — visible in BRR, faint in
    RGB); ``cloud_strength=1`` → overcast grey (BRR→0). The BRR channel thus
    carries the cloud signal far more strongly than the RGB global statistics.
    """
    bg = np.array([50.0, 80.0, 200.0])       # clear-sky blue
    grey = np.array([180.0, 180.0, 180.0])    # cloud grey
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = (size - 1) / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (size * 0.5)
    blob = np.exp(-r ** 2 / 0.25)             # localized gaussian core
    # Background mixes slightly, the core strongly → thin cloud = localized dip.
    mix = cloud_strength * (0.35 + 0.65 * blob)
    rgb = bg[None, None, :] * (1 - mix[..., None]) + grey[None, None, :] * mix[..., None]
    rgb = rgb + rng.normal(0.0, 2.0, rgb.shape)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)        # H, W, 3
    return rgb.transpose(2, 0, 1)                        # 3, H, W


def _build_synthetic_image_dataset(
    name: str,
    target_col: str,
    n_images: int = 240,
    horizon_hours: int = 1,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Bounded synthetic image-bearing frame + cube.

    Tabular spans ``n_images + horizon_hours`` hourly rows so every issue-time
    image has a future target at ``issue + horizon_hours``. The target at the
    forecast horizon depends on a latent ``cloud_strength`` state that is
    encoded in the image's BRR channel but is a NOISY function of ``target_now``
    — so the image (especially BRR) carries unique forecast signal. Sky-image
    timestamps are exactly aligned with the tabular issue timestamps so
    nearest-neighbor attach is exact (no tolerance confound).
    """
    rng = np.random.default_rng(seed)
    n_rows = n_images + horizon_hours
    times = pd.date_range("2020-06-01 06:00", periods=n_rows, freq="1h", tz="UTC")
    cs = np.clip(rng.uniform(0.0, 1.0, n_rows) * 0.5 + 0.5 * np.sin(np.arange(n_rows) / 7.0), 0.0, 1.0)
    cs_next = 0.85 * cs + 0.15 * rng.uniform(0.0, 1.0, n_rows)
    cs_next = np.clip(cs_next, 0.0, 1.0)
    clearsky = np.full(n_rows, 800.0)
    # target_now: noisy proxy of cs (weak), so endogenous lags alone struggle.
    target_now = clearsky * (1.0 - 0.6 * cs) + rng.normal(0.0, 160.0, n_rows)
    # forecast target: clean function of cs at the horizon → image-encoded.
    target_full = clearsky * (1.0 - 0.6 * cs_next) + rng.normal(0.0, 20.0, n_rows)
    target_full = np.clip(target_full, 0.0, None)
    naive_ts = times.tz_convert("UTC").tz_localize(None)

    df = pd.DataFrame({
        "timestamp_utc": naive_ts,
        "entity_id": f"{name.upper()}_E0",
        target_col: target_full,
        "ghi_clearsky": clearsky,
        "cos_zenith": 0.55,                 # daytime for every row
        "clearness_index": 1.0 - 0.6 * cs_next,
        "dataset_source": name,
    })
    # Temporal split: E3 uses a WITHIN-image-bearing-period split (the bounded
    # dev cubes cover only an early window — see plan). The split column is the
    # fallback; E3 itself re-splits on time over the image-bearing slice.
    n_img = n_images
    n_tr = int(n_img * 0.70)
    splits = ["train"] * n_tr + ["test"] * (n_img - n_tr) + ["test"] * horizon_hours
    df["split"] = splits

    # Cube: one frame per issue timestamp (exact-aligned). cloud_strength = cs
    # at the issue row → BRR encodes cs; RGB global stats are nearly constant.
    cube = np.stack([_make_sky_frame(cs[i], rng) for i in range(n_images)], axis=0)
    img_ts = np.array([naive_ts[i].isoformat() for i in range(n_images)])
    return df, cube, img_ts


@pytest.fixture
def fake_loader() -> _FakeLoader:
    df_f, cube_f, ts_f = _build_synthetic_image_dataset("folsom", "ghi_actual", seed=1)
    df_s, cube_s, ts_s = _build_synthetic_image_dataset("skippd", "pv_power_normalized", seed=2)
    # PV target rescaled to [0,1]-ish to mirror the real SKIPPD normalization
    # (the builder emits the forecast signal into the target column directly).
    df_s["pv_power_normalized"] = np.clip(df_s["pv_power_normalized"] / 1000.0, 0.0, 1.0) * 0.95
    df_s["ghi_actual"] = np.nan  # SKIPPD is PV-only (ghi_actual absent / unused)
    return _FakeLoader(
        tabular={"folsom": df_f, "skippd": df_s},
        images={"folsom": (cube_f, ts_f), "skippd": (cube_s, ts_s)},
    )


@pytest.fixture
def no_image_loader() -> _FakeLoader:
    """A loader whose datasets have NO sky-image coverage (Honest-skip path)."""
    df, _, _ = _build_synthetic_image_dataset("gefcom2014", "pv_power_normalized", seed=3)
    return _FakeLoader(tabular={"gefcom2014": df}, images={"gefcom2014": None})


# ---------------------------------------------------------------------------
# Contract tests for run(loader, out_dir)
# ---------------------------------------------------------------------------

def test_run_returns_figures_and_metrics_dict(fake_loader, tmp_path):
    res = e3_brr.run(fake_loader, tmp_path)
    assert isinstance(res, dict)
    assert set(res) >= {"figures", "metrics"}
    # both canonical figures written to disk
    gallery = tmp_path / "fig_brr_gallery.png"
    skill = tmp_path / "fig_brr_skill.png"
    assert gallery.exists() and gallery.stat().st_size > 0
    assert skill.exists() and skill.stat().st_size > 0
    # one results CSV under results/
    csvs = list((tmp_path / "results").glob("*.csv"))
    assert len(csvs) >= 1
    assert any("fig_brr_gallery.png" in p for p in res["figures"])
    assert any("fig_brr_skill.png" in p for p in res["figures"])


def test_brr_captures_signal_rgb_only_cannot(fake_loader, tmp_path):
    """The load-bearing hypothesis check on synthetic data.

    The thin-cloud signal is encoded in the BRR channel but kept nearly invisible
    to RGB global stats, so the RGB+BRR arm must beat the RGB-only arm
    (Δskill = skill_B - skill_A > 0). This proves the ablation mechanics capture
    BRR-informative signal — the whole point of E3.
    """
    res = e3_brr.run(fake_loader, tmp_path)
    # At least one image-bearing dataset evaluated.
    assert res["metrics"]["datasets"], "no dataset produced an ablation"
    for ds, arm in res["metrics"]["datasets"].items():
        overall = arm["overall"]
        d = overall["skill_brr"] - overall["skill_rgb"]
        assert np.isfinite(overall["skill_rgb"]) and np.isfinite(overall["skill_brr"]), ds
        assert d > 0.0, f"{ds}: BRR did not beat RGB-only (Δ={d:+.4f}); ablation broken"


def test_metrics_finite_and_skill_reported(fake_loader, tmp_path):
    res = e3_brr.run(fake_loader, tmp_path)
    for ds, arm in res["metrics"]["datasets"].items():
        for subset in ("overall", "thin_cloud"):
            s = arm[subset]
            for key in ("mae_rgb", "mae_brr", "mae_sp"):
                assert np.isfinite(s[key]), f"{ds}/{subset}/{key} not finite"
                assert s[key] >= 0, f"{ds}/{subset}/{key} negative"
            assert np.isfinite(s["skill_rgb"]), f"{ds}/{subset}: rgb skill not finite"
            assert np.isfinite(s["skill_brr"]), f"{ds}/{subset}: brr skill not finite"
            assert s["n_samples"] > 0, f"{ds}/{subset}: no samples"


def test_thin_cloud_subset_is_a_proper_subset(fake_loader, tmp_path):
    """Thin-cloud subset is defined, smaller than overall, and non-empty."""
    res = e3_brr.run(fake_loader, tmp_path)
    for ds, arm in res["metrics"]["datasets"].items():
        n_all = arm["overall"]["n_samples"]
        n_thin = arm["thin_cloud"]["n_samples"]
        assert 0 < n_thin < n_all, f"{ds}: thin-cloud subset {n_thin} not a proper subset of {n_all}"


def test_gallery_reports_three_regimes(fake_loader, tmp_path):
    res = e3_brr.run(fake_loader, tmp_path)
    gallery = res["metrics"]["gallery"]
    regimes = {g["regime"] for g in gallery["frames"]}
    assert regimes >= {"clear", "thin", "overcast"}, f"missing regimes: {regimes}"
    for g in gallery["frames"]:
        assert np.isfinite(g["brr_mean"]) and np.isfinite(g["brr_std"])
    # clear frames have higher mean BRR than overcast (blue sky vs grey).
    clear_mean = np.mean([g["brr_mean"] for g in gallery["frames"] if g["regime"] == "clear"])
    over_mean = np.mean([g["brr_mean"] for g in gallery["frames"] if g["regime"] == "overcast"])
    assert clear_mean > over_mean, "clear-sky BRR mean must exceed overcast"


def test_no_sky_images_honest_skip(no_image_loader, tmp_path):
    """A dataset with no sky-image coverage is Honest-skipped (no crash)."""
    res = e3_brr.run(no_image_loader, tmp_path)
    assert isinstance(res, dict)
    # No ablation possible; metrics.datasets is empty and the fact is reported.
    assert res["metrics"]["datasets"] == {}
    assert not res["figures"], "no figures expected when no sky images exist"


def test_run_does_not_mutate_loader_data(fake_loader, tmp_path):
    before_df = {k: v.copy(deep=True) for k, v in fake_loader._tabular.items()}
    before_cube = {k: v[0].copy() for k, v in fake_loader._images.items() if v is not None}
    e3_brr.run(fake_loader, tmp_path)
    for k, v in before_df.items():
        pd.testing.assert_frame_equal(v, fake_loader._tabular[k])
    for k, v in before_cube.items():
        np.testing.assert_array_equal(v, fake_loader._images[k][0])


def test_results_csv_round_trips(fake_loader, tmp_path):
    e3_brr.run(fake_loader, tmp_path)
    csv = list((tmp_path / "results").glob("*.csv"))[0]
    df = pd.read_csv(csv)
    assert {"dataset", "subset", "arm", "mae", "skill", "n_samples"}.issubset(df.columns)
    for ds in ("folsom", "skippd"):
        sub = df[df["dataset"] == ds]
        assert (sub["subset"] == "thin_cloud").any()
        assert (sub["arm"] == "rgb").any() and (sub["arm"] == "brr").any()


# ---------------------------------------------------------------------------
# Real-data RUN_SLOW integration
# ---------------------------------------------------------------------------

@slow
def test_run_real_standardized_packages(tmp_path):
    from solarunibench.loader import SolarUniBenchLoader

    loader = SolarUniBenchLoader(_REAL_DATA_ROOT)
    res = e3_brr.run(loader, tmp_path)
    assert isinstance(res, dict)
    assert (tmp_path / "fig_brr_gallery.png").exists()
    assert list((tmp_path / "results").glob("*.csv"))
    print("\n=== E3 real-data headline (Daytime skill vs Smart Persistence) ===")
    for ds, arm in res["metrics"]["datasets"].items():
        o, t = arm["overall"], arm["thin_cloud"]
        print(
            f"  {ds:8s} overall : rgb={o['skill_rgb']:+.4f} brr={o['skill_brr']:+.4f} "
            f"Δ={o['skill_brr'] - o['skill_rgb']:+.4f} (n={o['n_samples']})"
        )
        print(
            f"  {ds:8s} thin-cl: rgb={t['skill_rgb']:+.4f} brr={t['skill_brr']:+.4f} "
            f"Δ={t['skill_brr'] - t['skill_rgb']:+.4f} (n={t['n_samples']})"
        )
