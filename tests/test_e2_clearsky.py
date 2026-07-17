"""Tests for E2 — Clear-sky / clearness-index feature value (all 4 datasets).

Spec §11 (E2). Hypothesis: physics anchors (clear-sky / clearness-index) beat
raw endogenous features. The fast tests use a SYNTHETIC loader (mimics
:class:`SolarUniBenchLoader`) so they run in seconds; a ``RUN_SLOW``-gated test
runs E2 end-to-end on the real standardized packages.
"""
import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from solarunibench.experiments import e2_clearsky

# LightGBM is quiet by default; keep pytest output pristine regardless.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

slow = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW"),
    reason="set RUN_SLOW=1 to run the real-data E2 integration test",
)

_REAL_DATA_ROOT = "D:/monica/workspace/solar-unibench/data/standardized"


# ---------------------------------------------------------------------------
# Synthetic loader + frames
# ---------------------------------------------------------------------------

class _FakeLoader:
    """Mimics :class:`SolarUniBenchLoader.load_tabular` for the 4 datasets."""

    def __init__(self, frames: dict[str, pd.DataFrame], data_root: str = "/fake"):
        self._frames = frames
        self.data_root = data_root

    def load_tabular(self, dataset: str, split: str | None = None) -> pd.DataFrame:
        df = self._frames[dataset].copy()
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)
        return df


def _synth_dataset(
    name: str,
    target_col: str,
    n_entities: int = 2,
    n_days: int = 24,
    clearness_finite: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """A bounded synthetic standardized frame.

    The target is driven by the clear-sky bell so the clear-sky anchors carry
    real signal. When ``clearness_finite`` is False the ``clearness_index``
    column is entirely NaN — mirroring GEFCom/SKIPPD (PV-only, no measured
    ``ghi_actual`` to form the ratio) — and MUST be dropped per Honest-NaN.
    """
    rows = []
    for e in range(n_entities):
        eid = f"{name.upper()}_E{e}"
        n = n_days * 24
        times = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC")
        hod = times.hour.to_numpy()
        bell = np.clip(np.sin(np.pi * (hod - 6) / 13.0), 0.0, None)
        rng = np.random.default_rng(seed + e)
        ghi_clear = bell * 950.0
        cos_zen = bell  # proxy in [0, 1]; > 0.05 during the daytime band
        if target_col == "pv_power_normalized":
            target = np.clip(bell * 0.9 + rng.normal(0, 0.03, n), 0, 1)
        else:  # ghi_actual
            target = bell * 900.0 + rng.normal(0, 25, n)
        if clearness_finite:
            with np.errstate(divide="ignore", invalid="ignore"):
                kt = np.where(ghi_clear > 1, np.clip(target * 900.0 / ghi_clear, 0, 1.5), 0.0)
        else:
            kt = np.full(n, np.nan)  # Honest-NaN: dataset lacks clearness_index
        naive = times.tz_convert("UTC").tz_localize(None)
        n_tr = int(n * 0.70)
        n_va = int(n * 0.15)
        splits = ["train"] * n_tr + ["val"] * n_va + ["test"] * (n - n_tr - n_va)
        lat, lon = 35.0 - e, -100.0 + e
        for i in range(n):
            rows.append({
                "timestamp_utc": naive[i],
                "entity_id": eid,
                target_col: float(target[i]),
                "ghi_clearsky": float(ghi_clear[i]),
                "cos_zenith": float(cos_zen[i]),
                "clearness_index": float(kt[i]),
                "latitude": lat,
                "longitude": lon,
                "dataset_source": name,
                "split": splits[i],
            })
    return pd.DataFrame(rows)


@pytest.fixture
def fake_loader() -> _FakeLoader:
    return _FakeLoader({
        # PV datasets: clearness_index entirely NaN (Honest-NaN).
        "gefcom2014": _synth_dataset("gefcom2014", "pv_power_normalized", clearness_finite=False, seed=1),
        "skippd": _synth_dataset("skippd", "pv_power_normalized", clearness_finite=False, seed=2),
        # GHI datasets: clearness_index populated.
        "folsom": _synth_dataset("folsom", "ghi_actual", clearness_finite=True, seed=3),
        "sunlake": _synth_dataset("sunlake", "ghi_actual", clearness_finite=True, seed=4),
    })


# ---------------------------------------------------------------------------
# Contract tests for run(loader, out_dir)
# ---------------------------------------------------------------------------

def test_run_returns_figures_and_metrics_dict(fake_loader, tmp_path):
    res = e2_clearsky.run(fake_loader, tmp_path)
    assert isinstance(res, dict)
    assert set(res) >= {"figures", "metrics"}
    # the one canonical figure written to disk
    fig = tmp_path / "fig_clearsky_skill.png"
    assert fig.exists(), "missing fig_clearsky_skill.png"
    assert fig.stat().st_size > 0
    # one results CSV under results/
    csvs = list((tmp_path / "results").glob("*.csv"))
    assert len(csvs) >= 1
    # figures list reports the written path
    assert any("fig_clearsky_skill.png" in p for p in res["figures"])


def test_all_four_datasets_evaluated_with_correct_targets(fake_loader, tmp_path):
    res = e2_clearsky.run(fake_loader, tmp_path)
    datasets = res["metrics"]["datasets"]
    for ds in ("gefcom2014", "skippd", "folsom", "sunlake"):
        assert ds in datasets, f"missing dataset {ds}"
    assert datasets["gefcom2014"]["target"] == "pv_power_normalized"
    assert datasets["skippd"]["target"] == "pv_power_normalized"
    assert datasets["folsom"]["target"] == "ghi_actual"
    assert datasets["sunlake"]["target"] == "ghi_actual"


def test_clearness_index_honest_nan_handling(fake_loader, tmp_path):
    """GEFCom/SKIPPD (clearness_index all-NaN) omit it; Folsom/SunLake include it."""
    res = e2_clearsky.run(fake_loader, tmp_path)
    ds = res["metrics"]["datasets"]
    for pv in ("gefcom2014", "skippd"):
        used = ds[pv]["clearsky_features_used"]
        assert "ghi_clearsky" in used and "cos_zenith" in used
        assert "clearness_index" not in used  # Honest-NaN: dropped
    for ghi in ("folsom", "sunlake"):
        used = ds[ghi]["clearsky_features_used"]
        assert "clearness_index" in used
    # requested documents the full wishlist (Honest reporting of the omission).
    for pv in ("gefcom2014", "skippd"):
        assert "clearness_index" in ds[pv]["clearsky_features_requested"]


def test_baseline_is_truly_endogenous_no_clearsky_contamination(fake_loader, tmp_path):
    """The baseline arm must NOT receive clear-sky features.

    The feature engineer auto-builds a future clear-sky context block whenever
    both ghi_clearsky and cos_zenith are present. E2 drops ghi_clearsky for the
    baseline arm so it stays endogenous-only (cos_zenith kept for the daytime
    mask). This is the load-bearing correctness property of the ablation.
    """
    res = e2_clearsky.run(fake_loader, tmp_path)
    for ds_name, arm in res["metrics"]["datasets"].items():
        base = arm["baseline"]
        assert base["past_channels"] == 1, f"{ds_name}: baseline not endogenous-only"
        assert base["future_context"] is False, f"{ds_name}: baseline leaked clear-sky future"
        clr = arm["clearsky"]
        assert clr["future_context"] is True
        assert clr["past_channels"] == 1 + len(arm["clearsky_features_used"])


def test_metrics_finite_and_skill_reported(fake_loader, tmp_path):
    res = e2_clearsky.run(fake_loader, tmp_path)
    for ds_name, arm in res["metrics"]["datasets"].items():
        for key in ("baseline", "clearsky", "reference"):
            mae = arm[key]["daytime_mae"]
            assert np.isfinite(mae), f"{ds_name}/{key}: daytime_mae not finite ({mae})"
            assert mae >= 0, f"{ds_name}/{key}: negative MAE"
        # skill vs Smart Persistence reference: 0 = parity, positive = better.
        assert np.isfinite(arm["baseline"]["skill"])
        assert np.isfinite(arm["clearsky"]["skill"])
        assert np.isfinite(arm["skill_gain"])
        # daytime mask applied consistently across the three arms (same windows)
        n_base = arm["baseline"]["n_daytime"]
        n_clr = arm["clearsky"]["n_daytime"]
        assert n_base == n_clr > 0


def test_run_does_not_mutate_loader_data(fake_loader, tmp_path):
    before = {k: v.copy(deep=True) for k, v in fake_loader._frames.items()}
    e2_clearsky.run(fake_loader, tmp_path)
    for k, v in before.items():
        pd.testing.assert_frame_equal(v, fake_loader._frames[k])


def test_results_csv_round_trips(fake_loader, tmp_path):
    e2_clearsky.run(fake_loader, tmp_path)
    csv = list((tmp_path / "results").glob("*.csv"))[0]
    df = pd.read_csv(csv)
    # one row per (dataset, arm)
    assert {"dataset", "arm", "target", "daytime_mae", "skill"}.issubset(df.columns)
    for ds in ("gefcom2014", "skippd", "folsom", "sunlake"):
        sub = df[df["dataset"] == ds]
        assert (sub["arm"] == "baseline").any()
        assert (sub["arm"] == "clearsky").any()


# ---------------------------------------------------------------------------
# Real-data RUN_SLOW integration
# ---------------------------------------------------------------------------

@slow
def test_run_real_standardized_packages(tmp_path):
    from solarunibench.loader import SolarUniBenchLoader

    loader = SolarUniBenchLoader(_REAL_DATA_ROOT)
    res = e2_clearsky.run(loader, tmp_path)
    assert isinstance(res, dict)
    assert (tmp_path / "fig_clearsky_skill.png").exists()
    assert list((tmp_path / "results").glob("*.csv"))
    print("\n=== E2 real-data headline (skill gain, baseline -> +clearsky) ===")
    for ds, arm in res["metrics"]["datasets"].items():
        print(
            f"  {ds:11s} target={arm['target']:22s} "
            f"skill: base={arm['baseline']['skill']:+.4f} "
            f"clear={arm['clearsky']['skill']:+.4f} "
            f"gain={arm['skill_gain']:+.4f} "
            f"features={arm['clearsky_features_used']}"
        )
