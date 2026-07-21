"""Tests for E1 — Geo-recovery accuracy + TSR validation + downstream value (GEFCom).

The fast tests use a SYNTHETIC loader (mimics :class:`SolarUniBenchLoader`) so they
run in seconds. The heavy clear-sky global grid search is stubbed via monkeypatch;
the real TSR validation + the real (TINY) MLP downstream pipeline run on the
synthetic frames so the full wiring is exercised. A ``RUN_SLOW``-gated test runs
E1 end-to-end on the real standardized packages.
"""
import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from solarunibench.experiments import e1_geo
from solarunibench.operators import solar_geometry as sg

# The downstream arm deliberately uses a TINY MLP (max_iter=30) for fast dev
# iteration; it will not fully converge, and horizon=1 yields a single-output Y
# that sklearn warns about. Both are expected for the dev config, not bugs.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:Stochastic Optimizer:sklearn.exceptions.ConvergenceWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:A column-vector y was passed:sklearn.exceptions.DataConversionWarning"
    ),
]

slow = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW"),
    reason="set RUN_SLOW=1 to run the real-data E1 integration test",
)

_REAL_DATA_ROOT = "D:/monica/workspace/solar-unibench/data/standardized"


# ---------------------------------------------------------------------------
# Synthetic loader + frames
# ---------------------------------------------------------------------------

class _FakeLoader:
    """Mimics :class:`SolarUniBenchLoader.load_tabular` for the datasets E1 uses."""

    def __init__(self, frames: dict[str, pd.DataFrame], data_root: str = "/fake"):
        self._frames = frames
        self.data_root = data_root

    def load_tabular(self, dataset: str, split: str | None = None) -> pd.DataFrame:
        df = self._frames[dataset].copy()
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)
        return df


def _naive_utc(times: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Coerce a (possibly tz-aware) DatetimeIndex to naive UTC."""
    if times.tz is not None:
        return times.tz_convert("UTC").tz_localize(None)
    return times


def _synth_folsom(n_days: int = 12, true_lat: float = 38.67,
                  true_lon: float = -121.15) -> pd.DataFrame:
    """1 entity, KNOWN coords (geo_inferred=False), ghi_actual following a clear-sky bell."""
    n = n_days * 24
    times = pd.date_range("2013-06-01", periods=n, freq="1h", tz="UTC")
    hod = times.hour.to_numpy()
    bell = np.clip(np.sin(np.pi * (hod - 6) / 13.0), 0.0, None)
    rng = np.random.default_rng(0)
    ghi = bell * 900.0 + rng.normal(0, 20, n)
    naive = _naive_utc(times)
    return pd.DataFrame({
        "timestamp_utc": naive,
        "entity_id": "FOLSOM_STN",
        "ghi_actual": ghi,
        "pv_power_normalized": np.clip(bell * 0.9 + rng.normal(0, 0.02, n), 0, 1),
        "latitude": true_lat,
        "longitude": true_lon,
        "dataset_source": "folsom",
        "geo_inferred": False,
    })


def _synth_gefcom(n_days: int = 90, n_zones: int = 3) -> pd.DataFrame:
    """3 zones with recovered coords + a VAR178 (TSR) field whose daily total tracks
    theoretical TOA (so ``validate_with_tsr`` yields a finite, positive R^2)."""
    rows = []
    naive_idx = None
    for z in range(1, n_zones + 1):
        eid = f"GEFCOM_ZONE_{z}"
        lat, lon = -34.0 - 0.5 * z, 150.0
        n = n_days * 24
        times = pd.date_range("2013-01-01", periods=n, freq="1h", tz="UTC")
        hod = times.hour.to_numpy()
        doy = times.dayofyear.to_numpy().astype(float)
        bell = np.clip(np.sin(np.pi * (hod - 6) / 13.0), 0.0, None)
        # daily TOA total -> seasonal shape driving the VAR178 daily sum
        daily_toa = sg.toa_daily_irradiation(doy, lat)
        shape_day_sum = (
            pd.Series(bell, index=times)
            .groupby(times.dayofyear)
            .transform("sum")
            .to_numpy()
        )
        scale = np.where(shape_day_sum > 1e-6, daily_toa / shape_day_sum, 0.0)
        ghi_bell = bell * scale
        # ECMWF-style within-day cumulative (resets at midnight) -> diff gives the bell
        s_ghi = pd.Series(ghi_bell, index=times)
        var178 = s_ghi.groupby(s_ghi.index.date).cumsum().to_numpy()
        rng = np.random.default_rng(z)
        pv = np.clip(bell * 0.85 + rng.normal(0, 0.03, n), 0, 1)
        ghi_clear = bell * 950.0
        cos_zen = bell  # proxy in [0, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            kt = np.where(ghi_clear > 1, np.clip(pv * 900.0 / ghi_clear, 0, 1.5), 0.0)
        naive = _naive_utc(times)
        n_tr = int(n * 0.70)
        n_va = int(n * 0.15)
        splits = ["train"] * n_tr + ["val"] * n_va + ["test"] * (n - n_tr - n_va)
        for i in range(n):
            rows.append({
                "timestamp_utc": naive[i],
                "entity_id": eid,
                "pv_power_normalized": float(pv[i]),
                "ghi_clearsky": float(ghi_clear[i]),
                "ghi_actual": float(pv[i] * 900.0),
                "cos_zenith": float(cos_zen[i]),
                "clearness_index": float(kt[i]),
                "nwp_VAR178": float(var178[i]),
                "latitude": lat,
                "longitude": lon,
                "dataset_source": "gefcom2014",
                "split": splits[i],
            })
    return pd.DataFrame(rows)


@pytest.fixture
def fake_loader() -> _FakeLoader:
    return _FakeLoader(
        {"folsom": _synth_folsom(), "gefcom2014": _synth_gefcom()}
    )


@pytest.fixture
def patch_recovery(monkeypatch):
    """Stub the heavy pvlib clear-sky grid search -> instant + deterministic."""
    def _fake_recover(times, power, **kw):
        return {
            "latitude": 38.5,
            "longitude": -121.2,
            "details": {
                "method": "global_clearsky",
                "best_mse": 0.01,
                "top_n": [{"lat": 38.5, "lon": -121.2, "mse": 0.01}],
                "mirror_pairs": [],
                "has_mirror": False,
                "status": "ok",
            },
        }
    monkeypatch.setattr(e1_geo, "recover_geo_clearsky_global", _fake_recover)


# ---------------------------------------------------------------------------
# Unit test: haversine helper
# ---------------------------------------------------------------------------

def test_haversine_km_known_distance():
    # Sydney (-33.87, 151.21) -> Melbourne (-37.81, 144.96) is ~714 km.
    d = e1_geo._haversine_km(-33.87, 151.21, -37.81, 144.96)
    assert 690.0 < d < 740.0


# ---------------------------------------------------------------------------
# Contract tests for run(loader, out_dir)
# ---------------------------------------------------------------------------

def test_run_returns_figures_and_metrics_dict(fake_loader, patch_recovery, tmp_path):
    res = e1_geo.run(fake_loader, tmp_path)
    assert isinstance(res, dict)
    assert set(res) >= {"figures", "metrics"}
    # exactly the three canonical figures written to disk
    for name in (
        "fig_geo_blindtest.png",
        "fig_geo_tsr_validation.png",
        "fig_geo_skill_gain.png",
    ):
        p = tmp_path / name
        assert p.exists(), f"missing figure {name}"
        assert p.stat().st_size > 0
    # at least one results CSV under results/
    csvs = list((tmp_path / "results").glob("*.csv"))
    assert len(csvs) >= 1
    # figures list reports the written paths
    assert len(res["figures"]) >= 3
    assert all(str(tmp_path) in p for p in res["figures"])


def test_blind_test_reports_finite_error_and_true_coords(fake_loader, patch_recovery, tmp_path):
    res = e1_geo.run(fake_loader, tmp_path)
    bt = res["metrics"]["blind_test"]
    assert bt["true_lat"] == pytest.approx(38.67, abs=1e-6)
    assert bt["true_lon"] == pytest.approx(-121.15, abs=1e-6)
    assert bt["coords_known"] is True
    assert np.isfinite(bt["great_circle_error_km"])
    assert bt["great_circle_error_km"] >= 0.0
    assert bt["recovered_lat"] is not None and bt["recovered_lon"] is not None


def test_tsr_validation_reports_per_zone_r2(fake_loader, patch_recovery, tmp_path):
    res = e1_geo.run(fake_loader, tmp_path)
    tsr = res["metrics"]["tsr_validation"]
    assert isinstance(tsr, dict) and len(tsr) == 3
    for zone, v in tsr.items():
        assert "r2_latitude" in v and "rmse_latitude" in v
        assert "longitude_implied" in v
        # synthetic VAR178 was built proportional to theoretical TOA -> finite R^2
        assert np.isfinite(v["r2_latitude"])


def test_downstream_reports_both_arms_and_delta(fake_loader, patch_recovery, tmp_path):
    res = e1_geo.run(fake_loader, tmp_path)
    dn = res["metrics"]["downstream"]
    assert np.isfinite(dn["baseline_daytime_mae"])
    assert np.isfinite(dn["geo_daytime_mae"])
    assert "delta_daytime_mae" in dn
    assert "feature_cols" in dn  # documents which features the geo arm unlocked
    # Synthetic frame populates all 3 geo features -> all usable.
    assert dn["geo_features_used"] == ["ghi_clearsky", "cos_zenith", "clearness_index"]


def test_run_does_not_mutate_loader_data(fake_loader, patch_recovery, tmp_path):
    before_f = fake_loader._frames["folsom"].copy(deep=True)
    before_g = fake_loader._frames["gefcom2014"].copy(deep=True)
    e1_geo.run(fake_loader, tmp_path)
    pd.testing.assert_frame_equal(before_f, fake_loader._frames["folsom"])
    pd.testing.assert_frame_equal(before_g, fake_loader._frames["gefcom2014"])


def test_results_csv_round_trips(fake_loader, patch_recovery, tmp_path):
    e1_geo.run(fake_loader, tmp_path)
    csv = list((tmp_path / "results").glob("*.csv"))[0]
    df = pd.read_csv(csv)
    assert "section" in df.columns
    # one row per section-key
    assert (df["section"] == "blind_test").any()
    assert (df["section"] == "tsr_validation").sum() == 3
    assert (df["section"] == "downstream").sum() == 2


# ---------------------------------------------------------------------------
# Real-data RUN_SLOW integration
# ---------------------------------------------------------------------------

@slow
def test_run_real_standardized_packages(tmp_path):
    from solarunibench.loader import SolarUniBenchLoader

    loader = SolarUniBenchLoader(_REAL_DATA_ROOT)
    res = e1_geo.run(loader, tmp_path)
    assert isinstance(res, dict)
    for name in (
        "fig_geo_blindtest.png",
        "fig_geo_tsr_validation.png",
        "fig_geo_skill_gain.png",
    ):
        assert (tmp_path / name).exists()
    assert (tmp_path / "results").glob("*.csv")
    bt = res["metrics"]["blind_test"]
    # Honest reporting: error is finite when recovery succeeds; if the clear-sky
    # method failed (None coord), error_km is NaN and that IS the E1 finding.
    assert "great_circle_error_km" in bt
    tsr = res["metrics"]["tsr_validation"]
    assert len(tsr) >= 1
    dn = res["metrics"]["downstream"]
    assert "baseline_daytime_mae" in dn and "geo_daytime_mae" in dn
    # Print the E1 headline numbers for the report.
    print("\n=== E1 real-data headline ===")
    print(f"  blind-test great-circle MAE: {bt['great_circle_error_km']} km")
    print(f"  TSR R^2 per zone: { {z: v['r2_latitude'] for z, v in tsr.items()} }")
    print(f"  downstream Daytime_MAE: raw={dn['baseline_daytime_mae']} "
          f"geo={dn['geo_daytime_mae']} delta={dn['delta_daytime_mae']}")
