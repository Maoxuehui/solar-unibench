"""Tests for E7 — Data-defect diagnostics (descriptive, all datasets, NO model).

Spec §11 (E7) — the methodology's "缺陷热力图" (defect heatmap) diagnostic
reports. E7 is purely DESCRIPTIVE: it characterizes missingness, irregularity,
and outliers in the standardized packages WITHOUT training any model.

The fast tests use a SYNTHETIC loader with KNOWN defects (all-NaN columns, an
absent column, an injected gap, a duplicate timestamp, mixed quality_flag
values) so the assertions can be exact. A ``RUN_SLOW``-gated test runs E7
end-to-end on the real standardized packages.
"""
import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from solarunibench.experiments import e7_defects

slow = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW"),
    reason="set RUN_SLOW=1 to run the real-data E7 integration test",
)

_REAL_DATA_ROOT = "D:/monica/workspace/solar-unibench/data/standardized"


# ---------------------------------------------------------------------------
# Synthetic loader + frames with KNOWN defects
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


def _bell(idx: np.ndarray) -> np.ndarray:
    """Deterministic daytime bell (0 at night, peaks midday) from an index array."""
    hod = idx % 24
    return np.clip(np.sin(np.pi * (hod - 6) / 13.0), 0.0, None)


def _frame_gefcom() -> pd.DataFrame:
    """PV-only (ghi/dni/dhi/clearness all-NaN — Honest-NaN); hourly; 1 injected gap.

    Row index 10 is dropped so hour 9 is followed by hour 11 -> one 2-hour gap.
    Flags alternate 0 (good) / 3 (night) per row.
    """
    n = 48
    times = pd.date_range("2020-06-01", periods=n, freq="1h")
    keep = list(range(10)) + list(range(11, n))  # drop index 10 -> 2h gap
    times = times[keep]
    m = len(times)
    bell = _bell(np.arange(m))
    flags = np.where(np.arange(m) % 2 == 0, 0, 3).astype("int64")
    return pd.DataFrame({
        "timestamp_utc": times,
        "entity_id": "GEFCOM_Z1",
        "pv_power_normalized": bell * 0.9,
        "ghi_actual": np.nan,
        "dni_actual": np.nan,
        "dhi_actual": np.nan,
        "cos_zenith": bell.astype("float32"),
        "solar_azimuth": (bell * 180.0).astype("float32"),
        "ghi_clearsky": (bell * 950.0).astype("float32"),
        "dni_clearsky": (bell * 800.0).astype("float32"),
        "dhi_clearsky": (bell * 100.0).astype("float32"),
        "clearness_index": np.nan,  # Honest-NaN (PV-only, no ghi_actual to form ratio)
        "quality_flag": flags,
        "split": "train",
    })


def _frame_folsom() -> pd.DataFrame:
    """GHI-only (pv_power_normalized all-NaN — Honest-NaN); hourly; 1 duplicate ts."""
    n = 48
    times = pd.date_range("2020-06-01", periods=n, freq="1h")
    times = pd.DatetimeIndex(list(times) + [times[-1]])  # duplicate last timestamp
    m = len(times)
    bell = _bell(np.arange(m))
    ghi = bell * 900.0
    return pd.DataFrame({
        "timestamp_utc": times,
        "entity_id": "FOLSOM_STN",
        "pv_power_normalized": np.nan,  # Honest-NaN: no PV modality
        "ghi_actual": ghi,
        "dni_actual": ghi * 0.8,
        "dhi_actual": ghi * 0.2,
        "cos_zenith": bell.astype("float32"),
        "solar_azimuth": (bell * 180.0).astype("float32"),
        "ghi_clearsky": (bell * 950.0).astype("float32"),
        "dni_clearsky": (bell * 800.0).astype("float32"),
        "dhi_clearsky": (bell * 100.0).astype("float32"),
        "clearness_index": np.clip(ghi / 950.0, 0, 1.5).astype("float32"),
        "quality_flag": np.zeros(m, dtype="int64"),
        "split": "train",
    })


def _frame_skippd() -> pd.DataFrame:
    """PV-only, 15-min cadence, clean (no gaps/dups). ``dni_actual`` column is
    ABSENT entirely (Honest-NaN by absence — the loader-probing modality rule)."""
    n = 48
    times = pd.date_range("2020-06-01", periods=n, freq="15min")
    bell = _bell(np.arange(n))
    frame = pd.DataFrame({
        "timestamp_utc": times,
        "entity_id": "SKIPPD_STN",
        "pv_power_normalized": bell * 0.9,
        # NOTE: dni_actual intentionally OMITTED to exercise absent-column branch.
        "ghi_actual": np.nan,
        "dhi_actual": np.nan,
        "cos_zenith": bell.astype("float32"),
        "solar_azimuth": (bell * 180.0).astype("float32"),
        "ghi_clearsky": (bell * 950.0).astype("float32"),
        "dni_clearsky": (bell * 800.0).astype("float32"),
        "dhi_clearsky": (bell * 100.0).astype("float32"),
        "clearness_index": np.nan,  # Honest-NaN (PV-only)
        "quality_flag": np.zeros(n, dtype="int64"),
        "split": "train",
    })
    return frame


def _frame_sunlake() -> pd.DataFrame:
    """GHI-only, 30-min cadence, clean; all canonical columns present + populated."""
    n = 48
    times = pd.date_range("2020-06-01", periods=n, freq="30min")
    bell = _bell(np.arange(n))
    ghi = bell * 900.0
    return pd.DataFrame({
        "timestamp_utc": times,
        "entity_id": "SUNLAKE_S1",
        "pv_power_normalized": np.nan,  # Honest-NaN: no PV
        "ghi_actual": ghi,
        "dni_actual": ghi * 0.8,
        "dhi_actual": ghi * 0.2,
        "cos_zenith": bell.astype("float32"),
        "solar_azimuth": (bell * 180.0).astype("float32"),
        "ghi_clearsky": (bell * 950.0).astype("float32"),
        "dni_clearsky": (bell * 800.0).astype("float32"),
        "dhi_clearsky": (bell * 100.0).astype("float32"),
        "clearness_index": np.clip(ghi / 950.0, 0, 1.5).astype("float32"),
        "quality_flag": np.where(bell > 0, 0, 3).astype("int64"),
        "split": "train",
    })


@pytest.fixture
def fake_loader() -> _FakeLoader:
    return _FakeLoader({
        "gefcom2014": _frame_gefcom(),
        "folsom": _frame_folsom(),
        "skippd": _frame_skippd(),
        "sunlake": _frame_sunlake(),
    })


# ---------------------------------------------------------------------------
# Contract tests for run(loader, out_dir)
# ---------------------------------------------------------------------------

def test_run_returns_figures_and_metrics_dict(fake_loader, tmp_path):
    res = e7_defects.run(fake_loader, tmp_path)
    assert isinstance(res, dict)
    assert set(res) >= {"figures", "metrics"}
    # exactly the three canonical figures written to disk
    for name in (
        "fig_defect_missingness.png",
        "fig_defect_irregularity.png",
        "fig_defect_outliers.png",
    ):
        p = tmp_path / name
        assert p.exists(), f"missing figure {name}"
        assert p.stat().st_size > 0
    # the defect summary CSV under results/
    csvs = list((tmp_path / "results").glob("*.csv"))
    assert len(csvs) >= 1
    # figures list reports the written paths
    assert len(res["figures"]) >= 3
    assert all(str(tmp_path) in p for p in res["figures"])


def test_metrics_has_four_diagnostic_sections(fake_loader, tmp_path):
    res = e7_defects.run(fake_loader, tmp_path)
    metrics = res["metrics"]
    for section in ("missingness", "irregularity", "quality_flag", "outliers"):
        assert section in metrics, f"missing section {section}"
        assert isinstance(metrics[section], dict)


def test_all_four_datasets_evaluated(fake_loader, tmp_path):
    res = e7_defects.run(fake_loader, tmp_path)
    for section in ("missingness", "irregularity", "quality_flag", "outliers"):
        for ds in ("gefcom2014", "folsom", "skippd", "sunlake"):
            assert ds in res["metrics"][section], f"{section} missing {ds}"


# ---------------------------------------------------------------------------
# Missingness — Honest-NaN reporting (the load-bearing correctness property)
# ---------------------------------------------------------------------------

def test_missingness_all_nan_column_reported_as_100(fake_loader, tmp_path):
    """GEFCom ghi_actual / clearness_index are entirely NaN (PV-only) -> 100%."""
    res = e7_defects.run(fake_loader, tmp_path)
    m = res["metrics"]["missingness"]["gefcom2014"]
    assert m["ghi_actual"] == pytest.approx(100.0)
    assert m["clearness_index"] == pytest.approx(100.0)
    assert m["dni_actual"] == pytest.approx(100.0)
    assert m["dhi_actual"] == pytest.approx(100.0)


def test_missingness_absent_column_reported_as_100(fake_loader, tmp_path):
    """SKIPPD frame omits dni_actual entirely -> 100% (Honest-NaN by absence)."""
    res = e7_defects.run(fake_loader, tmp_path)
    m = res["metrics"]["missingness"]["skippd"]
    assert "dni_actual" in m  # canonical column reported even when absent
    assert m["dni_actual"] == pytest.approx(100.0)


def test_missingness_populated_column_reported_low(fake_loader, tmp_path):
    """Populated canonical columns report near-0% NaN."""
    res = e7_defects.run(fake_loader, tmp_path)
    assert res["metrics"]["missingness"]["gefcom2014"]["pv_power_normalized"] == pytest.approx(0.0)
    assert res["metrics"]["missingness"]["folsom"]["ghi_actual"] == pytest.approx(0.0)
    assert res["metrics"]["missingness"]["sunlake"]["clearness_index"] == pytest.approx(0.0)
    assert res["metrics"]["missingness"]["folsom"]["pv_power_normalized"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Irregularity — gap + duplicate detection
# ---------------------------------------------------------------------------

def test_irregularity_detects_injected_gap(fake_loader, tmp_path):
    """GEFCom frame has one 2h gap (dropped index 10) at hourly cadence."""
    res = e7_defects.run(fake_loader, tmp_path)
    ir = res["metrics"]["irregularity"]["gefcom2014"]
    assert ir["modal_cadence_seconds"] == pytest.approx(3600.0, rel=1e-6)
    assert ir["n_gaps"] == 1
    assert ir["n_duplicates"] == 0


def test_irregularity_detects_injected_duplicate(fake_loader, tmp_path):
    """Folsom frame duplicates the last timestamp -> 1 duplicate, 0 gaps."""
    res = e7_defects.run(fake_loader, tmp_path)
    ir = res["metrics"]["irregularity"]["folsom"]
    assert ir["n_duplicates"] == 1
    assert ir["n_gaps"] == 0
    assert ir["modal_cadence_seconds"] == pytest.approx(3600.0, rel=1e-6)


def test_irregularity_modal_cadence_per_dataset(fake_loader, tmp_path):
    """Each dataset's modal cadence matches its synthetic sampling freq."""
    res = e7_defects.run(fake_loader, tmp_path)
    ir = res["metrics"]["irregularity"]
    assert ir["gefcom2014"]["modal_cadence_seconds"] == pytest.approx(3600.0, rel=1e-6)
    assert ir["folsom"]["modal_cadence_seconds"] == pytest.approx(3600.0, rel=1e-6)
    assert ir["skippd"]["modal_cadence_seconds"] == pytest.approx(900.0, rel=1e-6)
    assert ir["sunlake"]["modal_cadence_seconds"] == pytest.approx(1800.0, rel=1e-6)
    # clean datasets: no gaps, no duplicates
    assert ir["skippd"]["n_gaps"] == 0 and ir["skippd"]["n_duplicates"] == 0
    assert ir["sunlake"]["n_gaps"] == 0 and ir["sunlake"]["n_duplicates"] == 0


# ---------------------------------------------------------------------------
# Quality flag composition + outliers
# ---------------------------------------------------------------------------

def test_quality_flag_composition_matches_synthetic(fake_loader, tmp_path):
    res = e7_defects.run(fake_loader, tmp_path)
    qf = res["metrics"]["quality_flag"]
    # GEFCom: alternating 0/3 -> both flags present, sums to row count
    g = qf["gefcom2014"]
    assert g["counts"].get("0", 0) > 0 and g["counts"].get("3", 0) > 0
    assert sum(g["counts"].values()) == g["n_total"]
    # Folsom: all-zero flags
    assert qf["folsom"]["counts"].get("0", 0) == qf["folsom"]["n_total"]
    assert "3" not in qf["folsom"]["counts"] or qf["folsom"]["counts"]["3"] == 0


def test_outliers_finite_on_populated_nan_on_all_nan(fake_loader, tmp_path):
    """Outlier fraction is finite on populated columns and NaN on all-NaN ones."""
    res = e7_defects.run(fake_loader, tmp_path)
    out = res["metrics"]["outliers"]
    # GEFCom: target (pv) populated -> finite; ghi_actual all-NaN -> NaN (honest)
    assert np.isfinite(out["gefcom2014"]["target_outlier_fraction"])
    assert 0.0 <= out["gefcom2014"]["target_outlier_fraction"] <= 0.25
    assert not np.isfinite(out["gefcom2014"]["ghi_outlier_fraction"])
    # Folsom: ghi populated -> finite; pv all-NaN -> target is ghi here, finite
    assert np.isfinite(out["folsom"]["ghi_outlier_fraction"])
    # SKIPPD: target (pv) finite; ghi_actual all-NaN -> NaN
    assert np.isfinite(out["skippd"]["target_outlier_fraction"])
    assert not np.isfinite(out["skippd"]["ghi_outlier_fraction"])


# ---------------------------------------------------------------------------
# Purity + CSV round-trip
# ---------------------------------------------------------------------------

def test_run_does_not_mutate_loader_data(fake_loader, tmp_path):
    before = {k: v.copy(deep=True) for k, v in fake_loader._frames.items()}
    e7_defects.run(fake_loader, tmp_path)
    for k, v in before.items():
        pd.testing.assert_frame_equal(v, fake_loader._frames[k])


def test_results_csv_has_expected_sections(fake_loader, tmp_path):
    e7_defects.run(fake_loader, tmp_path)
    csv = list((tmp_path / "results").glob("*.csv"))[0]
    df = pd.read_csv(csv)
    assert "dataset" in df.columns and "section" in df.columns
    for section in ("missingness", "irregularity", "quality_flag", "outliers"):
        assert (df["section"] == section).any(), f"CSV missing section {section}"
    # one missingness row per (dataset, canonical column)
    miss = df[df["section"] == "missingness"]
    n_datasets = miss["dataset"].nunique()
    assert n_datasets == 4
    # at least one row per dataset
    for ds in ("gefcom2014", "folsom", "skippd", "sunlake"):
        assert (miss["dataset"] == ds).any()


# ---------------------------------------------------------------------------
# Real-data RUN_SLOW integration
# ---------------------------------------------------------------------------

@slow
def test_run_real_standardized_packages(tmp_path):
    from solarunibench.loader import SolarUniBenchLoader

    loader = SolarUniBenchLoader(_REAL_DATA_ROOT)
    res = e7_defects.run(loader, tmp_path)
    assert isinstance(res, dict)
    for name in (
        "fig_defect_missingness.png",
        "fig_defect_irregularity.png",
        "fig_defect_outliers.png",
    ):
        assert (tmp_path / name).exists()
    assert list((tmp_path / "results").glob("*.csv"))
    # Print the E7 headline numbers for the report.
    print("\n=== E7 real-data defect summary ===")
    print("-- missingness (% NaN, canonical columns) --")
    for ds, cols in res["metrics"]["missingness"].items():
        hi = sorted(cols.items(), key=lambda kv: -kv[1])[:4]
        hi_str = ", ".join(f"{c}={v:.0f}%" for c, v in hi)
        print(f"  {ds:11s} top-NaN: {hi_str}")
    print("-- irregularity --")
    for ds, ir in res["metrics"]["irregularity"].items():
        cad = ir["modal_cadence_seconds"]
        cad_min = cad / 60.0 if np.isfinite(cad) else float("nan")
        print(
            f"  {ds:11s} cadence={cad_min:.1f}min gaps={ir['n_gaps']} "
            f"dups={ir['n_duplicates']} entities={ir['n_entities']}"
        )
    print("-- quality_flag composition --")
    for ds, qf in res["metrics"]["quality_flag"].items():
        print(f"  {ds:11s} {qf['counts']}")
    print("-- outliers (daytime IQR fraction) --")
    for ds, ou in res["metrics"]["outliers"].items():
        print(
            f"  {ds:11s} target={ou['target_outlier_fraction']} "
            f"ghi={ou['ghi_outlier_fraction']}"
        )
