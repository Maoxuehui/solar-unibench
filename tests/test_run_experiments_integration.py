"""Integration tests for ``scripts/run_experiments.py`` (Phase 7 Task 11).

Two tiers:

* **FAST (default pytest run):** hermetic mock-dispatch (no real data, no heavy
  compute) + a real dispatch over E7 (descriptive diagnostics, no model) on the
  standardized packages + unit tests for the default bounded ``raw_provider``
  (RUN_SLOW gating, bounded-to-GEFCom, pass-through wiring). These verify
  dispatch + output wiring + the raw-provider contract without the heavy
  real-data / deep-model runs.
* **RUN_SLOW:** the orchestrator over E1-E9 on the real standardized packages,
  asserting EACH selected experiment produced >=1 figure + (where applicable) a
  results CSV, and printing a per-experiment summary table. E9's real bounded
  RAW GEFCom snapshot is built in-run (via the orchestrator's default
  ``raw_provider``) so E9 actually executes rather than silently skipping.
"""
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
# Make solarunibench + scripts/ importable in-process.
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

import run_experiments as rx  # noqa: E402

slow = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW"),
    reason="set RUN_SLOW=1 to run the full orchestrator integration",
)

_REAL_DATA_ROOT = str(_REPO / "data" / "standardized")


# Per-experiment figure filename prefix (first figure) + results CSV name, for
# the RUN_SLOW integration assertion. Mirrors the ``save_fig`` / ``save_metrics``
# names each experiment module actually uses.
_EXP_FIGURE_GLOBS: dict[str, tuple[str, ...]] = {
    "E1": ("fig_geo_blindtest",),
    "E2": ("fig_clearsky_skill",),
    "E3": ("fig_brr_gallery",),
    "E4": ("fig_optflow_sequence",),
    "E5": ("fig_quality_flags",),
    "E6": ("fig_align_lag_hist",),
    "E7": ("fig_defect_missingness",),
    "E8": ("fig_benchmark_heatmap",),
    "E9": ("fig_governance_vs_complexity",),
}
_EXP_RESULT_CSV: dict[str, str] = {
    "E1": "results_e1_geo.csv",
    "E2": "results_e2_clearsky.csv",
    "E3": "results_e3_brr.csv",
    "E4": "results_e4_optflow.csv",
    "E5": "results_e5_quality.csv",
    "E6": "results_e6_alignment.csv",
    "E7": "defect_summary.csv",
    "E8": "results_benchmark.csv",
    "E9": "results_e9_governance.csv",
}


# ---------------------------------------------------------------------------
# FAST: hermetic mock dispatch (no real data, no heavy compute)
# ---------------------------------------------------------------------------

def test_dispatch_mock_experiments_writes_figures_and_csvs(tmp_path, monkeypatch):
    """A mock loader + mock experiments dispatch through the orchestrator and
    land figures + a results CSV under ``out_dir``. Verifies dispatch + output
    wiring without touching real data."""
    import matplotlib.pyplot as plt
    import pandas as pd
    from solarunibench.viz.io import save_fig, save_metrics

    seen = {}

    def mock_a(loader, out_dir):
        seen["a_loader"] = type(loader).__name__
        fig, ax = plt.subplots()
        ax.plot([0, 1])
        save_fig(fig, "mock_a_fig", out_dir)
        save_metrics(pd.DataFrame({"mae": [0.1]}), "mock_a", out_dir)
        return {"figures": ["mock_a_fig"], "metrics": {"mae": 0.1, "status": "ran"}}

    def mock_b(loader, out_dir, *, raw_provider=None):
        # Accepts raw_provider → the orchestrator must pass its default.
        seen["b_raw_provider"] = raw_provider is not None
        seen["b_provider_returns_none"] = raw_provider("gefcom2014") is None  # no RUN_SLOW
        fig, ax = plt.subplots()
        ax.plot([1, 2])
        save_fig(fig, "mock_b_fig", out_dir)
        return {"figures": ["mock_b_fig"], "metrics": {"status": "ran"}}

    monkeypatch.setattr(rx, "discover_experiments", lambda: {"MA": mock_a, "MB": mock_b})

    out = tmp_path / "out"
    rc = rx.main(["--only", "MA,MB", "--data-root", str(tmp_path), "--out", str(out)])
    assert rc == 0
    assert seen["a_loader"] == "SolarUniBenchLoader"
    # MB accepts raw_provider → orchestrator wired its default provider through.
    assert seen["b_raw_provider"] is True
    assert seen["b_provider_returns_none"] is True  # fast no-op without RUN_SLOW
    assert (out / "mock_a_fig.png").exists()
    assert (out / "results" / "mock_a.csv").exists()
    assert (out / "mock_b_fig.png").exists()
    # Summary JSON captures both experiments.
    summary = __import__("json").loads((out / "results" / "orchestrator_summary.json").read_text())
    ids = {e["id"] for e in summary["experiments"]}
    assert ids == {"MA", "MB"}


def test_quick_subset_selects_fast_experiments(tmp_path, monkeypatch):
    """``--quick`` selects the canonical fast/descriptive subset (E1, E2, E7)
    and nothing else, filtered to what's discoverable."""
    selected: list[str] = []

    def fake_loader_run(loader, out_dir):  # never actually invoked
        return {"figures": [], "metrics": {}}

    monkeypatch.setattr(
        rx, "discover_experiments",
        lambda: {eid: fake_loader_run for eid in ("E1", "E2", "E7", "E8")},
    )
    # Intercept _print_summary by capturing `selected` via the summary JSON.
    out = tmp_path / "out"
    rc = rx.main(["--quick", "--data-root", str(tmp_path), "--out", str(out)])
    assert rc == 0
    summary = __import__("json").loads((out / "results" / "orchestrator_summary.json").read_text())
    # --quick selects E1, E2, E7 (E8 is NOT in the quick subset).
    assert set(summary["selected"]) == {"E1", "E2", "E7"}


# ---------------------------------------------------------------------------
# FAST: real dispatch over E7 (descriptive, no model) on real standardized data
# ---------------------------------------------------------------------------

def test_real_e7_dispatch_produces_defect_figures(tmp_path):
    """E7 is fast + descriptive (no model). Real dispatch through the
    orchestrator produces ``fig_defect_*`` + a defect-summary CSV — proves the
    end-to-end wiring (loader → experiment → figures/CSV) on the real
    standardized packages without the heavy model-using experiments."""
    if not (Path(_REAL_DATA_ROOT) / "gefcom2014").exists():
        pytest.skip("standardized packages not built")
    out = tmp_path / "out"
    rc = rx.main(["--only", "E7", "--data-root", _REAL_DATA_ROOT, "--out", str(out)])
    assert rc == 0
    defects = sorted(p.name for p in out.glob("fig_defect_*"))
    assert len(defects) >= 1, f"no fig_defect_* under {out}"
    assert (out / "results" / "defect_summary.csv").exists()
    # Summary JSON records E7 as ran with >=1 figure.
    summary = __import__("json").loads((out / "results" / "orchestrator_summary.json").read_text())
    e7 = next(e for e in summary["experiments"] if e["id"] == "E7")
    assert e7["status"] == "ran" and e7["n_figures"] >= 1


# ---------------------------------------------------------------------------
# FAST: default raw_provider contract (RUN_SLOW gating + bounded-to-GEFCom)
# ---------------------------------------------------------------------------

def test_raw_provider_pass_through_detection():
    """``_pass_raw_provider`` detects E5/E9 (accept ``raw_provider``) but not
    E7 (descriptive, no raw arm) — so the orchestrator wires the provider ONLY
    where the experiment can use it."""
    registry = rx.discover_experiments()
    assert "E5" in registry and "E7" in registry and "E9" in registry
    assert rx._pass_raw_provider(registry["E5"]) is True
    assert rx._pass_raw_provider(registry["E9"]) is True
    assert rx._pass_raw_provider(registry["E7"]) is False


def test_default_raw_provider_noop_without_run_slow(tmp_path, monkeypatch):
    """Without ``RUN_SLOW`` the provider returns ``None`` fast (no build, no
    I/O) so the orchestrator default stays fast and E9/E5 honestly skip."""
    monkeypatch.delenv("RUN_SLOW", raising=False)
    provider = rx._build_default_raw_provider(
        Path("/nonexistent/source"), tmp_path / "cache"
    )
    assert provider("gefcom2014") is None
    # Memoized: a second call is also None without re-checking the source.
    assert provider("gefcom2014") is None


def test_default_raw_provider_bounded_to_gefcom(tmp_path, monkeypatch):
    """The bounded dev config scopes the RAW build to GEFCom (spec E9); other
    datasets return ``None`` even under ``RUN_SLOW``."""
    monkeypatch.setenv("RUN_SLOW", "1")
    provider = rx._build_default_raw_provider(
        Path("/nonexistent/source"), tmp_path / "cache"
    )
    assert provider("folsom") is None
    assert provider("skippd") is None
    assert provider("sunlake") is None


def test_default_raw_provider_skips_when_source_missing_under_run_slow(
    tmp_path, monkeypatch, recwarn,
):
    """Under ``RUN_SLOW`` but with no raw source present, the provider warns +
    returns ``None`` rather than crashing (honest skip when the raw benchmark
    dataset is not installed)."""
    monkeypatch.setenv("RUN_SLOW", "1")
    provider = rx._build_default_raw_provider(
        Path("/nonexistent/source"), tmp_path / "cache"
    )
    assert provider("gefcom2014") is None
    assert any("raw source not present" in str(w.message) for w in recwarn.list)


def test_default_raw_provider_uses_disk_cache(monkeypatch, tmp_path):
    """A pre-existing cached parquet is reused — no rebuild. Keeps the RAW
    build tractable across repeated orchestrator runs."""
    import pandas as pd

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    df = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(["2020-01-01", "2020-01-02"], utc=True),
            "entity_id": ["GEFCOM_E0", "GEFCOM_E0"],
            "pv_power_normalized": [0.1, 0.2],
            "ghi_clearsky": [800.0, 850.0],
            "cos_zenith": [0.5, 0.6],
            "capacity_kw": [1.0, 1.0],
        }
    )
    key = cache_dir / "gefcom2014_raw_bounded_limit1500.parquet"
    df.to_parquet(key)

    monkeypatch.setenv("RUN_SLOW", "1")
    provider = rx._build_default_raw_provider(Path("/nonexistent"), cache_dir)
    out = provider("gefcom2014")
    assert out is not None
    assert "pv_power_normalized" in out.columns
    pd.testing.assert_frame_equal(out, df)


# ---------------------------------------------------------------------------
# RUN_SLOW: full E1-E9 orchestrator integration on real standardized packages
# ---------------------------------------------------------------------------

@slow
def test_full_orchestrator_e1_e9(tmp_path):
    """Run the orchestrator over E1-E9 on the real standardized packages and
    assert EACH selected experiment produced >=1 figure + (where applicable) a
    results CSV. E9's real bounded RAW GEFCom snapshot is built in-run via the
    orchestrator's default ``raw_provider`` (``RUN_SLOW``-gated), so E9 actually
    executes. Prints a per-experiment summary table.
    """
    if not (Path(_REAL_DATA_ROOT) / "gefcom2014").exists():
        pytest.skip("standardized packages not built")

    out = tmp_path / "out"
    rc = rx.main(
        [
            "--data-root", _REAL_DATA_ROOT,
            "--out", str(out),
            "--raw-cache", str(tmp_path / "raw_cache"),
        ]
    )
    assert rc == 0

    summary = __import__("json").loads((out / "results" / "orchestrator_summary.json").read_text())
    by_id = {e["id"]: e for e in summary["experiments"]}

    print("\n=== Orchestrator E1-E9 integration (RUN_SLOW) ===")
    print(f"  {'exp':<4} {'figures':>8} {'csv':>6}  status")
    n_total_figures = 0
    for exp_id in ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"):
        entry = by_id.get(exp_id)
        assert entry is not None, f"{exp_id} missing from orchestrator summary"
        # Each selected experiment produced >=1 figure under out_dir.
        figs = []
        for prefix in _EXP_FIGURE_GLOBS[exp_id]:
            figs += list(out.glob(f"{prefix}*"))
        figs = sorted(set(p.name for p in figs))
        n_total_figures += len(figs)
        csv_name = _EXP_RESULT_CSV[exp_id]
        csv_ok = (out / "results" / csv_name).exists()
        print(f"  {exp_id:<4} {len(figs):>8} {'yes' if csv_ok else 'no':>6}  {entry['status']}")
        assert len(figs) >= 1, f"{exp_id} produced no figures (prefix {_EXP_FIGURE_GLOBS[exp_id]})"
        assert csv_ok, f"{exp_id} missing results CSV {csv_name}"
        assert entry["n_figures"] >= 1, f"{exp_id} summary recorded 0 figures"

    # The full figure set spans all 9 experiments.
    assert n_total_figures >= 9
    # E9 specifically: no longer silently skips — it produced governance figures
    # + a results CSV on the real bounded RAW GEFCom snapshot.
    assert by_id["E9"]["status"] == "ran"
    assert any(p.name.startswith("fig_governance_vs_complexity") for p in out.glob("fig_governance*"))
    assert any(p.name.startswith("fig_cleaning_delta") for p in out.glob("fig_cleaning_delta*"))
    print(f"  total figures: {n_total_figures} across {len(by_id)} experiments")
