"""Tests for the experiment orchestrator (scripts/run_experiments.py)."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
# Make solarunibench + scripts/ importable in-process.
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

import run_experiments as rx  # noqa: E402


def test_discover_returns_subset_of_E1_E9_and_never_raises():
    discovered = rx.discover_experiments()
    assert isinstance(discovered, dict)
    # Only ever exposes the canonical E1..E9 ids; missing modules are skipped.
    assert set(discovered).issubset({f"E{i}" for i in range(1, 10)})
    for fn in discovered.values():
        assert callable(fn)


def test_dispatch_only_selected_mock_experiment(tmp_path, monkeypatch):
    seen = {}

    def mock_run(loader, out_dir):
        seen["loader"] = type(loader).__name__
        seen["out_dir"] = str(out_dir)
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        from solarunibench.viz.io import save_fig, save_metrics
        save_fig(fig, "mock_fig", out_dir)
        save_metrics(pd.DataFrame({"mae": [0.1]}), "mock_metrics", out_dir)
        return {"figures": ["mock_fig"], "metrics": {"mae": 0.1}}

    monkeypatch.setattr(rx, "discover_experiments", lambda: {"MOCK": mock_run})

    out = tmp_path / "figures"
    rc = rx.main(["--only", "MOCK", "--data-root", str(tmp_path), "--out", str(out)])
    assert rc == 0
    assert seen["loader"] == "SolarUniBenchLoader"
    assert (out / "mock_fig.png").exists()
    assert (out / "results" / "mock_metrics.csv").exists()


def test_unknown_only_warns_but_does_not_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(rx, "discover_experiments", lambda: {})
    rc = rx.main(["--only", "NOPE", "--data-root", str(tmp_path), "--out", str(tmp_path)])
    assert rc == 0
