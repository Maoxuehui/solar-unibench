"""Tests for viz/io: save_fig (png/pdf) + save_metrics (csv)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from solarunibench.viz.io import save_fig, save_metrics


def test_save_fig_writes_png_and_closes(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    paths = save_fig(fig, "myfig", tmp_path)
    assert len(paths) == 1
    p = paths[0]
    assert p == tmp_path / "myfig.png"
    assert p.exists() and p.stat().st_size > 0
    assert not plt.fignum_exists(fig.number), "figure must be closed"


def test_save_fig_multi_format(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    paths = save_fig(fig, "f2", tmp_path, formats=("png", "pdf"))
    exts = sorted(p.suffix for p in paths)
    assert exts == [".pdf", ".png"]
    for p in paths:
        assert p.exists() and p.stat().st_size > 0


def test_save_fig_creates_missing_out_dir(tmp_path):
    out = tmp_path / "nested" / "deep"
    fig, ax = plt.subplots()
    paths = save_fig(fig, "g", out)
    assert paths[0].exists()


def test_save_metrics_writes_csv_in_results_dir(tmp_path):
    df = pd.DataFrame({"mae": [0.1, 0.2], "rmse": [0.3, 0.4]})
    p = save_metrics(df, "res", tmp_path)
    assert p == tmp_path / "results" / "res.csv"
    assert p.exists()
    got = pd.read_csv(p)
    assert list(got.columns) == ["mae", "rmse"]
    assert len(got) == 2
