"""Tests for the viz design system (palette + light/dark themes)."""
import matplotlib

matplotlib.use("Agg")  # no display required
import matplotlib.pyplot as plt

import pytest

from solarunibench.viz import style
from solarunibench.viz.style import PALETTE, MODEL_COLORS, DATASET_COLORS


def test_palette_has_six_distinct_hexes():
    assert len(PALETTE) == 6
    assert len(set(PALETTE)) == 6, "palette colors must be distinct"
    for c in PALETTE:
        assert isinstance(c, str) and c.startswith("#") and len(c) == 7, c


def test_set_style_changes_rcparams_and_dpi():
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    style.set_style("light")
    assert plt.rcParams["figure.dpi"] >= 150
    assert plt.rcParams["savefig.dpi"] >= 150
    light_bg = plt.rcParams["axes.facecolor"]
    # clean spines + grid on
    assert plt.rcParams["axes.grid"] is True
    assert plt.rcParams["axes.spines.top"] is False
    style.set_style("dark")
    dark_bg = plt.rcParams["axes.facecolor"]
    assert light_bg != dark_bg, "light/dark must differ in axes facecolor"
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def test_set_style_rejects_unknown_theme():
    with pytest.raises(ValueError):
        style.set_style("neon")


def test_model_and_dataset_colors_drawn_from_palette():
    assert len(DATASET_COLORS) >= 4, "need a color per dataset"
    for c in MODEL_COLORS.values():
        assert c in PALETTE
    for c in DATASET_COLORS.values():
        assert c in PALETTE
    # 4 datasets each get a distinct color
    assert len(set(DATASET_COLORS.values())) == len(DATASET_COLORS)


def test_with_style_context_restores_previous():
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    before = plt.rcParams["axes.facecolor"]
    with style.with_style("dark"):
        assert plt.rcParams["axes.facecolor"] != before
    assert plt.rcParams["axes.facecolor"] == before
