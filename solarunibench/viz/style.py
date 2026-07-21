"""Dataviz design system for Solar-UniBench figures.

Colorblind-safe palette + light/dark matplotlib themes so every figure rendered
through the pipeline reads as one system (spec §11).

Palette: Wong / Okabe-Ito colorblind-safe set
(https://journals.sagepub.com/doi/10.1177/0956797614529780 ; also
https://mikemol.github.io/technique/colorblind/2017/02/04/color-palette.html).
The six hues below are blue / orange / bluish-green / yellow / vermillion /
reddish-purple -- all distinguishable under deuteranopia/protanopia/tritanopia.
"""
from __future__ import annotations

from contextlib import contextmanager

import matplotlib.pyplot as plt

#: 6-color colorblind-safe sequence (Wong/Okabe-Ito).
PALETTE: tuple[str, ...] = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
)

#: Per-model color. SmartPersistence is the physical reference baseline, so it
#: gets the green anchor; downstream models cycle the remaining palette hues.
MODEL_COLORS: dict[str, str] = {
    "SmartPersistence": PALETTE[2],
    "LightGBM": PALETTE[0],
    "MLP": PALETTE[1],
    "LSTM": PALETTE[4],
    "PatchTST": PALETTE[5],
    "CrossViViT": PALETTE[3],
}

#: Per-dataset color (one per standardized package).
DATASET_COLORS: dict[str, str] = {
    "gefcom2014": PALETTE[0],
    "skippd": PALETTE[1],
    "folsom": PALETTE[2],
    "sunlake": PALETTE[3],
}

# Foreground / background per theme.
_THEMES: dict[str, dict[str, str]] = {
    "light": {
        "face": "#ffffff",
        "fg": "#1a1a1a",
        "grid": "#cccccc",
    },
    "dark": {
        "face": "#121212",
        "fg": "#e0e0e0",
        "grid": "#3a3a3a",
    },
}


def _theme_rcparams(theme: str) -> dict[str, object]:
    if theme not in _THEMES:
        raise ValueError(f"unknown theme: {theme!r} (expected 'light' or 'dark')")
    t = _THEMES[theme]
    return {
        # Typography
        "font.family": "DejaVu Sans",
        "font.size": 11.0,
        "axes.titlesize": 12.0,
        "axes.labelsize": 12.0,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 10.0,
        # Resolution (print-ready)
        "figure.dpi": 150,
        "savefig.dpi": 150,
        # Clean axes
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": True,
        "grid.color": t["grid"],
        "grid.alpha": 0.35,
        "grid.linewidth": 0.8,
        # Legend
        "legend.frameon": False,
        # Layout
        "figure.autolayout": True,
        # Theme colors
        "figure.facecolor": t["face"],
        "axes.facecolor": t["face"],
        "savefig.facecolor": t["face"],
        "axes.edgecolor": t["fg"],
        "axes.labelcolor": t["fg"],
        "xtick.color": t["fg"],
        "ytick.color": t["fg"],
        "text.color": t["fg"],
    }


def set_style(theme: str = "light") -> None:
    """Apply the Solar-UniBench matplotlib style globally.

    Parameters
    ----------
    theme : {"light", "dark"}
        Visual theme to apply.
    """
    plt.rcParams.update(_theme_rcparams(theme))


@contextmanager
def with_style(theme: str = "light"):
    """Context manager that applies ``theme`` and restores prior rcParams on exit."""
    with plt.rc_context(_theme_rcparams(theme)):
        yield
