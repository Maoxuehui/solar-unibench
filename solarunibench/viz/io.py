"""I/O helpers for persisting figures + numeric results.

All experiment figures flow through :func:`save_fig`; all numeric metrics flow
through :func:`save_metrics`, so output layout is uniform across E1-E9.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_fig(
    fig,
    name: str,
    out_dir,
    formats: tuple[str, ...] = ("png",),
) -> list[Path]:
    """Save ``fig`` to ``out_dir/<name>.<ext>`` for each requested format.

    The output directory is created if missing. The figure is closed afterwards
    so long-running experiment loops don't leak memory.

    Returns the list of written paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ext in formats:
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, format=ext)
        paths.append(p)
    plt.close(fig)
    return paths


def save_metrics(df: pd.DataFrame, name: str, out_dir) -> Path:
    """Write ``df`` to ``out_dir/results/<name>.csv`` (mkdir parents).

    Returns the written path.
    """
    out_dir = Path(out_dir)
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    p = results_dir / f"{name}.csv"
    df.to_csv(p, index=False)
    return p
