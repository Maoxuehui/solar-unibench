"""Solar-UniBench visualization design system (palette, themes, I/O)."""
from .style import (
    PALETTE,
    MODEL_COLORS,
    DATASET_COLORS,
    set_style,
    with_style,
)
from .io import save_fig, save_metrics

__all__ = [
    "PALETTE",
    "MODEL_COLORS",
    "DATASET_COLORS",
    "set_style",
    "with_style",
    "save_fig",
    "save_metrics",
]
