"""Eq. (2)-(3) — physical-bounds constraints & mask generation.

Canonical physics: ``operators.quality.apply_quality_audit`` (nocturnal
zeroing, clearness-index clamp k_t in [0, 1.5], per-row quality flags,
PV capacity clipping). Defaults live in ``config/default_curation.yaml``.
"""
from __future__ import annotations

import pandas as pd

from ..operators.quality import apply_quality_audit

__all__ = ["apply_physical_bounds", "apply_quality_audit"]

#: Eq. (2) — clearness-index clamp bounds (see config/default_curation.yaml)
KT_BOUNDS: tuple[float, float] = (0.0, 1.5)


def apply_physical_bounds(df_ts: pd.DataFrame,
                          capacity: float = 1.0) -> pd.DataFrame:
    """Nocturnal masking + k_t clamp + quality flags — Eq. (2)-(3).

    * values with ``cos_zenith <= 0`` are zeroed (nocturnal mask);
    * ``clearness_index`` is clipped to :data:`KT_BOUNDS`;
    * ``quality_flag`` marks suspect rows; PV is clipped to ``capacity``.

    Honest-NaN: entirely-absent modalities stay absent (no fabricated zeros).
    """
    return apply_quality_audit(df_ts, capacity=capacity)
