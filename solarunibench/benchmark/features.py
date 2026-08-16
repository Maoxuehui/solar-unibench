"""Feature prep for the E8 unified benchmark.

The benchmark fixes the input to F=4 channels
``[target, cos_zenith, ghi_clearsky, clearness_index_aligned]`` for every
dataset. The first three columns already exist in every standardized package
(see :mod:`solarunibench.contract`); this module derives the fourth —
``clearness_index_aligned`` — in a target-aware way so the same
:class:`~solarunibench.forecast.core.EndogenousFeatureEngineer` produces a
uniform design matrix across GHI and PV datasets.

Target-aware definition (Honest NaN — never fabricates measurements):

* **GHI datasets** (``target == "ghi_actual"``): the true clearness index,
  ``clip(ghi_actual / (ghi_clearsky + eps), 0, 1.5)``.

* **PV datasets** (``target == "pv_power_normalized"``): a
  ``target / (ghi_clearsky + eps)`` **proxy** (no ground GHI is available, so
  ``clearness_index`` in the contract is NaN for PV — we do not fabricate it).
  Night rows — where ``ghi_clearsky == 0`` — are defined as ``0`` to avoid
  division-by-zero; clipped to ``[0, 1.5]``.

In both branches NaN propagates: a missing ``ghi_actual`` / target / clearsky
yields NaN, never an invented value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Small floor inside the denominator to avoid division-by-zero when
# ``ghi_clearsky`` is exactly 0 in the GHI branch (the PV branch uses an
# explicit zero-rule). Physically negligible — clear-sky irradiance is in W/m².
_EPS = 1e-6

# Clearness index is bounded to [0, 1.5]: 0 = total overcast / night,
# 1 = clear sky, >1 only slightly (measurement noise / thin cloud brightening).
_CLEARNESS_CAP = 1.5

_GHI_TARGET = "ghi_actual"
_PV_TARGET = "pv_power_normalized"


def add_clearness_aligned(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Return ``df`` with a ``clearness_index_aligned`` column added.

    Parameters
    ----------
    df
        Tabular frame containing at least ``ghi_clearsky`` and the ``target``
        column. Other columns are passed through unchanged.
    target
        Forecast target column: ``"ghi_actual"`` selects the real clearness
        ratio; ``"pv_power_normalized"`` (or any other PV target) selects the
        proxy with the night-zero rule.

    Returns
    -------
    pandas.DataFrame
        A new frame (``df.assign``) with ``clearness_index_aligned`` as a
        ``float32`` column.

    Raises
    ------
    ValueError
        If ``ghi_clearsky`` or ``target`` is not a column (Honest NaN — we
        refuse to compute rather than fabricate).
    """
    if "ghi_clearsky" not in df.columns:
        raise ValueError(
            "add_clearness_aligned requires a 'ghi_clearsky' column "
            "(absent — refusing to fabricate the feature)."
        )
    if target not in df.columns:
        raise ValueError(
            f"add_clearness_aligned: target column '{target}' not in frame "
            f"(refusing to fabricate the feature)."
        )

    clearsky = df["ghi_clearsky"].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    denom = clearsky + _EPS

    if target == _GHI_TARGET:
        # Real clearness index. At night clearsky==0 and ghi_actual≈0 → 0/eps=0,
        # naturally 0 without a special case; NaN ghi_actual propagates.
        aligned = np.clip(y / denom, 0.0, _CLEARNESS_CAP)
    else:
        # PV proxy: no ground GHI available, so use target/clearsky. Define 0
        # at night (clearsky == 0) explicitly to avoid div-by-zero and give the
        # model a clean night signal. NaN clearsky / NaN target propagate.
        ratio = np.clip(y / denom, 0.0, _CLEARNESS_CAP)
        aligned = np.where(clearsky == 0, 0.0, ratio)

    return df.assign(clearness_index_aligned=aligned.astype(np.float32))
