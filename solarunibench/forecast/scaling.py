"""Global physical-constant feature scaler for the forecast pipeline.

:class:`PhysicalFeatureScaler` is applied to the raw tabular frame BEFORE
feature engineering (see :class:`solarunibench.forecast.core.SolarForecastPipeline`)
and implements three column-aware policies:

* **Irradiance columns** (regex ``^(ghi|dni|dhi)`` — matches ``ghi_clearsky``,
  ``ghi_actual``, ``dni_clearsky``, ``dhi_*``, …) are divided by a physical
  constant ``irradiance_max`` (default **1500.0 W/m^2**, a ceiling just above
  the solar constant at the top of the atmosphere). No ``fit`` is needed →
  there is no possibility of train/test leakage, and nighttime ``0.0`` is
  preserved exactly (``0 / 1500 == 0``). This fixes the MLP divergence caused
  by unscaled ``ghi_clearsky`` ∈ [0, ~1100] sitting next to a target ∈ [0, 1].

* **NWP columns** (regex ``^nwp_`` — e.g. ``nwp_VAR78``) are z-scored per
  column by a :class:`sklearn.preprocessing.StandardScaler` **fit on TRAIN
  μ/σ only**. ``fit`` is NaN-safe (NaNs are skipped when computing μ/σ);
  ``transform`` preserves NaN positions (``(NaN - μ) / σ == NaN``).

* **Everything else** (``cos_zenith`` ∈ [0, 1], ``clearness_index`` ∈ [0, 1],
  ``pv_power_normalized`` target ∈ [0, 1], ``timestamp_utc``, ``entity_id``,
  ``split``) is passed through unchanged.

When the frame contains no irradiance/NWP columns (e.g. the synthetic
``df_provider`` used by the forecast-core tests), the scaler is an exact
no-op — this keeps the default-on wiring from breaking pre-existing tests.

The scaler is injected into :class:`SolarForecastPipeline` as ``scaler=None``
(defaults to ``PhysicalFeatureScaler()``), so "raw vs scaled" is an ablation
axis: pass a no-op scaler to recover unscaled behavior.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Physical ceiling for surface irradiance: just above the solar constant
# (~1361 W/m^2 TOA, ≤ ~1000 W/m^2 at the surface). Dividing by it brings
# every realistic irradiance value into [0, 1] with no fit state.
DEFAULT_IRRADIANCE_MAX: float = 1500.0

# Regex anchors for the three column policies.
_IRRADIANCE_RE = re.compile(r"^(ghi|dni|dhi)")
_NWP_RE = re.compile(r"^nwp_")


class PhysicalFeatureScaler:
    """Global, train-fit, zero-leakage feature scaler.

    Parameters
    ----------
    irradiance_max:
        Physical constant used to normalize irradiance columns. Defaults to
        :data:`DEFAULT_IRRADIANCE_MAX` (1500.0 W/m^2).

    Notes
    -----
    Irradiance normalization is stateless (a constant divisor) and therefore
    cannot leak. NWP z-scoring IS stateful: ``fit`` must be called on the
    TRAIN frame only, and the fitted μ/σ are then applied to val/test.
    """

    def __init__(self, irradiance_max: float = DEFAULT_IRRADIANCE_MAX):
        self.irradiance_max = float(irradiance_max)
        # Per-column fitted StandardScalers for NWP cols discovered at fit time.
        self._nwp_scalers: dict[str, StandardScaler] = {}

    # ------------------------------------------------------------------ API

    def fit(self, df: pd.DataFrame) -> "PhysicalFeatureScaler":
        """Fit NWP z-score statistics on ``df`` (expected to be the TRAIN frame).

        Irradiance scaling has no fit state (constant divisor). NWP columns
        with no finite values are left as passthrough (mean=0, scale=1).
        NaNs in an NWP column are skipped when computing μ/σ.
        """
        self._nwp_scalers = {}
        for col in df.columns:
            if not _NWP_RE.match(col):
                continue
            values = df[col].to_numpy(dtype=float)
            finite = values[np.isfinite(values)].reshape(-1, 1)
            scaler = StandardScaler(with_mean=True, with_std=True)
            if finite.size:
                scaler.fit(finite)
                # Guard against zero-variance columns (div-by-zero on transform).
                # Recent sklearn versions do this internally; older ones do not.
                if scaler.scale_[0] == 0.0:
                    scaler.scale_ = np.array([1.0])
            else:
                # No finite data → degenerate scaler that leaves values unchanged.
                scaler.mean_ = np.array([0.0])
                scaler.scale_ = np.array([1.0])
                scaler.var_ = np.array([0.0])
                scaler.n_features_in_ = 1
            self._nwp_scalers[col] = scaler
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply irradiance ÷ constant and NWP z-scoring; passthrough everything else.

        Safe to call on a frame whose NWP columns were not present at ``fit``
        time (they are passed through) and on a frame with no irradiance/NWP
        columns at all (exact no-op).
        """
        out = df.copy()
        # Empty frame (e.g. a split with zero rows): nothing to scale, and
        # StandardScaler.transform rejects 0-sample inputs, so short-circuit.
        if not len(out):
            return out
        for col in out.columns:
            if _IRRADIANCE_RE.match(col):
                out[col] = out[col] / self.irradiance_max
            elif col in self._nwp_scalers:
                raw = out[[col]].to_numpy(dtype=float)
                out[col] = self._nwp_scalers[col].transform(raw).ravel()
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: ``self.fit(df).transform(df)``."""
        return self.fit(df).transform(df)
