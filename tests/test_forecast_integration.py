"""End-to-end integration: 3 forecast baselines on real standardized GEFCom.

RUN_SLOW-gated. Loads the real GEFCom2014 standardized package through
:class:`solarunibench.loader.SolarUniBenchLoader`, builds windows with
:class:`EndogenousFeatureEngineer`, and runs SmartPersistence + LightGBM + MLP
through :class:`SolarForecastPipeline`. Asserts finite MAE/RMSE for each model,
computes ``skill_vs_reference`` (SmartPersistence = reference) for the two
learned models, and prints a 3-row skill table to stdout.

Real-data accommodations (all operating on an in-memory copy — the on-disk
parquet is never mutated):
* **PV signal lives only in the train split.** In the current standardized
  GEFCom package, ``pv_power_normalized`` is identically zero in the ``val`` and
  ``test`` splits (a Phase 5 data-quality regression — see
  ``.superpowers/sdd/phase6a-task-6-report.md``). To exercise the framework on
  real signal we therefore keep only the train split, restrict to its
  signal-bearing period (2012-04 .. 2013-03) and apply a per-entity
  chronological 80/20 train/eval partition, overwriting the ``split`` column.
* **Night NaNs.** ``pv_power_normalized`` is NaN at night; tabular models that
  require a finite target (LightGBM, MLP) cannot fit on windows with NaN in
  ``Y``. We wrap ``EndogenousFeatureEngineer`` in a small filter that keeps only
  windows whose target is fully finite across lookback + horizon (standard
  daylight-only solar forecast evaluation).
* **``clearness_index`` is entirely NaN** in the current GEFCom package. We
  keep it in ``feature_cols`` per the task brief and ``fillna(0)`` so the column
  is harmless (LightGBM ignores it; MLP sees a constant). This is documented as
  a data finding, not a framework behavior.
"""
from __future__ import annotations

import functools
import os
import warnings

import numpy as np
import pandas as pd
import pytest

DATA_ROOT = "D:/monica/workspace/solar-unibench/data/standardized"
DATASET = "gefcom2014"
TARGET = "pv_power_normalized"
FEATURE_COLS = ["ghi_clearsky", "cos_zenith", "clearness_index"]

# Signal-bearing period for GEFCom PV: outside this window the target is
# identically zero in every split (Phase 5 data regression).
_SIGNAL_START = "2012-04-01"
_SIGNAL_END = "2013-04-01"

slow = pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="set RUN_SLOW=1")


class _FiniteTargetWindowFilter:
    """Wrap a FeatureEngineer; keep only windows with finite target throughout.

    Drops windows whose ``Y`` (target over the horizon) or whose target channel
    in ``X_past`` (``X_past[:, :, 0]``) contains any NaN. This is the standard
    "score daylight only" evaluation discipline for solar forecasting and is
    required here because tabular models (LightGBM, MLP) cannot fit on NaN
    targets. Other feature channels (e.g. an all-NaN ``clearness_index``) are
    left untouched — LightGBM handles them natively; the test fillna-s them
    before they reach the MLP.
    """

    def __init__(self, inner):
        self.inner = inner

    def create_features(self, df, lookback, horizon, stride):
        Xp, Y, T, Xf, cz_issue = self.inner.create_features(df, lookback, horizon, stride)
        keep = np.isfinite(Y).all(axis=1) & np.isfinite(Xp[:, :, 0]).all(axis=1)
        Xp, Y, T = Xp[keep], Y[keep], T[keep]
        if Xf is not None:
            Xf = Xf[keep]
        if cz_issue is not None:
            cz_issue = cz_issue[keep]
        return Xp, Y, T, Xf, cz_issue


@functools.lru_cache(maxsize=1)
def _gefcom_signal_frame() -> pd.DataFrame:
    """Load GEFCom train split, restrict to the PV signal period, re-split 80/20.

    Returns an in-memory copy; the on-disk standardized parquet is untouched.
    The returned frame has its ``split`` column overwritten to a per-entity
    chronological 80/20 train/test partition so the pipeline's standard
    train/test discipline still applies.
    """
    from solarunibench.loader import SolarUniBenchLoader

    full = SolarUniBenchLoader(DATA_ROOT).load_tabular(DATASET, split="train").copy()
    # Drop tz for clean comparison against the signal window.
    ts_naive = pd.to_datetime(full["timestamp_utc"]).dt.tz_convert(None)
    sig = full[(ts_naive >= _SIGNAL_START) & (ts_naive < _SIGNAL_END)].copy()

    # ``clearness_index`` is entirely NaN in this package → fill with 0 so the
    # tabular MLP (which cannot ingest NaN features) can fit. The column is a
    # documented data-quality casualty, not a framework behavior.
    for col in [TARGET, *FEATURE_COLS]:
        if col in sig.columns:
            sig[col] = sig[col].fillna(0.0)

    # Per-entity chronological 80/20 split.
    parts = []
    for _, g in sig.groupby("entity_id", sort=False):
        g = g.sort_values("timestamp_utc").reset_index(drop=True)
        cut = int(len(g) * 0.8)
        g.loc[:, "split"] = "train"
        g.loc[cut:, "split"] = "test"
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def _provider(_config: PipelineConfig) -> pd.DataFrame:
    """df_provider hook: returns the cleaned GEFCom signal frame."""
    return _gefcom_signal_frame()


@slow
def test_three_baselines_on_gefcom_skill_table(capsys):
    """Run SmartPersistence + LightGBM + MLP on real GEFCom; print skill table."""
    from solarunibench.forecast.core import (
        EndogenousFeatureEngineer,
        PipelineConfig,
        SolarForecastPipeline,
    )
    from solarunibench.forecast.metrics import mae, rmse, skill_vs_reference
    from solarunibench.forecast.models.lightgbm_model import LightGBMModel
    from solarunibench.forecast.models.mlp_model import MLPModel
    from solarunibench.forecast.models.smart_persistence import SmartPersistenceModel

    cfg = PipelineConfig(
        dataset=DATASET,
        target=TARGET,
        lookback=6,
        horizon=3,
        stride=6,
        feature_cols=FEATURE_COLS,
        data_root=DATA_ROOT,
    )
    fe = _FiniteTargetWindowFilter(
        EndogenousFeatureEngineer(target=TARGET, feature_cols=FEATURE_COLS)
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        res_sp = SolarForecastPipeline(cfg, fe, SmartPersistenceModel()).run(
            df_provider=_provider
        )
        res_lgb = SolarForecastPipeline(cfg, fe, LightGBMModel()).run(df_provider=_provider)
        res_mlp = SolarForecastPipeline(cfg, fe, MLPModel()).run(df_provider=_provider)

    # Shared ground truth (windows are deterministic).
    y = res_sp["y_true"]
    assert res_lgb["y_true"].shape == y.shape
    assert res_mlp["y_true"].shape == y.shape

    # --- Finite MAE/RMSE for each model + prediction shape matches horizon. ---
    results = {
        "SmartPersistence": res_sp,
        "LightGBM": res_lgb,
        "MLP": res_mlp,
    }
    for name, res in results.items():
        assert res["pred"].shape == y.shape, (
            f"{name} pred shape {res['pred'].shape} != y_true shape {y.shape}"
        )
        m = mae(res["y_true"], res["pred"])
        r = rmse(res["y_true"], res["pred"])
        assert np.isfinite(m), f"{name} MAE is not finite: {m}"
        assert np.isfinite(r), f"{name} RMSE is not finite: {r}"

    # --- Skill vs SmartPersistence for the two learned models. ---
    skill_lgb = skill_vs_reference(y, res_lgb["pred"], res_sp["pred"])
    skill_mlp = skill_vs_reference(y, res_mlp["pred"], res_sp["pred"])
    assert np.isfinite(skill_lgb), f"LightGBM skill is not finite: {skill_lgb}"
    assert np.isfinite(skill_mlp), f"MLP skill is not finite: {skill_mlp}"

    # --- Print the 3-row skill table (bypass pytest capture so it always shows). ---
    table = _format_skill_table(
        ("SmartPersistence", res_sp, float("nan")),
        ("LightGBM", res_lgb, skill_lgb),
        ("MLP", res_mlp, skill_mlp),
    )
    with capsys.disabled():
        print(table)


def _format_skill_table(*rows) -> str:
    from solarunibench.forecast.metrics import mae, rmse

    lines = [
        "",
        "=== GEFCom forecast skill (reference: SmartPersistence) ===",
        f"{'model':<20}{'MAE':>12}{'RMSE':>12}{'skill-vs-SP':>14}",
    ]
    for name, res, skill in rows:
        m = mae(res["y_true"], res["pred"])
        r = rmse(res["y_true"], res["pred"])
        skill_str = "(reference)" if name == "SmartPersistence" else f"{skill:+.4f}"
        lines.append(f"{name:<20}{m:>12.4f}{r:>12.4f}{skill_str:>14}")
    return "\n".join(lines)
