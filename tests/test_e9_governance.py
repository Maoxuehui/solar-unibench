"""Tests for E9 — Data governance vs model complexity (spec §11, E9; centerpiece).

*Hypothesis ("治理重于模型"):* cleaning + alignment buys more metric gain than
scaling model complexity — **a simple model on well-governed data can match or
beat a complex model on raw data.**

The fast tests use a SYNTHETIC pre-governance (RAW) frame with INJECTED label
noise — daytime over-capacity clipping, negative spikes, and night sensor
readings — that the quality audit removes. A ladder of models
{SmartPersistence, LightGBM, MLP} is run over the full 2-axis grid
{raw, cleaned} x {models}. The synthetic case is constructed so governance
clearly DOMINATES (heavy label noise -> cleaning gives a large gain; the model
ladder gives only modest gains), and the headline test asserts the crossover
there: the simplest non-trivial model on CLEANED >= the most complex model on
RAW. This proves the MEASUREMENT works (the thesis is empirically testable).

A ``RUN_SLOW``-gated test builds a bounded GEFCom RAW snapshot in-run (the
genuine pre-governance frame) and runs E9 end-to-end, reporting the REAL grid
honestly — the crossover may or may not hold at TINY dev configs; the test
asserts only that the grid is computed + finite, not that the thesis holds
(the publishable crossover comes at Phase 8 with full configs).
"""
import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from solarunibench.experiments import e9_governance
from solarunibench.forecast.models import (
    LightGBMModel,
    MLPModel,
    SmartPersistenceModel,
)
from solarunibench.operators.quality import apply_quality_audit

# LightGBM is quiet by default; keep pytest output pristine regardless.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

slow = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW"),
    reason="set RUN_SLOW=1 to run the real-data E9 integration test",
)

_REAL_DATA_ROOT = "D:/monica/workspace/solar-unibench/data/standardized"
_GEFCom_RAW_SOURCE = "D:/monica/workspace/solar_benchmark_dataset/GEFCom2014_data"
_REAL_GEFCom_LIMIT = 1500


# ---------------------------------------------------------------------------
# Synthetic raw + cleaned frames (E5 pattern, tuned so governance dominates)
# ---------------------------------------------------------------------------

class _FakeLoader:
    """Minimal loader stub (E9 sources both arms from ``raw_provider`` for
    rigorous row-alignment; the loader is the contract's first arg)."""

    def __init__(self, data_root: str = "/fake"):
        self.data_root = data_root

    def load_tabular(self, dataset: str, split: str | None = None) -> pd.DataFrame:
        raise KeyError(dataset)


def _synth_raw_frame(
    name: str,
    target_col: str,
    n_entities: int = 2,
    n_days: int = 40,
    seed: int = 0,
) -> pd.DataFrame:
    """A pre-governance frame (anchors merged) WITH heavy injected label noise.

    Mirrors the ``--raw`` snapshot: ``ghi_clearsky`` / ``cos_zenith`` anchors are
    present (computed BEFORE the audit), but the target carries substantial
    daytime noise the quality-audit removes:

    * **night sensor noise** — non-zero target where ``cos_zenith <= 0``
      (audit night-zeros these).
    * **daytime over-capacity clipping** (PV) — values > 1.0 (audit clips to
      capacity). Heavy fraction so the bias is visible to the daytime evaluator.
    * **daytime negative spikes** — sub-zero readings (audit clamps to 0).

    The daytime noise is heavy + EXTREME-OUT-OF-BOUNDS BY DESIGN so governance
    clearly dominates the model-complexity axis in the synthetic crossover
    assertion. The audit's daytime cleaning power is bounding (clip PV to
    capacity, clamp negatives to 0); a model trained on out-of-bounds extremes
    learns out-of-bounds predictions that fail catastrophically on the bounded
    cleaned test (large train-vs-test distribution mismatch — exactly the
    failure mode governance fixes in real data). The anchors
    (``ghi_clearsky``, ``cos_zenith``) are IDENTICAL in raw and cleaned — the
    ONLY difference between arms is the governed target.
    """
    rows = []
    scale = 1.0 if target_col == "pv_power_normalized" else 100.0
    for e in range(n_entities):
        eid = f"{name.upper()}_E{e}"
        n = n_days * 24
        times = pd.date_range("2020-01-01", periods=n, freq="1h")
        hod = times.hour.to_numpy()
        bell = np.clip(np.sin(np.pi * (hod - 6) / 13.0), 0.0, None)
        rng = np.random.default_rng(seed + e)
        ghi_clear = bell * 950.0
        cos_zen = bell  # proxy in [0, 1]; > 0.05 across the daytime band
        if target_col == "pv_power_normalized":
            clean = np.clip(bell * 0.9 + rng.normal(0, 0.02, n), 0, 1)
        else:  # ghi_actual
            clean = bell * 900.0 + rng.normal(0, 20, n)

        raw_target = clean.copy().astype(float)
        night = cos_zen <= 0
        day_idx = np.where(cos_zen > 0.05)[0]
        # Night sensor noise (audit night-zeros these).
        if night.any():
            raw_target[night] = np.abs(rng.normal(0.6 * scale, 0.3 * scale, int(night.sum())))
        # Daytime over-capacity (PV) — EXTREME (audit clips to capacity=1.0). A
        # heavy fraction at ~3.0 forces a large train-vs-test mismatch: the
        # raw-trained model learns to predict ~3.0 at these (high-bell midday)
        # points, but the cleaned test target is clipped to 1.0 there.
        if target_col == "pv_power_normalized" and len(day_idx):
            k = max(1, int(0.35 * len(day_idx)))
            sel = rng.choice(day_idx, size=k, replace=False)
            raw_target[sel] = 3.0 + rng.normal(0, 0.1, k)
        # Daytime negative spikes (EXTREME; audit clamps to 0). Scaled to the
        # target magnitude so the mismatch is comparable across PV/GHI datasets.
        if len(day_idx):
            k = max(1, int(0.25 * len(day_idx)))
            sel = rng.choice(day_idx, size=k, replace=False)
            raw_target[sel] = -np.abs(rng.normal(1.5 * scale, 0.3 * scale, k))

        for i in range(n):
            rows.append({
                "timestamp_utc": times[i],
                "entity_id": eid,
                target_col: float(raw_target[i]),
                "ghi_clearsky": float(ghi_clear[i]),
                "cos_zenith": float(cos_zen[i]),
                "capacity_kw": 1.0,
            })
    df = pd.DataFrame(rows)
    # Pre-governance: quality_flag / clearness_index are NA (audit not yet run).
    df["quality_flag"] = pd.NA
    df["clearness_index"] = pd.NA
    return df


def _derive_cleaned(raw_df: pd.DataFrame, capacity: float = 1.0) -> pd.DataFrame:
    """Run the quality audit on a raw frame -> the cleaned (governed) frame.

    Mirrors ``run_pipeline(raw=False)``: drop the NA ``quality_flag`` /
    ``clearness_index`` so the audit owns them, then run the audit. Row-aligned
    with ``raw_df`` and differs ONLY in governance.
    """
    base = raw_df.drop(
        columns=[c for c in ("quality_flag", "clearness_index") if c in raw_df.columns]
    )
    return apply_quality_audit(base, capacity=capacity)


_TARGETS = {
    "gefcom2014": "pv_power_normalized",
    "skippd": "pv_power_normalized",
    "folsom": "ghi_actual",
    "sunlake": "ghi_actual",
}


# Fast TINY model ladder (no torch fit loop) for the unit tests.
def _fast_ladder():
    return [
        SmartPersistenceModel(),
        LightGBMModel(params={"n_estimators": 40, "n_jobs": 1, "random_state": 0}),
        MLPModel(hidden_layer_sizes=(32,), max_iter=80, random_state=0),
    ]


@pytest.fixture
def synth_setup():
    """Return (loader, raw_provider, raw_frames, cleaned_frames) for one PV
    dataset (GEFCom — the representative fast PV dataset, spec E9)."""
    target = _TARGETS["gefcom2014"]
    raw_df = _synth_raw_frame("gefcom2014", target, seed=0)
    cleaned_df = _derive_cleaned(raw_df)
    loader = _FakeLoader()

    def raw_provider(dataset: str):
        return raw_df.copy() if dataset == "gefcom2014" else None

    return loader, raw_provider, raw_df, cleaned_df


# ---------------------------------------------------------------------------
# Contract tests for run(loader, out_dir, *, raw_provider=..., model_ladder=...)
# ---------------------------------------------------------------------------

def test_run_returns_figures_and_metrics_dict(synth_setup, tmp_path):
    loader, raw_provider, _, _ = synth_setup
    res = e9_governance.run(
        loader, tmp_path, raw_provider=raw_provider, model_ladder=_fast_ladder
    )
    assert isinstance(res, dict)
    assert set(res) >= {"figures", "metrics"}
    for fig in ("fig_governance_vs_complexity.png", "fig_cleaning_delta.png"):
        p = tmp_path / fig
        assert p.exists(), f"missing {fig}"
        assert p.stat().st_size > 0
    csvs = list((tmp_path / "results").glob("*.csv"))
    assert len(csvs) >= 1
    assert any("fig_governance_vs_complexity.png" in p for p in res["figures"])
    assert any("fig_cleaning_delta.png" in p for p in res["figures"])


def test_no_raw_provider_honest_skip(tmp_path):
    """Without raw_provider E9 cannot build the 2-axis grid (RAW data is not in
    the standardized packages); it honestly skips rather than fabricating."""
    loader = _FakeLoader()
    res = e9_governance.run(loader, tmp_path, model_ladder=_fast_ladder)
    assert res["metrics"]["status"] == "skipped"
    assert res["figures"] == []
    assert "note" in res["metrics"]


# ---------------------------------------------------------------------------
# Core: the 2-axis grid is complete (2 data states x N models)
# ---------------------------------------------------------------------------

def test_grid_complete_two_axes(synth_setup, tmp_path):
    """Every (data_state, model) cell is computed. The grid has 2 arms (raw,
    cleaned) x len(ladder) models, each with a finite skill."""
    loader, raw_provider, _, _ = synth_setup
    res = e9_governance.run(
        loader, tmp_path, raw_provider=raw_provider, model_ladder=_fast_ladder
    )
    grid = res["metrics"]["grid"]
    assert "gefcom2014" in grid
    cells = grid["gefcom2014"]
    # Ladder = SmartPersistence, LightGBM, MLP -> 3 model rows.
    assert set(cells) == {"SmartPersistence", "LightGBM", "MLP"}
    for model_name, cell in cells.items():
        assert set(cell) >= {"raw", "cleaned"}
        for arm in ("raw", "cleaned"):
            assert np.isfinite(cell[arm]["skill"]), (
                f"{model_name}/{arm}: skill not finite"
            )
            assert np.isfinite(cell[arm]["daytime_mae"])
            assert cell[arm]["n_daytime"] > 0


def test_complexity_ladder_order_recorded(synth_setup, tmp_path):
    """Each cell carries its complexity rank (0 = simplest) so the complexity
    axis is explicit, not implicit in row order."""
    loader, raw_provider, _, _ = synth_setup
    res = e9_governance.run(
        loader, tmp_path, raw_provider=raw_provider, model_ladder=_fast_ladder
    )
    cells = res["metrics"]["grid"]["gefcom2014"]
    ranks = {m: cells[m]["complexity_rank"] for m in cells}
    assert ranks["SmartPersistence"] == 0
    assert ranks["LightGBM"] == 1
    assert ranks["MLP"] == 2
    assert ranks["SmartPersistence"] < ranks["LightGBM"] < ranks["MLP"]


# ---------------------------------------------------------------------------
# Fair comparison: every cell evaluates on the SAME cleaned test windows
# ---------------------------------------------------------------------------

def test_arms_share_identical_daytime_windows(synth_setup, tmp_path):
    """All cells evaluate on the SAME cleaned test windows (clean ground truth,
    E5 pattern) so n_daytime is identical across the whole grid -> the skill
    comparison is apples-to-apples."""
    loader, raw_provider, _, _ = synth_setup
    res = e9_governance.run(
        loader, tmp_path, raw_provider=raw_provider, model_ladder=_fast_ladder
    )
    cells = res["metrics"]["grid"]["gefcom2014"]
    nds = {mn: c["raw"]["n_daytime"] for mn, c in cells.items()}
    nds.update({mn: c["cleaned"]["n_daytime"] for mn, c in cells.items()})
    assert len(set(nds.values())) == 1, (
        f"daytime window inconsistent across grid: {nds}"
    )


def test_skill_formula_consistent_with_reference(synth_setup, tmp_path):
    """skill = 1 - MAE_arm / MAE_SmartPersistence_cleaned (the shared
    reference). SmartPersistence on cleaned is the denominator -> its own
    cleaned-arm skill is exactly 0 (parity with itself)."""
    loader, raw_provider, _, _ = synth_setup
    res = e9_governance.run(
        loader, tmp_path, raw_provider=raw_provider, model_ladder=_fast_ladder
    )
    cells = res["metrics"]["grid"]["gefcom2014"]
    ref_mae = cells["SmartPersistence"]["cleaned"]["daytime_mae"]
    assert np.isfinite(ref_mae) and ref_mae > 0
    # SmartPersistence on cleaned = the reference -> skill exactly 0.
    assert cells["SmartPersistence"]["cleaned"]["skill"] == pytest.approx(0.0, abs=1e-9)
    for mn, cell in cells.items():
        for arm in ("raw", "cleaned"):
            expected = 1.0 - cell[arm]["daytime_mae"] / ref_mae
            assert cell[arm]["skill"] == pytest.approx(expected, abs=1e-9), (
                f"{mn}/{arm}: skill {cell[arm]['skill']} != formula {expected}"
            )


# ---------------------------------------------------------------------------
# Headline hypothesis: governance cuts label noise -> cleaned skill >= raw skill
# ---------------------------------------------------------------------------

def test_delta_hygiene_nonneg_per_model(synth_setup, tmp_path):
    """Δ-hygiene = skill(model, cleaned) - skill(model, raw) >= 0 for every
    model (cleaning never hurts; on the synthetic frame it removes real noise).
    """
    loader, raw_provider, _, _ = synth_setup
    res = e9_governance.run(
        loader, tmp_path, raw_provider=raw_provider, model_ladder=_fast_ladder
    )
    cells = res["metrics"]["grid"]["gefcom2014"]
    for mn, cell in cells.items():
        dh = cell["delta_hygiene"]
        assert np.isfinite(dh), f"{mn}: delta_hygiene not finite"
        assert dh >= -1e-9, (
            f"{mn}: delta_hygiene ({dh:.4f}) < 0 — cleaning hurt this model"
        )


def test_delta_complexity_computed_per_model(synth_setup, tmp_path):
    """Δ-complexity on each data state = skill(model) - skill(SmartPersistence)
    on that same data state (the gain from scaling complexity). Computed for
    both arms; == 0 for SmartPersistence by construction."""
    loader, raw_provider, _, _ = synth_setup
    res = e9_governance.run(
        loader, tmp_path, raw_provider=raw_provider, model_ladder=_fast_ladder
    )
    cells = res["metrics"]["grid"]["gefcom2014"]
    sp_raw = cells["SmartPersistence"]["raw"]["skill"]
    sp_cl = cells["SmartPersistence"]["cleaned"]["skill"]
    for mn, cell in cells.items():
        assert np.isfinite(cell["delta_complexity_raw"])
        assert np.isfinite(cell["delta_complexity_cleaned"])
        expected_raw = cell["raw"]["skill"] - sp_raw
        expected_cl = cell["cleaned"]["skill"] - sp_cl
        assert cell["delta_complexity_raw"] == pytest.approx(expected_raw, abs=1e-9)
        assert cell["delta_complexity_cleaned"] == pytest.approx(expected_cl, abs=1e-9)


# ---------------------------------------------------------------------------
# CROSSOVER — the centerpiece assertion (synthetic: governance dominates)
# ---------------------------------------------------------------------------

def test_crossover_metric_computed(synth_setup, tmp_path):
    """The crossover ('simple + clean >= complex + raw') is computed: a margin
    (skill_simplest_nontrivial_on_cleaned - skill_most_complex_on_raw) + a
    boolean holds flag, attributed per dataset."""
    loader, raw_provider, _, _ = synth_setup
    res = e9_governance.run(
        loader, tmp_path, raw_provider=raw_provider, model_ladder=_fast_ladder
    )
    cross = res["metrics"]["crossover"]
    assert "gefcom2014" in cross
    entry = cross["gefcom2014"]
    assert set(entry) >= {"margin", "holds", "simple_model", "complex_model"}
    assert np.isfinite(entry["margin"])
    assert isinstance(entry["holds"], bool)
    # Simplest non-trivial = LightGBM (rank 1); most complex = MLP (top rank).
    assert entry["simple_model"] == "LightGBM"
    assert entry["complex_model"] == "MLP"


def test_crossover_holds_when_governance_dominates(synth_setup, tmp_path):
    """THE headline E9 assertion. On the synthetic frame governance DOMINATES:
    heavy label noise is removed by cleaning, so the simplest non-trivial model
    on CLEANED matches or beats the MOST complex model on RAW.

    This proves the crossover MEASUREMENT works (the thesis is empirically
    testable). The real-data run may or may not cross over at TINY configs —
    that is reported honestly, not forced.
    """
    loader, raw_provider, _, _ = synth_setup
    res = e9_governance.run(
        loader, tmp_path, raw_provider=raw_provider, model_ladder=_fast_ladder
    )
    cells = res["metrics"]["grid"]["gefcom2014"]
    cross = res["metrics"]["crossover"]["gefcom2014"]
    simple_cleaned = cells["LightGBM"]["cleaned"]["skill"]
    complex_raw = cells["MLP"]["raw"]["skill"]
    assert simple_cleaned >= complex_raw, (
        f"crossover failed: LightGBM/cleaned skill {simple_cleaned:.4f} < "
        f"MLP/raw skill {complex_raw:.4f}"
    )
    assert cross["holds"] is True
    assert cross["margin"] == pytest.approx(simple_cleaned - complex_raw, abs=1e-9)


# ---------------------------------------------------------------------------
# Purity + reproducibility + CSV
# ---------------------------------------------------------------------------

def test_run_does_not_mutate_inputs(synth_setup, tmp_path):
    loader, raw_provider, raw_df, cleaned_df = synth_setup
    before_raw = raw_df.copy(deep=True)
    e9_governance.run(
        loader, tmp_path, raw_provider=raw_provider, model_ladder=_fast_ladder
    )
    pd.testing.assert_frame_equal(before_raw, raw_df)


def test_results_csv_schema_and_grid_rows(synth_setup, tmp_path):
    loader, raw_provider, _, _ = synth_setup
    e9_governance.run(
        loader, tmp_path, raw_provider=raw_provider, model_ladder=_fast_ladder
    )
    df = pd.read_csv(tmp_path / "results" / "results_e9_governance.csv")
    required = {
        "dataset", "model", "complexity_rank", "data_state",
        "daytime_mae", "daytime_skill", "n_daytime",
        "delta_hygiene", "delta_complexity_raw", "delta_complexity_cleaned",
    }
    assert required.issubset(df.columns), f"missing cols: {required - set(df.columns)}"
    # 3 models x 2 arms = 6 grid rows.
    assert len(df) == 3 * 2
    assert set(df["data_state"]) == {"raw", "cleaned"}
    assert set(df["model"]) == {"SmartPersistence", "LightGBM", "MLP"}


# ---------------------------------------------------------------------------
# Real-data RUN_SLOW integration (bounded RAW GEFCom snapshot, built in-run)
# ---------------------------------------------------------------------------

@slow
def test_run_real_gefcom_raw_vs_cleaned(tmp_path):
    """Build a bounded RAW GEFCom snapshot in-run, derive the cleaned frame via
    the quality audit, and run E9 end-to-end with the default model ladder.

    Reports the REAL grid honestly; asserts only that the grid is complete +
    finite (the crossover is NOT forced — at TINY dev configs it may be
    ambiguous; the publishable crossover comes at Phase 8 with full configs).
    """
    if not os.path.isdir(_GEFCom_RAW_SOURCE):
        pytest.skip(f"GEFCom raw source not present at {_GEFCom_RAW_SOURCE}")
    import shutil
    import tempfile

    from solarunibench.pipeline.adapters.gefcom import GEFCom2014Pipeline

    tmp = tempfile.mkdtemp()
    try:
        pipe = GEFCom2014Pipeline(
            raw_data_dir=_GEFCom_RAW_SOURCE,
            output_dir=tmp,
            dataset_name="gefcom2014_raw_bounded",
        )
        pipe.run_pipeline(raw=True, limit=_REAL_GEFCom_LIMIT)
        raw_df = pd.read_parquet(
            f"{tmp}/gefcom2014_raw_bounded/time_series.parquet"
        )
        raw_meta = pd.read_parquet(
            f"{tmp}/gefcom2014_raw_bounded/metadata.parquet"
        )
    finally:
        pass  # cleaned up after the test

    cap = (
        float(raw_meta["capacity_kw"].dropna().max())
        if raw_meta["capacity_kw"].notna().any()
        else 1.0
    )
    cleaned_df = _derive_cleaned(raw_df, capacity=cap)
    loader = _FakeLoader()

    def raw_provider(dataset: str):
        return raw_df.copy() if dataset == "gefcom2014" else None

    res = e9_governance.run(loader, tmp_path, raw_provider=raw_provider)
    shutil.rmtree(tmp, ignore_errors=True)

    assert (tmp_path / "fig_governance_vs_complexity.png").exists()
    assert (tmp_path / "fig_cleaning_delta.png").exists()
    cells = res["metrics"]["grid"]["gefcom2014"]
    print("\n=== E9 real-data (GEFCom, bounded RAW vs CLEANED) — 2-axis grid ===")
    print(f"  {'model':<16} {'skill_raw':>10} {'skill_clean':>12} {'Δhygiene':>10}")
    for mn, cell in cells.items():
        sr = cell["raw"]["skill"]
        sc = cell["cleaned"]["skill"]
        dh = cell["delta_hygiene"]
        print(f"  {mn:<16} {sr:>+10.4f} {sc:>+12.4f} {dh:>+10.4f}")
    cross = res["metrics"]["crossover"]["gefcom2014"]
    print(
        f"  crossover ({cross['simple_model']}/cleaned >= {cross['complex_model']}/raw): "
        f"margin={cross['margin']:+.4f}  holds={cross['holds']}"
    )
    # Honest: every cell finite (grid computed); the thesis is reported, not forced.
    for mn, cell in cells.items():
        for arm in ("raw", "cleaned"):
            assert np.isfinite(cell[arm]["skill"]), f"{mn}/{arm}: skill not finite"
