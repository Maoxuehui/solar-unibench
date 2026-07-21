"""Tests for E5 — Quality-audit cleaning value (spec §11, E5).

Hypothesis: ``quality_flag`` governance (night-zero, de-clip, fault removal) cuts
label noise and lifts forecast skill.

The fast tests use SYNTHETIC raw + cleaned frames: the RAW frame is a
pre-governance snapshot (anchors merged, ``quality_flag``/``clearness_index``
NA) with INJECTED label noise — night-time non-zero sensor readings,
over-capacity clipping artifacts, and negative values — that the quality audit
removes. The cleaned frame is derived by running :func:`apply_quality_audit` on
the raw frame (exactly what ``run_pipeline(raw=False)`` does to the
anchor-merged frame), so the two arms are row-aligned and differ ONLY in
governance. A ``RUN_SLOW``-gated test builds a bounded RAW snapshot for GEFCom
in-run (the real pre-governance frame) and runs the ablation end-to-end.
"""
import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from solarunibench.experiments import e5_quality
from solarunibench.operators.quality import apply_quality_audit

# LightGBM is quiet by default; keep pytest output pristine regardless.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

slow = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW"),
    reason="set RUN_SLOW=1 to run the real-data E5 integration test",
)

_REAL_DATA_ROOT = "D:/monica/workspace/solar-unibench/data/standardized"
_GEFCom_RAW_SOURCE = "D:/monica/workspace/solar_benchmark_dataset/GEFCom2014_data"


# ---------------------------------------------------------------------------
# Synthetic raw + cleaned frames
# ---------------------------------------------------------------------------

class _FakeLoader:
    """Mimics :class:`SolarUniBenchLoader.load_tabular` for the ablation datasets."""

    def __init__(self, frames: dict[str, pd.DataFrame], data_root: str = "/fake"):
        self._frames = frames
        self.data_root = data_root

    def load_tabular(self, dataset: str, split: str | None = None) -> pd.DataFrame:
        df = self._frames[dataset].copy()
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)
        return df


def _synth_raw_frame(
    name: str,
    target_col: str,
    n_entities: int = 2,
    n_days: int = 35,
    seed: int = 0,
) -> pd.DataFrame:
    """A pre-governance frame (anchors merged) WITH injected label noise.

    Mimics the ``--raw`` snapshot: ``ghi_clearsky`` / ``cos_zenith`` anchors are
    present (computed before the audit), but the target carries noise the
    quality-audit removes:

    * **night sensor noise** — non-zero target readings where ``cos_zenith <= 0``
      (audit night-zeros these).
    * **daytime over-capacity clipping** (PV) — values > 1.0 (audit clips to
      capacity). Substantial fraction so it is visible to the daytime evaluator.
    * **daytime negative spikes** — sub-zero readings (audit clamps to 0).
    * **daytime sensor spikes** — implausibly large values (audit clamps PV to
      capacity / leaves GHI but they perturb the raw lag features).

    The daytime noise is scaled to the target magnitude (PV ∈ [0,1], GHI ∈
    [0, ~900]) so the effect is comparable across datasets and robustly visible
    to the daytime-window evaluator (night rows are masked out of daytime MAE,
    so the DAYTIME label noise is what drives the skill delta). ``quality_flag``
    / ``clearness_index`` are NA (the audit has not run).
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
        cos_zen = bell  # proxy in [0, 1]; > 0.05 during the daytime band
        if target_col == "pv_power_normalized":
            clean = np.clip(bell * 0.9 + rng.normal(0, 0.02, n), 0, 1)
        else:  # ghi_actual
            clean = bell * 900.0 + rng.normal(0, 20, n)

        raw_target = clean.copy().astype(float)
        night = cos_zen <= 0
        day_idx = np.where(cos_zen > 0.05)[0]
        # Night sensor noise (audit night-zeros these → affects target std but
        # not daytime MAE since the evaluator masks night).
        if night.any():
            raw_target[night] = np.abs(rng.normal(0.6 * scale, 0.3 * scale, int(night.sum())))
        # Daytime over-capacity clipping (PV only — audit clips to capacity=1.0).
        if target_col == "pv_power_normalized" and len(day_idx):
            k = max(1, int(0.20 * len(day_idx)))
            sel = rng.choice(day_idx, size=k, replace=False)
            raw_target[sel] = 1.35 + rng.normal(0, 0.05, k)
        # Daytime negative spikes (audit clamps to 0; PV post-clip + irradiance).
        if len(day_idx):
            k = max(1, int(0.15 * len(day_idx)))
            sel = rng.choice(day_idx, size=k, replace=False)
            raw_target[sel] = -np.abs(rng.normal(0.6 * scale, 0.3 * scale, k))

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
    """Run the quality audit on a raw frame → the cleaned (governed) frame.

    Mirrors ``run_pipeline(raw=False)`` exactly: the audit runs on the
    anchor-merged frame (drop the NA ``quality_flag`` / ``clearness_index`` so
    the audit owns them), producing a row-aligned cleaned frame.
    """
    base = raw_df.drop(columns=[c for c in ("quality_flag", "clearness_index")
                                if c in raw_df.columns])
    return apply_quality_audit(base, capacity=capacity)


_TARGETS = {
    "gefcom2014": "pv_power_normalized",
    "skippd": "pv_power_normalized",
    "folsom": "ghi_actual",
    "sunlake": "ghi_actual",
}


@pytest.fixture
def synth_setup():
    """Return (loader, raw_provider) over 4 synthetic datasets.

    The loader serves CLEANED frames (so fig_quality_flags reads real
    quality_flag composition); the raw_provider returns the matching pre-
    governance RAW frames for the ablation.
    """
    raw_frames = {
        ds: _synth_raw_frame(ds, tgt, seed=i)
        for i, (ds, tgt) in enumerate(_TARGETS.items())
    }
    cleaned_frames = {ds: _derive_cleaned(r) for ds, r in raw_frames.items()}
    loader = _FakeLoader(cleaned_frames)

    def raw_provider(dataset: str):
        return raw_frames.get(dataset)

    return loader, raw_provider, raw_frames, cleaned_frames


# ---------------------------------------------------------------------------
# Contract + figure tests
# ---------------------------------------------------------------------------

def test_run_returns_figures_and_metrics_dict(synth_setup, tmp_path):
    loader, raw_provider, _, _ = synth_setup
    res = e5_quality.run(loader, tmp_path, raw_provider=raw_provider)
    assert isinstance(res, dict)
    assert set(res) >= {"figures", "metrics"}
    for fig in ("fig_quality_flags.png", "fig_quality_cleaning_effect.png"):
        p = tmp_path / fig
        assert p.exists(), f"missing {fig}"
        assert p.stat().st_size > 0
    csvs = list((tmp_path / "results").glob("*.csv"))
    assert len(csvs) >= 1
    assert any("fig_quality_flags.png" in p for p in res["figures"])
    assert any("fig_quality_cleaning_effect.png" in p for p in res["figures"])


def test_quality_flag_figure_produced_without_raw_provider(synth_setup, tmp_path):
    """Default path (no raw_provider) still produces fig_quality_flags.

    The cleaning-effect ablation is honestly skipped (no raw snapshot), but the
    quality_flag composition figure reads cleaned packages via the loader and is
    always produced.
    """
    loader, _, _, _ = synth_setup
    res = e5_quality.run(loader, tmp_path)  # no raw_provider
    assert (tmp_path / "fig_quality_flags.png").exists()
    # Ablation skipped honestly.
    assert res["metrics"]["ablation"]["status"] == "skipped"
    # Cleaning-effect figure NOT produced when no raw data.
    assert not (tmp_path / "fig_quality_cleaning_effect.png").exists()


def test_all_four_datasets_in_quality_flag_composition(synth_setup, tmp_path):
    loader, raw_provider, _, _ = synth_setup
    res = e5_quality.run(loader, tmp_path, raw_provider=raw_provider)
    qf = res["metrics"]["quality_flag"]
    for ds in ("gefcom2014", "skippd", "folsom", "sunlake"):
        assert ds in qf, f"missing dataset {ds} in quality_flag composition"
        assert qf[ds]["n_total"] > 0


# ---------------------------------------------------------------------------
# Core hypothesis: cleaned-arm skill >= raw-arm skill (label noise cut)
# ---------------------------------------------------------------------------

def test_cleaned_arm_beats_or_matches_raw_arm_skill(synth_setup, tmp_path):
    """The headline E5 assertion: governance cuts label noise → cleaned skill
    >= raw skill on the SAME daytime test windows (same model, same features,
    same horizon)."""
    loader, raw_provider, _, _ = synth_setup
    res = e5_quality.run(loader, tmp_path, raw_provider=raw_provider)
    ablation = res["metrics"]["ablation"]
    assert ablation["status"] == "ran"
    for ds, arm in ablation["datasets"].items():
        raw_skill = arm["raw"]["skill"]
        cl_skill = arm["cleaned"]["skill"]
        assert np.isfinite(raw_skill), f"{ds}: raw skill not finite"
        assert np.isfinite(cl_skill), f"{ds}: cleaned skill not finite"
        assert cl_skill >= raw_skill, (
            f"{ds}: cleaned skill ({cl_skill:.4f}) < raw skill ({raw_skill:.4f})"
        )


def test_arms_share_identical_test_windows(synth_setup, tmp_path):
    """Fair comparison: raw + cleaned arms evaluate on the SAME daytime test
    windows (same n_daytime) and the SAME Smart Persistence reference."""
    loader, raw_provider, _, _ = synth_setup
    res = e5_quality.run(loader, tmp_path, raw_provider=raw_provider)
    for ds, arm in res["metrics"]["ablation"]["datasets"].items():
        n_raw = arm["raw"]["n_daytime"]
        n_cl = arm["cleaned"]["n_daytime"]
        assert n_raw == n_cl > 0, f"{ds}: daytime window counts differ ({n_raw} vs {n_cl})"
        # Reference runs on the same cleaned test frame.
        assert arm["reference"]["n_daytime"] == n_cl


def test_metrics_finite_and_mae_nonneg(synth_setup, tmp_path):
    loader, raw_provider, _, _ = synth_setup
    res = e5_quality.run(loader, tmp_path, raw_provider=raw_provider)
    for ds, arm in res["metrics"]["ablation"]["datasets"].items():
        for key in ("raw", "cleaned", "reference"):
            mae = arm[key]["daytime_mae"]
            assert np.isfinite(mae), f"{ds}/{key}: daytime_mae not finite"
            assert mae >= 0, f"{ds}/{key}: negative MAE"


def test_target_distribution_shift_reported(synth_setup, tmp_path):
    """The fraction of rows flagged + target stats before/after governance are
    reported (the target-distribution shift)."""
    loader, raw_provider, raw_frames, cleaned_frames = synth_setup
    res = e5_quality.run(loader, tmp_path, raw_provider=raw_provider)
    for ds, shift in res["metrics"]["ablation"]["datasets"].items():
        # fraction_flagged in [0, 1]; raw std >= cleaned std (noise removed).
        assert 0.0 <= shift["fraction_flagged"] <= 1.0
        assert shift["n_flagged"] >= 0
        # fraction_target_changed: the honest label-modification measure.
        assert 0.0 <= shift["fraction_target_changed"] <= 1.0
        # Night-zeroing + clipping must change the target distribution.
        assert np.isfinite(shift["target_std_raw"])
        assert np.isfinite(shift["target_std_cleaned"])
        assert shift["target_std_raw"] >= shift["target_std_cleaned"]


def test_raw_arm_trained_on_raw_cleaned_arm_on_cleaned(synth_setup, tmp_path):
    """Design-matrix audit: raw arm trained on the noisy raw target, cleaned arm
    on the governed target (past_channels identical → fair comparison)."""
    loader, raw_provider, _, _ = synth_setup
    res = e5_quality.run(loader, tmp_path, raw_provider=raw_provider)
    for ds, arm in res["metrics"]["ablation"]["datasets"].items():
        assert arm["raw"]["train_source"] == "raw"
        assert arm["cleaned"]["train_source"] == "cleaned"
        assert arm["raw"]["past_channels"] == arm["cleaned"]["past_channels"]


def test_run_does_not_mutate_inputs(synth_setup, tmp_path):
    loader, raw_provider, raw_frames, cleaned_frames = synth_setup
    before_raw = {k: v.copy(deep=True) for k, v in raw_frames.items()}
    before_cl = {k: v.copy(deep=True) for k, v in cleaned_frames.items()}
    before_loader = {k: v.copy(deep=True) for k, v in loader._frames.items()}
    e5_quality.run(loader, tmp_path, raw_provider=raw_provider)
    for k, v in before_raw.items():
        pd.testing.assert_frame_equal(v, raw_frames[k])
    for k, v in before_cl.items():
        pd.testing.assert_frame_equal(v, cleaned_frames[k])
    for k, v in before_loader.items():
        pd.testing.assert_frame_equal(v, loader._frames[k])


def test_results_csv_round_trips(synth_setup, tmp_path):
    loader, raw_provider, _, _ = synth_setup
    e5_quality.run(loader, tmp_path, raw_provider=raw_provider)
    csv = list((tmp_path / "results").glob("*.csv"))
    assert csv
    df = pd.read_csv(csv[0])
    assert {"dataset", "arm", "daytime_mae", "skill"}.issubset(df.columns)
    # At least raw + cleaned + reference rows per ablation dataset.
    for a in ("raw", "cleaned", "reference"):
        assert (df["arm"] == a).any()


# ---------------------------------------------------------------------------
# Honest no-op: a raw frame whose audit changes nothing
# ---------------------------------------------------------------------------

def test_honest_noop_when_governance_changes_nothing(tmp_path):
    """If the raw snapshot has NO rows the audit modifies (no night, no
    clipping, no negatives), E5 reports the no-op honestly rather than
    fabricating a skill gain."""
    # A frame with target only at daytime, in [0,1], no negatives.
    rows = []
    times = pd.date_range("2020-01-01", periods=24 * 10, freq="1h")
    hod = times.hour.to_numpy()
    bell = np.clip(np.sin(np.pi * (hod - 6) / 13.0), 0.0, None)
    for i, t in enumerate(times):
        rows.append({
            "timestamp_utc": t, "entity_id": "GEFCOM_E0",
            "pv_power_normalized": float(bell[i] * 0.5),
            "ghi_clearsky": float(bell[i] * 950.0),
            "cos_zenith": float(bell[i]),
            "capacity_kw": 1.0,
        })
    raw = pd.DataFrame(rows)
    raw["quality_flag"] = pd.NA
    raw["clearness_index"] = pd.NA
    cleaned = _derive_cleaned(raw)
    # Use a canonical dataset name so the ablation queries it; only gefcom2014
    # is populated (others absent → empty quality_flag composition, honest).
    loader = _FakeLoader({"gefcom2014": cleaned})
    res = e5_quality.run(
        loader, tmp_path, raw_provider=lambda ds: raw if ds == "gefcom2014" else None
    )
    arm = res["metrics"]["ablation"]["datasets"]["gefcom2014"]
    # The audit MODIFIED no target → fraction_target_changed == 0 (honest no-op).
    # (quality_flag still marks the night band, but the labels are unchanged.)
    assert arm["fraction_target_changed"] == 0.0


# ---------------------------------------------------------------------------
# Real-data RUN_SLOW integration (bounded RAW GEFCom snapshot, built in-run)
# ---------------------------------------------------------------------------

@slow
def test_run_real_gefcom_raw_vs_cleaned(tmp_path):
    """Build a bounded RAW GEFCom snapshot in-run, derive the cleaned frame via
    the quality audit, and run the E5 ablation end-to-end.

    The RAW build runs the GEFCom adapter (geo-recovery + anchor merge) WITHOUT
    the quality audit — the genuine pre-governance frame. GEFCom is chosen
    because it is tabular-only (no image tensors) and the smallest package.
    The build is RUN_SLOW-gated (geo-recovery dominates the cost, ~minutes).
    """
    if not os.path.isdir(_GEFCom_RAW_SOURCE):
        pytest.skip(f"GEFCom raw source not present at {_GEFcom_RAW_SOURCE}")
    from solarunibench.loader import SolarUniBenchLoader
    from solarunibench.pipeline.adapters.gefcom import GEFCom2014Pipeline

    import shutil
    import tempfile

    tmp = tempfile.mkdtemp()
    try:
        pipe = GEFCom2014Pipeline(
            raw_data_dir=_GEFCom_RAW_SOURCE, output_dir=tmp,
            dataset_name="gefcom2014_raw_bounded",
        )
        pipe.run_pipeline(raw=True, limit=_REAL_GEFcom_LIMIT)
        raw_df = pd.read_parquet(
            f"{tmp}/gefcom2014_raw_bounded/time_series.parquet"
        )
        raw_meta = pd.read_parquet(
            f"{tmp}/gefcom2014_raw_bounded/metadata.parquet"
        )
    finally:
        pass  # keep tmp until the test ends; cleaned below removes it

    # Derive the cleaned frame via the audit (row-aligned with raw).
    cap = float(raw_meta["capacity_kw"].dropna().max()) if raw_meta["capacity_kw"].notna().any() else 1.0
    cleaned_df = _derive_cleaned(raw_df, capacity=cap)

    # Real loader for fig_quality_flags (all 4 standardized packages) + the
    # derived cleaned GEFCom frame fed via raw_provider's companion.
    loader = SolarUniBenchLoader(_REAL_DATA_ROOT)

    def raw_provider(dataset: str):
        return raw_df if dataset == "gefcom2014" else None

    # Override the loader's gefcom2014 with the derived bounded cleaned frame so
    # the ablation pairs the bounded raw with its exact cleaned counterpart.
    orig_load = loader.load_tabular
    loader.load_tabular = lambda dataset, split=None: (
        cleaned_df.copy() if dataset == "gefcom2014"
        else orig_load(dataset, split)
    )

    res = e5_quality.run(loader, tmp_path, raw_provider=raw_provider)
    shutil.rmtree(tmp, ignore_errors=True)

    assert (tmp_path / "fig_quality_flags.png").exists()
    assert (tmp_path / "fig_quality_cleaning_effect.png").exists()
    arm = res["metrics"]["ablation"]["datasets"]["gefcom2014"]
    raw_skill = arm["raw"]["skill"]
    cl_skill = arm["cleaned"]["skill"]
    print("\n=== E5 real-data (GEFCom, bounded RAW vs CLEANED) ===")
    print(f"  raw   : daytime_mae={arm['raw']['daytime_mae']:.4f}  skill={raw_skill:+.4f}")
    print(f"  clean : daytime_mae={arm['cleaned']['daytime_mae']:.4f}  skill={cl_skill:+.4f}")
    print(f"  gain  : {cl_skill - raw_skill:+.4f}  (fraction_flagged={arm['fraction_flagged']:.3f})")
    assert np.isfinite(raw_skill) and np.isfinite(cl_skill)


_REAL_GEFcom_LIMIT = 1500
