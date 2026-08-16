"""Tests for ``solarunibench.benchmark.summary`` — Task 5.

Pivots the E8 runner's flat ``list[dict]`` rows into a journal-style
``(Dataset, Metric, Horizon) × 9-model`` MultiIndex DataFrame and exports it to
a LaTeX ``\\multirow`` / ``\\cmidrule`` table (plus a multi-index CSV).

The input row schema mirrors :func:`solarunibench.benchmark.run_e8.run_benchmark`:
``{dataset, horizon, model, MAE, RMSE, Skill_MAE_pct, Skill_RMSE_pct, n_daytime}``.
CrossViViT is absent for non-image datasets (gefcom2014/sunlake) so those cells
must surface as ``NaN`` in the DataFrame and render as "—" in the LaTeX table.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from solarunibench.benchmark.summary import (
    MODELS,
    build_results_table,
    save_results,
    to_latex_multirow,
)

# The fixed 9-model battery, in stable column order. SmartPersistence is the
# reference (always first); CrossViViT is the multimodal model (always last).
_EXPECTED_MODELS: list[str] = [
    "SmartPersistence",
    "LightGBM",
    "MLP",
    "DLinear",
    "LSTM",
    "iTransformer",
    "PatchTST-CI",
    "PatchTST-CD",
    "CrossViViT",
]


# ---------------------------------------------------------------------------
# Synthetic rows: 2 datasets x 2 horizons x a subset of models.
# `folsom` is an image dataset (CrossViViT present); `sunlake` is non-image
# (CrossViViT absent — the missing-cell case the spec calls out).
# ---------------------------------------------------------------------------


def _row(dataset: str, horizon: int, model: str, skill: float, mae: float) -> dict:
    return {
        "dataset": dataset,
        "horizon": horizon,
        "model": model,
        "MAE": mae,
        "RMSE": mae * 1.5,
        "Skill_MAE_pct": skill,
        "Skill_RMSE_pct": skill - 1.0,
        "n_daytime": 100,
    }


@pytest.fixture
def synthetic_rows() -> list[dict]:
    rows: list[dict] = []
    # folsom (image): 2 horizons × 4 models incl. CrossViViT.
    for hz, base in [(5, 10.0), (10, 20.0)]:
        for i, m in enumerate(
            ["SmartPersistence", "LightGBM", "MLP", "CrossViViT"]
        ):
            rows.append(_row("folsom", hz, m, base - i, 0.1 * (i + 1)))
    # sunlake (non-image): 2 horizons × 3 models, NO CrossViViT.
    for hz, base in [(1, 30.0), (2, 40.0)]:
        for i, m in enumerate(
            ["SmartPersistence", "LightGBM", "PatchTST-CI"]
        ):
            rows.append(_row("sunlake", hz, m, base - i, 0.2 * (i + 1)))
    return rows


# ---------------------------------------------------------------------------
# MODELS constant
# ---------------------------------------------------------------------------


def test_models_constant_is_the_nine_model_battery_in_stable_order() -> None:
    assert MODELS == _EXPECTED_MODELS
    assert len(MODELS) == 9


# ---------------------------------------------------------------------------
# build_results_table — shape, index, columns
# ---------------------------------------------------------------------------


def test_results_table_has_three_level_row_index(synthetic_rows: list[dict]) -> None:
    df = build_results_table(synthetic_rows)

    assert df.index.names == ["Dataset", "Metric", "Horizon"]
    # Metric level is the default Skill_MAE_pct.
    assert (df.index.get_level_values("Metric") == "Skill_MAE_pct").all()


def test_results_table_columns_are_the_nine_models_in_stable_order(
    synthetic_rows: list[dict],
) -> None:
    df = build_results_table(synthetic_rows)
    assert list(df.columns) == _EXPECTED_MODELS


def test_results_table_has_one_row_per_dataset_horizon_cell(
    synthetic_rows: list[dict],
) -> None:
    df = build_results_table(synthetic_rows)
    # 2 datasets × 2 horizons = 4 rows; metric is a row level, not extra rows.
    assert len(df) == 4


def test_results_table_horizons_sorted_ascending_within_dataset(
    synthetic_rows: list[dict],
) -> None:
    df = build_results_table(synthetic_rows)
    hz_folsom = (
        df.xs("folsom", level="Dataset").index.get_level_values("Horizon").tolist()
    )
    assert hz_folsom == [5, 10]


def test_results_table_values_match_input_skill(synthetic_rows: list[dict]) -> None:
    df = build_results_table(synthetic_rows)
    # folsom hz=5, LightGBM (index 1) → skill = 10.0 - 1 = 9.0.
    val = df.loc[("folsom", "Skill_MAE_pct", 5), "LightGBM"]
    assert np.isclose(val, 9.0)


# ---------------------------------------------------------------------------
# Missing cells: CrossViViT absent on non-image datasets
# ---------------------------------------------------------------------------


def test_results_table_missing_model_is_nan(synthetic_rows: list[dict]) -> None:
    df = build_results_table(synthetic_rows)
    # sunlake never ran CrossViViT → NaN across all its horizons.
    sunlake = df.xs("sunlake", level="Dataset")
    assert sunlake["CrossViViT"].isna().all()
    # folsom DID run CrossViViT → finite at every horizon.
    folsom = df.xs("folsom", level="Dataset")
    assert folsom["CrossViViT"].notna().all()
    # Models nobody ran in the synthetic fixture (DLinear, LSTM, iTransformer,
    # PatchTST-CD) are NaN everywhere.
    for unused in ["DLinear", "LSTM", "iTransformer", "PatchTST-CD"]:
        assert df[unused].isna().all(), unused


def test_results_table_metric_switch_to_mae(synthetic_rows: list[dict]) -> None:
    df = build_results_table(synthetic_rows, metric="MAE")
    assert (df.index.get_level_values("Metric") == "MAE").all()
    # MAE for folsom hz=5 MLP (index 2) → 0.1 * 3 = 0.3.
    val = df.loc[("folsom", "MAE", 5), "MLP"]
    assert np.isclose(val, 0.3)


def test_results_table_metric_switch_to_rmse(synthetic_rows: list[dict]) -> None:
    df = build_results_table(synthetic_rows, metric="Skill_RMSE_pct")
    assert (df.index.get_level_values("Metric") == "Skill_RMSE_pct").all()


def test_results_table_empty_rows_yields_empty_frame_with_correct_columns() -> None:
    df = build_results_table([])
    assert list(df.columns) == _EXPECTED_MODELS
    assert df.index.names == ["Dataset", "Metric", "Horizon"]
    assert len(df) == 0


def test_results_table_unknown_metric_raises(synthetic_rows: list[dict]) -> None:
    with pytest.raises(ValueError):
        build_results_table(synthetic_rows, metric="Nope")


# ---------------------------------------------------------------------------
# to_latex_multirow
# ---------------------------------------------------------------------------


def test_latex_contains_multirow_and_cmidrule_and_dataset_names(
    synthetic_rows: list[dict],
) -> None:
    df = build_results_table(synthetic_rows)
    tex = to_latex_multirow(df)

    assert r"\multirow" in tex
    # \cmidrule must be present (translated from pandas' \cline).
    assert r"\cmidrule" in tex
    # Both dataset names appear in the rendered table.
    assert "folsom" in tex
    assert "sunlake" in tex
    # Tabular environment + booktabs rules present.
    assert r"\begin{tabular}" in tex
    assert r"\toprule" in tex and r"\bottomrule" in tex


def test_latex_missing_cells_rendered_as_dash(synthetic_rows: list[dict]) -> None:
    df = build_results_table(synthetic_rows)
    tex = to_latex_multirow(df)
    # NaN cells must render as "—" (em dash), not "nan" / "NaN".
    assert "—" in tex
    assert "nan" not in tex.lower()


def test_latex_escape_disabled_so_model_names_with_dash_survive(
    synthetic_rows: list[dict],
) -> None:
    df = build_results_table(synthetic_rows)
    tex = to_latex_multirow(df)
    # PatchTST-CI / PatchTST-CD column headers appear verbatim (unescaped).
    assert "PatchTST-CI" in tex
    assert "PatchTST-CD" in tex


def test_latex_no_cline_left_behind(synthetic_rows: list[dict]) -> None:
    """All \\cline must be upgraded to \\cmidrule (no raw \\cline remains)."""
    df = build_results_table(synthetic_rows)
    tex = to_latex_multirow(df)
    assert r"\cline" not in tex


# ---------------------------------------------------------------------------
# save_results
# ---------------------------------------------------------------------------


def test_save_results_writes_csv_and_tex(
    synthetic_rows: list[dict], tmp_path: Path
) -> None:
    paths = save_results(synthetic_rows, str(tmp_path))

    csv_path = Path(paths["csv"])
    tex_path = Path(paths["tex"])
    assert csv_path.exists() and tex_path.exists()
    assert csv_path.suffix == ".csv"
    assert tex_path.suffix == ".tex"

    # CSV round-trips back to the same MultiIndex shape + columns.
    df = pd.read_csv(csv_path, index_col=[0, 1, 2])
    assert df.index.names == ["Dataset", "Metric", "Horizon"]
    assert list(df.columns) == _EXPECTED_MODELS

    # .tex contains the required multirow/cmidrule tokens.
    tex = tex_path.read_text(encoding="utf-8")
    assert r"\multirow" in tex and r"\cmidrule" in tex


def test_save_results_creates_out_dir_if_missing(
    synthetic_rows: list[dict], tmp_path: Path
) -> None:
    out_dir = tmp_path / "new_dir" / "results"
    paths = save_results(synthetic_rows, str(out_dir))
    assert Path(paths["csv"]).exists()
    assert Path(paths["tex"]).exists()
