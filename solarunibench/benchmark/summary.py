"""LaTeX ``\\multirow`` results summary for the E8 unified benchmark (Task 5).

Pivots the flat ``list[dict]`` rows emitted by
:func:`solarunibench.benchmark.run_e8.run_benchmark` into a journal-style
``(Dataset, Metric, Horizon) × 9-model`` MultiIndex DataFrame and exports it to:

* a **multi-index CSV** (``e8_summary.csv``), and
* a **LaTeX table** (``e8_summary.tex``) using ``\\multirow`` + ``\\cmidrule``
  (the latter requires the ``booktabs`` and ``multirow`` LaTeX packages).

Row schema (one per ``(dataset, horizon, model)`` cell):
``{dataset, horizon, model, MAE, RMSE, Skill_MAE_pct, Skill_RMSE_pct, n_daytime}``.

CrossViViT is skipped on non-image datasets (gefcom2014 / sunlake) by the
runner, so those cells surface as ``NaN`` in the DataFrame and render as "—" in
the LaTeX table.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

__all__ = [
    "MODELS",
    "build_results_table",
    "to_latex_multirow",
    "save_results",
]

# The fixed 9-model battery, in stable column order. SmartPersistence is the
# skill reference (always first); CrossViViT is the multimodal model (always
# last) and is absent for non-image datasets.
MODELS: list[str] = [
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

# Metrics the pivot can spread into the table. The `metric` argument selects
# which key from each runner row becomes the cell value (and the Metric row
# level).
_VALID_METRICS: frozenset[str] = frozenset(
    {"MAE", "RMSE", "Skill_MAE_pct", "Skill_RMSE_pct"}
)

# Placeholder rendered for missing cells (model not run on that dataset).
_MISSING: str = "—"

# File basenames written by :func:`save_results`.
_CSV_NAME: str = "e8_summary.csv"
_TEX_NAME: str = "e8_summary.tex"


def build_results_table(
    rows: list[dict], metric: str = "Skill_MAE_pct"
) -> pd.DataFrame:
    """Pivot flat benchmark rows into a ``(Dataset, Metric, Horizon) × models`` frame.

    Parameters
    ----------
    rows
        Flat rows from :func:`run_benchmark` (one per cell).
    metric
        Row key to spread into the cell values; also becomes the constant
        ``Metric`` row level. One of ``MAE``, ``RMSE``, ``Skill_MAE_pct``,
        ``Skill_RMSE_pct`` (default ``Skill_MAE_pct`` — the headline metric).

    Returns
    -------
    pandas.DataFrame
        MultiIndex rows ``("Dataset", "Metric", "Horizon")``; columns are the
        9 models in :data:`MODELS` order (stable; missing models → ``NaN``).
        Datasets follow first-appearance order; horizons sort ascending within
        each dataset.
    """
    if metric not in _VALID_METRICS:
        raise ValueError(
            f"metric must be one of {sorted(_VALID_METRICS)}, got {metric!r}"
        )

    # Empty input → well-typed empty frame (right index + columns) so downstream
    # LaTeX/CSV paths don't crash on a no-op benchmark.
    empty_idx = pd.MultiIndex.from_tuples([], names=["Dataset", "Metric", "Horizon"])
    if not rows:
        return pd.DataFrame(columns=MODELS, index=empty_idx, dtype=float)

    flat = pd.DataFrame(rows)

    # Pivot: rows = (dataset, horizon), cols = model, values = metric.
    # aggfunc="first" is a guard against any accidental duplicate cell; the
    # runner emits exactly one row per cell so it is a no-op in practice.
    table = (
        flat.pivot_table(
            index=["dataset", "horizon"],
            columns="model",
            values=metric,
            aggfunc="first",
        )
        # Stable column order; models nobody ran (or that were skipped on every
        # dataset, e.g. CrossViViT on an all-non-image fixture) become NaN cols.
        .reindex(columns=MODELS)
    )

    # Stable row ordering: datasets by first appearance, horizons ascending.
    datasets_in_order = list(dict.fromkeys(flat["dataset"].tolist()))
    ordered = [
        (ds, hz)
        for ds in datasets_in_order
        for hz in sorted(flat.loc[flat["dataset"] == ds, "horizon"].unique())
    ]
    table = table.reindex(
        pd.MultiIndex.from_tuples(ordered, names=["dataset", "horizon"])
    )

    # Promote to the spec'd (Dataset, Metric, Horizon) MultiIndex. Metric is a
    # constant row level so the rendered table self-documents which metric the
    # numbers are (the caller may switch metrics between runs).
    table.index = pd.MultiIndex.from_arrays(
        [
            table.index.get_level_values("dataset"),
            pd.Index([metric] * len(table), name="Metric"),
            table.index.get_level_values("horizon"),
        ],
        names=["Dataset", "Metric", "Horizon"],
    )
    table.columns.name = None
    return table


def to_latex_multirow(df: pd.DataFrame) -> str:
    """Render a MultiIndex DataFrame to a ``\\multirow`` / ``\\cmidrule`` LaTeX table.

    Uses ``df.to_latex(multirow=True, escape=False, na_rep="—")`` and upgrades
    the resulting ``\\cline{a-b}`` rules to ``\\cmidrule(lr){a-b}`` (pandas
    emits ``\\cline`` rather than ``\\cmidrule``). ``escape=False`` preserves
    model names verbatim (e.g. ``PatchTST-CI``). Missing cells render as "—".

    Requires the LaTeX packages ``booktabs`` and ``multirow``.
    """
    # Column format: one left-aligned per index level, one right-aligned per
    # data column (numeric skill/MAE/RMSE values read better right-aligned).
    col_format = "l" * df.index.nlevels + "r" * len(df.columns)

    latex = df.to_latex(
        multirow=True,
        escape=False,
        na_rep=_MISSING,
        column_format=col_format,
    )
    # Upgrade \cline{a-b} → \cmidrule(lr){a-b} for journal-standard rules.
    latex = re.sub(r"\\cline\{([^}]*)\}", r"\\cmidrule(lr){\1}", latex)
    return latex


def save_results(
    rows: list[dict],
    out_dir: str | Path,
    metric: str = "Skill_MAE_pct",
) -> dict[str, str]:
    """Write the multi-index CSV + ``.tex`` summary for ``rows`` to ``out_dir``.

    Creates ``out_dir`` if it does not exist. Returns a mapping with the written
    ``"csv"`` and ``"tex"`` absolute paths.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = build_results_table(rows, metric=metric)
    csv_path = out / _CSV_NAME
    tex_path = out / _TEX_NAME
    df.to_csv(csv_path)
    tex_path.write_text(to_latex_multirow(df), encoding="utf-8")
    return {"csv": str(csv_path.resolve()), "tex": str(tex_path.resolve())}
