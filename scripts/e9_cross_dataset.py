"""Phase 3 — E9 cross-dataset replication (GEFCom + Folsom).

Builds a bounded RAW snapshot per dataset (``adapter.run_pipeline(raw=True)``),
derives the CLEANED frame (``apply_quality_audit``), runs E9's 2-axis governance
grid (raw / cleaned × complexity ladder), and reports the crossover per dataset.
Extended ladder: SmartPersistence, LightGBM, MLP, DLinear, LSTM, iTransformer,
PatchTST.

Crossover (the thesis): does the SIMPLEST non-trivial model on CLEANED beat the
MOST complex model on RAW? If it holds across datasets, governance > complexity
generalizes beyond GEFCom.

Output: ``figures_full/results/results_e9_cross_dataset.csv`` + printed table.
"""
import os

os.environ.setdefault("SUB_FULL_CONFIG", "1")  # full-capacity presets (100ep etc.)
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from solarunibench.experiments import e9_governance as e9
from solarunibench.experiments._config import (
    DLINEAR_PARAMS, LGBM_PARAMS, LSTM_PARAMS, MLP_PARAMS,
    PATCHTST_PARAMS, iTRANSFORMER_PARAMS,
)
from solarunibench.forecast.models import (
    DLinearModel, iTransformerModel, LightGBMModel, LSTMModel, MLPModel,
    PatchTSTModel, SmartPersistenceModel,
)
from solarunibench.pipeline.adapters.folsom import FolsomPipeline
from solarunibench.pipeline.adapters.gefcom import GEFCom2014Pipeline

DATAROOT = "D:/monica/workspace/solar_benchmark_dataset"
# GEFCom is hourly; Folsom tabular is 1-minute cadence, so it needs a much
# larger row bound to cover enough day/night cycles for a valid temporal split.
RAW_LIMIT = {"gefcom2014": 2000, "folsom": 20000}
TARGETS = {"gefcom2014": "pv_power_normalized", "folsom": "ghi_actual"}
ADAPTERS = {"gefcom2014": GEFCom2014Pipeline, "folsom": FolsomPipeline}
RAWSUB = {"gefcom2014": "GEFCom2014_data", "folsom": "Folsom_data"}


def build_ladder() -> list:
    """Extended complexity ladder (low → high): SP, LightGBM, MLP, DLinear,
    LSTM, iTransformer, PatchTST."""
    return [
        SmartPersistenceModel(),
        LightGBMModel(params=LGBM_PARAMS),
        MLPModel(**MLP_PARAMS),
        DLinearModel(**DLINEAR_PARAMS),
        LSTMModel(**LSTM_PARAMS),
        iTransformerModel(**iTRANSFORMER_PARAMS),
        PatchTSTModel(**PATCHTST_PARAMS),
    ]


def build_raw(dataset: str) -> pd.DataFrame:
    """Bounded RAW snapshot for one dataset via its adapter (raw=True)."""
    tmp = tempfile.mkdtemp(prefix=f"{dataset}_raw_", dir="D:/tmp")
    adapter = ADAPTERS[dataset]
    pipe = adapter(
        raw_data_dir=f"{DATAROOT}/{RAWSUB[dataset]}",
        output_dir=tmp, dataset_name=f"{dataset}_raw",
    )
    pipe.run_pipeline(raw=True, limit=RAW_LIMIT[dataset])
    return pd.read_parquet(f"{tmp}/{dataset}_raw/time_series.parquet")


def main() -> int:
    rows = []
    summaries = []
    for ds in ("gefcom2014", "folsom"):
        target = TARGETS[ds]
        print(f"\n=== {ds} (target={target}) ===", flush=True)
        raw_df = build_raw(ds)
        cleaned_df = e9._derive_cleaned(raw_df)
        print(f"  raw rows={len(raw_df)}, cleaned rows={len(cleaned_df)}", flush=True)
        cells = e9._evaluate_dataset(raw_df, cleaned_df, target, build_ladder)
        summary = e9._crossover_summary(cells)
        summaries.append({"dataset": ds, **summary})
        print(f"  CROSSOVER: simplest({summary['simple_model']})+cleaned vs "
              f"most-complex({summary['complex_model']})+raw -> margin={summary['margin']:+.3f} "
              f"holds={summary['holds']}", flush=True)
        for name, cell in cells.items():
            rows.append({
                "dataset": ds, "model": name, "target": target,
                "complexity_rank": cell["complexity_rank"],
                "skill_raw": cell["raw"]["skill"],
                "skill_cleaned": cell["cleaned"]["skill"],
                "delta_hygiene": cell["delta_hygiene"],
                "daytime_mae_raw": cell["raw"]["daytime_mae"],
                "daytime_mae_cleaned": cell["cleaned"]["daytime_mae"],
            })
    out = Path("figures_full/results/results_e9_cross_dataset.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    print("\n=== Crossover summary across datasets ===", flush=True)
    print(pd.DataFrame(summaries).to_string(index=False,
          float_format=lambda x: f"{x:+.3f}" if isinstance(x, float) else str(x)), flush=True)
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
