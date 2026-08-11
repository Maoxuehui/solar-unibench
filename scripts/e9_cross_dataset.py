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
from solarunibench.pipeline.adapters.skippd import SKIPPDPipeline
from solarunibench.pipeline.adapters.sunlake import SunLakePipeline

DATAROOT = "D:/monica/workspace/solar_benchmark_dataset"
# Per-dataset row bound for the bounded RAW snapshot. GEFCom is hourly (3
# entities); Folsom + SKIPPD tabular are 1-minute; SunLake is 30-min (6 entities).
# 1-minute datasets need a large bound to cover enough day/night cycles.
RAW_LIMIT = None  # load FULL raw (all 3 years), then uniform-subsample below
SUBSAMPLE_TARGET = {"gefcom2014": 2000, "folsom": 40000, "skippd": 40000, "sunlake": 4000}
TARGETS = {
    "gefcom2014": "pv_power_normalized",
    "folsom": "ghi_actual",
    "skippd": "pv_power_normalized",
    "sunlake": "ghi_actual",
}
ADAPTERS = {
    "gefcom2014": GEFCom2014Pipeline,
    "folsom": FolsomPipeline,
    "skippd": SKIPPDPipeline,
    "sunlake": SunLakePipeline,
}
RAWSUB = {
    "gefcom2014": "GEFCom2014_data",
    "folsom": "Folsom_data",
    "skippd": "SKIPPD_all",
    "sunlake": "SunLake_data",
}
#: SunLake's tabular raw is local, but process_and_save_tensors pulls satellite
#: from the deeplake hub (network). For the tabular-only E9 grid we skip that.
_SAT_SKIPPED = False
def _skip_sunlake_satellite():
    global _SAT_SKIPPED
    if _SAT_SKIPPED:
        return
    import solarunibench.imaging.sunlake_satellite as _sat
    _sat.build_sunlake_satellite_zarr = lambda *a, **k: None
    _SAT_SKIPPED = True


# -- Cadence-aligned grid (the fairness fix) --------------------------------
# E9's default grid used fixed ROW params (lookback=12, horizon=1), which on 1-min
# data means a 1-MINUTE-ahead forecast where persistence is near-perfect (all
# models lose catastrophically). Align to a 12-HOUR lookback + 1-HOUR horizon in
# PHYSICAL time for every dataset, scaling row-params by each dataset's cadence:
#   GEFCom (hourly):  L=12,  H=1    (12h lookback, 1h ahead)
#   SunLake (30min):  L=24,  H=2    (12h lookback, 1h ahead)
#   Folsom/SKIPPD (1min): L=720, H=60 (12h lookback, 1h ahead)
# so the raw-vs-cleaned comparison is fair (not a cadence artifact).
CADENCE_MIN = {"gefcom2014": 60, "sunlake": 30, "folsom": 1, "skippd": 1}
ROW_PARAMS = {
    "gefcom2014": (12, 1),
    "sunlake": (24, 2),
    "folsom": (720, 60),
    "skippd": (720, 60),
}


def row_params(dataset: str) -> tuple[int, int, int]:
    """(lookback, horizon, stride) in ROWS for the cadence-aligned 1h-ahead grid.
    stride = horizon (non-overlapping windows)."""
    lb, hz = ROW_PARAMS[dataset]
    return lb, hz, max(1, hz)


def build_ladder(lookback: int = 12) -> list:
    """Extended complexity ladder (low → high): SP, LightGBM, MLP, DLinear,
    LSTM, iTransformer, PatchTST.

    For long lookbacks (the 1-min datasets at L=720), PatchTST's patch_len is
    raised (fewer tokens → less attention memory) and the torch batch size is
    cut so the 720-step sequences fit on the GPU without OOM. iTransformer is
    intrinsically light (one token per channel) so only its batch is cut.
    """
    long = lookback >= 120
    pt = {**PATCHTST_PARAMS, "patch_len": 32, "stride": 16, "batch_size": 32} if long else PATCHTST_PARAMS
    itr = {**iTRANSFORMER_PARAMS, "batch_size": 32} if long else iTRANSFORMER_PARAMS
    lstm = {**LSTM_PARAMS, "batch_size": 32} if long else LSTM_PARAMS
    return [
        SmartPersistenceModel(),
        LightGBMModel(params=LGBM_PARAMS),
        MLPModel(**MLP_PARAMS),
        DLinearModel(**DLINEAR_PARAMS),
        LSTMModel(**lstm),
        iTransformerModel(**itr),
        PatchTSTModel(**pt),
    ]


def build_raw(dataset: str) -> pd.DataFrame:
    """Bounded RAW snapshot for one dataset via its adapter (raw=True)."""
    if dataset == "sunlake":
        _skip_sunlake_satellite()  # tabular-only: skip the hub satellite pull
    tmp = tempfile.mkdtemp(prefix=f"{dataset}_raw_", dir="D:/tmp")
    adapter = ADAPTERS[dataset]
    pipe = adapter(
        raw_data_dir=f"{DATAROOT}/{RAWSUB[dataset]}",
        output_dir=tmp, dataset_name=f"{dataset}_raw",
    )
    pipe.run_pipeline(raw=True, limit=RAW_LIMIT)  # None = load full multi-year
    df = pd.read_parquet(f"{tmp}/{dataset}_raw/time_series.parquet")
    # Uniform-stride subsample to target (covers all seasons, not just early Jan).
    target = SUBSAMPLE_TARGET[dataset]
    if len(df) > target:
        stride = len(df) / target
        idx = np.arange(0, len(df), stride).round().astype(np.intp)[:target]
        df = df.iloc[idx].reset_index(drop=True)
    return df


def main() -> int:
    # Optional argv: run a SUBSET of datasets (e.g. to keep each background job
    # short — the environment reaps very long-running tasks; GEFCom's geo-
    # recovery is the slow part). Each dataset appends to the same CSV.
    datasets = sys.argv[1:] if len(sys.argv) > 1 else ["gefcom2014", "folsom", "skippd", "sunlake"]
    rows = []
    summaries = []
    for ds in datasets:
        target = TARGETS[ds]
        # Cadence-aligned grid: 1-hour-ahead forecast for ALL datasets (fairness).
        lb, hz, st = row_params(ds)
        e9._LOOKBACK, e9._HORIZON, e9._STRIDE = lb, hz, st
        print(f"\n=== {ds} (target={target}, cadence={CADENCE_MIN[ds]}min, "
              f"lookback={lb} horizon={hz} stride={st} rows) ===", flush=True)
        raw_df = build_raw(ds)
        cleaned_df = e9._derive_cleaned(raw_df)
        print(f"  raw rows={len(raw_df)}, cleaned rows={len(cleaned_df)}", flush=True)
        cells = e9._evaluate_dataset(raw_df, cleaned_df, target, lambda: build_ladder(lb))
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
    new_df = pd.DataFrame(rows)
    # Append-merge: keep rows from datasets NOT in this run, add/replace this run's.
    if out.exists():
        old = pd.read_csv(out)
        old = old[~old["dataset"].isin(datasets)]
        new_df = pd.concat([old, new_df], ignore_index=True)
    new_df.to_csv(out, index=False)

    print("\n=== Crossover summary across datasets ===", flush=True)
    print(pd.DataFrame(summaries).to_string(index=False,
          float_format=lambda x: f"{x:+.3f}" if isinstance(x, float) else str(x)), flush=True)
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
