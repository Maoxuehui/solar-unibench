"""CLI: run one forecast baseline on one standardized dataset; print metrics.

Minimal driver used by Phase 7/8 to sanity-check a baseline end-to-end without
spinning up the full benchmark harness. It wires the Phase 6a forecast framework
(:class:`SolarForecastPipeline`) to the Phase 5 loader for a single
(dataset, target, model) triple and prints MAE / RMSE / skill-vs-SmartPersistence.

Examples
--------
    python scripts/run_baseline.py --dataset gefcom2014 --model smartpersistence
    python scripts/run_baseline.py --dataset skippd --model lightgbm \\
        --target pv_power_normalized --lookback 6 --horizon 3
"""
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from solarunibench.forecast.core import (
    EndogenousFeatureEngineer,
    PipelineConfig,
    SolarForecastPipeline,
)
from solarunibench.forecast.metrics import mae, rmse, skill_vs_reference
from solarunibench.forecast.models.lightgbm_model import LightGBMModel
from solarunibench.forecast.models.mlp_model import MLPModel
from solarunibench.forecast.models.smart_persistence import SmartPersistenceModel

# Per-dataset canonical target (spec §10): PV for SKIPPD/GEFCom, GHI for the rest.
_DEFAULT_TARGET = {
    "gefcom2014": "pv_power_normalized",
    "skippd": "pv_power_normalized",
    "folsom": "ghi_actual",
    "sunlake": "ghi_actual",
}
_MODELS = {
    "smartpersistence": SmartPersistenceModel,
    "lightgbm": LightGBMModel,
    "mlp": MLPModel,
}


def main(argv=None):
    repo = Path(__file__).resolve().parents[2]
    import yaml
    default_root = yaml.safe_load((repo / "configs" / "paths.yaml").read_text())["output_root"]

    ap = argparse.ArgumentParser(description="Run one forecast baseline; print metrics.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument(
        "--model",
        required=True,
        choices=sorted(_MODELS),
        help="Baseline to run.",
    )
    ap.add_argument("--target", default=None, help="Override target column.")
    ap.add_argument("--data-root", default=default_root)
    ap.add_argument("--lookback", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument(
        "--feature-cols",
        nargs="+",
        default=["ghi_clearsky", "cos_zenith", "clearness_index"],
    )
    args = ap.parse_args(argv)

    target = args.target or _DEFAULT_TARGET.get(args.dataset)
    if target is None:
        ap.error(f"--target required (no default for dataset '{args.dataset}')")

    cfg = PipelineConfig(
        dataset=args.dataset,
        target=target,
        lookback=args.lookback,
        horizon=args.horizon,
        stride=args.stride,
        feature_cols=args.feature_cols,
        data_root=args.data_root,
    )
    fe = EndogenousFeatureEngineer(target=target, feature_cols=args.feature_cols)
    model = _MODELS[args.model]()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        res = SolarForecastPipeline(cfg, fe, model).run()

    y = res["y_true"]
    print(f"\n=== {model.name} on {args.dataset} (target={target}) ===")
    print(f"  samples={len(y)} horizon={y.shape[1] if y.ndim > 1 else 1}")
    print(f"  MAE  = {mae(y, res['pred']):.6f}")
    print(f"  RMSE = {rmse(y, res['pred']):.6f}")

    if args.model != "smartpersistence":
        # Skill vs the physical reference, computed on the same test windows.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ref = SolarForecastPipeline(
                cfg, fe, SmartPersistenceModel()
            ).run()
        skill = skill_vs_reference(y, res["pred"], ref["pred"])
        print(f"  skill-vs-SmartPersistence = {skill:+.4f}")


if __name__ == "__main__":
    main()
