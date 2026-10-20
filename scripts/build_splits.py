"""CLI: build frozen splits.parquet for a dataset from config/splits.yaml."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from solarunibench.splits import build


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    cfg_all = yaml.safe_load((root / "config" / "splits.yaml").read_text())
    # Per-dataset keys exclude the top-level version stamp and the benchmark-level
    # `spatial` cross-domain folds entry (not a per-dataset config).
    dataset_choices = sorted(k for k in cfg_all if k not in ("splits_version", "spatial"))

    ap = argparse.ArgumentParser(description="Build frozen train/val/test splits")
    ap.add_argument("--dataset", required=True, choices=dataset_choices)
    ap.add_argument("--output-root", default=None)
    args = ap.parse_args(argv)

    version = cfg_all.pop("splits_version", "1.0")
    paths = yaml.safe_load((root / "config" / "paths.yaml").read_text())
    out_root = Path(args.output_root or paths["output_root"])
    manifest = build(out_root / args.dataset, cfg_all[args.dataset], version)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
