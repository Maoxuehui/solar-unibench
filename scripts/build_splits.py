"""CLI: build frozen splits.parquet for a dataset from configs/splits.yaml."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from solarunibench.splits import build


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build frozen train/val/test splits")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output-root", default=None)
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    cfg_all = yaml.safe_load((root / "configs" / "splits.yaml").read_text())
    version = cfg_all.pop("splits_version", "1.0")
    paths = yaml.safe_load((root / "configs" / "paths.yaml").read_text())
    out_root = Path(args.output_root or paths["output_root"])
    manifest = build(out_root / args.dataset, cfg_all[args.dataset], version)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
