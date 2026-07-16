"""CLI: build a standardized Solar-UniBench package for one dataset."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from solarunibench.pipeline.adapters.gefcom import GEFCom2014Pipeline

ADAPTERS = {"gefcom2014": GEFCom2014Pipeline}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build standardized Solar-UniBench package")
    ap.add_argument("--dataset", required=True, choices=sorted(ADAPTERS))
    ap.add_argument("--raw", action="store_true", help="emit pre-governance snapshot (E9 control)")
    ap.add_argument("--limit", type=int, default=None, help="rows per entity (debug)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--output-root", default=None)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "paths.yaml").read_text())
    data_root = args.data_root or cfg["data_root"]
    output_root = args.output_root or cfg["output_root"]

    adapter_cls = ADAPTERS[args.dataset]
    # dataset-specific raw subdir
    raw_dir = str(Path(data_root) / {"gefcom2014": "GEFCom2014_data"}.get(args.dataset, ""))
    pipe = adapter_cls(raw_data_dir=raw_dir, output_dir=output_root, dataset_name=args.dataset)
    manifest = pipe.run_pipeline(raw=args.raw, limit=args.limit)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
