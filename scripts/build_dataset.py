"""CLI: build a standardized Solar-UniBench package for one dataset."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from solarunibench.pipeline.adapters.gefcom import GEFCom2014Pipeline
from solarunibench.pipeline.adapters.folsom import FolsomPipeline
from solarunibench.pipeline.adapters.skippd import SKIPPDPipeline
from solarunibench.pipeline.adapters.sunlake import SunLakePipeline

ADAPTERS = {
    "gefcom2014": GEFCom2014Pipeline,
    "folsom": FolsomPipeline,
    "skippd": SKIPPDPipeline,
    "sunlake": SunLakePipeline,
}
RAW_SUBDIRS = {
    "gefcom2014": "GEFCom2014_data",
    "folsom": "Folsom_data",
    "skippd": "SKIPPD_all",
    "sunlake": "SunLake_data",
}


def _run_sample() -> None:
    """1-minute demo: rebuild the shipped folsom 1-day sample from its raw side.

    Stages data/sample/raw/folsom_1day into the file shape the Folsom adapter
    expects (Folsom_irradiance.csv + Folsom_sky_images_2016.tar.bz2), runs the
    REAL curation pipeline on it, installs the result under
    data/sample/standardized_rebuild/folsom/ (gitignored) and prints a
    comparison against the shipped data/sample/standardized/folsom reference.
    """
    import shutil
    import tarfile
    import tempfile

    repo = Path(__file__).resolve().parents[1]
    raw = repo / "data" / "sample" / "raw" / "folsom_1day"
    ref = repo / "data" / "sample" / "standardized" / "folsom"
    rebuild = repo / "data" / "sample" / "standardized_rebuild" / "folsom"

    with tempfile.TemporaryDirectory(prefix="folsom_sample_") as tmp:
        stage = Path(tmp)
        shutil.copy(raw / "folsom_irradiance_20160201.csv",
                    stage / "Folsom_irradiance.csv")
        jpgs = sorted((raw / "sky_images_20160201").glob("*.jpg"))
        with tarfile.open(stage / "Folsom_sky_images_2016.tar.bz2",
                          "w:bz2") as tar:
            for p in jpgs:
                tar.add(p, arcname=f"2016/02/01/{p.name}")
        print(f"[sample] staged {len(jpgs)} frames + irradiance slice", flush=True)
        pipe = FolsomPipeline(raw_data_dir=str(stage), output_dir=Path(tmp) / "out",
                              dataset_name="folsom")
        pipe.run_pipeline()
        if rebuild.exists():
            shutil.rmtree(rebuild)
        rebuild.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(Path(tmp) / "out" / "folsom"), str(rebuild))

    import pandas as pd

    for name in ("time_series.parquet", "metadata.parquet"):
        try:
            a, b = pd.read_parquet(ref / name), pd.read_parquet(rebuild / name)
            print(f"[sample] {name}: reference {len(a)} rows / rebuild "
                  f"{len(b)} rows", flush=True)
        except Exception as e:                                   # noqa: BLE001
            print(f"[sample] {name}: compare skipped ({e})", flush=True)
    n_ref = len(list((ref / "sky_images.zarr" / "images").glob("[0-9]*"))) \
        if (ref / "sky_images.zarr").is_dir() else "n/a"
    print(f"[sample] rebuild installed at {rebuild} (reference zarr frames: "
          f"{n_ref})", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build standardized Solar-UniBench package")
    ap.add_argument("--dataset", default=None, choices=sorted(ADAPTERS))
    ap.add_argument("--raw", action="store_true", help="emit pre-governance snapshot (E9 control)")
    ap.add_argument("--limit", type=int, default=None, help="rows per entity (debug)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--sample", action="store_true",
                    help="1-minute demo: rebuild data/sample from its raw side "
                         "through the real curation pipeline (ignores --dataset "
                         "and path overrides; result lands in "
                         "data/sample/standardized_rebuild/)")
    args = ap.parse_args(argv)

    if args.sample:
        _run_sample()
        return
    if not args.dataset:
        ap.error("--dataset is required (or use --sample)")

    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "paths.yaml").read_text())
    data_root = args.data_root or cfg["data_root"]
    output_root = args.output_root or cfg["output_root"]

    adapter_cls = ADAPTERS[args.dataset]
    # dataset-specific raw subdir
    raw_dir = str(Path(data_root) / RAW_SUBDIRS[args.dataset])
    pipe = adapter_cls(raw_data_dir=raw_dir, output_dir=output_root, dataset_name=args.dataset)
    manifest = pipe.run_pipeline(raw=args.raw, limit=args.limit)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
