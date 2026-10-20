"""Demo: an agent curates the shipped Folsom sample end-to-end.

Builds the tool context from data/sample (times/power for geolocation,
the standardized frame for physical QC, a short sky-image stack for the
visual features), then lets SolarCurationAgent plan and dispatch the
tools from a natural-language instruction, printing a ReAct-style trace
and exporting the governed frame to the canonical Parquet contract.

Framework note: this uses the built-in deterministic planner. To drive the
identical tools with an LLM agent, wrap them with your platform's tool
adapter (example for LangChain in solarunibench/agent/__init__.py).

Run (conda env solar_uni):
    python scripts/run_agent_curation.py
    python scripts/run_agent_curation.py --instruction "recover the site coordinates only"
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import zarr
from PIL import Image

from solarunibench.agent import SolarCurationAgent, export_canonical


def build_context(repo: Path, n_frames: int = 6) -> dict:
    """Assemble the tool context from the 1-day Folsom sample."""
    pkg = repo / "data" / "sample" / "standardized" / "folsom"
    ts = pd.read_parquet(pkg / "time_series.parquet")
    ts = ts.set_index("timestamp_utc").sort_index()
    target = ts["ghi_actual"].fillna(ts.get("pv_power_normalized"))
    day = target.notna() & (target > 0)

    g = zarr.open_group(str(pkg / "sky_images.zarr"), mode="r")
    frames = []
    for i in range(min(n_frames, g["images"].shape[0])):
        img = np.asarray(Image.open(io.BytesIO(bytes(g["images"][i]))))
        frames.append(img.astype(np.float32) / 255.0)

    return {
        "times": ts.index,
        "power": target.where(day, 0.0).to_numpy(),
        "df_ts": ts.reset_index(),
        "image_tensor": np.stack(frames),
    }


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--instruction",
        default=("Please curate the raw Folsom dataset: recover the site "
                 "coordinates, apply the physical quality bounds, and "
                 "extract the visual physics features."),
        help="natural-language curation instruction")
    ap.add_argument("--export",
                    default="results/sample/agent_curated.parquet",
                    help="canonical Parquet export path for the governed "
                         "frame ('' disables)")
    args = ap.parse_args()

    context = build_context(repo)
    agent = SolarCurationAgent()
    report = agent.run(args.instruction, context)

    if args.export and "physical_qc" in report["observations"]:
        df = report["observations"]["physical_qc"]
        if hasattr(df, "to_parquet"):
            out = export_canonical(df, repo / args.export)
            print(f"[export] governed frame -> {out}")

    print("[summary] steps executed:", report["plan"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
