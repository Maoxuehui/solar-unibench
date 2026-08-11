"""Resumable E8 runner: runs one dataset at a time, appends to CSV, skips
datasets already present. Restart after an env kill to continue.

Usage: python scripts/e8_resume.py [dataset1 dataset2 ...]
Default: all 4 datasets in speed order (hourly first, 1-min last).
"""
import os, sys
os.environ.setdefault("SUB_FULL_CONFIG", "1")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from solarunibench.loader import SolarUniBenchLoader
from solarunibench.experiments import e8_benchmark as e8
from solarunibench.viz.style import set_style

OUT = Path("figures_full")
CSV = OUT / "results" / "results_benchmark.csv"
DATASETS = sys.argv[1:] if len(sys.argv) > 1 else ["gefcom2014", "sunlake", "folsom", "skippd"]

def main():
    set_style("light")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    loader = SolarUniBenchLoader("D:/monica/workspace/solar-unibench/data/standardized")

    done = set()
    if CSV.exists():
        done = set(pd.read_csv(CSV)["dataset"].unique())
        print(f"Already in CSV: {done}", flush=True)

    for ds in DATASETS:
        if ds in done:
            print(f"SKIP {ds} (already done)", flush=True)
            continue
        print(f"\n=== E8 {ds} (uniform 3-year sampling) ===", flush=True)
        # Use a per-dataset temp dir so e8.run's internal _save_results_csv
        # doesn't overwrite the shared benchmark CSV.
        tmp_out = OUT / f"_e8_{ds}"
        res = e8.run(loader, str(tmp_out), datasets=[ds])
        new_rows = res["metrics"]["rows"]
        new_df = pd.DataFrame(new_rows)
        # Append-merge: keep other datasets, replace this one.
        if CSV.exists():
            old = pd.read_csv(CSV)
            old = old[old["dataset"] != ds]
            combined = pd.concat([old, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(CSV, index=False)
        print(f"  {ds}: wrote {len(new_rows)} rows -> {CSV}", flush=True)
        done.add(ds)

    # Regenerate heatmap from combined results.
    all_df = pd.read_csv(CSV)
    if len(all_df):
        figures = e8._plot_heatmap(all_df, OUT)
        print(f"regenerated heatmap: {figures}", flush=True)
    print(f"\nAll datasets done: {done}", flush=True)

if __name__ == "__main__":
    main()
