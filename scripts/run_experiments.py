"""CLI orchestrator for Solar-UniBench experiments E1-E9.

Discovers each experiment's ``run(loader, out_dir) -> dict`` callable lazily and
dispatches the selected subset, sharing one :class:`SolarUniBenchLoader` and one
output directory. Experiments that have not landed yet are skipped with a
warning, so the harness runs incrementally as E1-E9 are added in Phase 7.

Examples
--------
    python scripts/run_experiments.py                       # all implemented
    python scripts/run_experiments.py --only E1,E3          # subset
    python scripts/run_experiments.py --out figures/        # default
"""
import argparse
import importlib
import sys
import warnings
from pathlib import Path

# Make the repo importable when run as ``python scripts/run_experiments.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solarunibench.loader import SolarUniBenchLoader  # noqa: E402

# Canonical experiment id -> module path (filenames from the Phase 7 plan).
# Discovery tries each lazily and skips modules that don't exist yet.
_EXPERIMENT_MODULES: dict[str, str] = {
    "E1": "solarunibench.experiments.e1_geo",
    "E2": "solarunibench.experiments.e2_clearsky",
    "E3": "solarunibench.experiments.e3_brr",
    "E4": "solarunibench.experiments.e4_optflow",
    "E5": "solarunibench.experiments.e5_quality",
    "E6": "solarunibench.experiments.e6_alignment",
    "E7": "solarunibench.experiments.e7_defects",
    "E8": "solarunibench.experiments.e8_benchmark",
    "E9": "solarunibench.experiments.e9_governance",
}


def discover_experiments() -> dict[str, "callable"]:
    """Lazily import each experiment module and return ``{id: run}``.

    Modules that are not yet implemented are skipped (with a warning) rather
    than raising, so the orchestrator works incrementally.
    """
    found: dict[str, "callable"] = {}
    for exp_id, mod_path in _EXPERIMENT_MODULES.items():
        try:
            mod = importlib.import_module(mod_path)
        except ModuleNotFoundError:
            warnings.warn(f"{exp_id} not implemented yet ({mod_path}); skipping.")
            continue
        run = getattr(mod, "run", None)
        if not callable(run):
            warnings.warn(f"{exp_id} ({mod_path}) has no callable `run`; skipping.")
            continue
        found[exp_id] = run
    return found


def _parse_only(spec: str | None) -> list[str] | None:
    if not spec:
        return None
    ids = [s.strip().upper() for s in spec.split(",") if s.strip()]
    return ids or None


def main(argv=None) -> int:
    repo = Path(__file__).resolve().parents[1]
    import yaml

    default_root = yaml.safe_load((repo / "configs" / "paths.yaml").read_text())["output_root"]

    ap = argparse.ArgumentParser(description="Run Solar-UniBench experiments E1-E9.")
    ap.add_argument(
        "--only",
        default=None,
        help="Comma-separated experiment ids to run (e.g. E1,E3). Default: all.",
    )
    ap.add_argument("--data-root", default=default_root, help="Standardized packages root.")
    ap.add_argument("--out", default="figures", help="Output directory for figures + results.")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    registry = discover_experiments()
    selected = _parse_only(args.only)
    if selected is None:
        selected = list(_EXPERIMENT_MODULES.keys())

    loader = SolarUniBenchLoader(args.data_root)

    results: list[tuple[str, dict]] = []
    for exp_id in selected:
        run = registry.get(exp_id)
        if run is None:
            warnings.warn(f"{exp_id} is not available; skipping.")
            continue
        res = run(loader, out_dir)
        results.append((exp_id, res or {}))

    _print_summary(results, out_dir)
    return 0


def _print_summary(results: list[tuple[str, dict]], out_dir: Path) -> None:
    print(f"\n=== Solar-UniBench experiments -> {out_dir} ===")
    if not results:
        print("  (no experiments ran)")
        return
    for exp_id, res in results:
        figures = res.get("figures", [])
        metrics = res.get("metrics", {})
        n_fig = len(figures) if isinstance(figures, (list, tuple)) else 1
        n_met = len(metrics) if hasattr(metrics, "__len__") else 1
        print(f"  {exp_id}: {n_fig} figure(s), {n_met} metric(s)")
    print(f"  total: {len(results)} experiment(s) completed.")


if __name__ == "__main__":
    sys.exit(main())
