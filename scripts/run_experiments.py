"""CLI orchestrator for Solar-UniBench experiments E1-E9.

Discovers each experiment's ``run(loader, out_dir) -> dict`` callable lazily and
dispatches the selected subset, sharing one :class:`SolarUniBenchLoader` and one
output directory. Experiments that have not landed yet are skipped with a
warning, so the harness runs incrementally as E1-E9 are added in Phase 7.

**Raw-provider wiring (Phase 7 Task 11).** E9 (governance vs complexity) and E5
(quality cleaning) need a pre-governance RAW frame that is NOT in the
standardized packages. The orchestrator supplies a default bounded
``raw_provider`` (GEFCom only, ``RUN_SLOW``-gated real build, cached) and passes
it to every experiment whose ``run`` signature accepts ``raw_provider`` (detected
via :func:`inspect.signature`), so dispatch stays uniform. Without ``RUN_SLOW``
the real build is skipped (fast default) and E9/E5 honestly skip their
raw-vs-cleaned axis; set ``RUN_SLOW=1`` to build the real bounded RAW snapshot.

Examples
--------
    python scripts/run_experiments.py                       # all implemented
    python scripts/run_experiments.py --only E1,E3          # subset
    python scripts/run_experiments.py --quick               # fast smoke (E1,E2,E7)
    python scripts/run_experiments.py --out figures/        # default out dir
    RUN_SLOW=1 python scripts/run_experiments.py --only E9  # real bounded RAW build
"""
import argparse
import importlib
import inspect
import json
import os
import shutil
import sys
import tempfile
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

#: Fast/descriptive subset for a <1-min smoke run (E7 = descriptive diagnostics,
#: E1 = geo blind-test + TSR + downstream, E2 = clear-sky feature ablation).
_QUICK_SUBSET: tuple[str, ...] = ("E1", "E2", "E7")

#: Bounded RAW snapshot is built for GEFCom only (spec E9 dev config: PV, fixed
#: data, tabular-only — the representative fast dataset). Phase 8 extends it.
_RAW_DATASET: str = "gefcom2014"
#: Bounded row cap per entity for the in-run RAW snapshot (matches the E9 test).
_RAW_LIMIT: int = 1500
#: Raw-source subdir under ``data_root`` (mirrors ``scripts/build_dataset.py``).
_RAW_SUBDIRS: dict[str, str] = {
    "gefcom2014": "GEFCom2014_data",
    "folsom": "Folsom_data",
    "skippd": "SKIPPD_all",
    "sunlake": "SunLake_data",
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


# ---------------------------------------------------------------------------
# Default bounded RAW provider (E9/E5 raw-vs-cleaned axis)
# ---------------------------------------------------------------------------

def _build_default_raw_provider(
    raw_source: Path | None,
    cache_dir: Path,
    *,
    dataset: str = _RAW_DATASET,
    limit: int = _RAW_LIMIT,
):
    """Return a ``raw_provider(dataset) -> df | None`` for bounded RAW snapshots.

    Builds a bounded RAW snapshot for ONE dataset (GEFCom by default) on demand:
    the heavy real build is ``RUN_SLOW``-gated + cached (on disk + in memory);
    without ``RUN_SLOW`` the provider returns ``None`` fast (no I/O) so E9/E5
    honestly skip their raw-vs-cleaned axis and the orchestrator stays fast by
    default.

    The snapshot is written under ``cache_dir`` (NOT the standardized packages
    root) so no real standardized data is mutated. Other datasets return
    ``None`` — the bounded dev config scopes E9 to GEFCom (Phase 8 extends it).

    The returned provider is closed over an in-memory cache, so a single
    orchestrator run builds the snapshot at most once even if multiple
    experiments request it.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = cache_dir / f"{dataset}_raw_bounded_limit{limit}.parquet"
    memo: dict[str, object] = {}

    def provider(name: str):
        if name != dataset:
            return None  # bounded: one dataset (spec E9 dev config)
        if name in memo:
            df = memo[name]
            return None if df is None else df.copy()
        # Fast no-op path: skip the heavy build unless RUN_SLOW is set.
        if not os.environ.get("RUN_SLOW"):
            memo[name] = None
            return None
        # Disk cache hit → reuse (no rebuild across runs). A cached snapshot is
        # served even when the raw source is no longer present — the cache makes
        # repeated orchestrator runs tractable and source-optional.
        if cache_key.exists():
            import pandas as pd

            df = pd.read_parquet(cache_key)
            memo[name] = df
            return df.copy()
        if raw_source is None or not Path(raw_source).exists():
            warnings.warn(
                f"E9/E5 raw source not present at {raw_source}; "
                f"raw-vs-cleaned axis will skip."
            )
            memo[name] = None
            return None
        # Real bounded build into a tmp dir under cache_dir, then hoist the
        # time_series.parquet to cache_key for reuse.
        try:
            from solarunibench.pipeline.adapters.gefcom import GEFCom2014Pipeline

            import pandas as pd
        except ImportError as exc:  # pragma: no cover - defensive
            warnings.warn(f"RAW snapshot build failed ({exc!r}); skipping.")
            memo[name] = None
            return None
        tmp = tempfile.mkdtemp(prefix=f"{dataset}_raw_", dir=str(cache_dir))
        try:
            pipe = GEFCom2014Pipeline(
                raw_data_dir=str(raw_source),
                output_dir=tmp,
                dataset_name=f"{dataset}_raw_bounded",
            )
            pipe.run_pipeline(raw=True, limit=limit)
            df = pd.read_parquet(
                Path(tmp) / f"{dataset}_raw_bounded" / "time_series.parquet"
            )
        except Exception as exc:  # pragma: no cover - defensive build contract
            warnings.warn(f"RAW snapshot build failed ({exc!r}); skipping.")
            memo[name] = None
            return None
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        df.to_parquet(cache_key)  # cache for the next run
        memo[name] = df
        return df.copy()

    return provider


# ---------------------------------------------------------------------------
# Dispatch + summary
# ---------------------------------------------------------------------------

def _pass_raw_provider(run: "callable") -> bool:
    """True iff ``run`` accepts a ``raw_provider`` keyword (E5, E9)."""
    try:
        params = inspect.signature(run).parameters
    except (ValueError, TypeError):
        return False
    return "raw_provider" in params


def _write_summary_json(
    results: list[tuple[str, dict]], selected: list[str], out_dir: Path
) -> Path:
    """Persist a per-experiment summary JSON under ``out_dir/results/``.

    Captures each experiment's status (``ran``/``skipped``/...), figure paths,
    and metric keys — the structured record the integration test + downstream
    tooling consume (complementary to the human-readable printed summary).
    """
    items = []
    for exp_id, res in results:
        figures = res.get("figures", [])
        metrics = res.get("metrics", {})
        status = metrics.get("status") if isinstance(metrics, dict) else None
        items.append(
            {
                "id": exp_id,
                "status": status or ("ran" if figures else "skipped"),
                "figures": [str(p) for p in figures] if figures else [],
                "n_figures": len(figures) if isinstance(figures, (list, tuple)) else 0,
                "metric_keys": list(metrics.keys()) if isinstance(metrics, dict) else [],
            }
        )
    payload = {"out_dir": str(out_dir), "selected": selected, "experiments": items}
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    path = out_dir / "results" / "orchestrator_summary.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


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
        status = metrics.get("status") if isinstance(metrics, dict) else None
        tag = f" [{status}]" if status in ("skipped", "failed") and n_fig == 0 else ""
        print(f"  {exp_id}: {n_fig} figure(s), {n_met} metric(s){tag}")
    print(f"  total: {len(results)} experiment(s) completed.")


def main(argv=None) -> int:
    repo = Path(__file__).resolve().parents[1]
    import yaml

    paths_cfg = yaml.safe_load((repo / "configs" / "paths.yaml").read_text())
    default_root = paths_cfg["output_root"]
    default_raw_source = Path(paths_cfg.get("data_root", "")) / _RAW_SUBDIRS.get(
        _RAW_DATASET, ""
    )

    ap = argparse.ArgumentParser(description="Run Solar-UniBench experiments E1-E9.")
    ap.add_argument(
        "--only",
        default=None,
        help="Comma-separated experiment ids to run (e.g. E1,E3). Default: all.",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help=f"Fast smoke run: select the quick subset {list(_QUICK_SUBSET)} (<1 min).",
    )
    ap.add_argument("--data-root", default=default_root, help="Standardized packages root.")
    ap.add_argument("--out", default="figures", help="Output directory for figures + results.")
    ap.add_argument(
        "--raw-source",
        default=str(default_raw_source),
        help=f"RAW source dir for the bounded E9/E5 snapshot (default: <data_root>/{_RAW_SUBDIRS.get(_RAW_DATASET)}).",
    )
    ap.add_argument(
        "--raw-cache",
        default=None,
        help="Cache dir for the bounded RAW snapshot (default: <out>/.raw_cache).",
    )
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    registry = discover_experiments()
    if args.quick:
        selected = [e for e in _QUICK_SUBSET if e in registry]
        if not selected:
            warnings.warn("--quick selected no available experiments; running nothing.")
    else:
        selected = _parse_only(args.only) or list(_EXPERIMENT_MODULES.keys())

    loader = SolarUniBenchLoader(args.data_root)
    raw_provider = _build_default_raw_provider(
        Path(args.raw_source) if args.raw_source else None,
        Path(args.raw_cache) if args.raw_cache else out_dir / ".raw_cache",
    )

    results: list[tuple[str, dict]] = []
    for exp_id in selected:
        run = registry.get(exp_id)
        if run is None:
            warnings.warn(f"{exp_id} is not available; skipping.")
            continue
        kwargs = {}
        if _pass_raw_provider(run):
            kwargs["raw_provider"] = raw_provider
        res = run(loader, out_dir, **kwargs)
        results.append((exp_id, res or {}))

    _print_summary(results, out_dir)
    _write_summary_json(results, selected, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
