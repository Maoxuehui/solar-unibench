#!/usr/bin/env python
"""Select the ~6-8 paper figures from ``figures/`` and export print-quality copies.

This is the **Stream C "safe-now" packaging step** for the ESSD figure bundle. It
reads each selected source figure (PNG, produced by ``scripts/run_experiments.py``
via :func:`solarunibench.viz.io.save_fig`) and writes a print-ready **PDF + 300 dpi
PNG** pair into ``figures/paper/``.

What this script does (and does *not* do)
-----------------------------------------
- It **packages existing raster figures** at print resolution. It re-samples the
  source PNG (assumed rendered at matplotlib's default 100 dpi) to the requested
  output dpi (default 300) using Lanczos interpolation and embeds it in a PDF.
- It **does not re-render** the figures from source and **does not fabricate any
  data or metric**. The dev-config numbers currently in ``figures/`` are honest
  but bounded; the publication / full-config figure set is produced by Phase 8
  Stream B (``run_experiments.py`` at full model configs on full data). Re-run
  this script after Stream B to refresh the bundle.

Robustness
----------
Missing source figures are **skipped with a warning** (not a hard error) so the
script stays useful while the figure set is still landing. A non-zero exit code
is returned only if *no* figure exported successfully.

Examples
--------
    python scripts/select_figures.py                       # defaults
    python scripts/select_figures.py --src figures --out figures/paper
    python scripts/select_figures.py --dpi 600             # higher-res PNG/PDF
    python scripts/select_figures.py --list                # just print the registry
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

try:
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover - env dependency
    Image = None

try:  # Pillow >= 9.1
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover - older Pillow
    _LANCZOS = Image.LANCZOS

# ---------------------------------------------------------------------------
# Paper-figure registry.
# Canonical selection per the Phase 8 Stream C plan:
#   E9 governance = Fig. 1 (centerpiece); E1 geo, E8 benchmark, E7 defects,
#   E4 optical-flow, E6 alignment as supporting figures.
# Each entry: (output_stem, source_filename, one_line_caption).
# ---------------------------------------------------------------------------
PAPER_FIGURES: list[tuple[str, str, str]] = [
    (
        "fig1_governance_vs_complexity",
        "fig_governance_vs_complexity.png",
        "E9 - Data governance vs model complexity (centerpiece): "
        "skill over {data state x model}; the 'governance > complexity' crossover.",
    ),
    (
        "fig2_cleaning_delta",
        "fig_cleaning_delta.png",
        "E9 - Per-dataset skill gain attributable to data hygiene vs to model upgrade.",
    ),
    (
        "fig3_geo_blindtest",
        "fig_geo_blindtest.png",
        "E1 - Clear-sky geo-recovery blind test (Folsom, known coords) + GEFCom zones.",
    ),
    (
        "fig3b_geo_validation",
        "fig_geo_validation.png",
        "ESSD data validation - geo phase-shift ablation + Folsom blind inversion: "
        "(a) clear-sky phase misalignment, (b) downstream skill response, (c) 2D "
        "clear-sky residual contour (honest hemisphere-ambiguity limitation).",
    ),
    (
        "fig4_benchmark_heatmap",
        "fig_benchmark_heatmap.png",
        "E8 - Unified benchmark: Forecast Skill vs Smart Persistence "
        "(baselines x datasets x horizons).",
    ),
    (
        "fig5_defect_missingness",
        "fig_defect_missingness.png",
        "E7 - Data-defect diagnostics: missingness heatmap across all datasets.",
    ),
    (
        "fig6_optflow_sequence",
        "fig_optflow_sequence.png",
        "E4 - Optical-flow extrapolation: context -> extrapolated -> truth + uncertainty.",
    ),
    (
        "fig7_align_lag_hist",
        "fig_align_lag_hist.png",
        "E6 - Cross-modal alignment: image<->scalar timestamp diff before vs after.",
    ),
    (
        "fig8_e9_cross_dataset_crossover",
        "fig_e9_cross_dataset_crossover.png",
        "Phase 3 - E9 cross-dataset crossover (physics-aligned 1h-ahead): "
        "margin per dataset; bar >= 0 = governance > complexity holds.",
    ),
    (
        "fig9_e9_waterfall",
        "fig_e9_waterfall.png",
        "Phase 3 - Governance waterfall (Folsom, DLinear): clearsky anchor features "
        "dominate the governance gain (-68% MAE).",
    ),
    (
        "fig10_e9_granularity_sp_collapse",
        "fig_e9_granularity_sp_collapse.png",
        "Phase 3 - Smart-Persistence skill collapses (MAE rises) at coarser temporal "
        "granularity (Folsom 1min->1h); DLinear capacity caveat shown.",
    ),
    (
        "fig11_physics_closure_parity",
        "fig_physics_closure_parity.png",
        "ESSD data validation - physical-closure parity (Folsom): measured GHI vs "
        "DNI*cos(z)+DHI; R^2=0.9998 (irradiance modality is physically consistent).",
    ),
]


def _resolve(path: str | Path) -> Path:
    """Resolve a path relative to the repo root if it is not already absolute."""
    p = Path(path)
    return p if p.is_absolute() else (REPO / p)


def _export_one(
    stem: str,
    source: Path,
    out_dir: Path,
    dpi: int,
    source_dpi: int,
) -> tuple[Path, Path] | None:
    """Export one figure as ``<stem>.pdf`` + ``<stem>.png`` at ``dpi``.

    Returns ``(png_path, pdf_path)`` on success, or ``None`` if the source is
    missing/unreadable (caller warns).
    """
    if Image is None:
        raise RuntimeError(
            "Pillow is required (pip install pillow) but is not importable."
        )
    if not source.exists():
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as img:
        img.load()
        # Flatten to RGB for print (drop alpha / palette).
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        # Preserve the source's physical size but emit at the target dpi. Prefer
        # the dpi metadata embedded in the source PNG; fall back to ``source_dpi``
        # (matplotlib's savefig default 100) when none is present. Then
        # physical_inches = px / src_dpi and output px = physical_inches * dpi,
        # which upsamples resolution without changing the laid-out size and
        # without inventing new signal.
        w, h = img.size
        src_dpi_xy = img.info.get("dpi")
        if src_dpi_xy and src_dpi_xy[0]:
            src_dpi = float(src_dpi_xy[0])
        else:
            src_dpi = float(source_dpi)
        phys_w = w / src_dpi
        phys_h = h / src_dpi
        out_w = max(1, round(phys_w * dpi))
        out_h = max(1, round(phys_h * dpi))
        if (out_w, out_h) != (w, h):
            img = img.resize((out_w, out_h), _LANCZOS)
        png_path = out_dir / f"{stem}.png"
        pdf_path = out_dir / f"{stem}.pdf"
        img.save(png_path, dpi=(dpi, dpi))  # PNG: dpi metadata
        img.save(pdf_path, "PDF", resolution=float(dpi))  # PDF: raster embedded @ dpi
    return png_path, pdf_path


def _print_registry() -> None:
    print(f"Paper-figure registry ({len(PAPER_FIGURES)} entries):")
    for stem, src, caption in PAPER_FIGURES:
        print(f"  {stem:<32} <- {src}")
        print(f"    {caption}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Select ~6-8 paper figures from figures/ and export print-quality "
            "PDF + 300 dpi PNG copies to figures/paper/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Notes: packages existing raster figures (no re-render, no fabricated "
            "data). Re-run after Phase 8 Stream B to refresh with full-config figures."
        ),
    )
    ap.add_argument(
        "--src",
        default="figures",
        help="Source directory of dev figures (default: figures).",
    )
    ap.add_argument(
        "--out",
        default="figures/paper",
        help="Output directory for print-quality figures (default: figures/paper).",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output resolution in dpi (default: 300).",
    )
    ap.add_argument(
        "--source-dpi",
        type=int,
        default=100,
        help=(
            "Assumed dpi of the input PNGs, used to preserve physical size "
            "when re-sampling (default: 100, matplotlib's savefig default)."
        ),
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="Print the paper-figure registry and exit (no export).",
    )
    args = ap.parse_args(argv)

    if args.list:
        _print_registry()
        return 0

    if Image is None:
        ap.error(
            "Pillow is not installed in this environment (pip install pillow)."
        )

    src_dir = _resolve(args.src)
    out_dir = _resolve(args.out)
    if not src_dir.is_dir():
        warnings.warn(f"source directory not found: {src_dir}")

    print(f"Exporting paper figures: {src_dir} -> {out_dir} @ {args.dpi} dpi")
    exported = 0
    skipped = 0
    for stem, src_name, caption in PAPER_FIGURES:
        source = src_dir / src_name
        try:
            result = _export_one(stem, source, out_dir, args.dpi, args.source_dpi)
        except Exception as exc:  # pragma: no cover - defensive per-figure
            warnings.warn(f"{stem}: failed to export ({exc!r}); skipping.")
            skipped += 1
            continue
        if result is None:
            warnings.warn(f"{stem}: source missing ({source}); skipping.")
            skipped += 1
            continue
        png_path, pdf_path = result
        print(f"  {stem}: {png_path.name} + {pdf_path.name}")
        exported += 1

    total = len(PAPER_FIGURES)
    print(
        f"\nDone: {exported}/{total} exported"
        + (f", {skipped} skipped" if skipped else "")
        + f" -> {out_dir}"
    )
    return 0 if exported > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
