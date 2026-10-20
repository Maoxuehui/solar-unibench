"""Generate ESSD paper figures (fig01-fig10; fig04 is hand-drawn).

Usage (conda env solar_uni):
    python scripts/generate_paper_figures.py            # all figures
    python scripts/generate_paper_figures.py --fig 6    # just fig06

Each plots/plot_figNN.py module exposes run() and writes its output
(PNG/PDF or component panels) into figures/paper/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plots import (
    plot_fig01, plot_fig02, plot_fig03,
    plot_fig05, plot_fig06, plot_fig07,
    plot_fig08, plot_fig09, plot_fig10,
)


def main():
    parser = argparse.ArgumentParser(description="Generate ESSD Paper Figures")
    parser.add_argument("--fig", type=str, default="all",
                        help="Figure number to generate (e.g., 6 or 'all')")
    args = parser.parse_args()

    figures = {
        "1": plot_fig01, "2": plot_fig02, "3": plot_fig03,
        "5": plot_fig05, "6": plot_fig06, "7": plot_fig07,
        "8": plot_fig08, "9": plot_fig09, "10": plot_fig10,
    }

    if args.fig == "all":
        for num, func in figures.items():
            print(f"Generating Fig {num}...", flush=True)
            func.run()
    elif args.fig in figures:
        figures[args.fig].run()
    else:
        parser.error(f"unknown --fig {args.fig!r} "
                     f"(valid: {', '.join(figures)}, or 'all')")


if __name__ == "__main__":
    main()
