"""CLI alias: run the benchmark / experiment suite (see run_experiments.py).

Paper-facing name for the same entry point (1-h dispatch evaluation and the
multi-horizon sensitivity grid live in the E8/E9 experiments):

    python scripts/run_benchmark.py --only E8,E9      # core experiments
    python scripts/run_benchmark.py --quick           # smoke (<1 min)
    python scripts/run_benchmark.py --full            # publishable config
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_experiments import main

if __name__ == "__main__":
    sys.exit(main())
