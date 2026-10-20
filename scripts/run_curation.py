"""CLI alias: run the curation pipeline (see build_dataset.py).

Paper-facing name for the same entry point:

    python scripts/run_curation.py --dataset folsom      # full build
    python scripts/run_curation.py --sample              # 1-minute demo
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_dataset import main

if __name__ == "__main__":
    sys.exit(main())
