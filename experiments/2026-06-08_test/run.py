"""Experiment 2026-06-08_test.

Auto-scaffolded by new.py. Edit freely. Output PNGs/CSVs to PLOTS_DIR.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core import io, metrics, loaders  # noqa: F401 — pull what you need

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print(f"hello from 2026-06-08_test; write plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
