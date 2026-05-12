"""Run all four trainings in sequence. Used on the school PC.

Usage:
    python scripts/train_all.py

Runs: BiLSTM → lab transformer → DistilBERT → ClimateBERT.
Each script saves its own predictions + metrics + figures.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SCRIPTS = [
    "scripts/train_bilstm.py",
    "scripts/train_lab_transformer.py",
    "scripts/train_distilbert.py",
    "scripts/train_climatebert.py",
]


def main() -> None:
    overall_start = time.time()
    for script in SCRIPTS:
        print(f"\n{'=' * 72}", flush=True)
        print(f"  RUNNING  {script}", flush=True)
        print('=' * 72, flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, script], cwd=str(PROJECT_ROOT))
        elapsed = time.time() - t0
        if r.returncode != 0:
            print(f"\n!!! {script} failed (exit {r.returncode}) after {elapsed:.1f}s — stopping.", flush=True)
            sys.exit(r.returncode)
        print(f"\n  ✓ {script} done in {elapsed / 60:.1f} min", flush=True)
    print(f"\nAll trainings complete. Total: {(time.time() - overall_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
