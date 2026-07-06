"""Entrypoint. Reads /input/tasks.json, writes /output/results.json, exits 0.

The only contractually acceptable failure mode is fallback captions — never a
missing file, never malformed JSON, never a non-zero exit after tasks were read.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
from pathlib import Path

from .contract import read_tasks, write_results
from .pipeline import run_batch

INPUT = os.environ.get("STYLEREEL_INPUT", "/input/tasks.json")
OUTPUT = os.environ.get("STYLEREEL_OUTPUT", "/output/results.json")
BUDGET_S = float(os.environ.get("STYLEREEL_BUDGET_S", "510"))  # 8.5 min of the 10

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("stylereel")


def main() -> int:
    try:
        tasks = read_tasks(INPUT)
    except Exception as exc:
        log.error("cannot read tasks: %s", exc)
        return 1  # nothing sensible to write without task ids

    results: dict = {}
    try:
        with tempfile.TemporaryDirectory() as td:
            results = asyncio.run(run_batch(tasks, Path(td), BUDGET_S))
    except Exception as exc:
        log.error("batch failed: %s", exc)

    try:
        write_results(OUTPUT, results, tasks)
    except Exception as exc:  # last resort: write fallbacks only
        log.error("write_results failed (%s); writing pure fallbacks", exc)
        write_results(OUTPUT, {}, tasks)
    log.info("wrote %s for %d tasks", OUTPUT, len(tasks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
