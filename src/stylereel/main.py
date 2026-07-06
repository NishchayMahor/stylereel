"""Entrypoint. Reads /input/tasks.json, writes /output/results.json, exits 0.

Results are written INSIDE the event loop (before asyncio.run tears down and
joins executor threads), and the process then hard-exits: an abandoned whisper
or ffmpeg thread must never delay the output write past the container's
10-minute kill. STYLEREEL_HARD_EXIT=0 disables the hard exit for tests.
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
BUDGET_S = float(os.environ.get("STYLEREEL_BUDGET_S", "480"))  # 8 min of the 10

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("stylereel")


def _safe_write(tasks, results: dict) -> None:
    try:
        write_results(OUTPUT, results, tasks)
    except Exception as exc:  # last resort: write pure fallbacks
        log.error("write_results failed (%s); writing pure fallbacks", exc)
        try:
            write_results(OUTPUT, {}, tasks)
        except Exception as exc2:
            log.error("fallback write also failed: %s", exc2)


async def _run(tasks) -> None:
    results: dict = {}

    def checkpoint(current: dict) -> None:
        # Skeleton-first: a schema-valid, complete results.json exists on disk
        # from t=0 and is atomically refreshed after every finished clip.
        write_results(OUTPUT, current, tasks)

    try:
        with tempfile.TemporaryDirectory() as td:
            results = await asyncio.wait_for(
                run_batch(tasks, Path(td), BUDGET_S, checkpoint=checkpoint),
                timeout=BUDGET_S + 45)
    except Exception as exc:
        log.error("batch failed: %s", exc)
    # Write while the loop is alive — before any executor-thread join can block.
    _safe_write(tasks, results)
    log.info("wrote %s for %d tasks", OUTPUT, len(tasks))


def main() -> int:
    try:
        tasks = read_tasks(INPUT)
    except Exception as exc:
        log.error("cannot read tasks: %s", exc)
        return 1  # nothing sensible to write without task ids

    try:
        asyncio.run(_run(tasks))
    except Exception as exc:
        log.error("event loop failed (%s); ensuring output exists", exc)
        _safe_write(tasks, {})

    if os.environ.get("STYLEREEL_HARD_EXIT", "1") == "1":
        # Skip interpreter shutdown (which joins non-cancellable executor
        # threads) — results are already on disk and fsync'd by os.replace.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
