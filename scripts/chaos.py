"""Chaos suite: run the container/pipeline through hostile inputs and assert the
contract holds (complete valid JSON, every style present, exit 0 semantics).

Runs in-process (no API key needed — all model calls fail and the degraded
ladder must still produce a complete, valid file).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]

SCENARIOS = {
    "dead_urls": [
        {"task_id": "d1", "video_url": "http://nope.invalid/a.mp4", "styles": STYLES},
    ],
    "odd_styles": [
        {"task_id": "o1", "video_url": "http://nope.invalid/b.mp4",
         "styles": ["Formal", "poetic", "sarcastic"]},
    ],
    "empty_styles": [
        {"task_id": "e1", "video_url": "http://nope.invalid/c.mp4", "styles": []},
    ],
    "wrapped_dict": {"tasks": [
        {"task_id": "w1", "video_url": "http://nope.invalid/d.mp4", "styles": STYLES}]},
    "malformed_entry": [
        {"task_id": "m1", "video_url": "http://nope.invalid/e.mp4", "styles": STYLES},
        {"task_id": "m2"},  # no url — must be skipped, others still processed
    ],
}


def run_scenario(name: str, tasks_obj) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        (tdp / "input").mkdir()
        (tdp / "output").mkdir()
        (tdp / "input/tasks.json").write_text(json.dumps(tasks_obj))
        env = {**os.environ, "STYLEREEL_INPUT": str(tdp / "input/tasks.json"),
               "STYLEREEL_OUTPUT": str(tdp / "output/results.json"),
               "STYLEREEL_BUDGET_S": "20", "STYLEREEL_HARD_EXIT": "0",
               "FIREWORKS_API_KEY": "dummy-key-forces-degraded"}
        proc = subprocess.run([sys.executable, "-m", "stylereel.main"],
                              env={**env, "PYTHONPATH": str(REPO / "src")},
                              capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            return False, f"exit {proc.returncode}: {proc.stderr[-300:]}"
        out = tdp / "output/results.json"
        if not out.exists():
            return False, "no results.json"
        try:
            data = json.loads(out.read_text())
        except Exception as exc:
            return False, f"malformed JSON: {exc}"

        # every requested style key present and non-empty for every parseable task
        raw = tasks_obj["tasks"] if isinstance(tasks_obj, dict) else tasks_obj
        want = {t["task_id"]: (t.get("styles") or STYLES)
                for t in raw if isinstance(t, dict) and "video_url" in t}
        got = {d["task_id"]: d["captions"] for d in data}
        for tid, styles in want.items():
            if tid not in got:
                return False, f"task {tid} missing from output"
            for s in styles:
                if not got[tid].get(s, "").strip():
                    return False, f"task {tid} style {s} empty/missing"
        return True, f"{len(data)} tasks, all styles present"


def main() -> int:
    failures = 0
    for name, tasks in SCENARIOS.items():
        ok, msg = run_scenario(name, tasks)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")
        failures += not ok
    print(f"\n{len(SCENARIOS) - failures}/{len(SCENARIOS)} chaos scenarios passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
