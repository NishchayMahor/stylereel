"""Run the pipeline on the dev set, then score it. The moat.

Usage:
  python harness/run.py                 # run pipeline in-process on N clips, then score
  python harness/run.py --n 6           # subset for fast iteration
  python harness/run.py --score-only    # just re-score existing harness/out/results.json
  python harness/run.py --baseline      # save current scores as baseline.json
  python harness/run.py --local-clips   # use downloaded clips instead of GCS URLs (fast)

Reads FIREWORKS_API_KEY from env or the .fireworks_key file next to the repo.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "harness/out"
DEVSET = REPO / "harness/devset/tasks.json"
CLIPS = REPO / "harness/devset/clips"


def _load_key() -> None:
    if os.environ.get("FIREWORKS_API_KEY"):
        return
    for cand in (REPO / ".fireworks_key", REPO.parent / ".fireworks_key"):
        if cand.exists():
            os.environ["FIREWORKS_API_KEY"] = cand.read_text().strip()
            return


def _build_tasks(n: int | None, local: bool) -> list[dict]:
    tasks = json.loads(DEVSET.read_text())
    if n:
        tasks = tasks[:n]
    if local:
        for t in tasks:
            local_path = CLIPS / Path(t["video_url"]).name
            t["video_url"] = local_path.resolve().as_uri()
    return tasks


async def _run_pipeline(tasks: list[dict]) -> dict:
    from stylereel.contract import Task, write_results
    from stylereel.pipeline import run_batch

    task_objs = [Task(t["task_id"], t["video_url"], t["styles"]) for t in tasks]
    import tempfile
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        results = await run_batch(task_objs, Path(td), budget_s=480,
                                  checkpoint=lambda r: write_results(OUT / "results.json", r, task_objs))
    write_results(OUT / "results.json", results, task_objs)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--local-clips", action="store_true")
    args = ap.parse_args()
    _load_key()
    if not os.environ.get("FIREWORKS_API_KEY"):
        sys.exit("No FIREWORKS_API_KEY (env or .fireworks_key file). See README.")

    tasks = _build_tasks(args.n, args.local_clips)
    (OUT).mkdir(parents=True, exist_ok=True)
    (OUT / "devset_used.json").write_text(json.dumps(tasks, indent=2))

    if not args.score_only:
        t0 = time.time()
        asyncio.run(_run_pipeline(tasks))
        print(f"pipeline done in {time.time()-t0:.0f}s")

    from score import score_results  # noqa: E402
    report = asyncio.run(score_results(OUT / "results.json", OUT / "devset_used.json"))
    (OUT / "scores.json").write_text(json.dumps(report, indent=2))

    ov = report["overall"]
    print("\n=== OVERALL ===")
    print(f"  combined={ov['combined']:.3f}  accuracy={ov['accuracy']:.3f}  "
          f"style={ov['style']:.3f}  blind_style_id={ov['blind_style_id']:.3f}")
    print("per style (accuracy / style):")
    for s, v in report["per_style"].items():
        print(f"  {s:20s} {v['accuracy']:.3f} / {v['style']:.3f}")

    base = OUT / "baseline.json"
    if args.baseline:
        base.write_text(json.dumps(report, indent=2))
        print(f"\nsaved baseline -> {base}")
    elif base.exists():
        b = json.loads(base.read_text())["overall"]
        d = ov["combined"] - b["combined"]
        print(f"\nvs baseline: combined {b['combined']:.3f} -> {ov['combined']:.3f} "
              f"({'+' if d >= 0 else ''}{d:.3f})")


if __name__ == "__main__":
    main()
