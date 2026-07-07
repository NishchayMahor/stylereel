"""Ablation study: quantify what each pipeline stage contributes.

Runs the dev set through several configurations and scores each with the same
judge, producing a table. This is the evidence that the architecture is
load-bearing (not decoration) — and the frame the Gemma arm slots into.

Arms:
  full            describe(+verify) -> stylize best-of-5 -> judge   (shipping config)
  no_verify       describe(no verify) -> stylize best-of-5 -> judge
  no_bestof       describe(+verify) -> stylize n=1 (no judge)
  blind           NO describe; single-shot caption straight from frames per style
  (gemma)         same as full but describe runs on GEMMA_ENDPOINT (add when live)

Usage: python harness/ablation.py [--n N] [--arms full,blind,...]
Reads FIREWORKS_API_KEY / .fireworks_key.
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
sys.path.insert(0, str(REPO / "harness"))

from stylereel.contract import STYLES, Task  # noqa: E402
from stylereel.fw import ModelClient  # noqa: E402
from stylereel.frames import extract_frames  # noqa: E402
from stylereel.stages import describe, pick_best, stylize  # noqa: E402
from stylereel.pipeline import _single_shot, download  # noqa: E402

import httpx  # noqa: E402
from score import score_results  # noqa: E402

OUT = REPO / "harness/out/ablation"
CLIPS = REPO / "harness/devset/clips"
DEVSET = REPO / "harness/devset/tasks.json"


def _load_key():
    if os.environ.get("FIREWORKS_API_KEY"):
        return
    for c in (REPO / ".fireworks_key", REPO.parent / ".fireworks_key"):
        if c.exists():
            os.environ["FIREWORKS_API_KEY"] = c.read_text().strip()
            return


async def _caption_clip(client, http, task, workdir, arm, transcript_cache):
    caps: dict[str, str] = {}
    video = workdir / f"{task.task_id}.mp4"
    try:
        await download(http, task.video_url, video)
    except Exception:
        return caps
    frames = await asyncio.get_running_loop().run_in_executor(None, extract_frames, str(video))
    if not frames:
        return caps
    # transcript cached across arms (same clip) to save time/money
    if task.task_id not in transcript_cache:
        from stylereel.audio import transcribe
        transcript_cache[task.task_id] = await asyncio.get_running_loop().run_in_executor(
            None, transcribe, str(video))
    transcript = transcript_cache[task.task_id]

    if arm == "blind":
        for s in task.styles:
            try:
                caps[s] = (await _single_shot(client, frames, transcript, s)).strip()
            except Exception:
                pass
        return caps

    verify = arm in ("full", "gemma")
    desc = await describe(client, frames, transcript, verify=verify)
    for s in task.styles:
        try:
            if arm == "no_bestof":
                caps[s] = (await stylize(client, desc, s, n=1))[0]
            else:
                cands = await stylize(client, desc, s, n=5 if s != "formal" else 3)
                caps[s] = await pick_best(client, desc, s, cands)
        except Exception:
            pass
    return caps


async def run_arm(arm: str, tasks: list[Task], transcript_cache: dict) -> Path:
    client = ModelClient()
    http = httpx.AsyncClient(timeout=httpx.Timeout(60, connect=15), follow_redirects=True)
    results = []
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        for t in tasks:
            caps = await _caption_clip(client, http, t, Path(td), arm, transcript_cache)
            results.append({"task_id": t.task_id,
                            "captions": {s: caps.get(s, "") for s in t.styles}})
    await client.close()
    await http.aclose()
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"results_{arm}.json"
    p.write_text(json.dumps(results, indent=2))
    return p


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--arms", default="full,no_verify,no_bestof,blind")
    args = ap.parse_args()
    _load_key()
    if not os.environ.get("FIREWORKS_API_KEY"):
        sys.exit("No FIREWORKS_API_KEY")

    tasks_raw = json.loads(DEVSET.read_text())
    if args.n:
        tasks_raw = tasks_raw[: args.n]
    for t in tasks_raw:  # use local clips
        t["video_url"] = (CLIPS / Path(t["video_url"]).name).resolve().as_uri()
    tasks = [Task(t["task_id"], t["video_url"], t["styles"]) for t in tasks_raw]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tasks.json").write_text(json.dumps(tasks_raw, indent=2))

    transcript_cache: dict = {}
    table = {}
    for arm in args.arms.split(","):
        t0 = time.time()
        rp = await run_arm(arm, tasks, transcript_cache)
        report = await score_results(rp, OUT / "tasks.json")
        table[arm] = report["overall"]
        ov = report["overall"]
        print(f"[{arm:10s}] combined={ov['combined']:.3f}  acc={ov['accuracy']:.3f}  "
              f"style={ov['style']:.3f}  blind_id={ov['blind_style_id']:.3f}  ({time.time()-t0:.0f}s)")

    (OUT / "table.json").write_text(json.dumps(table, indent=2))
    print("\n=== ABLATION TABLE ===")
    print(f"{'arm':12s} {'combined':>9s} {'accuracy':>9s} {'style':>7s}")
    for arm, ov in table.items():
        print(f"{arm:12s} {ov['combined']:>9.3f} {ov['accuracy']:>9.3f} {ov['style']:>7.3f}")
    print(f"\nsaved -> {OUT/'table.json'}")


if __name__ == "__main__":
    asyncio.run(main())
