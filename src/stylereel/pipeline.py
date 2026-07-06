"""Per-clip pipeline with degraded-mode ladder, plus the batch orchestrator.

Ladder per clip:
  (a) full: frames -> transcript -> describe -> stylize(best-of-3) -> judge
  (b) low time: stylize n=1, skip judge
  (c) describe failed: single-shot caption per style from 8 uniform frames
  (d) everything failed: contract-level fallback captions (handled by write_results)

The global deadline manager shrinks per-clip ambition rather than dropping clips.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from pathlib import Path

import httpx

from .contract import Task
from .frames import extract_frames
from .fw import ModelClient
from .stages import STYLE_DEFINITIONS, describe, pick_best, stylize

log = logging.getLogger(__name__)


async def download(url: str, dest: Path, attempts: int = 3) -> Path:
    async with httpx.AsyncClient(timeout=httpx.Timeout(90, connect=15),
                                 follow_redirects=True) as client:
        last: Exception | None = None
        for i in range(attempts):
            try:
                async with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    with open(dest, "wb") as f:
                        async for chunk in resp.aiter_bytes(1 << 20):
                            f.write(chunk)
                return dest
            except Exception as exc:
                last = exc
                await asyncio.sleep(1.5 * (i + 1))
        raise RuntimeError(f"download failed for {url}: {last}")


class Deadline:
    def __init__(self, budget_s: float) -> None:
        self.t0 = time.monotonic()
        self.budget = budget_s

    def remaining(self) -> float:
        return self.budget - (time.monotonic() - self.t0)

    @property
    def comfortable(self) -> bool:
        return self.remaining() > 120

    @property
    def critical(self) -> bool:
        return self.remaining() < 45


async def _single_shot(client: ModelClient, frames, style: str) -> str:
    """Ladder rung (c): one caption directly from frames, no describe stage."""
    from .fw import frames_to_content

    content = frames_to_content(frames[:8], "")
    content.append({
        "type": "text",
        "text": (f"Write a caption for this video in a {STYLE_DEFINITIONS[style]} style. "
                 "2-3 sentences describing what actually happens. Output only the caption."),
    })
    return await client.chat([{"role": "user", "content": content}],
                             vision=True, max_tokens=250, temperature=0.7)


async def process_clip(client: ModelClient, task: Task, workdir: Path,
                       deadline: Deadline) -> dict[str, str]:
    captions: dict[str, str] = {}
    video = workdir / f"{task.task_id}.mp4"
    try:
        await download(task.video_url, video)
    except Exception as exc:
        log.error("clip %s: download failed: %s", task.task_id, exc)
        return captions  # (d) contract fallback

    loop = asyncio.get_running_loop()
    try:
        frames = await loop.run_in_executor(None, extract_frames, str(video))
    except Exception as exc:
        log.error("clip %s: frame extraction failed: %s", task.task_id, exc)
        return captions
    if not frames:
        return captions

    transcript = ""
    if deadline.remaining() > 90:
        from .audio import transcribe

        try:
            transcript = await asyncio.wait_for(
                loop.run_in_executor(None, transcribe, str(video)), timeout=75)
        except Exception as exc:
            log.warning("clip %s: transcript skipped: %s", task.task_id, exc)

    desc: str | None = None
    try:
        desc = await asyncio.wait_for(describe(client, frames, transcript), timeout=80)
    except Exception as exc:
        log.warning("clip %s: describe failed: %s", task.task_id, exc)

    async def do_style(style: str) -> None:
        try:
            if desc is None:
                captions[style] = (await _single_shot(client, frames, style)).strip()
                return
            n = 3 if deadline.comfortable else 1
            cands = await stylize(client, desc, style, n=n)
            if n > 1 and not deadline.critical:
                captions[style] = await pick_best(client, desc, style, cands)
            else:
                captions[style] = cands[0]
        except Exception as exc:
            log.warning("clip %s style %s failed: %s", task.task_id, style, exc)

    await asyncio.gather(*[do_style(s) for s in task.styles])
    try:
        video.unlink(missing_ok=True)
    except OSError:
        pass
    return captions


async def run_batch(tasks: list[Task], workdir: Path, budget_s: float,
                    concurrency: int = 4) -> dict[str, dict[str, str]]:
    deadline = Deadline(budget_s)
    client = ModelClient()
    sem = asyncio.Semaphore(concurrency)
    results: dict[str, dict[str, str]] = {}

    async def worker(task: Task) -> None:
        async with sem:
            if deadline.remaining() < 20:
                log.error("clip %s skipped: out of time", task.task_id)
                return
            try:
                per_clip = max(30.0, deadline.remaining() - 10)
                results[task.task_id] = await asyncio.wait_for(
                    process_clip(client, task, workdir, deadline), timeout=per_clip)
            except Exception as exc:
                log.error("clip %s failed entirely: %s", task.task_id, exc)

    try:
        await asyncio.gather(*[worker(t) for t in tasks])
    finally:
        await client.close()
    return results
