"""Per-clip pipeline with degraded-mode ladder, plus the batch orchestrator.

Ladder per clip:
  (a) full: frames -> transcript -> describe -> stylize(best-of-3) -> judge
  (b) low time: stylize n=1, skip judge
  (c) describe failed: single-shot caption per style from 8 uniform frames
  (d) everything failed: contract-level fallback captions (write_results fills)

Design decisions that protect the score:
  - results[task_id] holds a LIVE dict mutated by process_clip, so a per-clip
    timeout keeps every caption completed before the timeout fired.
  - Per-clip budget is fair-share (budget / waves), so one pathological clip
    cannot starve the clips queued behind its semaphore slot.
  - CPU work (frames, whisper) runs on a dedicated small executor to bound
    contention; timeouts abandon a task but the pool caps zombie threads.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from .contract import Task
from .frames import extract_frames
from .fw import ModelClient, frames_to_content
from .stages import describe, pick_best, style_definition, stylize

log = logging.getLogger(__name__)

MAX_DOWNLOAD_BYTES = 600 * 1024 * 1024  # sanity cap
CPU_EXECUTOR = ThreadPoolExecutor(max_workers=3, thread_name_prefix="stylereel-cpu")


async def download(client: httpx.AsyncClient, url: str, dest: Path,
                   attempts: int = 3, attempt_timeout: float = 50) -> Path:
    if url.startswith("file://"):
        import shutil
        from urllib.parse import unquote, urlparse

        shutil.copy(unquote(urlparse(url).path), dest)
        return dest
    last: Exception | None = None
    for i in range(attempts):
        try:
            async def _one() -> None:
                got = 0
                async with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    with open(dest, "wb") as f:
                        async for chunk in resp.aiter_bytes(1 << 20):
                            got += len(chunk)
                            if got > MAX_DOWNLOAD_BYTES:
                                raise RuntimeError("download exceeds size cap")
                            f.write(chunk)

            await asyncio.wait_for(_one(), timeout=attempt_timeout)
            return dest
        except Exception as exc:
            last = exc
            await asyncio.sleep(1.0 * (i + 1))
    raise RuntimeError(f"download failed for {url}: {last!r}")


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


async def _single_shot(client: ModelClient, frames, transcript: str, style: str) -> str:
    """Ladder rung (c): one caption directly from frames, no describe stage."""
    content = frames_to_content(frames[:8], transcript)
    content.append({
        "type": "text",
        "text": (f"Write a caption for this video in a {style_definition(style)} style. "
                 "2-3 sentences describing what actually happens. Output only the caption."),
    })
    return await client.chat([{"role": "user", "content": content}],
                             vision=True, max_tokens=250, temperature=0.7)


async def process_clip(client: ModelClient, http: httpx.AsyncClient, task: Task,
                       workdir: Path, deadline: Deadline,
                       captions: dict[str, str]) -> None:
    """Mutates `captions` in place so partial progress survives an outer timeout."""
    video = workdir / f"{task.task_id}.mp4"
    try:
        await download(http, task.video_url, video)
    except Exception as exc:
        log.error("clip %s: download failed: %s", task.task_id, exc)
        return  # (d) contract fallback

    loop = asyncio.get_running_loop()
    try:
        frames = await asyncio.wait_for(
            loop.run_in_executor(CPU_EXECUTOR, extract_frames, str(video)), timeout=90)
    except Exception as exc:
        log.error("clip %s: frame extraction failed: %s", task.task_id, exc)
        return
    if not frames:
        return

    transcript = ""
    if deadline.remaining() > 90:
        from .audio import transcribe

        try:
            transcript = await asyncio.wait_for(
                loop.run_in_executor(CPU_EXECUTOR, transcribe, str(video)), timeout=60)
        except Exception as exc:
            log.warning("clip %s: transcript skipped: %s", task.task_id, exc)

    desc: str | None = None
    try:
        # Verify pass (extra vision call) only when there is time headroom.
        verify = deadline.comfortable
        # 200s covers draft + verify each with a fallback attempt
        desc = await asyncio.wait_for(
            describe(client, frames, transcript, verify=verify),
            timeout=200 if verify else 130)
    except Exception as exc:
        log.warning("clip %s: describe failed: %s", task.task_id, exc)

    from .contract import canonical_style
    volatile = {"sarcastic", "humorous_tech", "humorous_non_tech"}

    async def do_style(style: str) -> None:
        try:
            if desc is None:
                captions[style] = (await _single_shot(client, frames, transcript, style)).strip()
                return
            if deadline.comfortable:
                # humor/sarcasm vary most in accuracy -> more candidates for the judge to pick from
                n = 5 if canonical_style(style) in volatile else 3
            else:
                n = 1
            cands = await stylize(client, desc, style, n=n)
            captions[style] = cands[0]  # bank a caption immediately
            if n > 1 and not deadline.critical:
                captions[style] = await pick_best(client, desc, style, cands)
        except Exception as exc:
            log.warning("clip %s style %s failed: %s", task.task_id, style, exc)

    await asyncio.gather(*[do_style(s) for s in task.styles])
    try:
        video.unlink(missing_ok=True)
    except OSError:
        pass


async def run_batch(tasks: list[Task], workdir: Path, budget_s: float,
                    concurrency: int = 4,
                    checkpoint=None) -> dict[str, dict[str, str]]:
    """`checkpoint(results)` is called after every clip so a valid results file
    exists on disk from the first completion onward (skeleton-first strategy)."""
    deadline = Deadline(budget_s)
    client = ModelClient()
    http = httpx.AsyncClient(timeout=httpx.Timeout(45, connect=15), follow_redirects=True)
    sem = asyncio.Semaphore(concurrency)
    results: dict[str, dict[str, str]] = {t.task_id: {} for t in tasks}

    if checkpoint:
        checkpoint(results)  # skeleton: every task/style present as fallback

    waves = max(1, math.ceil(len(tasks) / concurrency))
    fair_share = max(75.0, (budget_s - 30) / waves)

    async def worker(task: Task) -> None:
        async with sem:
            if deadline.remaining() < 25:
                log.error("clip %s skipped: out of time", task.task_id)
                return
            per_clip = min(fair_share, max(30.0, deadline.remaining() - 10))
            try:
                await asyncio.wait_for(
                    process_clip(client, http, task, workdir, deadline,
                                 results[task.task_id]),
                    timeout=per_clip)
            except TimeoutError:
                done = sum(1 for v in results[task.task_id].values() if v)
                log.warning("clip %s hit per-clip budget %.0fs with %d/%d styles done",
                            task.task_id, per_clip, done, len(task.styles))
            except Exception as exc:
                log.error("clip %s failed entirely: %s", task.task_id, exc)
            if checkpoint:
                try:
                    checkpoint(results)
                except Exception as exc:
                    log.warning("checkpoint write failed: %s", exc)

    try:
        await asyncio.gather(*[worker(t) for t in tasks])
    finally:
        await client.close()
        await http.aclose()
    return results
