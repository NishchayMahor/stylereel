"""Pipeline stages: describe (vision) -> stylize x4 (text) -> judge (text).

All stages take the shared ModelClient; all prompts live in prompts/ so they can
be iterated without touching code.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

from .fw import JUDGE_CHAIN, ModelClient, frames_to_content

log = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

STYLE_DEFINITIONS = {
    "formal": "professional, objective, factual tone",
    "sarcastic": "dry, ironic, lightly mocking",
    "humorous_tech": "funny, with technology or programming references",
    "humorous_non_tech": "funny, everyday humour with no technical jargon",
}


def _load(rel: str) -> str:
    return (PROMPTS_DIR / rel).read_text()


async def describe(client: ModelClient, frames, transcript: str) -> str:
    messages = [
        {"role": "system", "content": _load("describe.txt")},
        {"role": "user", "content": frames_to_content(frames, transcript)},
    ]
    return await client.chat(messages, vision=True, max_tokens=1200, temperature=0.2)


async def stylize(client: ModelClient, description: str, style: str, n: int = 3) -> list[str]:
    system = _load(f"styles/{style}.txt")
    user = f"FACTS (the complete factual description of the video):\n\n{description}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    outs = await asyncio.gather(
        *[client.chat(messages, max_tokens=300, temperature=0.9 if "humor" in style or style == "sarcastic" else 0.5)
          for _ in range(n)],
        return_exceptions=True,
    )
    captions = [o.strip().strip('"') for o in outs if isinstance(o, str) and o.strip()]
    if not captions:
        raise RuntimeError(f"stylize produced no candidates for {style}")
    return captions


def _parse_judge_json(text: str) -> list[dict]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no json in judge output")
    return json.loads(m.group(0))["scores"]


async def pick_best(client: ModelClient, description: str, style: str,
                    candidates: list[str]) -> str:
    """Judge candidates; hard accuracy gate, tone breaks ties. One revision if all inaccurate."""
    if len(candidates) == 1:
        return candidates[0]
    system = (_load("judge.txt")
              .replace("{style}", style)
              .replace("{style_definition}", STYLE_DEFINITIONS[style]))
    numbered = "\n\n".join(f"CAPTION {i}:\n{c}" for i, c in enumerate(candidates))
    user = f"DESCRIPTION:\n{description}\n\n{numbered}"
    try:
        raw = await client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            chain=JUDGE_CHAIN, max_tokens=800, temperature=0.0, use_gemma=False,
        )
        scores = _parse_judge_json(raw)
    except Exception as exc:
        log.warning("judge failed (%s); returning first candidate", exc)
        return candidates[0]

    valid = [s for s in scores if isinstance(s.get("i"), int) and 0 <= s["i"] < len(candidates)]
    if not valid:
        return candidates[0]
    accurate = [s for s in valid if s.get("accuracy", 0) >= 4]
    if accurate:
        best = max(accurate, key=lambda s: (s.get("tone", 0), s.get("accuracy", 0)))
        return candidates[best["i"]]

    # all candidates failed the accuracy gate -> one revision pass on the best-toned one
    best = max(valid, key=lambda s: (s.get("tone", 0), s.get("accuracy", 0)))
    contradictions = "; ".join(best.get("contradictions", [])[:5]) or "unverifiable claims"
    try:
        fixed = await client.chat(
            [{"role": "system", "content": _load(f"styles/{style}.txt")},
             {"role": "user", "content":
              f"FACTS:\n\n{description}\n\nYour previous caption:\n{candidates[best['i']]}\n\n"
              f"It contains inaccuracies: {contradictions}\n"
              "Rewrite the caption fixing these while keeping the same tone. Output only the caption."}],
            max_tokens=300, temperature=0.6,
        )
        return fixed.strip().strip('"') or candidates[best["i"]]
    except Exception:
        return candidates[best["i"]]
