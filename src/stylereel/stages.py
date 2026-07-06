"""Pipeline stages: describe (vision) -> stylize x4 (text) -> judge (text).

All stages take the shared ModelClient; all prompts live in prompts/ so they can
be iterated without touching code. Style names are handled verbatim: requested
names map to canonical prompt files when recognizable, otherwise a generic
style-by-name prompt keeps the output key intact.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from .contract import canonical_style
from .fw import JUDGE_CHAIN, ModelClient, frames_to_content

log = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

STYLE_DEFINITIONS = {
    "formal": "professional, objective, factual tone",
    "sarcastic": "dry, ironic, lightly mocking",
    "humorous_tech": "funny, with technology or programming references",
    "humorous_non_tech": "funny, everyday humour with no technical jargon",
}

GENERIC_STYLE_PROMPT = """PERSONA: You are a skilled caption writer.

Write a caption for a video in the style "{style_name}". Interpret the style name
faithfully and make the tone unmistakable in every sentence.

RULES:
- 2-3 sentences, information-dense, describing what actually happens in the video.
- Every claim must come from the FACTS below. Do not invent.

Write ONLY the caption text, nothing else."""


def _load(rel: str) -> str:
    return (PROMPTS_DIR / rel).read_text()


def style_definition(style: str) -> str:
    canon = canonical_style(style)
    return STYLE_DEFINITIONS[canon] if canon else f'the style called "{style}"'


def _style_system_prompt(style: str) -> str:
    canon = canonical_style(style)
    if canon:
        return _load(f"styles/{canon}.txt")
    return GENERIC_STYLE_PROMPT.replace("{style_name}", style)


async def describe(client: ModelClient, frames, transcript: str) -> str:
    messages = [
        {"role": "system", "content": _load("describe.txt")},
        {"role": "user", "content": frames_to_content(frames, transcript)},
    ]
    return await client.chat(messages, vision=True, max_tokens=1200,
                             temperature=0.2, read_timeout=55)


async def stylize(client: ModelClient, description: str, style: str, n: int = 3) -> list[str]:
    canon = canonical_style(style)
    creative = canon in ("sarcastic", "humorous_tech", "humorous_non_tech") or canon is None
    system = _style_system_prompt(style)
    user = f"FACTS (the complete factual description of the video):\n\n{description}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    outs = await asyncio.gather(
        *[client.chat(messages, max_tokens=300,
                      temperature=0.9 if creative else 0.5, read_timeout=45)
          for _ in range(n)],
        return_exceptions=True,
    )
    captions = [o.strip().strip('"') for o in outs if isinstance(o, str) and o.strip()]
    if not captions:
        raise RuntimeError(f"stylize produced no candidates for {style}")
    return captions


def _parse_judge_json(text: str) -> list[dict]:
    """Extract the first JSON object via raw_decode (tolerates trailing prose)."""
    decoder = json.JSONDecoder()
    idx = text.find("{")
    while idx != -1:
        try:
            obj, _end = decoder.raw_decode(text[idx:])
            scores = obj.get("scores") if isinstance(obj, dict) else None
            if isinstance(scores, list):
                return [s for s in scores if isinstance(s, dict)]
        except json.JSONDecodeError:
            pass
        idx = text.find("{", idx + 1)
    raise ValueError("no scores object in judge output")


async def pick_best(client: ModelClient, description: str, style: str,
                    candidates: list[str]) -> str:
    """Judge candidates; hard accuracy gate, tone breaks ties. One revision if all inaccurate.

    Any failure anywhere returns candidates[0] — this stage may only improve, never lose.
    """
    if len(candidates) == 1:
        return candidates[0]
    try:
        system = (_load("judge.txt")
                  .replace("{style}", style)
                  .replace("{style_definition}", style_definition(style)))
        numbered = "\n\n".join(f"CAPTION {i}:\n{c}" for i, c in enumerate(candidates))
        user = f"DESCRIPTION:\n{description}\n\n{numbered}"
        raw = await client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            chain=JUDGE_CHAIN, max_tokens=800, temperature=0.0, use_gemma=False,
            read_timeout=45,
        )
        scores = _parse_judge_json(raw)
        valid = [s for s in scores
                 if isinstance(s.get("i"), int) and 0 <= s["i"] < len(candidates)
                 and isinstance(s.get("accuracy"), (int, float))
                 and isinstance(s.get("tone"), (int, float))]
        if not valid:
            return candidates[0]
        accurate = [s for s in valid if s["accuracy"] >= 4]
        if accurate:
            best = max(accurate, key=lambda s: (s["tone"], s["accuracy"]))
            return candidates[best["i"]]

        # all candidates failed the accuracy gate -> one revision pass on the best-toned one
        best = max(valid, key=lambda s: (s["tone"], s["accuracy"]))
        contradictions = "; ".join(str(c) for c in best.get("contradictions", [])[:5]) \
            or "unverifiable claims"
        fixed = await client.chat(
            [{"role": "system", "content": _style_system_prompt(style)},
             {"role": "user", "content":
              f"FACTS:\n\n{description}\n\nYour previous caption:\n{candidates[best['i']]}\n\n"
              f"It contains inaccuracies: {contradictions}\n"
              "Rewrite the caption fixing these while keeping the same tone. Output only the caption."}],
            max_tokens=300, temperature=0.6, read_timeout=45,
        )
        return fixed.strip().strip('"') or candidates[best["i"]]
    except Exception as exc:
        log.warning("judge stage failed (%s); returning first candidate", exc)
        return candidates[0]
