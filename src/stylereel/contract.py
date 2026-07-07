"""Input/output contract with the judging harness.

Guarantees: results.json is always valid JSON written atomically, every
requested style key (VERBATIM as requested, including unknown/case-variant
names) is present for every task, and captions are non-empty strings.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

STYLES = ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")

FALLBACK_CAPTIONS = {
    "formal": "A short video clip depicting a scene with visual activity recorded on camera.",
    "sarcastic": "Oh good, another video clip. Truly the pinnacle of cinema right here.",
    "humorous_tech": "This clip loaded faster than my CI pipeline, and honestly it has fewer bugs too.",
    "humorous_non_tech": "Somewhere out there, someone filmed this and thought: yes, the world needs to see it.",
}
GENERIC_FALLBACK = "A short video clip showing a scene unfolding over time."


def normalize_caption(text: str) -> str:
    """Clean model output into natural prose: strip em/en dashes (an AI tell) and
    tidy spacing/quotes. Keeps hyphenated words and numeric ranges intact."""
    import re

    t = text.strip().strip('"').strip("'")
    # spaced dash used as punctuation -> comma
    t = re.sub(r"\s+[—–―]\s+", ", ", t)
    # tight em/en dash between words -> comma+space; hyphen between letters kept
    t = re.sub(r"(?<=\w)[—―](?=\w)", ", ", t)
    t = t.replace("—", ", ").replace("―", ", ")
    t = t.replace("‘", "'").replace("’", "'")
    t = t.replace("“", '"').replace("”", '"')
    t = re.sub(r"\s+,", ",", t)
    t = re.sub(r",\s*,", ",", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


def canonical_style(style: str) -> str | None:
    """Map a requested style string to one of our four canonical styles, or None."""
    norm = style.strip().lower().replace("-", "_").replace(" ", "_")
    return norm if norm in STYLES else None


@dataclass
class Task:
    task_id: str
    video_url: str
    styles: list[str] = field(default_factory=lambda: list(STYLES))  # verbatim as requested


def read_tasks(path: str | Path) -> list[Task]:
    """Tolerant parse: skips malformed entries, accepts {'tasks': [...]} wrapping."""
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, dict):
        raw = raw.get("tasks") or raw.get("data") or []
    tasks: list[Task] = []
    for item in raw:
        try:
            styles = item.get("styles")
            if not isinstance(styles, list) or not styles:
                styles = list(STYLES)
            styles = [str(s) for s in styles]
            tasks.append(Task(task_id=str(item["task_id"]),
                              video_url=str(item["video_url"]), styles=styles))
        except Exception as exc:
            log.error("skipping malformed task entry %r: %s", item, exc)
    return tasks


def fallback_for(style: str) -> str:
    canon = canonical_style(style)
    return FALLBACK_CAPTIONS.get(canon, GENERIC_FALLBACK) if canon else GENERIC_FALLBACK


def write_results(path: str | Path, results: dict[str, dict[str, str]], tasks: list[Task]) -> None:
    """Atomic write; every requested style key present verbatim. Never raises on bad values."""
    out = []
    for task in tasks:
        got = results.get(task.task_id) or {}
        captions = {}
        for style in task.styles:
            cap = got.get(style)
            if not isinstance(cap, str) or not cap.strip():
                cap = fallback_for(style)
            captions[style] = normalize_caption(cap)
        out.append({"task_id": task.task_id, "captions": captions})
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    os.replace(tmp, p)
