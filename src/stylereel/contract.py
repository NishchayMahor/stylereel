"""Input/output contract with the judging harness.

Guarantees: results.json is always valid JSON, every requested style for every
task is present, and captions are non-empty strings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

STYLES = ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")

FALLBACK_CAPTIONS = {
    "formal": "A short video clip depicting a scene with visual activity recorded on camera.",
    "sarcastic": "Oh good, another video clip. Truly the pinnacle of cinema right here.",
    "humorous_tech": "This clip loaded faster than my CI pipeline, and honestly it has fewer bugs too.",
    "humorous_non_tech": "Somewhere out there, someone filmed this and thought: yes, the world needs to see it.",
}


@dataclass
class Task:
    task_id: str
    video_url: str
    styles: list[str] = field(default_factory=lambda: list(STYLES))


def read_tasks(path: str | Path) -> list[Task]:
    raw = json.loads(Path(path).read_text())
    tasks = []
    for item in raw:
        styles = [s for s in item.get("styles", STYLES) if s in STYLES]
        if not styles:
            styles = list(STYLES)
        tasks.append(Task(task_id=str(item["task_id"]), video_url=item["video_url"], styles=styles))
    return tasks


def write_results(path: str | Path, results: dict[str, dict[str, str]], tasks: list[Task]) -> None:
    """Write results, filling any hole with a fallback caption. Never raises on bad input."""
    out = []
    for task in tasks:
        got = results.get(task.task_id) or {}
        captions = {}
        for style in task.styles:
            cap = got.get(style)
            if not isinstance(cap, str) or not cap.strip():
                cap = FALLBACK_CAPTIONS[style]
            captions[style] = cap.strip()
        out.append({"task_id": task.task_id, "captions": captions})
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2))
