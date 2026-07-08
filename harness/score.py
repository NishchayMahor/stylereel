"""Local judge that mirrors the competition's LLM-Judge.

Scores each caption on two 0-1 dimensions:
  - accuracy: faithfulness to the video content (judge sees the SAME frames)
  - style:    match to the requested tone

Uses a different model family than the pipeline generator (Kimi/DeepSeek) to
reduce self-preference bias, and — critically — the judge WATCHES THE CLIP
(samples its own frames) rather than trusting a reference description, so it is
an independent check exactly like the real judge.

Also computes a "blind style ID" metric: given the 4 unlabeled captions, can an
LLM assign all 4 styles correctly? (distinctness proxy)

Requires FIREWORKS_API_KEY (or .fireworks_key alongside the repo).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from stylereel.contract import STYLES  # noqa: E402
from stylereel.frames import extract_frames  # noqa: E402
from stylereel.fw import ModelClient  # noqa: E402

CLIPS = REPO / "harness/devset/clips"

STYLE_DEFS = {
    "formal": "professional, objective, factual — no humor, no opinion",
    "sarcastic": "dry, ironic, lightly mocking via sentiment incongruity",
    "humorous_tech": "funny with genuine technology/programming references",
    "humorous_non_tech": "funny, everyday humor with zero technical jargon",
}

# Accuracy judge must watch the clip -> needs vision (only kimi has it here).
# Style/blind judges are text-only -> use a different family to cut self-preference.
VISION_JUDGE = ["accounts/fireworks/models/kimi-k2p6"]
TEXT_JUDGE = ["accounts/fireworks/models/deepseek-v4-pro"]

ACC_PROMPT = """You are scoring a video caption for ACCURACY. You are shown frames from the video.
Caption ({style}): "{caption}"

Rate 0.0-1.0 how faithfully the caption reflects what is actually visible in the frames.
1.0 = every claim is supported and the main subject+action are captured; subtract for each
hallucinated/contradicted detail and for missing the main subject. Reply ONLY JSON:
{{"accuracy": <float>, "reason": "<10 words>"}}"""

STYLE_PROMPT = """You are scoring how well a caption matches a requested STYLE.
Requested style: {style} ({style_def})
Caption: "{caption}"

Rate 0.0-1.0 how strongly and consistently the caption embodies that exact style (every
sentence should carry the tone; wrong style = low). Reply ONLY JSON:
{{"style": <float>, "reason": "<10 words>"}}"""

BLIND_PROMPT = """Here are four captions of the same video, each written in one of these styles:
formal, sarcastic, humorous_tech, humorous_non_tech. Assign each caption to exactly one style.
{captions}
Reply ONLY JSON mapping caption index to style: {{"0": "...", "1": "...", "2": "...", "3": "..."}}"""


def _frames_content(frames):
    content = [{"type": "text", "text": "Video frames (chronological):"}]
    # 10 frames balances coverage vs judge latency (16 frames times out on Kimi vision)
    step = max(1, len(frames) // 10)
    for f in frames[::step][:10]:
        b64 = base64.b64encode(f.jpeg).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return content


async def _robust(coro_fn, tries=3):
    """Retry a judge call; return '' on repeated failure so scoring never crashes."""
    for i in range(tries):
        try:
            return await coro_fn()
        except Exception:
            if i == tries - 1:
                return ""
            await asyncio.sleep(1.5 * (i + 1))
    return ""


def _num(text: str, key: str) -> float:
    dec = json.JSONDecoder()
    i = text.find("{")
    while i != -1:
        try:
            obj, _ = dec.raw_decode(text[i:])
            if isinstance(obj, dict) and key in obj:
                return max(0.0, min(1.0, float(obj[key])))
        except Exception:
            pass
        i = text.find("{", i + 1)
    return 0.0


async def score_clip(client: ModelClient, task_id: str, video_path: Path,
                     captions: dict[str, str]) -> dict:
    frames = await asyncio.get_running_loop().run_in_executor(
        None, extract_frames, str(video_path))
    per_style = {}
    for style, cap in captions.items():
        acc_msgs = [{"role": "user", "content": _frames_content(frames) +
                     [{"type": "text", "text": ACC_PROMPT.format(style=style, caption=cap)}]}]
        sty_msgs = [{"role": "user", "content": STYLE_PROMPT.format(
            style=style, style_def=STYLE_DEFS.get(style, style), caption=cap)}]
        acc_raw, sty_raw = await asyncio.gather(
            _robust(lambda: client.chat(acc_msgs, vision=True, max_tokens=120, temperature=0.0,
                                        use_gemma=False, chain=VISION_JUDGE, read_timeout=90)),
            _robust(lambda: client.chat(sty_msgs, max_tokens=120, temperature=0.0,
                                        use_gemma=False, chain=TEXT_JUDGE, read_timeout=90)),
        )
        per_style[style] = {"accuracy": _num(acc_raw, "accuracy"),
                            "style": _num(sty_raw, "style")}

    # blind style-ID distinctness metric
    ordered = [s for s in STYLES if s in captions]
    blind = 0.0
    if len(ordered) == 4:
        listing = "\n".join(f"[{i}] {captions[s]}" for i, s in enumerate(ordered))
        raw = await client.chat(
            [{"role": "user", "content": BLIND_PROMPT.format(captions=listing)}],
            max_tokens=120, temperature=0.0, use_gemma=False, chain=TEXT_JUDGE)
        try:
            dec = json.JSONDecoder()
            j = raw.find("{")
            mapping, _ = dec.raw_decode(raw[j:])
            correct = sum(1 for i, s in enumerate(ordered)
                          if str(mapping.get(str(i), "")).strip() == s)
            blind = correct / 4
        except Exception:
            blind = 0.0
    return {"task_id": task_id, "styles": per_style, "blind_style_id": blind}


async def score_results(results_path: Path, tasks_path: Path) -> dict:
    results = {d["task_id"]: d["captions"] for d in json.loads(results_path.read_text())}
    tasks = {t["task_id"]: t for t in json.loads(tasks_path.read_text())}
    client = ModelClient()
    rows = []
    try:
        for tid, caps in results.items():
            name = Path(tasks[tid]["video_url"]).name
            vp = CLIPS / name
            if not vp.exists():
                continue
            rows.append(await score_clip(client, tid, vp, caps))
    finally:
        await client.close()

    # aggregate
    agg = {s: {"accuracy": [], "style": []} for s in STYLES}
    blinds = []
    for r in rows:
        blinds.append(r["blind_style_id"])
        for s, sc in r["styles"].items():
            if s in agg:
                agg[s]["accuracy"].append(sc["accuracy"])
                agg[s]["style"].append(sc["style"])

    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    summary = {s: {"accuracy": mean(v["accuracy"]), "style": mean(v["style"])}
               for s, v in agg.items()}
    all_acc = [x for v in agg.values() for x in v["accuracy"]]
    all_sty = [x for v in agg.values() for x in v["style"]]
    overall = {"accuracy": mean(all_acc), "style": mean(all_sty),
               "combined": round((mean(all_acc) + mean(all_sty)) / 2, 3),
               "blind_style_id": mean(blinds)}
    return {"overall": overall, "per_style": summary, "per_clip": rows}


if __name__ == "__main__":
    rp = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "harness/out/results.json"
    tp = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "harness/devset/tasks.json"
    report = asyncio.run(score_results(rp, tp))
    print(json.dumps(report["overall"], indent=2))
    print("\nper style:")
    for s, v in report["per_style"].items():
        print(f"  {s:20s} acc={v['accuracy']:.3f}  style={v['style']:.3f}")
    outp = REPO / "harness/out/scores.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=2))
    print(f"\nfull report -> {outp}")
