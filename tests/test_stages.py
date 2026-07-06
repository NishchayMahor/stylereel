import json

import pytest

from stylereel.frames import Frame
from stylereel.stages import describe, pick_best, stylize


class FakeClient:
    """Scriptable ModelClient double: pops queued responses, records requests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


async def test_describe_builds_vision_request():
    client = FakeClient(["SUBJECTS:\n- a cat"])
    frames = [Frame(ts=0.0, jpeg=b"x"), Frame(ts=5.0, jpeg=b"y")]
    out = await describe(client, frames, "[00:01] meow")
    assert "SUBJECTS" in out
    call = client.calls[0]
    assert call["kwargs"]["vision"] is True
    content = call["messages"][1]["content"]
    assert sum(1 for c in content if c["type"] == "image_url") == 2
    system = call["messages"][0]["content"]
    assert "INCONGRUITY" in system and "ON-SCREEN TEXT" in system


async def test_stylize_returns_candidates_and_style_prompt():
    client = FakeClient(["cap one", "cap two", "cap three"])
    outs = await stylize(client, "desc", "humorous_tech", n=3)
    assert outs == ["cap one", "cap two", "cap three"]
    system = client.calls[0]["messages"][0]["content"]
    assert "engineer" in system  # persona present
    assert "FACTS" in client.calls[0]["messages"][1]["content"]


async def test_stylize_all_fail_raises():
    client = FakeClient([RuntimeError("x"), RuntimeError("x"), RuntimeError("x")])
    with pytest.raises(RuntimeError):
        await stylize(client, "desc", "formal", n=3)


async def test_pick_best_accuracy_gate():
    verdict = json.dumps({"scores": [
        {"i": 0, "accuracy": 5, "tone": 3, "contradictions": []},
        {"i": 1, "accuracy": 2, "tone": 5, "contradictions": ["invented a dog"]},
        {"i": 2, "accuracy": 4, "tone": 4, "contradictions": []},
    ]})
    client = FakeClient([verdict])
    best = await pick_best(client, "desc", "sarcastic", ["a", "b", "c"])
    assert best == "c"  # highest tone among accurate (>=4)


async def test_pick_best_all_inaccurate_triggers_revision():
    verdict = json.dumps({"scores": [
        {"i": 0, "accuracy": 2, "tone": 4, "contradictions": ["wrong color"]},
        {"i": 1, "accuracy": 1, "tone": 2, "contradictions": ["wrong animal"]},
    ]})
    client = FakeClient([verdict, "revised caption"])
    best = await pick_best(client, "desc", "formal", ["a", "b"])
    assert best == "revised caption"
    assert "wrong color" in client.calls[1]["messages"][1]["content"]


async def test_pick_best_judge_garbage_returns_first():
    client = FakeClient(["not json at all"])
    best = await pick_best(client, "desc", "formal", ["a", "b"])
    assert best == "a"
