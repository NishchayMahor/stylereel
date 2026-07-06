import json
from pathlib import Path

import pytest

import stylereel.pipeline as pl
from stylereel.contract import STYLES, Task
from stylereel.frames import Frame


class FakeClient:
    def __init__(self, response="ok caption"):
        self.response = response
        self.closed = False

    async def chat(self, messages, **kwargs):
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    async def close(self):
        self.closed = True


@pytest.fixture
def patched(monkeypatch, tmp_path):
    """Neutralize IO-heavy pieces; each test overrides what it exercises."""
    async def fake_download(url, dest, attempts=3):
        Path(dest).write_bytes(b"fake")
        return dest

    monkeypatch.setattr(pl, "download", fake_download)
    monkeypatch.setattr(pl, "extract_frames",
                        lambda p, max_frames=16: [Frame(0.0, b"j"), Frame(1.0, b"k")])
    return tmp_path


async def test_full_ladder_happy(patched, monkeypatch):
    monkeypatch.setattr("stylereel.audio.transcribe", lambda p: "")
    client = FakeClient("a fine caption")
    deadline = pl.Deadline(600)
    caps = await pl.process_clip(client, Task("v1", "http://x"), patched, deadline)
    assert set(caps) == set(STYLES)
    assert all(caps[s] for s in STYLES)


async def test_describe_dead_single_shot_rung(patched, monkeypatch):
    monkeypatch.setattr("stylereel.audio.transcribe", lambda p: "")

    class DescribeFails(FakeClient):
        async def chat(self, messages, **kwargs):
            # describe stage sends system prompt containing INCONGRUITY
            first = messages[0]
            if isinstance(first.get("content"), str) and "INCONGRUITY" in first["content"]:
                raise RuntimeError("vlm down")
            return "single-shot caption"

    caps = await pl.process_clip(DescribeFails(), Task("v1", "http://x"),
                                 patched, pl.Deadline(600))
    assert set(caps) == set(STYLES)
    assert all(c == "single-shot caption" for c in caps.values())


async def test_download_dead_returns_empty(patched, monkeypatch):
    async def bad_download(url, dest, attempts=3):
        raise RuntimeError("404")

    monkeypatch.setattr(pl, "download", bad_download)
    caps = await pl.process_clip(FakeClient(), Task("v1", "http://x"),
                                 patched, pl.Deadline(600))
    assert caps == {}


async def test_run_batch_all_exploding_still_returns(patched, monkeypatch):
    async def bad_download(url, dest, attempts=3):
        raise RuntimeError("network gone")

    monkeypatch.setattr(pl, "download", bad_download)
    tasks = [Task("v1", "u1"), Task("v2", "u2")]
    results = await pl.run_batch(tasks, patched, budget_s=600)
    assert isinstance(results, dict)  # write_results fills the holes downstream


async def test_main_writes_complete_output_when_everything_fails(tmp_path, monkeypatch):
    tasks_file = tmp_path / "tasks.json"
    tasks_file.write_text(json.dumps(
        [{"task_id": "v1", "video_url": "http://nope.invalid/x.mp4",
          "styles": list(STYLES)}]))
    out_file = tmp_path / "results.json"
    monkeypatch.setenv("FIREWORKS_API_KEY", "k")
    import importlib

    import stylereel.main as m
    monkeypatch.setattr(m, "INPUT", str(tasks_file))
    monkeypatch.setattr(m, "OUTPUT", str(out_file))
    monkeypatch.setattr(m, "BUDGET_S", 5.0)
    rc = m.main()
    assert rc == 0
    data = json.loads(out_file.read_text())
    assert data[0]["task_id"] == "v1"
    assert set(data[0]["captions"]) == set(STYLES)
