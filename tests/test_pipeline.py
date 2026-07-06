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

    async def chat(self, messages, *, chain=None, vision=False, max_tokens=1024,
                   temperature=0.7, use_gemma=True, read_timeout=55):
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    async def close(self):
        self.closed = True


class FakeHttp:
    async def aclose(self):
        pass


@pytest.fixture
def patched(monkeypatch, tmp_path):
    """Neutralize IO-heavy pieces; each test overrides what it exercises."""
    async def fake_download(client, url, dest, attempts=3, attempt_timeout=50):
        Path(dest).write_bytes(b"fake")
        return dest

    monkeypatch.setattr(pl, "download", fake_download)
    monkeypatch.setattr(pl, "extract_frames",
                        lambda p, max_frames=16: [Frame(0.0, b"j"), Frame(1.0, b"k")])
    monkeypatch.setattr("stylereel.audio.transcribe", lambda p: "")
    return tmp_path


async def test_full_ladder_happy(patched):
    client = FakeClient("a fine caption")
    caps: dict = {}
    await pl.process_clip(client, FakeHttp(), Task("v1", "http://x"),
                          patched, pl.Deadline(600), caps)
    assert set(caps) == set(STYLES)
    assert all(caps[s] for s in STYLES)


async def test_describe_dead_single_shot_rung(patched):
    class DescribeFails(FakeClient):
        async def chat(self, messages, **kwargs):
            first = messages[0]
            if isinstance(first.get("content"), str) and "INCONGRUITY" in first["content"]:
                raise RuntimeError("vlm down")
            return "single-shot caption"

    caps: dict = {}
    await pl.process_clip(DescribeFails(), FakeHttp(), Task("v1", "http://x"),
                          patched, pl.Deadline(600), caps)
    assert set(caps) == set(STYLES)
    assert all(c == "single-shot caption" for c in caps.values())


async def test_download_dead_leaves_captions_empty(patched, monkeypatch):
    async def bad_download(client, url, dest, attempts=3, attempt_timeout=50):
        raise RuntimeError("404")

    monkeypatch.setattr(pl, "download", bad_download)
    caps: dict = {}
    await pl.process_clip(FakeClient(), FakeHttp(), Task("v1", "http://x"),
                          patched, pl.Deadline(600), caps)
    assert caps == {}


async def test_partial_captions_survive_shared_dict(patched):
    """Captions banked in the shared dict persist even if processing is cut short."""
    class SlowAfterTwo(FakeClient):
        def __init__(self):
            super().__init__()
            self.n = 0

        async def chat(self, messages, **kwargs):
            first = messages[0]
            if isinstance(first.get("content"), str) and "INCONGRUITY" in first["content"]:
                return "SUBJECTS: a thing"
            self.n += 1
            if self.n > 6:  # first two styles' worth of calls succeed
                raise RuntimeError("rate limited")
            return "banked caption"

    caps: dict = {}
    await pl.process_clip(SlowAfterTwo(), FakeHttp(), Task("v1", "http://x"),
                          patched, pl.Deadline(600), caps)
    assert any(v == "banked caption" for v in caps.values())  # partials kept


async def test_run_batch_checkpoint_called(patched, monkeypatch):
    async def bad_download(client, url, dest, attempts=3, attempt_timeout=50):
        raise RuntimeError("network gone")

    monkeypatch.setattr(pl, "download", bad_download)
    seen = []
    tasks = [Task("v1", "u1"), Task("v2", "u2")]
    results = await pl.run_batch(tasks, patched, budget_s=600,
                                 checkpoint=lambda r: seen.append(len(r)))
    assert isinstance(results, dict)
    assert len(seen) >= 3  # skeleton + one per clip


def test_main_writes_complete_output_when_everything_fails(tmp_path, monkeypatch):
    tasks_file = tmp_path / "tasks.json"
    tasks_file.write_text(json.dumps(
        [{"task_id": "v1", "video_url": "http://nope.invalid/x.mp4",
          "styles": list(STYLES)}]))
    out_file = tmp_path / "results.json"
    monkeypatch.setenv("FIREWORKS_API_KEY", "k")
    monkeypatch.setenv("STYLEREEL_HARD_EXIT", "0")
    import stylereel.main as m
    monkeypatch.setattr(m, "INPUT", str(tasks_file))
    monkeypatch.setattr(m, "OUTPUT", str(out_file))
    monkeypatch.setattr(m, "BUDGET_S", 5.0)
    rc = m.main()
    assert rc == 0
    data = json.loads(out_file.read_text())
    assert data[0]["task_id"] == "v1"
    assert set(data[0]["captions"]) == set(STYLES)
