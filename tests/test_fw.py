import httpx
import pytest
import respx

from stylereel.fw import FIREWORKS_BASE, FWError, ModelClient, frames_to_content
from stylereel.frames import Frame
from stylereel.keybox import _deobfuscate, obfuscate

URL = f"{FIREWORKS_BASE}/chat/completions"


def _ok(text="hello"):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.delenv("GEMMA_ENDPOINT", raising=False)


@respx.mock
async def test_happy_path():
    respx.post(URL).mock(return_value=_ok("caption text"))
    client = ModelClient()
    out = await client.chat([{"role": "user", "content": "hi"}])
    assert out == "caption text"
    await client.close()


@respx.mock
async def test_fallback_on_500():
    calls = []

    def handler(request):
        calls.append(request.read())
        if len(calls) <= 2:  # first model, 2 attempts -> 500
            return httpx.Response(500, json={"error": "boom"})
        return _ok("from fallback")

    respx.post(URL).mock(side_effect=handler)
    client = ModelClient()
    out = await client.chat([{"role": "user", "content": "hi"}])
    assert out == "from fallback"
    assert len(calls) == 3
    await client.close()


@respx.mock
async def test_all_fail_raises():
    respx.post(URL).mock(return_value=httpx.Response(500, json={}))
    client = ModelClient()
    with pytest.raises(FWError):
        await client.chat([{"role": "user", "content": "hi"}], chain=["m1"])
    await client.close()


@respx.mock
async def test_gemma_preferred_then_failover(monkeypatch):
    monkeypatch.setenv("GEMMA_ENDPOINT", "http://gemma.local/v1")
    respx.post("http://gemma.local/v1/chat/completions").mock(
        side_effect=httpx.ConnectError("down"))
    respx.post(URL).mock(return_value=_ok("fw answer"))
    client = ModelClient()
    out = await client.chat([{"role": "user", "content": "hi"}])
    assert out == "fw answer"
    # gemma marked down: second call goes straight to fireworks
    out2 = await client.chat([{"role": "user", "content": "hi"}])
    assert out2 == "fw answer"
    assert client._gemma_down is True
    await client.close()


def test_keybox_roundtrip():
    parts = obfuscate("fw-secret-123", n_parts=3)
    assert len(parts) == 3
    assert "fw-secret-123" not in "".join(parts)
    assert _deobfuscate(parts) == "fw-secret-123"


def test_frames_to_content():
    frames = [Frame(ts=1.0, jpeg=b"AA"), Frame(ts=65.0, jpeg=b"BB")]
    content = frames_to_content(frames, "[00:01] hi there")
    images = [c for c in content if c["type"] == "image_url"]
    assert len(images) == 2
    assert any("[01:05]" in c.get("text", "") for c in content)
    assert any("hi there" in c.get("text", "") for c in content)
