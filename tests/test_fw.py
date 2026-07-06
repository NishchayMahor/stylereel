import httpx
import pytest
import respx

from stylereel.frames import Frame
from stylereel.fw import FIREWORKS_BASE, FWError, ModelClient, frames_to_content
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
        calls.append(1)
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
async def test_empty_choices_falls_through_chain():
    """200 with empty choices must NOT abort the chain (IndexError path)."""
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) <= 2:
            return httpx.Response(200, json={"choices": []})
        return _ok("recovered")

    respx.post(URL).mock(side_effect=handler)
    client = ModelClient()
    out = await client.chat([{"role": "user", "content": "hi"}])
    assert out == "recovered"
    await client.close()


@respx.mock
async def test_non_json_body_falls_through_chain():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) <= 2:
            return httpx.Response(200, text="<html>gateway</html>")
        return _ok("recovered")

    respx.post(URL).mock(side_effect=handler)
    client = ModelClient()
    out = await client.chat([{"role": "user", "content": "hi"}])
    assert out == "recovered"
    await client.close()


@respx.mock
async def test_all_fail_raises():
    respx.post(URL).mock(return_value=httpx.Response(500, json={}))
    client = ModelClient()
    with pytest.raises(FWError):
        await client.chat([{"role": "user", "content": "hi"}], chain=["m1"])
    await client.close()


@respx.mock
async def test_gemma_transport_failures_accumulate(monkeypatch):
    monkeypatch.setenv("GEMMA_ENDPOINT", "http://gemma.local/v1")
    respx.post("http://gemma.local/v1/chat/completions").mock(
        side_effect=httpx.ConnectError("down"))
    respx.post(URL).mock(return_value=_ok("fw answer"))
    client = ModelClient()
    for _ in range(3):
        assert await client.chat([{"role": "user", "content": "hi"}]) == "fw answer"
    assert client._gemma_down is True  # 3 transport failures -> down
    # subsequent calls skip gemma entirely
    assert await client.chat([{"role": "user", "content": "hi"}]) == "fw answer"
    await client.close()


@respx.mock
async def test_gemma_4xx_does_not_mark_down(monkeypatch):
    monkeypatch.setenv("GEMMA_ENDPOINT", "http://gemma.local/v1")
    gemma_route = respx.post("http://gemma.local/v1/chat/completions")
    gemma_route.mock(return_value=httpx.Response(400, json={"error": "too large"}))
    respx.post(URL).mock(return_value=_ok("fw answer"))
    client = ModelClient()
    assert await client.chat([{"role": "user", "content": "hi"}]) == "fw answer"
    assert client._gemma_down is False  # per-request rejection, endpoint healthy
    # next call tries gemma again
    gemma_route.mock(return_value=_ok("gemma answer"))
    assert await client.chat([{"role": "user", "content": "hi"}]) == "gemma answer"
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
