"""Model client: OpenAI-compatible chat with retry, model fallback, and an
optional preferred backend (Gemma on AMD) that fails over to Fireworks.

Backend order for every call:
  1. GEMMA_ENDPOINT (if configured) — self-hosted google/gemma-3-27b-it on AMD
     MI300X via vLLM. Marked down only after repeated transport failures, not on
     a single per-request 4xx.
  2. Fireworks serverless fallback chain.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os

import httpx

from .keybox import get_key

log = logging.getLogger(__name__)

FIREWORKS_BASE = "https://api.fireworks.ai/inference/v1"

# Verified against the live /models endpoint for this account (2026-07-06).
# Only kimi-k2p6 / kimi-k2p5 accept image input on serverless here.
VISION_CHAIN = [
    "accounts/fireworks/models/kimi-k2p6",
    "accounts/fireworks/models/kimi-k2p5",
]
TEXT_CHAIN = [
    "accounts/fireworks/models/kimi-k2p6",
    "accounts/fireworks/models/deepseek-v4-pro",
    "accounts/fireworks/models/glm-5p2",
]
# Text-only, different family from the kimi generator -> judge independence.
JUDGE_CHAIN = [
    "accounts/fireworks/models/deepseek-v4-pro",
    "accounts/fireworks/models/glm-5p2",
]

GEMMA_DOWN_THRESHOLD = 3  # consecutive transport failures before giving up

# Exceptions that mean "this attempt failed, try again/next model" — includes
# IndexError (empty choices) and ValueError (JSONDecodeError) from bad 200s.
RETRIABLE = (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError, ValueError)


class FWError(RuntimeError):
    pass


def _strip_reasoning(text: str) -> str:
    """Belt-and-suspenders: remove <think>…</think> blocks some models still emit
    even with reasoning_effort=none."""
    import re

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # drop an unmatched leading '<think>...' with no closing tag
    if "<think>" in text.lower():
        text = re.split(r"</?think>", text, flags=re.IGNORECASE)[-1]
    return text.strip()


class ModelClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(60, connect=15))
        self._gemma_url = os.environ.get("GEMMA_ENDPOINT", "").rstrip("/")
        self._gemma_model = os.environ.get("GEMMA_MODEL", "google/gemma-3-27b-it")
        self._gemma_transport_failures = 0

    async def close(self) -> None:
        await self._client.aclose()

    @property
    def _gemma_down(self) -> bool:
        return self._gemma_transport_failures >= GEMMA_DOWN_THRESHOLD

    async def _post(self, base_url: str, api_key: str, model: str,
                    messages: list, max_tokens: int, temperature: float,
                    read_timeout: float, connect_timeout: float = 15) -> str:
        # reasoning_effort=none: the serverless models here (Kimi, DeepSeek, GLM)
        # are reasoning models that otherwise inline their chain-of-thought into
        # `content` with no delimiter — this keeps captions clean.
        resp = await self._client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "messages": messages, "max_tokens": max_tokens,
                  "temperature": temperature, "reasoning_effort": "none"},
            timeout=httpx.Timeout(read_timeout, connect=connect_timeout),
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise FWError("empty completion")
        return _strip_reasoning(content)

    async def chat(self, messages: list, *, chain: list[str] | None = None,
                   vision: bool = False, max_tokens: int = 1024,
                   temperature: float = 0.7, use_gemma: bool = True,
                   read_timeout: float = 55) -> str:
        # Preferred backend: self-hosted Gemma on AMD
        if use_gemma and self._gemma_url and not self._gemma_down:
            try:
                out = await self._post(
                    self._gemma_url, os.environ.get("GEMMA_API_KEY", "unused"),
                    self._gemma_model, messages, max_tokens, temperature,
                    read_timeout=read_timeout, connect_timeout=10,
                )
                self._gemma_transport_failures = 0
                return out
            except httpx.HTTPStatusError as exc:
                # endpoint alive but rejected THIS request (payload too large etc.)
                log.warning("Gemma rejected request (%s); using Fireworks for this call",
                            exc.response.status_code)
            except Exception as exc:
                self._gemma_transport_failures += 1
                log.warning("Gemma transport failure %d/%d (%s)",
                            self._gemma_transport_failures, GEMMA_DOWN_THRESHOLD, exc)

        models = chain or (VISION_CHAIN if vision else TEXT_CHAIN)
        key = get_key()
        last: Exception | None = None
        for model in models:
            for attempt in range(2):
                try:
                    return await self._post(FIREWORKS_BASE, key, model, messages,
                                            max_tokens, temperature,
                                            read_timeout=read_timeout)
                except FWError as exc:
                    last = exc
                except RETRIABLE as exc:
                    last = exc
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    if status is not None and 400 <= status < 500 and status != 429:
                        break  # 4xx (except rate limit): don't retry same model
                await asyncio.sleep(0.4 * (2 ** attempt))
        raise FWError(f"all models failed: {last!r}")


def frames_to_content(frames, transcript: str) -> list:
    """Build multimodal content: timestamped frames + optional transcript."""
    content: list = [{"type": "text",
                      "text": f"Video frames in chronological order ({len(frames)} frames):"}]
    for f in frames:
        m, s = divmod(int(f.ts), 60)
        content.append({"type": "text", "text": f"Frame at [{m:02d}:{s:02d}]:"})
        b64 = base64.b64encode(f.jpeg).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    if transcript:
        content.append({"type": "text",
                        "text": f"Audio transcript (timestamped):\n{transcript}"})
    else:
        content.append({"type": "text",
                        "text": "Audio: no intelligible speech (silent, ambient, or music only)."})
    return content
