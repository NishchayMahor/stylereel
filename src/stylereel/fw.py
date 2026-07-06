"""Model client: OpenAI-compatible chat with retry, model fallback, and an
optional preferred backend (Gemma on AMD) that fails over to Fireworks.

Backend order for every call:
  1. GEMMA_ENDPOINT (if configured) — self-hosted google/gemma-3-27b-it on AMD
     MI300X via vLLM; short connect timeout so an unreachable endpoint costs ~10s
     once (then it is marked down for the rest of the run).
  2. Fireworks serverless fallback chain.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os

import httpx

from .keybox import get_key

log = logging.getLogger(__name__)

FIREWORKS_BASE = "https://api.fireworks.ai/inference/v1"

VISION_CHAIN = [
    "accounts/fireworks/models/qwen3p7-plus",
    "accounts/fireworks/models/minimax-m3",
    "accounts/fireworks/models/kimi-k2p6",
]
TEXT_CHAIN = [
    "accounts/fireworks/models/kimi-k2p6",
    "accounts/fireworks/models/qwen3p7-plus",
    "accounts/fireworks/models/minimax-m3",
]
JUDGE_CHAIN = [
    "accounts/fireworks/models/deepseek-v4-flash",
    "accounts/fireworks/models/qwen3p7-plus",
]


class FWError(RuntimeError):
    pass


class ModelClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(90, connect=15))
        self._gemma_url = os.environ.get("GEMMA_ENDPOINT", "").rstrip("/")
        self._gemma_model = os.environ.get("GEMMA_MODEL", "google/gemma-3-27b-it")
        self._gemma_down = False

    async def close(self) -> None:
        await self._client.aclose()

    async def _post(self, base_url: str, api_key: str, model: str,
                    messages: list, max_tokens: int, temperature: float,
                    connect_timeout: float | None = None) -> str:
        timeout = httpx.Timeout(90, connect=connect_timeout or 15)
        resp = await self._client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "messages": messages,
                  "max_tokens": max_tokens, "temperature": temperature},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise FWError("empty completion")
        return content

    async def chat(self, messages: list, *, chain: list[str] | None = None,
                   vision: bool = False, max_tokens: int = 1024,
                   temperature: float = 0.7, use_gemma: bool = True) -> str:
        # Preferred backend: self-hosted Gemma on AMD
        if use_gemma and self._gemma_url and not self._gemma_down:
            try:
                return await self._post(
                    self._gemma_url, os.environ.get("GEMMA_API_KEY", "unused"),
                    self._gemma_model, messages, max_tokens, temperature,
                    connect_timeout=10,
                )
            except Exception as exc:
                log.warning("Gemma endpoint failed (%s); failing over to Fireworks", exc)
                self._gemma_down = True

        models = chain or (VISION_CHAIN if vision else TEXT_CHAIN)
        key = get_key()
        last: Exception | None = None
        for model in models:
            for attempt in range(2):
                try:
                    return await self._post(FIREWORKS_BASE, key, model,
                                            messages, max_tokens, temperature)
                except (httpx.HTTPStatusError, httpx.TransportError, FWError, KeyError) as exc:
                    last = exc
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    if status is not None and 400 <= status < 500 and status != 429:
                        break  # 4xx (except rate limit): don't retry same model
                    await asyncio.sleep(0.5 * (2 ** attempt))
        raise FWError(f"all models failed: {last}")


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
