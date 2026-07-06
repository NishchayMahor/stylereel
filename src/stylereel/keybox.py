"""API key handling.

Priority: FIREWORKS_API_KEY env var (local dev) -> baked obfuscated parts (judged
container; Track 2 injects no env). Obfuscation is XOR+base64 split into parts —
enough to defeat registry scrapers scanning for key patterns; the key itself has
a capped credit balance and is rotated before final submission.

Generate parts with: python scripts/obfuscate_key.py <key>
"""

from __future__ import annotations

import base64
import os

_XOR = 0x5A

# Filled by scripts/obfuscate_key.py before the final image build.
_PARTS: list[str] = []


def _deobfuscate(parts: list[str]) -> str:
    blob = base64.b64decode("".join(parts))
    return bytes(b ^ _XOR for b in blob).decode()


def obfuscate(key: str, n_parts: int = 3) -> list[str]:
    blob = bytes(b ^ _XOR for b in key.encode())
    b64 = base64.b64encode(blob).decode()
    step = -(-len(b64) // n_parts)
    return [b64[i:i + step] for i in range(0, len(b64), step)]


def get_key() -> str:
    env = os.environ.get("FIREWORKS_API_KEY")
    if env:
        return env
    if _PARTS:
        return _deobfuscate(_PARTS)
    raise RuntimeError("No Fireworks API key available (env FIREWORKS_API_KEY unset, no baked key)")
