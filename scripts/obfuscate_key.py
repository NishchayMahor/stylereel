#!/usr/bin/env python3
"""Bake an obfuscated Fireworks key into keybox.py before the final image build.

Usage: python scripts/obfuscate_key.py fw_xxxxxxxx
Rewrites the `_PARTS: list[str] = [...]` line in src/stylereel/keybox.py.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from stylereel.keybox import _deobfuscate, obfuscate  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: obfuscate_key.py <fireworks-api-key>")
    key = sys.argv[1]
    parts = obfuscate(key)
    assert _deobfuscate(parts) == key
    keybox = Path(__file__).resolve().parent.parent / "src/stylereel/keybox.py"
    src = keybox.read_text()
    new = re.sub(r"_PARTS: list\[str\] = \[.*?\]",
                 f"_PARTS: list[str] = {parts!r}", src, count=1, flags=re.DOTALL)
    keybox.write_text(new)
    print(f"baked {len(parts)} parts into {keybox}")


if __name__ == "__main__":
    main()
