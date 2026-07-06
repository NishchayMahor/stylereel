# StyleReel — Track 2 video captioning agent
# Judging VM runs linux/amd64. Build with:
#   docker buildx build --platform linux/amd64 -t ghcr.io/<user>/stylereel:latest --push .
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Layer-cache deps separately from source
COPY pyproject.toml README.md* ./
COPY src ./src
RUN pip install --no-cache-dir .

# Bake model weights into the image: the judging environment must never
# download at runtime (60s readiness rule; network may be restricted).
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')" \
    && python - <<'EOF'
# warm silero-vad (bundled with faster-whisper >=1.0, no download needed) and
# fail the build if the import chain is broken
from faster_whisper.vad import VadOptions, get_speech_timestamps
import numpy as np
get_speech_timestamps(np.zeros(16000, dtype=np.float32), VadOptions())
print("vad ok")
EOF

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "stylereel.main"]
