"""Audio leg: extract audio, gate on VAD, transcribe with faster-whisper.

Returns "" whenever there is nothing useful (silent clip, music only, any error)
— the pipeline treats transcript as optional context.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

_whisper_model = None

MIN_SPEECH_SECONDS = 1.5


def _extract_wav(video_path: str, wav_path: str) -> bool:
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000",
         "-f", "wav", wav_path],
        capture_output=True, timeout=60,
    )
    return proc.returncode == 0 and Path(wav_path).stat().st_size > 1000


def _speech_seconds(wav_path: str) -> float:
    """Silero VAD via faster-whisper's bundled implementation."""
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    audio = decode_audio(wav_path, sampling_rate=16000)
    ts = get_speech_timestamps(audio, VadOptions(min_speech_duration_ms=250))
    return sum((t["end"] - t["start"]) for t in ts) / 16000.0


def _get_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model


def transcribe(video_path: str) -> str:
    """Timestamped transcript lines '[MM:SS] text', or '' if no usable speech."""
    try:
        with tempfile.TemporaryDirectory() as td:
            wav = str(Path(td) / "a.wav")
            if not _extract_wav(video_path, wav):
                return ""
            if _speech_seconds(wav) < MIN_SPEECH_SECONDS:
                return ""
            segments, _info = _get_model().transcribe(wav, language="en", vad_filter=True)
            lines = []
            for seg in segments:
                text = seg.text.strip()
                if text:
                    m, s = divmod(int(seg.start), 60)
                    lines.append(f"[{m:02d}:{s:02d}] {text}")
            return "\n".join(lines)
    except Exception as exc:
        log.warning("transcription failed: %s", exc)
        return ""
