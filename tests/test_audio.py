import subprocess

import pytest

from stylereel.audio import transcribe


@pytest.fixture(scope="module")
def silent_video(tmp_path_factory):
    """2s video with a silent audio track."""
    path = str(tmp_path_factory.mktemp("aud") / "silent.mp4")
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", "color=c=gray:s=160x120:d=2",
         "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
         "-shortest", "-c:v", "mpeg4", "-c:a", "aac", path],
        check=True, capture_output=True,
    )
    return path


def test_silent_video_returns_empty(silent_video):
    assert transcribe(silent_video) == ""


def test_missing_file_returns_empty():
    assert transcribe("/nonexistent/nope.mp4") == ""
