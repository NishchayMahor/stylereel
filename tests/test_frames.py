import subprocess

import cv2
import numpy as np
import pytest

from stylereel.frames import Frame, extract_frames


@pytest.fixture(scope="module")
def scene_video(tmp_path_factory):
    """3 visually distinct 'scenes', 2s each @ 10fps, 320x240, via ffmpeg-compatible writer."""
    path = str(tmp_path_factory.mktemp("vid") / "scenes.mp4")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    rng = np.random.default_rng(7)
    for color in ((255, 0, 0), (0, 255, 0), (0, 0, 255)):
        base = np.full((240, 320, 3), color, np.uint8)
        for _ in range(20):
            frame = base.copy()
            noise = rng.integers(0, 60, (240, 320, 3), dtype=np.uint8)
            frame = cv2.add(frame, noise)
            w.write(frame)
    w.release()
    return path


def test_extract_frames_covers_scenes(scene_video):
    frames = extract_frames(scene_video, max_frames=16)
    assert 4 <= len(frames) <= 16
    ts = [f.ts for f in frames]
    assert ts == sorted(ts)
    # all three color scenes should be represented (diversity selection)
    dominant = set()
    for f in frames:
        img = cv2.imdecode(np.frombuffer(f.jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert img is not None
        assert max(img.shape[:2]) <= 768
        dominant.add(int(np.argmax(img.reshape(-1, 3).mean(axis=0))))
    assert len(dominant) == 3


def test_cv2_fallback_when_ffmpeg_breaks(scene_video, monkeypatch):
    import stylereel.frames as fr

    def boom(*args, **kwargs):
        raise subprocess.SubprocessError("ffmpeg exploded")

    monkeypatch.setattr(fr, "_ffmpeg_sample", boom)
    frames = extract_frames(scene_video, max_frames=16)
    assert len(frames) >= 1
    assert all(isinstance(f, Frame) for f in frames)


def test_everything_broken_returns_empty(tmp_path):
    bad = tmp_path / "not_a_video.mp4"
    bad.write_bytes(b"garbage")
    assert extract_frames(str(bad)) == []
