import cv2
import numpy as np
import pytest

from stylereel.frames import Frame, extract_frames


@pytest.fixture(scope="module")
def scene_video(tmp_path_factory):
    """3 visually distinct 'scenes', 2s each @ 10fps, 320x240."""
    path = str(tmp_path_factory.mktemp("vid") / "scenes.mp4")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    rng = np.random.default_rng(7)
    for color in ((255, 0, 0), (0, 255, 0), (0, 0, 255)):
        base = np.full((240, 320, 3), color, np.uint8)
        for _ in range(20):
            frame = base.copy()
            # noise so frames aren't identical and sharpness varies
            noise = rng.integers(0, 60, (240, 320, 3), dtype=np.uint8)
            frame = cv2.add(frame, noise)
            w.write(frame)
    w.release()
    return path


def test_extract_frames_scene_video(scene_video):
    frames = extract_frames(scene_video, max_frames=16)
    assert 3 <= len(frames) <= 16
    ts = [f.ts for f in frames]
    assert ts == sorted(ts)
    for f in frames:
        img = cv2.imdecode(np.frombuffer(f.jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert img is not None
        assert max(img.shape[:2]) <= 768


def test_uniform_fallback_when_scenedetect_breaks(scene_video, monkeypatch):
    import stylereel.frames as fr

    def boom(path):
        raise RuntimeError("codec explosion")

    monkeypatch.setattr(fr, "_scene_boundaries", boom)
    frames = extract_frames(scene_video, max_frames=16)
    assert len(frames) == 12  # uniform fallback count
    assert all(isinstance(f, Frame) for f in frames)
