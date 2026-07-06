"""Smart frame extraction: scene detection + sharpness-based keyframe selection.

VLMs get <=16 frames; picking the right ones matters more than picking many.
Strategy: PySceneDetect content cuts -> sharpest frame near each scene middle ->
pad with uniform samples; hard fallback to pure uniform sampling on any error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)

MAX_LONG_SIDE = 768
JPEG_QUALITY = 82


@dataclass
class Frame:
    ts: float  # seconds
    jpeg: bytes


def _encode(img: np.ndarray) -> bytes:
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side > MAX_LONG_SIDE:
        scale = MAX_LONG_SIDE / long_side
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def _sharpness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _grab(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    return img if ok else None


def _scene_boundaries(video_path: str) -> list[tuple[int, int]]:
    from scenedetect import ContentDetector, detect

    scenes = detect(video_path, ContentDetector(threshold=27.0))
    return [(s.get_frames(), e.get_frames()) for s, e in scenes]


def _uniform_indices(total: int, n: int) -> list[int]:
    if total <= 0:
        return []
    n = min(n, total)
    return sorted({int((i + 0.5) * total / n) for i in range(n)})


def extract_frames(video_path: str, max_frames: int = 16) -> list[Frame]:
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            fps = 25.0

        indices: list[int] = []
        try:
            scenes = _scene_boundaries(video_path)
            if len(scenes) >= 2:
                per_scene = max(1, max_frames // len(scenes)) if len(scenes) <= max_frames else 1
                for start, end in scenes[:max_frames]:
                    span = max(end - start, 1)
                    # candidates around scene middle; pick sharpest
                    mid = start + span // 2
                    cands = [c for c in (mid, start + span // 4, start + 3 * span // 4) if start <= c < end]
                    best, best_sharp = None, -1.0
                    for c in cands[: max(per_scene * 2, 2)]:
                        img = _grab(cap, c)
                        if img is None:
                            continue
                        s = _sharpness(img)
                        if s > best_sharp:
                            best, best_sharp = c, s
                    if best is not None:
                        indices.append(best)
        except Exception as exc:  # scenedetect can fail on odd codecs — never fatal
            log.warning("scene detection failed (%s); using uniform sampling", exc)

        if len(indices) < 4:
            indices = _uniform_indices(total, 12)
        elif len(indices) < max_frames:
            extra = _uniform_indices(total, max_frames - len(indices))
            indices = sorted(set(indices) | set(extra))
        indices = sorted(set(indices))[:max_frames]

        frames: list[Frame] = []
        for idx in indices:
            img = _grab(cap, idx)
            if img is None:
                continue
            frames.append(Frame(ts=idx / fps, jpeg=_encode(img)))
        return frames
    finally:
        cap.release()
