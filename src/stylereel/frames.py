"""Smart frame extraction, ffmpeg-first.

The judging clips are UHD (2.5K-4K). Full-decode scene detection burns minutes
per clip, so instead: ffmpeg samples N downscaled candidate thumbnails in a
single fast pass (fps filter + scale), then we pick a diverse, sharp subset by
HSV-histogram greedy max-min diversity + Laplacian sharpness. No dependence on
container frame-count metadata; VFR/fragmented MP4s work.

Fallback ladder: ffmpeg sampling -> cv2 sequential-read sampling.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

MAX_LONG_SIDE = 768
JPEG_QUALITY = 82
CANDIDATES = 28  # sampled by ffmpeg, then reduced to max_frames


@dataclass
class Frame:
    ts: float  # seconds
    jpeg: bytes


def _duration_s(video_path: str) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, timeout=15, text=True,
        )
        return max(float(out.stdout.strip()), 0.5)
    except Exception:
        return 60.0  # sane default for 30s-2min clips


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
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _histogram(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def _ffmpeg_sample(video_path: str, n: int, duration: float, outdir: str) -> list[tuple[float, str]]:
    """One fast pass: n downscaled thumbnails, evenly spaced. Returns (ts, path)."""
    fps = n / duration
    pattern = str(Path(outdir) / "f%04d.jpg")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-vf", f"fps={fps:.6f},scale='if(gt(iw,ih),{MAX_LONG_SIDE},-2)':'if(gt(iw,ih),-2,{MAX_LONG_SIDE})'",
         "-q:v", "3", pattern],
        capture_output=True, timeout=120, check=True,
    )
    files = sorted(Path(outdir).glob("f*.jpg"))
    step = duration / max(len(files), 1)
    return [(round((i + 0.5) * step, 2), str(p)) for i, p in enumerate(files)]


def _select_diverse(imgs: list[tuple[float, np.ndarray]], k: int) -> list[tuple[float, np.ndarray]]:
    """Greedy max-min diversity on HSV histograms, sharpness-weighted seed.

    Keeps temporal order in the returned list.
    """
    if len(imgs) <= k:
        return imgs
    hists = [_histogram(im) for _, im in imgs]
    sharps = [_sharpness(im) for _, im in imgs]
    chosen = [int(np.argmax(sharps))]
    while len(chosen) < k:
        best_i, best_d = -1, -1.0
        for i in range(len(imgs)):
            if i in chosen:
                continue
            d = min(float(cv2.compareHist(hists[i], hists[j], cv2.HISTCMP_BHATTACHARYYA))
                    for j in chosen)
            # slight bonus for sharpness so we don't pick diverse-but-blurry
            d += 0.05 * (sharps[i] / (max(sharps) + 1e-6))
            if d > best_d:
                best_i, best_d = i, d
        chosen.append(best_i)
    return [imgs[i] for i in sorted(chosen)]


def _cv2_fallback(video_path: str, n: int) -> list[Frame]:
    """Sequential read (no seeks, no frame-count dependence), sample every Nth."""
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps > 240:
            fps = 25.0
        frames: list[Frame] = []
        stride = int(fps * 2)  # one candidate every ~2s
        i = 0
        kept: list[tuple[float, np.ndarray]] = []
        while True:
            ok = cap.grab()
            if not ok:
                break
            if i % max(stride, 1) == 0:
                ok, img = cap.retrieve()
                if ok and img is not None:
                    kept.append((i / fps, img))
                if len(kept) >= 60:  # cap work on very long inputs
                    break
            i += 1
        kept = _select_diverse(kept, n)
        for ts, img in kept:
            frames.append(Frame(ts=ts, jpeg=_encode(img)))
        return frames
    finally:
        cap.release()


def extract_frames(video_path: str, max_frames: int = 16) -> list[Frame]:
    duration = _duration_s(video_path)
    try:
        with tempfile.TemporaryDirectory() as td:
            sampled = _ffmpeg_sample(video_path, CANDIDATES, duration, td)
            imgs: list[tuple[float, np.ndarray]] = []
            for ts, p in sampled:
                img = cv2.imread(p)
                if img is not None:
                    imgs.append((ts, img))
            if len(imgs) >= 4:
                picked = _select_diverse(imgs, max_frames)
                return [Frame(ts=ts, jpeg=_encode(im)) for ts, im in picked]
            log.warning("ffmpeg sampling produced %d frames; falling back to cv2", len(imgs))
    except Exception as exc:
        log.warning("ffmpeg sampling failed (%s); falling back to cv2", exc)
    try:
        return _cv2_fallback(video_path, max_frames)
    except Exception as exc:
        log.error("cv2 fallback failed too: %s", exc)
        return []
