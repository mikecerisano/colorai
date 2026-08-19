"""Temporal face tracking and mask propagation for a shot.

The deterministic engine measures skin on a single representative still. That
is a point sample; this module turns it into a *temporally robust* sample and a
*stable mask* over the shot:

* :func:`track_face` follows a face across sampled frames by re-detecting
  (YuNet) and matching boxes (drift-free, no fragile tracker state).
* :func:`temporal_skin_metrics` collapses the track into a median skin
  signature + a stability score (lower = the face's skin is more consistent).
* :func:`stable_skin_mask` is the temporal majority of per-frame skin masks —
  the "propagated mask" that stays put instead of flickering frame to frame.

All images are HxWx3 BGR uint8; boxes are ``(x, y, w, h)``.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from colorai.face import detect_faces, skin_metrics_in_region
from colorai.frames import extract_frame
from colorai.skin import skin_mask


@dataclass(frozen=True)
class FaceTrack:
    """A face followed across sampled frames of a shot."""

    seed_box: tuple[int, int, int, int]
    samples: tuple[tuple[int, tuple[int, int, int, int]], ...]  # (frame_index, box)

    @property
    def tracked_frames(self) -> int:
        return len(self.samples)


def sample_frames(start: int, end: int, n: int = 8) -> list[int]:
    """Evenly spaced, inclusive, de-duplicated frame indices within ``[start, end]``."""
    if end < start:
        return []
    if n <= 1:
        return [start]
    return sorted({round(start + (end - start) * i / (n - 1)) for i in range(n)})


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter_w = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    inter_h = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_box(
    candidates: list[tuple[int, int, int, int]],
    target: tuple[int, int, int, int],
    *,
    min_iou: float = 0.0,
) -> tuple[int, int, int, int] | None:
    """Pick the candidate box that best overlaps ``target`` (or ``None``)."""
    best: tuple[int, int, int, int] | None = None
    best_iou = min_iou
    for box in candidates:
        iou = _iou(box, target)
        if iou > best_iou:
            best_iou, best = iou, box
    return best


def track_face(
    video_path: str | Path,
    start: int,
    end: int,
    seed_box: tuple[int, int, int, int],
    fps: float,
    *,
    samples: int = 8,
    scale: int | None = 480,
    extract: Callable[..., Path] = extract_frame,
    detect: Callable[..., list[tuple[int, int, int, int]]] = detect_faces,
) -> FaceTrack:
    """Follow ``seed_box`` across ``samples`` frames of ``[start, end]``.

    Re-detects faces each frame and matches by IoU to the previous box, so the
    track does not accumulate drift. ``extract``/``detect`` are injectable for
    tests.
    """
    probe_dir = Path(tempfile.mkdtemp(prefix="colorai_track_"))
    try:
        previous = seed_box
        track: list[tuple[int, tuple[int, int, int, int]]] = []
        for frame_index in sample_frames(start, end, samples):
            still = extract(
                video_path, frame_index, probe_dir / f"{frame_index}.png",
                fps=fps, scale=scale,
            )
            image = cv2.imread(str(still), cv2.IMREAD_COLOR)
            if image is None:
                continue
            box = match_box(detect(image), previous)
            if box is None:
                continue
            track.append((frame_index, box))
            previous = box
        return FaceTrack(seed_box, tuple(track))
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)


def stable_skin_mask(masks: list[np.ndarray]) -> np.ndarray:
    """Temporal majority of same-shaped binary skin masks (the propagated mask)."""
    if not masks:
        raise ValueError("no masks to propagate")
    shape = masks[0].shape
    if any(m.shape != shape for m in masks):
        raise ValueError("all masks must share a shape")
    return (np.stack(masks).astype(np.float32).mean(axis=0) >= 0.5).astype(np.uint8)


def temporal_skin_metrics(signatures: list[tuple[float, float, float]]) -> dict:
    """Collapse per-frame skin BGR signatures into a median + stability score.

    ``stability`` is the max per-channel standard deviation; lower is more
    temporally consistent.
    """
    if not signatures:
        raise ValueError("no skin signatures")
    arr = np.asarray(signatures, dtype=np.float64)
    return {
        "median_bgr": [float(v) for v in np.median(arr, axis=0)],
        "stability": float(arr.std(axis=0).max()),
        "samples": len(signatures),
    }


def propagate_shot_mask(
    video_path: str | Path,
    start: int,
    end: int,
    face_index: int,
    fps: float,
    *,
    samples: int = 8,
    scale: int | None = 480,
    seed_frame: int | None = None,
) -> dict:
    """Track a face (by ``face_index``) and return its temporally robust sample.

    The seed face is detected at the *same* ``scale`` used for tracking, so box
    coordinates stay consistent. Returns the track, median skin signature,
    stability, and the propagated (temporal-majority) skin mask.
    """
    probe_dir = Path(tempfile.mkdtemp(prefix="colorai_mask_"))
    try:
        seed_idx = seed_frame if seed_frame is not None else (start + end) // 2
        seed_still = extract_frame(video_path, seed_idx, probe_dir / "seed.png", fps=fps, scale=scale)
        seed_image = cv2.imread(str(seed_still), cv2.IMREAD_COLOR)
        faces = detect_faces(seed_image) if seed_image is not None else []
        if face_index >= len(faces):
            return {"tracked_frames": 0, "error": f"face {face_index} not found"}

        track = track_face(
            video_path, start, end, faces[face_index], fps, samples=samples, scale=scale
        )
        signatures: list[tuple[float, float, float]] = []
        masks: list[np.ndarray] = []
        for frame_index, box in track.samples:
            still = extract_frame(
                video_path, frame_index, probe_dir / f"{frame_index}.png",
                fps=fps, scale=scale,
            )
            image = cv2.imread(str(still), cv2.IMREAD_COLOR)
            if image is None:
                continue
            skin = skin_metrics_in_region(image, box)
            if skin is None:
                continue
            signatures.append(tuple(skin["mean_bgr"]))
            x, y, w, h = box
            masks.append(skin_mask(image[y : y + h, x : x + w]))
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)

    if not signatures:
        return {"tracked_frames": 0}

    metrics = temporal_skin_metrics(signatures)
    seed_w, seed_h = track.seed_box[2], track.seed_box[3]
    resized = [
        cv2.resize(m, (seed_w, seed_h), interpolation=cv2.INTER_NEAREST) for m in masks
    ]
    return {
        "tracked_frames": len(signatures),
        "seed_box": list(track.seed_box),
        "median_bgr": metrics["median_bgr"],
        "stability": metrics["stability"],
        "mask": stable_skin_mask(resized),
    }
