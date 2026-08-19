"""Face detection and face-region skin sampling.

Two detectors sit behind one narrow interface:

* **YuNet** (OpenCV ``FaceDetectorYN``, bundled ONNX) — fast, always
  available, gives a face bounding box.
* **MediaPipe FaceMesh** (optional, 468 landmarks) — precise facial geometry
  used to sample skin from the forehead and cheeks while avoiding eyes/lips/
  hair. MediaPipe is a heavier optional dependency (``colorai[face]``); when it
  is not installed, sampling falls back to the YuNet bounding box.

Skin metrics are computed only within detected face regions, which is where
skin-tone QC should be measured. All images are HxWx3 BGR uint8.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from colorai.skin import skin_mask

_MODEL_PATH = Path(__file__).parent / "models" / "face_detection_yunet_2023mar.onnx"
_DEFAULT_SCORE = 0.9

# FaceMesh landmark indices for skin-safe regions: forehead and cheeks.
_FOREHEAD = (10, 108, 151, 337, 299, 333, 297, 301, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0)
_CHEEKS = (234, 127, 162, 21, 54, 103, 67, 109, 116, 143, 156, 70, 63, 105, 66, 107)

_face_mesh = None


def detect_faces(
    image_bgr: np.ndarray, *, score_threshold: float = _DEFAULT_SCORE
) -> list[tuple[int, int, int, int]]:
    """Return detected faces as ``(x, y, width, height)`` boxes (empty if none)."""
    if not _MODEL_PATH.exists() or image_bgr.size == 0:
        return []
    height, width = image_bgr.shape[:2]
    detector = cv2.FaceDetectorYN.create(
        str(_MODEL_PATH), "", (width, height), score_threshold, 0.3, 5000
    )
    detector.setInputSize((width, height))
    _, faces = detector.detect(image_bgr)
    if faces is None:
        return []
    return [(int(x), int(y), int(fw), int(fh)) for x, y, fw, fh in faces[:, :4]]


def _get_face_mesh():
    """Lazily construct the MediaPipe FaceMesh, or ``None`` if unavailable."""
    global _face_mesh
    if _face_mesh is None:
        try:
            os.environ.setdefault("GLOG_minloglevel", "2")
            import mediapipe as mp

            _face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=10,
                min_detection_confidence=0.5,
                refine_landmarks=False,
            )
        except Exception:  # mediapipe not installed or failed to init
            _face_mesh = False
    return _face_mesh if _face_mesh is not False else None


def face_landmarks(image_bgr: np.ndarray) -> list[np.ndarray]:
    """Return per-face ``Nx3`` normalized landmarks ``(x, y, z)`` (empty if none)."""
    mesh = _get_face_mesh()
    if mesh is None or image_bgr.size == 0:
        return []
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = mesh.process(rgb)
    if not result.multi_face_landmarks:
        return []
    return [
        np.array([[lm.x, lm.y, lm.z] for lm in face.landmark], dtype=np.float64)
        for face in result.multi_face_landmarks
    ]


def skin_metrics_in_region(
    image_bgr: np.ndarray, bbox: tuple[int, int, int, int]
) -> dict | None:
    """Skin statistics within an arbitrary ``(x, y, w, h)`` box.

    Returns ``None`` when the box contains no skin pixels. ``mean_bgr`` is the
    mean of skin pixels in BGR order (0..255).
    """
    x, y, w, h = (int(v) for v in bbox)
    region = image_bgr[y : y + h, x : x + w]
    if region.size == 0:
        return None
    mask = skin_mask(region)
    count = int(mask.sum())
    if count == 0:
        return None
    skin_pixels = region[mask]
    return {
        "bbox": [x, y, w, h],
        "skin_coverage": float(mask.mean()),
        "mean_bgr": [float(v) for v in skin_pixels.mean(axis=0)],
    }


def skin_sample_from_landmarks(
    image_bgr: np.ndarray, landmarks: np.ndarray, *, radius: int = 3
) -> dict | None:
    """Sample skin color around forehead/cheek landmarks (precise skin tone).

    ``landmarks`` is an ``Nx3`` array of normalized ``(x, y, z)`` coordinates
    from MediaPipe FaceMesh. Returns ``None`` if no skin pixels are found near
    the sampled landmarks.
    """
    height, width = image_bgr.shape[:2]
    full_mask = skin_mask(image_bgr)
    samples: list[np.ndarray] = []
    for idx in _FOREHEAD + _CHEEKS:
        if idx >= len(landmarks):
            continue
        px = int(round(landmarks[idx, 0] * width))
        py = int(round(landmarks[idx, 1] * height))
        x0, x1 = max(0, px - radius), min(width, px + radius + 1)
        y0, y1 = max(0, py - radius), min(height, py + radius + 1)
        patch_mask = full_mask[y0:y1, x0:x1]
        patch = image_bgr[y0:y1, x0:x1]
        skin = patch[patch_mask]
        if skin.size:
            samples.append(skin)
    if not samples:
        return None
    all_skin = np.concatenate(samples, axis=0)
    return {
        "mean_bgr": [float(v) for v in all_skin.mean(axis=0)],
        "sample_pixels": int(len(all_skin)),
    }


def face_skin_metrics(image_bgr: np.ndarray) -> list[dict]:
    """Skin metrics for every detected face.

    Prefers MediaPipe landmark sampling when available; falls back to YuNet
    bounding-box sampling otherwise.
    """
    landmarks_list = face_landmarks(image_bgr)
    if landmarks_list:
        metrics = [
            skin_sample_from_landmarks(image_bgr, landmarks)
            for landmarks in landmarks_list
        ]
        return [m for m in metrics if m is not None]
    return [
        m
        for m in (
            skin_metrics_in_region(image_bgr, box) for box in detect_faces(image_bgr)
        )
        if m is not None
    ]
