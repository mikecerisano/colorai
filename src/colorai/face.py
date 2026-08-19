"""Face detection and face-region skin sampling.

Uses OpenCV's YuNet DNN face detector (``FaceDetectorYN``) with a bundled
ONNX model (~230 KB), so detection is real, local, deterministic, and needs no
runtime download. It supersedes the color-only heuristic in
:mod:`colorai.skin` for *locating* skin: skin metrics are computed only within
detected face regions, which is where skin-tone QC should be measured.

The detector interface is narrow, so a different local model can be swapped in
later without changing callers — see ``docs/research-notes.md``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from colorai.skin import skin_mask

_MODEL_PATH = Path(__file__).parent / "models" / "face_detection_yunet_2023mar.onnx"
_DEFAULT_SCORE = 0.9


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
    return [
        (int(x), int(y), int(fw), int(fh)) for x, y, fw, fh in faces[:, :4]
    ]


def skin_metrics_in_region(
    image_bgr: np.ndarray, bbox: tuple[int, int, int, int]
) -> dict | None:
    """Skin statistics within an arbitrary ``(x, y, w, h)`` box.

    Returns ``None`` when the box contains no skin pixels, so callers can tell
    "no face" from "face with no measurable skin". ``mean_bgr`` is the mean of
    skin pixels in BGR order (0..255).
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


def face_skin_metrics(image_bgr: np.ndarray) -> list[dict]:
    """Skin metrics for every detected face in the image."""
    return [
        metrics
        for metrics in (
            skin_metrics_in_region(image_bgr, box) for box in detect_faces(image_bgr)
        )
        if metrics is not None
    ]
