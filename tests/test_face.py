"""Tests for face detection and face-region skin sampling."""

from __future__ import annotations

import numpy as np
import pytest

from colorai.face import detect_faces, face_skin_metrics, skin_metrics_in_region

SKIN = (140, 150, 180)  # BGR, verified as skin by the YCrCb heuristic


def test_skin_metrics_in_skin_region():
    img = np.full((40, 40, 3), SKIN, dtype=np.uint8)
    metrics = skin_metrics_in_region(img, (0, 0, 40, 40))
    assert metrics is not None
    assert metrics["skin_coverage"] == pytest.approx(1.0)
    assert metrics["mean_bgr"] == pytest.approx(list(SKIN), abs=2)


def test_skin_metrics_returns_none_for_no_skin():
    img = np.full((40, 40, 3), (0, 0, 0), dtype=np.uint8)
    assert skin_metrics_in_region(img, (0, 0, 40, 40)) is None


def test_skin_metrics_empty_region():
    img = np.full((4, 4, 3), SKIN, dtype=np.uint8)
    assert skin_metrics_in_region(img, (10, 10, 4, 4)) is None


def test_detect_faces_on_blank_is_empty():
    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    assert detect_faces(blank) == []


def test_face_skin_metrics_on_blank_is_empty():
    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    assert face_skin_metrics(blank) == []


def test_face_model_is_bundled():
    from pathlib import Path

    from colorai import face

    model = Path(face.__file__).parent / "models" / "face_detection_yunet_2023mar.onnx"
    assert model.exists()
    assert model.stat().st_size > 100_000
