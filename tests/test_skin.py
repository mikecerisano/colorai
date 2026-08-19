"""Tests for the experimental skin-segmentation heuristic."""

from __future__ import annotations

import numpy as np
import pytest

from colorai.skin import skin_coverage, skin_mask

# BGR values verified against the YCrCb thresholds in skin.py:
# skin -> YCrCb [158, 144, 118], non-skin -> [176, 117, 142].
SKIN = (140, 150, 180)
NON_SKIN = (200, 180, 160)


def _solid(bgr, size=(16, 16)):
    return np.full((size[0], size[1], 3), bgr, dtype=np.uint8)


def test_skin_detected():
    assert skin_coverage(_solid(SKIN)) == 1.0


def test_non_skin_rejected():
    assert skin_coverage(_solid(NON_SKIN)) == 0.0


@pytest.mark.parametrize(
    "bgr", [(0, 0, 0), (255, 255, 255), (128, 128, 128)]
)
def test_neutral_colors_not_skin(bgr):
    assert skin_coverage(_solid(bgr)) == 0.0


def test_mixed_image_coverage():
    top = _solid(SKIN, size=(8, 16))
    bottom = _solid((0, 0, 0), size=(8, 16))
    img = np.vstack([top, bottom])
    assert skin_coverage(img) == pytest.approx(0.5)


def test_mask_shape_and_dtype():
    mask = skin_mask(_solid(SKIN))
    assert mask.shape == (16, 16)
    assert mask.dtype == np.bool_
    assert mask.all()


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        skin_mask(np.zeros((16, 16), dtype=np.uint8))
