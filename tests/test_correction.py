"""Tests for deterministic correction transforms."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from colorai.correction import (
    apply_correction,
    apply_corrections,
    preview_correction,
    validate_correction,
)
from colorai.project import Correction, ProjectStore, make_representative_frame, make_shots


def _px(r, g, b):
    return np.array([[[r, g, b]]], dtype=np.float32)


def test_exposure():
    out = apply_correction(_px(0.5, 0.5, 0.5), "exposure", {"gain": 2.0})
    assert out[0, 0].tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_offset_positive_and_negative():
    out = apply_correction(_px(0.5, 0.5, 0.5), "offset", {"value": 0.1})
    assert out[0, 0].tolist() == pytest.approx([0.6, 0.6, 0.6])
    out = apply_correction(_px(0.5, 0.5, 0.5), "offset", {"value": -0.2})
    assert out[0, 0].tolist() == pytest.approx([0.3, 0.3, 0.3])


def test_cdl_slope():
    out = apply_correction(
        _px(0.5, 0.5, 0.5), "cdl", {"slope": [2.0, 1.0, 1.0]}
    )
    assert out[0, 0].tolist() == pytest.approx([1.0, 0.5, 0.5])


def test_cdl_power():
    # offset 0.25 with power 2 -> sqrt(0.25) = 0.5.
    out = apply_correction(
        _px(0.0, 0.0, 0.0), "cdl", {"offset": [0.25, 0.25, 0.25], "power": [2.0, 2.0, 2.0]}
    )
    assert out[0, 0].tolist() == pytest.approx([0.5, 0.5, 0.5], abs=1e-3)


def test_rgb_balance():
    out = apply_correction(_px(0.5, 0.5, 0.5), "rgb_balance", {"gain": [2.0, 1.0, 0.5]})
    assert out[0, 0].tolist() == pytest.approx([1.0, 0.5, 0.25])


def test_contrast():
    out = apply_correction(_px(0.25, 0.25, 0.25), "contrast", {"amount": 2.0, "pivot": 0.5})
    assert out[0, 0].tolist() == pytest.approx([0.0, 0.0, 0.0])
    out = apply_correction(_px(0.75, 0.75, 0.75), "contrast", {"amount": 2.0, "pivot": 0.5})
    assert out[0, 0].tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_saturation_zero_makes_gray():
    out = apply_correction(_px(1.0, 0.0, 0.0), "saturation", {"amount": 0.0})
    r, g, b = out[0, 0]
    assert r == pytest.approx(g) == pytest.approx(b)


def test_saturation_identity():
    img = _px(1.0, 0.0, 0.0)
    out = apply_correction(img, "saturation", {"amount": 1.0})
    assert out[0, 0].tolist() == pytest.approx([1.0, 0.0, 0.0], abs=1e-4)


def test_hue_rotate_red_to_green():
    out = apply_correction(_px(1.0, 0.0, 0.0), "hue_rotate", {"degrees": 120.0})
    r, g, b = out[0, 0]
    assert g > 0.9 and r < 0.1 and b < 0.1


def test_apply_corrections_order_and_disabled():
    seq = [("exposure", {"gain": 2.0}), ("offset", {"value": -0.5})]
    out = apply_corrections(_px(0.5, 0.5, 0.5), seq)
    assert out[0, 0].tolist() == pytest.approx([0.5, 0.5, 0.5])

    disabled = Correction(kind="exposure", parameters={"gain": 2.0}, enabled=False)
    out = apply_corrections(_px(0.5, 0.5, 0.5), [disabled])
    assert out[0, 0].tolist() == pytest.approx([0.5, 0.5, 0.5])


@pytest.mark.parametrize(
    ("kind", "params"),
    [
        ("bogus", {}),
        ("cdl", {"power": [1.0, 0.0, 1.0]}),  # power must be > 0
        ("rgb_balance", {"gain": [1.0, 1.0]}),  # wrong length
        ("contrast", {"pivot": 1.5}),  # pivot out of range
        ("saturation", {"amount": -1.0}),  # negative amount
        ("exposure", {"gain": -1.0}),  # negative gain
    ],
)
def test_validate_correction_rejects(kind, params):
    with pytest.raises(ValueError):
        validate_correction(kind, params)


def test_validate_correction_accepts_defaults():
    for kind in ("cdl", "exposure", "offset", "rgb_balance", "contrast", "saturation", "hue_rotate"):
        validate_correction(kind, {})  # defaults are valid


@pytest.mark.parametrize(
    ("kind", "params"),
    [
        ("exposure", {"gain": float("nan")}),
        ("exposure", {"gain": float("inf")}),
        ("offset", {"value": float("nan")}),
        ("rgb_balance", {"gain": [1.0, float("nan"), 1.0]}),
        ("contrast", {"amount": float("nan")}),
        ("contrast", {"pivot": float("inf")}),
        ("saturation", {"amount": float("inf")}),
        ("hue_rotate", {"degrees": float("nan")}),
        ("cdl", {"slope": [1.0, float("nan"), 1.0]}),
    ],
)
def test_validate_correction_rejects_non_finite(kind, params):
    with pytest.raises(ValueError):
        validate_correction(kind, params)


def test_preview_correction(tmp_path):
    store = ProjectStore.create(":memory:")
    project = store.create_project("correction test")
    asset = store.add_asset(
        project.id, source_path="/media/m.mov", frame_rate=25.0
    )
    shot = make_shots(asset, [(0, 24)])[0]
    with store.session() as session:
        session.add(shot)
        session.flush()
        session.refresh(shot)

    still = tmp_path / "still.png"
    cv2.imwrite(str(still), np.full((8, 8, 3), [128, 128, 128], dtype=np.uint8))  # mid-gray BGR
    with store.session() as session:
        session.add(make_representative_frame(shot, 0, image_path=str(still), frame_rate=25.0))
        session.add(Correction(shot_id=shot.id, kind="exposure", parameters={"gain": 2.0}))
        session.commit()

    out = preview_correction(store, shot, tmp_path / "preview.png")
    result = cv2.imread(str(out))
    assert result is not None
    assert result.mean() > 250  # mid-gray * 2 -> clipped to white
