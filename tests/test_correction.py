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
from colorai.color import bt709_to_linear, linear_to_bt709
from colorai.project import Correction, ProjectStore, make_representative_frame, make_shots


def _px(r, g, b):
    return np.array([[[r, g, b]]], dtype=np.float32)


def test_exposure_is_linear_gain():
    out = apply_correction(_px(0.5, 0.5, 0.5), "exposure", {"gain": 2.0})
    expected = float(linear_to_bt709(bt709_to_linear(0.5) * 2.0))
    assert out[0, 0].tolist() == pytest.approx([expected] * 3)


def test_offset_is_linear_lift():
    out = apply_correction(_px(0.5, 0.5, 0.5), "offset", {"value": 0.1})
    expected = float(linear_to_bt709(bt709_to_linear(0.5) + 0.1))
    assert out[0, 0].tolist() == pytest.approx([expected] * 3)
    out = apply_correction(_px(0.5, 0.5, 0.5), "offset", {"value": -0.05})
    expected = float(linear_to_bt709(bt709_to_linear(0.5) - 0.05))
    assert out[0, 0].tolist() == pytest.approx([expected] * 3)


def test_cdl_slope_in_linear():
    out = apply_correction(_px(0.5, 0.5, 0.5), "cdl", {"slope": [2.0, 1.0, 1.0]})
    lin = bt709_to_linear(0.5)
    expected = linear_to_bt709(lin * np.array([2.0, 1.0, 1.0]))
    assert out[0, 0].tolist() == pytest.approx(expected.tolist(), abs=1e-3)


def test_cdl_power_in_linear():
    # linear offset 0.25 with power 2 -> sqrt(0.25) = 0.5 linear.
    out = apply_correction(
        _px(0.0, 0.0, 0.0), "cdl", {"offset": [0.25, 0.25, 0.25], "power": [2.0, 2.0, 2.0]}
    )
    expected = float(linear_to_bt709(np.sqrt(0.25)))
    assert out[0, 0].tolist() == pytest.approx([expected] * 3, abs=1e-3)


def test_rgb_balance_in_linear():
    out = apply_correction(_px(0.5, 0.5, 0.5), "rgb_balance", {"gain": [2.0, 1.0, 0.5]})
    lin = bt709_to_linear(0.5)
    expected = linear_to_bt709(lin * np.array([2.0, 1.0, 0.5]))
    assert out[0, 0].tolist() == pytest.approx(expected.tolist(), abs=1e-3)


def test_contrast_in_linear():
    out = apply_correction(_px(0.25, 0.25, 0.25), "contrast", {"amount": 2.0, "pivot": 0.5})
    lin = bt709_to_linear(0.25)
    expected = float(linear_to_bt709((lin - 0.5) * 2.0 + 0.5))
    assert out[0, 0].tolist() == pytest.approx([expected] * 3, abs=1e-3)


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
    lin_mid = float(bt709_to_linear(0.5))
    # +1 stop, then subtract the same linear lift -> back to mid-gray.
    seq = [("exposure", {"gain": 2.0}), ("offset", {"value": -lin_mid})]
    out = apply_corrections(_px(0.5, 0.5, 0.5), seq)
    assert out[0, 0].tolist() == pytest.approx([0.5, 0.5, 0.5], abs=1e-3)

    disabled = Correction(kind="exposure", parameters={"gain": 2.0}, enabled=False)
    out = apply_corrections(_px(0.5, 0.5, 0.5), [disabled])
    assert out[0, 0].tolist() == pytest.approx([0.5, 0.5, 0.5])


def test_stacked_corrections_compose_in_one_float_pass():
    # Two tiny linear lifts, each individually below the uint8 quantization
    # step: composed in float they must survive a single final quantize,
    # instead of being rounded away after every step.
    img = np.full((1, 1, 3), 128, dtype=np.uint8)
    one = apply_corrections(img, [("offset", {"value": 0.001})])
    both = apply_corrections(
        img, [("offset", {"value": 0.001}), ("offset", {"value": 0.001})]
    )
    assert (one == 128).all()  # a single lift rounds away
    assert (both > 128).all()  # composition is preserved across the stack


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
    for kind in ("cdl", "exposure", "offset", "rgb_balance", "contrast", "saturation", "hue_rotate", "curve"):
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
    # Mid-gray 128/255 decoded to linear, doubled, then re-encoded.
    expected = float(linear_to_bt709(bt709_to_linear(128 / 255) * 2.0))
    assert result.mean() == pytest.approx(expected * 255, abs=5)


def test_preview_rejects_log_transfer(tmp_path):
    store = ProjectStore.create(":memory:")
    project = store.create_project("log asset")
    asset = store.add_asset(
        project.id,
        source_path="/media/log.mov",
        frame_rate=25.0,
        transfer="slog3",  # camera log — no known pair, never guessed
    )
    shot = make_shots(asset, [(0, 24)])[0]
    with store.session() as session:
        session.add(shot)
        session.flush()
        session.refresh(shot)

    still = tmp_path / "still.png"
    cv2.imwrite(str(still), np.full((8, 8, 3), [128, 128, 128], dtype=np.uint8))
    with store.session() as session:
        session.add(make_representative_frame(shot, 0, image_path=str(still), frame_rate=25.0))
        session.commit()

    with pytest.raises(ValueError, match="never guessed"):
        preview_correction(store, shot, tmp_path / "preview.png")


def test_apply_corrections_transfer_native_identity():
    import numpy as np

    from colorai.correction import apply_corrections

    rng = np.random.default_rng(3)
    pixels = rng.random((8, 10, 3))
    for transfer in ("bt709", "pq", "hlg", None):
        out = np.asarray(apply_corrections(pixels, [], transfer=transfer))
        assert out == pytest.approx(pixels, abs=1e-6)


def test_apply_corrections_pq_differs_from_bt709():
    import numpy as np

    from colorai.correction import apply_corrections

    pixels = np.full((4, 4, 3), 0.5)
    bt709 = np.asarray(apply_corrections(pixels, [("exposure", {"gain": 1.5})], transfer="bt709"))
    pq = np.asarray(apply_corrections(pixels, [("exposure", {"gain": 1.5})], transfer="pq"))
    assert not np.allclose(bt709, pq, atol=1e-6)  # different EOTFs, different grades
    assert (pq > 0.5).all() and (bt709 > 0.5).all()  # both lift mid-gray


def test_preview_grades_pq_asset(tmp_path):
    store = ProjectStore.create(":memory:")
    project = store.create_project("pq asset")
    asset = store.add_asset(
        project.id, source_path="/media/pq.mov", frame_rate=25.0, transfer="smpte2084",
    )
    shot = make_shots(asset, [(0, 24)])[0]
    with store.session() as session:
        session.add(shot)
        session.flush()
        session.refresh(shot)
        session.add(Correction(shot_id=shot.id, kind="exposure", parameters={"gain": 1.5}))

    still = tmp_path / "still.png"
    cv2.imwrite(str(still), np.full((8, 8, 3), [128, 128, 128], dtype=np.uint8))
    with store.session() as session:
        session.add(make_representative_frame(shot, 0, image_path=str(still), frame_rate=25.0))
        session.commit()

    out = preview_correction(store, shot, tmp_path / "preview.png")
    assert out.exists()
    assert cv2.imread(str(out)).mean() > 128  # lifted, transfer-natively


def test_describe_correction_kinds():
    from colorai.correction import describe_correction

    assert describe_correction("exposure", {"gain": 2.0}) == "exposure 2.00× (+1.00 stops)"
    assert describe_correction("offset", {"value": -0.02}) == "lift -0.020"
    assert "R +10.0%" in describe_correction("rgb_balance", {"gain": [1.1, 1.0, 0.9]})
    assert describe_correction("contrast", {"amount": 1.2, "pivot": 0.5}) == "contrast ×1.20 @ 0.50"
    assert describe_correction("saturation", {"amount": 0.8}) == "saturation ×0.80"
    assert describe_correction("hue_rotate", {"degrees": 15}) == "hue +15.0°"
    assert describe_correction("cdl", {"slope": [1.1, 1.0, 1.0]}).startswith("CDL S[1.100")
    assert describe_correction("curve", {"mode": "luma", "points": [[0, 0], [1, 1]]}) == "curve luma (2 pts)"
    assert describe_correction("lut", {"path": "/l/look.cube", "space": "display"}) == "LUT look.cube [display]"
    assert describe_correction("mystery", {"a": 1}).startswith("mystery")
