"""Tests for .cube LUT parsing and the lut/curve correction kinds."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from colorai.color import bt709_to_linear, linear_to_bt709
from colorai.correction import (
    apply_correction,
    normalize_parameters,
    validate_correction,
)
from colorai.lutcube import (
    apply_cube,
    cube_content_hash,
    load_cube,
    parse_cube,
    parse_cube_file,
)
from colorai.project import Correction, ProjectStore, make_representative_frame, make_shots


def _px(r, g, b):
    return np.array([[[r, g, b]]], dtype=np.float32)


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


IDENTITY_1D = "LUT_1D_SIZE 2\n0.0 0.0 0.0\n1.0 1.0 1.0\n"

# Size-2 identity, .cube ordering (red fastest, blue slowest).
IDENTITY_3D = (
    "LUT_3D_SIZE 2\n"
    "0.0 0.0 0.0\n1.0 0.0 0.0\n"
    "0.0 1.0 0.0\n1.0 1.0 0.0\n"
    "0.0 0.0 1.0\n1.0 0.0 1.0\n"
    "0.0 1.0 1.0\n1.0 1.0 1.0\n"
)

# Linear-domain gain-by-two (clamped at 1.0).
GAIN_1D = "LUT_1D_SIZE 3\n0.0 0.0 0.0\n1.0 1.0 1.0\n1.0 1.0 1.0\n"


# -- parser ---------------------------------------------------------------

def test_parse_1d_lut():
    lut = parse_cube(IDENTITY_1D)
    assert not lut.is_3d and lut.size == 2
    assert lut.table.shape == (2, 3)
    assert lut.domain_min == (0.0, 0.0, 0.0)


def test_parse_3d_lut_ordering():
    lut = parse_cube(IDENTITY_3D)
    assert lut.is_3d and lut.size == 2
    assert lut.table.shape == (2, 2, 2, 3)
    # table[r, g, b] should be the (r, g, b) triple for the identity LUT.
    assert lut.table[1, 0, 0].tolist() == [1.0, 0.0, 0.0]
    assert lut.table[0, 1, 0].tolist() == [0.0, 1.0, 0.0]
    assert lut.table[1, 1, 1].tolist() == [1.0, 1.0, 1.0]


def test_parse_domain_and_comments():
    text = (
        "# comment\n"
        'TITLE "test"\n'
        "LUT_1D_SIZE 2\n"
        "DOMAIN_MIN 0.2 0.2 0.2\n"
        "DOMAIN_MAX 0.8 0.8 0.8\n"
        "0.0 0.0 0.0\n1.0 1.0 1.0\n"
    )
    lut = parse_cube(text)
    assert lut.domain_min == (0.2, 0.2, 0.2)
    assert lut.domain_max == (0.8, 0.8, 0.8)


def test_parse_input_range_variant():
    lut = parse_cube("LUT_1D_SIZE 2\nLUT_1D_INPUT_RANGE 0.0 1.0\n0 0 0\n1 1 1\n")
    assert lut.domain_min == (0.0, 0.0, 0.0)
    assert lut.domain_max == (1.0, 1.0, 1.0)


def test_parse_rejects_missing_size():
    with pytest.raises(ValueError, match="missing"):
        parse_cube("0.0 0.0 0.0\n1.0 1.0 1.0\n")


def test_parse_rejects_wrong_entry_count():
    with pytest.raises(ValueError, match="expected"):
        parse_cube("LUT_1D_SIZE 4\n0.0 0.0 0.0\n1.0 1.0 1.0\n")


# -- content hash + cache --------------------------------------------------

def test_content_hash_deterministic_and_sensitive(tmp_path):
    p = _write(tmp_path, "a.cube", IDENTITY_1D)
    h = cube_content_hash(p)
    assert h == cube_content_hash(p)
    p.write_text(IDENTITY_3D)
    assert cube_content_hash(p) != h


def test_load_cube_invalidates_on_change(tmp_path):
    p = _write(tmp_path, "a.cube", IDENTITY_1D)
    assert load_cube(p).table.shape == (2, 3)
    p.write_text(IDENTITY_3D)
    assert load_cube(p).table.shape == (2, 2, 2, 3)


# -- application (linear domain) ------------------------------------------

def test_apply_cube_1d_identity():
    lut = parse_cube(IDENTITY_1D)
    rgb = np.array([[[0.25, 0.5, 0.75]]], dtype=np.float32)
    assert apply_cube(lut, rgb) == pytest.approx(rgb, abs=1e-6)


def test_apply_cube_1d_gain_and_clamp():
    lut = parse_cube(GAIN_1D)
    # 0.25 -> 0.5 (doubled), 0.75 -> 1.0 (clamped).
    out = apply_cube(lut, np.array([[[0.25, 0.75, 0.1]]], dtype=np.float32))
    assert out[0, 0].tolist() == pytest.approx([0.5, 1.0, 0.2], abs=1e-6)


def test_apply_cube_3d_identity():
    lut = parse_cube(IDENTITY_3D)
    rgb = np.array([[[0.4, 0.6, 0.8]]], dtype=np.float32)
    assert apply_cube(lut, rgb) == pytest.approx(rgb, abs=1e-5)


def test_apply_cube_1d_domain_clamps():
    lut = parse_cube(
        "LUT_1D_SIZE 2\nDOMAIN_MIN 0.2 0.2 0.2\nDOMAIN_MAX 0.8 0.8 0.8\n0 0 0\n1 1 1\n"
    )
    assert apply_cube(lut, np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)).tolist() == [
        [[0.0, 0.5, 1.0]]
    ]


# -- correction kinds -----------------------------------------------------

def test_curve_identity():
    out = apply_correction(_px(0.5, 0.25, 0.75), "curve", {"points": [[0, 0], [1, 1]]})
    assert out[0, 0].tolist() == pytest.approx([0.5, 0.25, 0.75], abs=1e-4)


def test_curve_lift_in_linear():
    # Lift black point to 0.1 (linear): out = 0.1 + 0.9 * in_linear.
    params = {"points": [[0.0, 0.1], [1.0, 1.0]]}
    out = apply_correction(_px(0.5, 0.5, 0.5), "curve", params)
    lin = bt709_to_linear(0.5)
    expected = float(linear_to_bt709(0.1 + 0.9 * lin))
    assert out[0, 0].tolist() == pytest.approx([expected] * 3, abs=1e-4)


def test_curve_luma_preserves_chromaticity():
    # A luma-only curve must scale all channels equally (ratio preserved).
    img = _px(0.5, 0.25, 0.1)
    out = apply_correction(img, "curve", {"mode": "luma", "points": [[0, 0], [1, 1]]})
    assert out[0, 0].tolist() == pytest.approx([0.5, 0.25, 0.1], abs=1e-4)


def test_curve_per_channel():
    params = {
        "mode": "per_channel",
        "points": {"r": [[0, 0], [1, 1]], "g": [[0, 0], [1, 0]], "b": [[0, 0], [1, 1]]},
    }
    out = apply_correction(_px(0.5, 0.5, 0.5), "curve", params)
    r, g, b = out[0, 0]
    assert r == pytest.approx(0.5, abs=1e-4)
    assert g == pytest.approx(0.0, abs=1e-4)
    assert b == pytest.approx(0.5, abs=1e-4)


def test_lut_identity_roundtrip(tmp_path):
    p = _write(tmp_path, "id.cube", IDENTITY_1D)
    out = apply_correction(_px(0.5, 0.25, 0.75), "lut", {"path": str(p)})
    assert out[0, 0].tolist() == pytest.approx([0.5, 0.25, 0.75], abs=1e-4)


def test_lut_gain_matches_exposure_in_linear(tmp_path):
    # A linear-domain x2 LUT should equal exposure gain 2 (both linear ops).
    p = _write(tmp_path, "gain.cube", GAIN_1D)
    lut_out = apply_correction(_px(0.5, 0.5, 0.5), "lut", {"path": str(p)})
    exp_out = apply_correction(_px(0.5, 0.5, 0.5), "exposure", {"gain": 2.0})
    assert np.allclose(lut_out, exp_out, atol=1e-4)


def test_lut_display_space_applies_in_gamma(tmp_path):
    # A display-space x2 LUT doubles the *gamma* value: 0.5 -> 1.0 (clamped),
    # rather than the linear-domain result (~0.69 after re-encode).
    p = _write(tmp_path, "gain.cube", GAIN_1D)
    out = apply_correction(_px(0.5, 0.5, 0.5), "lut", {"path": str(p), "space": "display"})
    assert out[0, 0].tolist() == pytest.approx([1.0, 1.0, 1.0], abs=1e-4)


def test_lut_space_validation(tmp_path):
    p = _write(tmp_path, "id.cube", IDENTITY_1D)
    validate_correction("lut", {"path": str(p), "space": "display"})  # accepted
    with pytest.raises(ValueError):
        validate_correction("lut", {"path": str(p), "space": "log"})


# -- validation -----------------------------------------------------------

@pytest.mark.parametrize(
    "params",
    [
        {"points": [[0, 0], [1, -0.1]]},  # out of range
        {"points": [[0, 0], [0.5, 0.5], [0.25, 0.75]]},  # x not increasing
        {"points": [[0, 1], [1, 0]]},  # non-monotonic (decreasing)
        {"points": [[0, float("nan")], [1, 1]]},  # non-finite
        {"mode": "bogus", "points": [[0, 0], [1, 1]]},
    ],
)
def test_curve_validation_rejects(params):
    with pytest.raises(ValueError):
        validate_correction("curve", params)


def test_lut_validation_requires_path():
    with pytest.raises(ValueError):
        validate_correction("lut", {})
    with pytest.raises(ValueError):
        validate_correction("lut", {"path": ""})
    with pytest.raises(ValueError):
        validate_correction("lut", {"path": "/x.cube", "space": "log"})


# -- persistence + non-destructiveness ------------------------------------

def test_normalize_parameters_adds_content_hash(tmp_path):
    p = _write(tmp_path, "a.cube", IDENTITY_1D)
    before = p.read_bytes()
    params = normalize_parameters("lut", {"path": str(p)})
    assert params["content_hash"] == cube_content_hash(p)
    assert p.read_bytes() == before  # LUT file never modified


# -- preview / render parity ----------------------------------------------

def test_curve_flows_through_preview(tmp_path):
    store = ProjectStore.create(":memory:")
    project = store.create_project("curve preview")
    asset = store.add_asset(project.id, source_path="/media/m.mov", frame_rate=25.0)
    shot = make_shots(asset, [(0, 24)])[0]
    with store.session() as session:
        session.add(shot)
        session.flush()
        session.refresh(shot)

    still = tmp_path / "still.png"
    cv2.imwrite(str(still), np.full((8, 8, 3), [128, 128, 128], dtype=np.uint8))
    with store.session() as session:
        session.add(make_representative_frame(shot, 0, image_path=str(still), frame_rate=25.0))
        session.add(Correction(shot_id=shot.id, kind="curve", parameters={"points": [[0, 0.1], [1, 1]]}))
        session.commit()

    from colorai.correction import load_corrected_still

    result = load_corrected_still(store, shot)  # BGR uint8
    lin = bt709_to_linear(128 / 255)
    expected = float(linear_to_bt709(0.1 + 0.9 * lin))
    assert result.mean() == pytest.approx(expected * 255, abs=6)


def test_lut_flows_through_render_spans(tmp_path):
    from colorai.render import build_shot_spans, corrections_for_frame

    store = ProjectStore.create(":memory:")
    project = store.create_project("lut render")
    asset = store.add_asset(project.id, source_path="/media/m.mov", frame_rate=25.0)
    shot = make_shots(asset, [(0, 24)])[0]
    with store.session() as session:
        session.add(shot)
        session.flush()
        session.refresh(shot)

    cube = _write(tmp_path, "id.cube", IDENTITY_1D)
    with store.session() as session:
        session.add(Correction(shot_id=shot.id, kind="lut", parameters={"path": str(cube), "space": "linear"}))
        session.commit()

    spans = build_shot_spans(store, asset.id)
    assert corrections_for_frame(0, spans)[0][0] == "lut"
