"""Tests for BT.709 transfer functions."""

from __future__ import annotations

import numpy as np
import pytest

from colorai.color import (
    bt709_oetf,
    bt709_oetf_inverse,
    bt709_to_linear,
    decode_transfer,
    describe_working_space,
    encode_transfer,
    hlg_oetf,
    hlg_oetf_inverse,
    is_gradeable_transfer,
    linear_to_bt709,
    non_gradeable_reason,
    normalize_color_space,
    normalize_transfer,
    pq_eotf,
    pq_oetf,
    srgb_eotf,
    srgb_oetf,
)


def test_anchors():
    assert bt709_to_linear(0.0) == pytest.approx(0.0)
    assert bt709_to_linear(1.0) == pytest.approx(1.0)
    # Mid-gray (18% reflectance) -> ~0.18 linear.
    assert linear_to_bt709(0.18) == pytest.approx(0.461, abs=1e-3)
    assert bt709_to_linear(0.461) == pytest.approx(0.18, abs=1e-3)


def test_roundtrip():
    xs = np.array([0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    assert linear_to_bt709(bt709_to_linear(xs)) == pytest.approx(xs, abs=1e-6)


def test_linear_light_monotonic():
    xs = np.linspace(0.0, 1.0, 100)
    linear = bt709_to_linear(xs)
    assert (np.diff(linear) >= 0).all()


def test_vectorized_rgb():
    rgb = np.array([[[0.5, 0.25, 0.75]]], dtype=np.float64)
    out = bt709_to_linear(rgb)
    assert out.shape == rgb.shape
    assert out[0, 0, 0] > out[0, 0, 1]  # 0.5 gamma -> brighter linear than 0.25


def test_normalize_transfer_untagged_defaults_to_bt709():
    assert normalize_transfer(None) == "bt709"
    assert normalize_transfer("") == "bt709"
    assert normalize_transfer("unknown") == "bt709"
    assert normalize_transfer("unspecified") == "bt709"


def test_normalize_transfer_canonicalizes():
    assert normalize_transfer("bt709") == "bt709"
    assert normalize_transfer("iec61966-2-1") == "bt709"  # sRGB
    assert normalize_transfer("smpte2084") == "pq"
    assert normalize_transfer("arib-std-b67") == "hlg"
    assert normalize_transfer("bt2020-10") == "bt2020"
    # Unrecognized values pass through unchanged.
    assert normalize_transfer("something-exotic") == "something-exotic"


def test_normalize_color_space_defaults_to_bt709():
    assert normalize_color_space(None) == "bt709"
    assert normalize_color_space("unknown") == "bt709"
    assert normalize_color_space("bt2020nc") == "bt2020"


def test_is_gradeable_transfer():
    assert is_gradeable_transfer(None) is True  # untagged -> assumed BT.709
    assert is_gradeable_transfer("bt709") is True
    assert is_gradeable_transfer("iec61966-2-1") is True
    assert is_gradeable_transfer("smpte2084") is True  # PQ: transfer-native grade
    assert is_gradeable_transfer("hlg") is True
    assert is_gradeable_transfer("slog3") is False  # log: never guessed
    assert is_gradeable_transfer("something-exotic") is False


def test_describe_working_space():
    assert "BT.709" in describe_working_space()


def test_srgb_eotf_standard_value():
    # sRGB / BT.1886 display EOTF: code 0.5 -> ~0.2140 display-linear.
    assert srgb_eotf(0.5) == pytest.approx(0.2140, abs=1e-3)
    assert srgb_oetf(srgb_eotf(0.5)) == pytest.approx(0.5, abs=1e-6)
    # The pipeline aliases are the display-referred pair.
    assert bt709_to_linear(0.5) == pytest.approx(srgb_eotf(0.5))
    assert linear_to_bt709(0.2140) == pytest.approx(0.5, abs=1e-3)


def test_bt709_camera_oetf_standard_values():
    # Rec. ITU-R BT.709 camera OETF: 4.5 knee below 0.018, power above.
    assert bt709_oetf(0.01) == pytest.approx(0.045, abs=1e-9)  # knee: 4.5 * L
    assert bt709_oetf(1.0) == pytest.approx(1.0, abs=1e-6)
    assert bt709_oetf(0.18) == pytest.approx(0.4089, abs=1e-3)
    # Just above the knee the power branch applies (spec-defined ~0.0812).
    assert bt709_oetf(0.018) == pytest.approx(0.0812, abs=1e-3)
    # Inverse: code 0.5 -> ~0.2596 scene-linear (NOT the display ~0.2140).
    assert bt709_oetf_inverse(0.5) == pytest.approx(0.2596, abs=1e-3)
    assert bt709_oetf_inverse(0.045) == pytest.approx(0.01, abs=1e-5)
    # Round-trip (tolerance absorbs the spec's knee discontinuity).
    xs = np.linspace(0.0, 1.0, 50)
    assert bt709_oetf_inverse(bt709_oetf(xs)) == pytest.approx(xs, abs=1e-4)


def test_pq_eotf_standard_values():
    # SMPTE ST 2084: black stays black, peak normalizes to 1.0, and 50%
    # code lands at ~94 nits (0.0094 of the 10 000-nit peak).
    assert pq_eotf(0.0) == pytest.approx(0.0, abs=1e-9)
    assert pq_eotf(1.0) == pytest.approx(1.0, abs=1e-9)
    assert pq_eotf(0.5) == pytest.approx(0.0094, abs=2e-3)
    xs = np.linspace(0.0, 1.0, 50)
    assert (np.diff(pq_eotf(xs)) >= 0).all()


def test_hlg_oetf_roundtrip_and_anchor():
    # The 0.5 hinge decodes to exactly 1/12 scene-linear.
    assert hlg_oetf_inverse(0.5) == pytest.approx(1.0 / 12.0, abs=1e-9)
    assert hlg_oetf(1.0 / 12.0) == pytest.approx(0.5, abs=1e-9)
    xs = np.linspace(0.0, 1.0, 50)
    assert hlg_oetf_inverse(hlg_oetf(xs)) == pytest.approx(xs, abs=1e-6)


def test_non_gradeable_reason():
    assert non_gradeable_reason(None) is None
    assert non_gradeable_reason("bt709") is None
    assert non_gradeable_reason("smpte2084") is None
    assert non_gradeable_reason("hlg") is None
    reason = non_gradeable_reason("slog3")
    assert reason is not None and "slog3" in reason and "never guessed" in reason


def test_pq_oetf_anchors_and_roundtrip():
    assert pq_oetf(1.0) == pytest.approx(1.0, abs=1e-9)
    assert pq_oetf(0.0) == pytest.approx(0.0, abs=1e-5)  # curve never touches zero
    xs = np.linspace(0.0, 1.0, 50)
    assert pq_eotf(pq_oetf(xs)) == pytest.approx(xs, abs=1e-4)
    assert pq_oetf(pq_eotf(xs)) == pytest.approx(xs, abs=1e-4)


def test_transfer_pair_dispatch_roundtrips():
    xs = np.linspace(0.02, 0.98, 40)
    for transfer in ("bt709", "pq", "hlg", None, "smpte2084", "arib-std-b67"):
        assert encode_transfer(decode_transfer(xs, transfer), transfer) == pytest.approx(xs, abs=2e-3)


def test_transfer_pair_rejects_log():
    with pytest.raises(ValueError, match="never guessed"):
        decode_transfer(0.5, "slog3")
    with pytest.raises(ValueError, match="never guessed"):
        encode_transfer(0.5, "slog3")
