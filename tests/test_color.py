"""Tests for BT.709 transfer functions."""

from __future__ import annotations

import numpy as np
import pytest

from colorai.color import (
    bt709_to_linear,
    describe_working_space,
    is_gradeable_transfer,
    linear_to_bt709,
    normalize_color_space,
    normalize_transfer,
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
    assert is_gradeable_transfer("smpte2084") is False
    assert is_gradeable_transfer("hlg") is False


def test_describe_working_space():
    assert "BT.709" in describe_working_space()
