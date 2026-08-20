"""Tests for the pure OKLab skin-appearance math."""

from __future__ import annotations

import numpy as np
import pytest

from colorai.skin_appearance import (
    SkinAppearanceParameters,
    apply_skin_appearance,
    derive_skin_appearance_parameters,
    masked_skin_profile,
    oklab_to_rgb_linear,
    rgb_linear_to_oklab,
    validate_skin_appearance_parameters,
)


def _params(offset=(-0.012, 0.004), scale=(1.0, 1.0), strength=1.0):
    return SkinAppearanceParameters(
        version=1, luma_mode="preserve", ab_offset=offset, ab_scale=scale, strength=strength
    )


def test_oklab_round_trip_is_close_for_linear_rgb():
    rgb = np.array([[[0.12, 0.38, 0.71], [0.84, 0.31, 0.09]]], dtype=np.float64)
    assert np.allclose(oklab_to_rgb_linear(rgb_linear_to_oklab(rgb)), rgb, atol=1e-6)


def test_chroma_only_transform_preserves_oklab_lightness():
    rgb = np.full((8, 8, 3), [0.55, 0.30, 0.21])
    out = apply_skin_appearance(rgb, _params(offset=(-0.012, 0.004)), np.ones((8, 8)))
    assert np.allclose(rgb_linear_to_oklab(out)[..., 0], rgb_linear_to_oklab(rgb)[..., 0], atol=1e-6)


def test_apply_zero_alpha_is_identity():
    rgb = np.random.default_rng(0).random((16, 16, 3))
    out = apply_skin_appearance(rgb, _params(offset=(-0.02, 0.02)), np.zeros((16, 16)))
    assert np.allclose(out, rgb, atol=1e-6)


def test_validate_rejects_nan_ab_offset():
    with pytest.raises(ValueError, match="ab_offset"):
        validate_skin_appearance_parameters({"version": 1, "ab_offset": [float("nan"), 0]})


def test_validate_rejects_out_of_bounds_offset_and_scale():
    with pytest.raises(ValueError, match="ab_offset"):
        validate_skin_appearance_parameters({"version": 1, "ab_offset": [0.05, 0]})
    with pytest.raises(ValueError, match="ab_scale"):
        validate_skin_appearance_parameters({"version": 1, "ab_offset": [0, 0], "ab_scale": [1.3, 1]})


def test_validate_rejects_wrong_space_and_luma_mode():
    with pytest.raises(ValueError, match="space"):
        validate_skin_appearance_parameters({"version": 1, "space": "lab", "luma_mode": "preserve", "ab_offset": [0, 0], "ab_scale": [1, 1], "strength": 1})
    with pytest.raises(ValueError, match="luma_mode"):
        validate_skin_appearance_parameters({"version": 1, "space": "oklab", "luma_mode": "match", "ab_offset": [0, 0], "ab_scale": [1, 1], "strength": 1})


def test_validate_returns_typed_parameters():
    params = validate_skin_appearance_parameters(
        {"version": 1, "space": "oklab", "luma_mode": "preserve", "ab_offset": [-0.01, 0.02], "ab_scale": [0.94, 1.05], "strength": 0.5}
    )
    assert params.ab_offset == (-0.01, 0.02)
    assert params.ab_scale == (0.94, 1.05)
    assert params.strength == 0.5


def test_masked_profile_excludes_low_alpha_pixels():
    rgb = np.zeros((4, 4, 3))
    rgb[0, 0] = [0.5, 0.2, 0.1]  # this pixel is masked out
    rgb[1:, 1:] = [0.2, 0.4, 0.6]
    alpha = np.zeros((4, 4))
    alpha[1:, 1:] = 1.0
    profile = masked_skin_profile(rgb, alpha)
    # The profile reflects only the high-alpha (non-[0,0]) pixels.
    lab_full = rgb_linear_to_oklab(rgb[1:, 1:])
    assert profile["mean_ab"] == pytest.approx([float(np.median(lab_full[..., 1])), float(np.median(lab_full[..., 2]))], abs=1e-9)


def test_excess_positive_a_derives_negative_bounded_offset():
    candidate = {"mean_ab": [0.03, 0.01], "spread_ab": [0.02, 0.02]}
    target = {"mean_ab": [0.0, 0.01], "spread_ab": [0.02, 0.02]}
    params = derive_skin_appearance_parameters(candidate, target, strength=1.0)
    assert params.ab_offset[0] < 0.0
    assert params.ab_offset[0] == pytest.approx(-0.03)
    assert -0.04 <= params.ab_offset[0] <= 0.04
    assert -0.04 <= params.ab_offset[1] <= 0.04


def test_derive_clips_offset_and_softens_scale_by_strength():
    candidate = {"mean_ab": [0.0, 0.0], "spread_ab": [0.01, 0.01]}
    target = {"mean_ab": [0.1, 0.0], "spread_ab": [0.02, 0.01]}
    params = derive_skin_appearance_parameters(candidate, target, strength=0.5)
    # offset = 0.1 * 0.5 = 0.05 -> clipped to 0.04
    assert params.ab_offset[0] == pytest.approx(0.04)
    # scale raw = 2.0 -> clipped to 1.15 -> softened: 1 + (1.15-1)*0.5 = 1.075
    assert params.ab_scale[0] == pytest.approx(1.075)
