"""Tests for restoration primitives and the proposal boundary."""

from __future__ import annotations

import numpy as np
import pytest

from colorai.restoration import (
    METHOD_BLEND,
    METHOD_GENERATIVE,
    METHOD_NEAREST,
    blend_frames,
    generative_restore,
    propose_restoration,
    replace_damaged_frames,
    temporal_median,
)


def _frame(value):
    return np.full((4, 4, 3), value, dtype=np.uint8)


def test_blend_frames_midpoint():
    out = blend_frames(_frame(0), _frame(255), 0.5)
    assert out.mean() == pytest.approx(127.5, abs=1)


def test_blend_frames_endpoints():
    black, white = _frame(0), _frame(255)
    assert (blend_frames(black, white, 0.0) == black).all()
    assert (blend_frames(black, white, 1.0) == white).all()


def test_blend_frames_requires_same_shape():
    with pytest.raises(ValueError):
        blend_frames(np.zeros((4, 4, 3), np.uint8), np.zeros((5, 5, 3), np.uint8), 0.5)


def test_temporal_median_suppresses_outlier():
    stack = [_frame(0), _frame(255), _frame(0)]
    out = temporal_median(stack)
    assert (out == 0).all()


def test_temporal_median_requires_frames():
    with pytest.raises(ValueError):
        temporal_median([])


def test_replace_damaged_frames_blend():
    frames = {0: _frame(0), 2: _frame(255)}
    out = replace_damaged_frames(frames, [1], method=METHOD_BLEND)
    assert 1 in out
    assert out[1].mean() == pytest.approx(127.5, abs=1)


def test_replace_damaged_frames_nearest():
    frames = {0: _frame(0), 2: _frame(255)}
    out = replace_damaged_frames(frames, [1], method=METHOD_NEAREST)
    assert out[1].mean() == 0  # equidistant -> before


def test_replace_damaged_frames_single_side():
    out = replace_damaged_frames({0: _frame(255)}, [1, 2], method=METHOD_NEAREST)
    assert out[1].mean() == 255
    assert out[2].mean() == 255


def test_propose_restoration_prefers_deterministic():
    p = propose_restoration(10, 20, has_before=True, has_after=True)
    assert p.method == METHOD_BLEND
    assert p.requires_approval

    p = propose_restoration(10, 20, has_before=True, has_after=False)
    assert p.method == METHOD_NEAREST


def test_propose_restoration_generative_gate():
    p = propose_restoration(10, 20, has_before=False, has_after=False, allow_generative=False)
    assert p.method == METHOD_GENERATIVE

    p = propose_restoration(10, 20, has_before=False, has_after=False, allow_generative=True)
    assert p.method == METHOD_GENERATIVE


def test_generative_restore_not_implemented():
    with pytest.raises(NotImplementedError):
        generative_restore()
