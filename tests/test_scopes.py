"""Tests for review scopes (waveform + vectorscope)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from colorai.scopes import vectorscope, waveform


def _gradient(h=32, w=64):
    cols = (np.arange(w) / (w - 1) * 255).astype(np.uint8)
    gray = np.tile(cols, (h, 1))
    return np.stack([gray, gray, gray], axis=-1)


def test_waveform_tracks_horizontal_gradient():
    wf = waveform(_gradient(), width=64)
    assert wf["width"] == 64
    assert wf["luma_max"][0] == pytest.approx(0, abs=8)
    assert wf["luma_min"][-1] == pytest.approx(255, abs=8)
    assert wf["luma_min"][0] <= wf["luma_median"][0] <= wf["luma_max"][0]
    assert wf["r_mean"] == pytest.approx(wf["luma_median"], abs=8)  # gray ramp
    json.dumps(wf)


def test_waveform_rejects_bad_input():
    with pytest.raises(ValueError, match="HxWx3"):
        waveform(np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="width"):
        waveform(_gradient(), width=2)


def test_vectorscope_neutral_peaks_at_center():
    gray = np.full((32, 32, 3), 128, dtype=np.uint8)
    vs = vectorscope(gray, size=48)
    assert vs["size"] == 48
    cells = np.asarray(vs["cells"])
    assert cells.shape == (48, 48)
    assert cells.max() == pytest.approx(1.0)
    peak = np.unravel_index(cells.argmax(), cells.shape)
    assert abs(peak[0] - 24) <= 2 and abs(peak[1] - 24) <= 2  # near neutral
    json.dumps(vs)


def test_vectorscope_saturated_color_moves_off_center():
    red = np.zeros((32, 32, 3), dtype=np.uint8)
    red[..., 2] = 255  # BGR red
    vs = vectorscope(red, size=48)
    cells = np.asarray(vs["cells"])
    peak = np.unravel_index(cells.argmax(), cells.shape)
    assert abs(peak[0] - 24) + abs(peak[1] - 24) > 6
