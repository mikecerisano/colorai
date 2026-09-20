"""Tests for the generative restoration loader and inference I/O."""

from __future__ import annotations

import numpy as np
import pytest

from colorai.generative import (
    GenerativeModelError,
    generative_models_status,
    lama_inpaint,
    load_session,
    model_dir,
    rife_interpolate,
)


def test_model_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("COLORAI_GENERATIVE_MODEL_DIR", str(tmp_path))
    assert model_dir() == tmp_path


def test_status_reports_not_ready_without_models(monkeypatch, tmp_path):
    monkeypatch.setenv("COLORAI_GENERATIVE_MODEL_DIR", str(tmp_path))
    status = generative_models_status()
    assert status["ready"] is False
    assert status["model_dir"] == str(tmp_path)


def test_load_session_rejects_unknown_name():
    with pytest.raises(GenerativeModelError, match="unknown generative model"):
        load_session("bogus.onnx")


def test_load_session_reports_missing_runtime_or_model(monkeypatch, tmp_path):
    monkeypatch.setenv("COLORAI_GENERATIVE_MODEL_DIR", str(tmp_path))
    with pytest.raises(GenerativeModelError):
        load_session("rife.onnx")


def test_generative_restore_raises_actionable_error():
    from colorai.restoration import generative_restore

    with pytest.raises(NotImplementedError, match="not installed"):
        generative_restore()


class _Meta:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class _Session:
    """Minimal ONNX-session double: declared inputs + canned output."""

    def __init__(self, inputs, output):
        self._inputs = inputs
        self._output = output
        self.seen_feed = None

    def get_inputs(self):
        return self._inputs

    def run(self, _names, feed):
        self.seen_feed = dict(feed)
        return [self._output]


def _frame(value: int, h: int = 8, w: int = 16) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_rife_split_inputs_nchw_output():
    out = np.full((1, 3, 8, 16), 0.5, dtype=np.float32)
    sess = _Session(
        [_Meta("img0", [1, 3, 8, 16]), _Meta("img1", [1, 3, 8, 16])], out
    )
    result = rife_interpolate(_frame(0), _frame(255), 0.5, session=sess)
    assert result.shape == (8, 16, 3)
    assert result.dtype == np.uint8
    assert set(sess.seen_feed) == {"img0", "img1"}
    # Mid-gray model output decodes to ~128 in every channel.
    assert result == pytest.approx(128, abs=2)


def test_rife_stacked_input_with_timestep():
    out = np.full((1, 3, 8, 16), 1.0, dtype=np.float32)
    sess = _Session(
        [_Meta("input", [1, 6, 8, 16]), _Meta("timestep", [1])], out
    )
    result = rife_interpolate(_frame(10), _frame(20), 0.25, session=sess)
    assert sess.seen_feed["input"].shape == (1, 6, 8, 16)
    assert sess.seen_feed["timestep"] == pytest.approx([0.25])
    assert result.shape == (8, 16, 3)


def test_rife_nhwc_output_parsed():
    out = np.full((1, 8, 16, 3), 0.0, dtype=np.float32)
    sess = _Session(
        [_Meta("left", [1, 3, 8, 16]), _Meta("right", [1, 3, 8, 16])], out
    )
    result = rife_interpolate(_frame(0), _frame(0), session=sess)
    assert result.shape == (8, 16, 3)
    assert (result == 0).all()


def test_rife_rejects_bad_frames_and_t():
    sess = _Session([_Meta("img0", [1, 3, 8, 16])], [])
    with pytest.raises(ValueError, match="same shape"):
        rife_interpolate(_frame(0), np.zeros((4, 4, 3), dtype=np.uint8), session=sess)
    with pytest.raises(ValueError, match="t must be"):
        rife_interpolate(_frame(0), _frame(0), t=2.0, session=sess)
    with pytest.raises(GenerativeModelError, match="unrecognized generative output"):
        rife_interpolate(
            _frame(0),
            _frame(0),
            session=_Session(
                [_Meta("a", [1, 3, 8, 16]), _Meta("b", [1, 3, 8, 16])],
                np.zeros((1, 5, 8, 16), dtype=np.float32),
            ),
        )


def test_lama_split_and_stacked_inputs():
    out = np.full((1, 3, 8, 16), 0.25, dtype=np.float32)
    split = _Session(
        [_Meta("image", [1, 3, 8, 16]), _Meta("mask", [1, 1, 8, 16])], out
    )
    mask = np.zeros((8, 16), dtype=np.uint8)
    mask[2:4, 2:4] = 255
    result = lama_inpaint(_frame(100), mask, session=split)
    assert result.shape == (8, 16, 3)
    assert split.seen_feed["mask"].shape == (1, 1, 8, 16)

    stacked = _Session([_Meta("input", [1, 4, 8, 16])], out)
    result2 = lama_inpaint(_frame(100), mask, session=stacked)
    assert stacked.seen_feed["input"].shape == (1, 4, 8, 16)
    assert result2.shape == (8, 16, 3)


def test_lama_rejects_shape_mismatch():
    sess = _Session([_Meta("image", [1, 3, 8, 16]), _Meta("mask", [1, 1, 8, 16])], [])
    with pytest.raises(ValueError, match="mask must be"):
        lama_inpaint(_frame(0), np.zeros((4, 4), dtype=np.uint8), session=sess)


def test_generative_restore_dispatch(monkeypatch):
    import colorai.generative as gen
    import colorai.restoration as rest

    monkeypatch.setattr(
        gen, "generative_models_status",
        lambda: {"onnxruntime": True, "rife": True, "lama": True,
                 "model_dir": "/models", "ready": True},
    )
    out = np.full((1, 3, 8, 16), 0.5, dtype=np.float32)
    monkeypatch.setattr(
        gen, "load_session",
        lambda name: _Session(
            [_Meta("img0", [1, 3, 8, 16]), _Meta("img1", [1, 3, 8, 16])], out
        ),
    )
    result = rest.generative_restore("rife", frame0=_frame(0), frame1=_frame(255))
    assert result.shape == (8, 16, 3)
    with pytest.raises(ValueError, match="unknown generative mode"):
        rest.generative_restore("bogus", frame0=_frame(0), frame1=_frame(0))
