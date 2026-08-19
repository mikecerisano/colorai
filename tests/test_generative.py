"""Tests for the generative restoration loader and status surface."""

from __future__ import annotations

import pytest

from colorai.generative import (
    GenerativeModelError,
    generative_models_status,
    load_session,
    model_dir,
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
