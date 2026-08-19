"""Generative restoration loader (RIFE + LaMa, ONNX).

The deterministic restoration tier (``restoration.py``) is complete. The
generative tier is the approval-gated fallback for intervals deterministic
recovery cannot repair. Model selection is decided — **RIFE** for temporal
frame interpolation and **LaMa** for spatial inpainting, both ONNX, run
locally via ONNX Runtime — but the model files are *not bundled* (they are
large and gitignored; see ``docs/research-notes.md``).

This module wires the *loader* and a status surface so the system can tell,
precisely, what is missing. Model files are resolved from
``COLORAI_GENERATIVE_MODEL_DIR`` (default: ``colorai/models/generative``).

The per-model inference I/O (pre/post-processing) is deliberately not faked
here: it depends on the exact ONNX export contract of the chosen RIFE/LaMa
checkpoint, so it lands with the model files.
"""

from __future__ import annotations

import os
from pathlib import Path

MODEL_NAMES = ("rife.onnx", "lama.onnx")
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models" / "generative"


class GenerativeModelError(RuntimeError):
    """Raised when a generative model (or its runtime) is unavailable."""


def model_dir() -> Path:
    """Directory containing the generative ONNX models."""
    return Path(os.environ.get("COLORAI_GENERATIVE_MODEL_DIR", str(DEFAULT_MODEL_DIR)))


def _ort():
    try:
        import onnxruntime as ort  # noqa: PLC0415
    except ImportError:
        return None
    return ort


def generative_models_status() -> dict:
    """Describe what is present for the generative tier (and whether it's ready)."""
    d = model_dir()
    has_ort = _ort() is not None
    has_rife = (d / "rife.onnx").exists()
    has_lama = (d / "lama.onnx").exists()
    return {
        "model_dir": str(d),
        "onnxruntime": has_ort,
        "rife": has_rife,
        "lama": has_lama,
        "ready": has_ort and has_rife and has_lama,
    }


def load_session(name: str):
    """Load an ONNX ``InferenceSession`` for ``name`` from the model dir.

    Raises :class:`GenerativeModelError` with an actionable message when the
    runtime or the model file is missing. ``name`` is one of ``MODEL_NAMES``.
    """
    if name not in MODEL_NAMES:
        raise GenerativeModelError(f"unknown generative model {name!r}")
    ort = _ort()
    if ort is None:
        raise GenerativeModelError(
            "onnxruntime is not installed; install with `pip install 'colorai[generative]'`"
        )
    path = model_dir() / name
    if not path.exists():
        raise GenerativeModelError(
            f"generative model {name!r} not found in {model_dir()}; "
            "see docs/research-notes.md for model acquisition"
        )
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
