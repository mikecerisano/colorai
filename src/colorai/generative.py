"""Generative restoration loader + inference I/O (RIFE + LaMa, ONNX).

The deterministic restoration tier (``restoration.py``) is complete. The
generative tier is the approval-gated fallback for intervals deterministic
recovery cannot repair. Model selection is decided — **RIFE** for temporal
frame interpolation and **LaMa** for spatial inpainting, both ONNX, run
locally via ONNX Runtime — but the model files are *not bundled* (they are
large and gitignored; see ``docs/research-notes.md``).

Model files are resolved from ``COLORAI_GENERATIVE_MODEL_DIR`` (default:
``colorai/models/generative``). The status surface tells precisely what is
missing; the inference helpers below (:func:`rife_interpolate`,
:func:`lama_inpaint`) are contract-adaptive — they inspect the loaded
session's input names/shapes and build the feed for the common RIFE/LaMa
export variants (split vs. stacked image inputs, optional timestep/mask)
instead of assuming one exact export. Anything unrecognized raises
:class:`GenerativeModelError` rather than guessing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

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


# ---------------------------------------------------------------------------
# Contract-adaptive inference I/O
# ---------------------------------------------------------------------------
#
# RIFE/LaMa ONNX exports differ by source (split vs. stacked inputs, optional
# timestep/mask, NCHW vs. NHWC). These helpers inspect the session's declared
# inputs and build the matching feed; anything outside the recognized variants
# raises instead of running with a miswired feed.

def _bgr_to_nchw(frame: np.ndarray) -> np.ndarray:
    """BGR uint8 ``HxWx3`` -> float32 NCHW RGB in ``[0, 1]``."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))[None]


def _output_to_bgr(output: Any, height: int, width: int) -> np.ndarray:
    """Parse a model output into a BGR uint8 ``HxWx3`` frame.

    Accepts NCHW / NHWC / CHW / HWC float outputs in ``[0, 1]`` (or
    ``[0, 255]``). Raises :class:`GenerativeModelError` when the output has
    no recognizable 3-channel image.
    """
    arr = np.asarray(output)
    if arr.ndim == 4:
        if arr.shape[1] == 3:
            img = arr[0].transpose(1, 2, 0)
        elif arr.shape[-1] == 3:
            img = arr[0]
        else:
            raise GenerativeModelError(
                f"unrecognized generative output shape {arr.shape!r}"
            )
    elif arr.ndim == 3:
        if arr.shape[0] == 3:
            img = arr.transpose(1, 2, 0)
        elif arr.shape[-1] == 3:
            img = arr
        else:
            raise GenerativeModelError(
                f"unrecognized generative output shape {arr.shape!r}"
            )
    else:
        raise GenerativeModelError(
            f"unrecognized generative output shape {arr.shape!r}"
        )
    img = np.asarray(img, dtype=np.float64)
    if img.max() > 1.5:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)
    if img.shape[0] != height or img.shape[1] != width:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    rgb8 = (img * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)


def _check_frame(frame: np.ndarray, *, label: str) -> tuple[int, int]:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"{label} must be an HxWx3 image")
    return int(frame.shape[0]), int(frame.shape[1])


def _is_timestep_input(meta: Any) -> bool:
    name = str(getattr(meta, "name", "")).lower()
    tokens = name.replace("-", "_").replace(".", "_").split("_")
    if not any(
        tag in tokens for tag in ("t", "dt", "time", "timestep", "ratio", "timestep")
    ):
        return False
    shape = list(getattr(meta, "shape", []) or [])
    dims = [d for d in shape if isinstance(d, int)]
    # A timestep is scalar-ish: no large spatial dims.
    return not dims or max(dims) <= 4


def _image_inputs(session: Any) -> list[Any]:
    return [m for m in session.get_inputs() if not _is_timestep_input(m)]


def _channel_count(meta: Any) -> int | None:
    shape = list(getattr(meta, "shape", []) or [])
    # Prefer the channel position (NCHW dim 1, NHWC last) so the batch dim
    # (usually 1) is never mistaken for a 1-channel mask input.
    if len(shape) >= 2:
        for candidate in (shape[1], shape[-1]):
            if isinstance(candidate, int) and candidate in (1, 3, 4, 6):
                return candidate
    for d in shape:
        if isinstance(d, int) and d in (3, 4, 6):
            return d
    for d in shape:
        if d == 1:
            return 1
    return None


def rife_interpolate(
    frame0: np.ndarray,
    frame1: np.ndarray,
    t: float = 0.5,
    *,
    session: Any = None,
) -> np.ndarray:
    """Interpolate between ``frame0`` and ``frame1`` at time ``t`` (RIFE).

    ``frame0``/``frame1`` are BGR uint8 ``HxWx3``; ``t`` in ``[0, 1]``.
    ``session`` defaults to the bundled ``rife.onnx`` via :func:`load_session`.
    The feed adapts to the session contract: two 3-channel image inputs, one
    stacked 6-channel input, and an optional scalar timestep input.
    """
    h, w = _check_frame(frame0, label="frame0")
    h1, w1 = _check_frame(frame1, label="frame1")
    if (h, w) != (h1, w1):
        raise ValueError("frame0 and frame1 must have the same shape")
    if not 0.0 <= float(t) <= 1.0:
        raise ValueError("t must be in [0, 1]")
    sess = session if session is not None else load_session("rife.onnx")

    a = _bgr_to_nchw(frame0)
    b = _bgr_to_nchw(frame1)
    image_metas = _image_inputs(sess)
    timestep_metas = [m for m in sess.get_inputs() if _is_timestep_input(m)]
    feed: dict[str, Any] = {}
    if len(image_metas) == 2:
        first, second = image_metas[0], image_metas[1]
        n0, n1 = str(first.name), str(second.name)
        # Order by name hint (…0/…1, left/right, first/second) when present.
        low0, low1 = n0.lower(), n1.lower()
        swap = ("1" in low0 and "0" in low1) or (
            ("second" in low0 or "right" in low0)
            and not ("second" in low1 or "right" in low1)
        )
        if swap:
            feed = {n0: b, n1: a}
        else:
            feed = {n0: a, n1: b}
    elif len(image_metas) == 1:
        meta = image_metas[0]
        channels = _channel_count(meta)
        if channels == 6:
            feed = {str(meta.name): np.concatenate([a, b], axis=1)}
        elif channels == 3:
            raise GenerativeModelError(
                f"RIFE session {str(meta.name)!r} takes a single 3-channel input; "
                "interpolation needs two frames"
            )
        else:
            raise GenerativeModelError(
                f"unrecognized RIFE input shape {list(getattr(meta, 'shape', []))!r}"
            )
    else:
        raise GenerativeModelError(
            "unrecognized RIFE input contract: "
            f"{[str(m.name) for m in sess.get_inputs()]!r}"
        )
    for meta in timestep_metas:
        feed[str(meta.name)] = np.array([float(t)], dtype=np.float32)
    outputs = sess.run(None, feed)
    if not outputs:
        raise GenerativeModelError("RIFE session returned no outputs")
    return _output_to_bgr(outputs[0], h, w)


def lama_inpaint(
    frame: np.ndarray,
    mask: np.ndarray,
    *,
    session: Any = None,
) -> np.ndarray:
    """Inpaint the masked region of ``frame`` (LaMa).

    ``frame`` is BGR uint8 ``HxWx3``; ``mask`` is ``HxW`` (nonzero = hole).
    ``session`` defaults to ``lama.onnx`` via :func:`load_session`. The feed
    adapts to split image+mask inputs or a single stacked 4-channel input.
    """
    h, w = _check_frame(frame, label="frame")
    m = np.asarray(mask)
    if m.ndim == 3 and m.shape[2] == 1:
        m = m[..., 0]
    if m.ndim != 2 or m.shape != (h, w):
        raise ValueError("mask must be HxW matching frame")
    sess = session if session is not None else load_session("lama.onnx")

    image = _bgr_to_nchw(frame)
    hole = (m > 0).astype(np.float32)[None, None]
    image_metas = _image_inputs(sess)
    feed: dict[str, Any] = {}
    if len(image_metas) == 2:
        # Split variant: the 3-channel input takes the image, 1-channel the mask.
        ordered = sorted(
            image_metas, key=lambda mm: (_channel_count(mm) or 0), reverse=True
        )
        feed = {str(ordered[0].name): image, str(ordered[1].name): hole}
    elif len(image_metas) == 1:
        meta = image_metas[0]
        channels = _channel_count(meta)
        if channels == 4:
            feed = {str(meta.name): np.concatenate([image, hole], axis=1)}
        else:
            raise GenerativeModelError(
                f"unrecognized LaMa input shape {list(getattr(meta, 'shape', []))!r}"
            )
    else:
        raise GenerativeModelError(
            "unrecognized LaMa input contract: "
            f"{[str(m.name) for m in sess.get_inputs()]!r}"
        )
    outputs = sess.run(None, feed)
    if not outputs:
        raise GenerativeModelError("LaMa session returned no outputs")
    return _output_to_bgr(outputs[0], h, w)
