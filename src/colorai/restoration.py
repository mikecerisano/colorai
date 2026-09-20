"""Restoration for genuinely damaged temporal intervals.

Tiered by design (see ``docs/architecture.md``): deterministic recovery first,
generative reconstruction only where deterministic recovery cannot recover the
missing image, and always behind human approval. This module implements the
deterministic primitives and the proposal boundary; the generative tier is an
explicit, unimplemented interface until a local model is chosen
(``docs/research-notes.md``).

All images are HxWx3 BGR uint8 arrays, matching OpenCV output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

# Deterministic recovery methods.
METHOD_NEAREST = "nearest"
METHOD_BLEND = "blend"
METHOD_MEDIAN = "median"
# Generative tier — reserved, not implemented without a local model.
METHOD_GENERATIVE = "generative"


@dataclass(frozen=True)
class RestorationProposal:
    """A plan for repairing one damaged frame interval."""

    start_frame: int
    end_frame: int  # inclusive
    method: str
    description: str
    requires_approval: bool = True


def blend_frames(before: np.ndarray, after: np.ndarray, t: float) -> np.ndarray:
    """Linear cross-dissolve between two same-shaped frames (``t`` in ``[0, 1]``)."""
    if before.shape != after.shape:
        raise ValueError("frames must have the same shape")
    blended = (1.0 - t) * before.astype(np.float64) + t * after.astype(np.float64)
    return np.clip(blended, 0, 255).round().astype(np.uint8)


def temporal_median(stack: Iterable[np.ndarray]) -> np.ndarray:
    """Pixelwise median across a stack of same-shaped frames.

    Deterministic dead-pixel / flicker removal: out-of-family pixels (hot or
    stuck pixels, brief flicker) are suppressed by the median.
    """
    frames = list(stack)
    if not frames:
        raise ValueError("temporal_median needs at least one frame")
    return np.median(np.stack(frames).astype(np.float64), axis=0).round().astype(np.uint8)


def replace_damaged_frames(
    frames: Mapping[int, np.ndarray],
    damaged: Iterable[int],
    *,
    method: str = METHOD_BLEND,
) -> dict[int, np.ndarray]:
    """Replace damaged frames using their good neighbors.

    ``frames`` maps frame index -> BGR image (good frames only). Returns a new
    mapping with the damaged indices filled in. ``blend`` interpolates between
    the surrounding good frames; ``nearest`` copies the closest good frame.
    """
    if method not in (METHOD_BLEND, METHOD_NEAREST):
        raise ValueError(f"unsupported deterministic method: {method!r}")

    good = sorted(frames)
    if not good:
        raise ValueError("at least one good frame is required")

    result = dict(frames)
    for d in sorted(damaged):
        before = max((g for g in good if g < d), default=None)
        after = min((g for g in good if g > d), default=None)
        if method == METHOD_BLEND and before is not None and after is not None:
            t = (d - before) / (after - before)
            result[d] = blend_frames(frames[before], frames[after], t)
        elif before is not None and after is not None:
            result[d] = frames[before] if (d - before) <= (after - d) else frames[after]
        elif before is not None:
            result[d] = frames[before]
        else:
            result[d] = frames[after]
    return result


def propose_restoration(
    start_frame: int,
    end_frame: int,
    *,
    has_before: bool,
    has_after: bool,
    allow_generative: bool = False,
) -> RestorationProposal:
    """Choose a restoration method for an inclusive damaged frame interval.

    Deterministic methods are preferred when a good frame is available on
    either side; generative reconstruction is proposed only when it is
    explicitly allowed and no deterministic method can recover the interval.
    """
    if has_before or has_after:
        method = METHOD_BLEND if (has_before and has_after) else METHOD_NEAREST
        return RestorationProposal(
            start_frame,
            end_frame,
            method,
            f"deterministic {method} from surrounding good frames",
        )
    if allow_generative:
        return RestorationProposal(
            start_frame,
            end_frame,
            METHOD_GENERATIVE,
            "generative reconstruction (requires a local model and approval)",
        )
    return RestorationProposal(
        start_frame,
        end_frame,
        METHOD_GENERATIVE,
        "no deterministic recovery possible; generative tier is not enabled",
    )


def _missing_generative(status: dict) -> list[str]:
    return [
        name
        for name, present in (
            ("onnxruntime", status["onnxruntime"]),
            ("rife.onnx", status["rife"]),
            ("lama.onnx", status["lama"]),
        )
        if not present
    ]


def generative_restore(
    mode: str = "rife",
    *args,  # noqa: ANN002
    **kwargs,  # noqa: ANN003
) -> np.ndarray:
    """Generative reconstruction (approval-gated, model-backed).

    Selected local models (see ``docs/research-notes.md``):

    * temporal — **RIFE** (frame interpolation) for missing/damaged frames:
      ``generative_restore("rife", frame0=..., frame1=..., t=0.5)``
      (``before``/``after`` are accepted as aliases).
    * spatial — **LaMa** (inpainting) for damaged regions within a frame:
      ``generative_restore("lama", frame=..., mask=...)``.

    Both run locally via ONNX Runtime on Apple Silicon (see
    :mod:`colorai.generative`); until the models are installed this raises
    rather than falling back silently. Deterministic recovery
    (blend/nearest/median) is available now.
    """
    from colorai.generative import generative_models_status

    status = generative_models_status()
    missing = _missing_generative(status)
    if missing:
        raise NotImplementedError(
            "generative restoration models not installed — missing: "
            + ", ".join(missing)
            + f" (model dir: {status['model_dir']}). Deterministic recovery "
            "(blend/nearest/median) is available now."
        )
    from colorai.generative import lama_inpaint, rife_interpolate

    if mode == "rife":
        frame0 = kwargs.get("frame0", kwargs.get("before"))
        frame1 = kwargs.get("frame1", kwargs.get("after"))
        if frame0 is None and args:
            frame0 = args[0]
        if frame1 is None and len(args) > 1:
            frame1 = args[1]
        if frame0 is None or frame1 is None:
            raise ValueError("rife mode needs frame0 and frame1 (or before/after)")
        return rife_interpolate(frame0, frame1, kwargs.get("t", 0.5))
    if mode == "lama":
        frame = kwargs.get("frame", args[0] if args else None)
        mask = kwargs.get("mask", args[1] if len(args) > 1 else None)
        if frame is None or mask is None:
            raise ValueError("lama mode needs frame and mask")
        return lama_inpaint(frame, mask)
    raise ValueError(f"unknown generative mode {mode!r} (expected 'rife' or 'lama')")
