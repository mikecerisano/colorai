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


def generative_restore(*args, **kwargs) -> np.ndarray:  # noqa: ANN002, ANN003
    """Generative reconstruction interface (NOT IMPLEMENTED).

    Selected local models (see ``docs/research-notes.md``):

    * temporal — **RIFE** (frame interpolation) for missing/damaged frames
    * spatial  — **LaMa** (inpainting) for damaged regions within a frame

    Both run locally via ONNX Runtime on Apple Silicon. The loader will read
    model files from ``COLORAI_GENERATIVE_MODEL_DIR``; until a model is
    present this raises rather than falling back silently.
    """
    raise NotImplementedError(
        "generative restoration models not installed — selected: RIFE (temporal) "
        "and LaMa (spatial), both ONNX; deterministic recovery "
        "(blend/nearest/median) is available now"
    )
