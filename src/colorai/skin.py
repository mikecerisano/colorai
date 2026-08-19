"""Experimental skin-pixel segmentation (color heuristic).

A deterministic YCrCb threshold heuristic, included as an *initial experiment*
only. It is color-based and has no face/context awareness, so it will
false-positive on skin-toned non-skin pixels and miss skin outside the
threshold ranges. Use it for coarse skin-coverage measurement; see
``docs/research-notes.md`` for the planned segmentation path.

Thresholds follow Chai & Ngan (1999), "Face segmentation using skin-color map
in videophone applications", IEEE Trans. Circuits Syst. Video Technol.
"""

from __future__ import annotations

import cv2
import numpy as np

_Y_MIN = 80
_CR_RANGE = (133, 173)
_CB_RANGE = (77, 127)


def skin_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Return a boolean skin mask (same HxW shape) for an HxWx3 BGR image."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("expected an HxWx3 BGR image")
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = ycrcb[..., 0], ycrcb[..., 1], ycrcb[..., 2]
    return (
        (cr >= _CR_RANGE[0])
        & (cr <= _CR_RANGE[1])
        & (cb >= _CB_RANGE[0])
        & (cb <= _CB_RANGE[1])
        & (y >= _Y_MIN)
    )


def skin_coverage(image_bgr: np.ndarray) -> float:
    """Fraction of pixels classified as skin, in ``[0, 1]``."""
    return float(skin_mask(image_bgr).mean())
