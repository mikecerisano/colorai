"""Image metrics for representative frames.

The initial metric set is intentionally small and deterministic: luminance
percentiles and dispersion, per-channel RGB means, and a mean chroma-magnitude
proxy for saturation. All values are normalized to ``[0, 1]`` assuming
full-range 8-bit input. These feed shot-to-shot consistency comparisons;
they are measurements, not decisions (see the product brief: a statistical
difference is not itself a visual error).

Image arrays are expected in OpenCV BGR order (``cv2.imread``) with values
``[0, 255]``; grayscale (2-D) input is also accepted.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# BT.709 luma coefficients.
_LUMA = (0.2126, 0.7152, 0.0722)


def compute_frame_metrics(image_bgr: np.ndarray) -> dict[str, float]:
    """Compute normalized image statistics for one frame.

    ``image_bgr``: HxWx3 uint8 BGR, or HxW uint8 grayscale.
    """
    array = image_bgr.astype(np.float64) / 255.0
    if array.ndim == 2:
        r = g = b = array
    else:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    luma = _LUMA[0] * r + _LUMA[1] * g + _LUMA[2] * b

    # Chroma distance from the luma axis: a rough Cb/Cr-like magnitude.
    cb = 0.5 * (b - luma)
    cr = 0.5 * (r - luma)
    chroma = np.sqrt(cb**2 + cr**2)

    return {
        "luma_min": float(luma.min()),
        "luma_p5": float(np.percentile(luma, 5)),
        "luma_mean": float(luma.mean()),
        "luma_median": float(np.percentile(luma, 50)),
        "luma_p95": float(np.percentile(luma, 95)),
        "luma_max": float(luma.max()),
        "luma_std": float(luma.std()),
        "r_mean": float(r.mean()),
        "g_mean": float(g.mean()),
        "b_mean": float(b.mean()),
        "saturation_mean": float(chroma.mean()),
    }


def metrics_from_path(path: str) -> dict[str, float]:
    """Load a still with OpenCV and compute its metrics."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {path!r}")
    return compute_frame_metrics(image)


def frame_sharpness(image_bgr: np.ndarray) -> float:
    """Sharpness proxy: variance of the Laplacian of the luma plane.

    Higher = sharper/more edge energy. Used for content-aware representative
    frame selection (a flat frame scores ~0, an in-focus textured frame scores
    higher).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def store_frame_metrics(
    store: ProjectStore, shot: Shot, frame_index: int, metrics: dict[str, Any]
) -> Any:
    """Persist ``metrics`` for a shot/frame (see :class:`FrameMetrics`)."""
    from colorai.project.models import FrameMetrics

    row = FrameMetrics(shot_id=shot.id, frame_index=frame_index, **metrics)
    with store.session() as session:
        session.add(row)
        session.flush()
        session.refresh(row)
    return row
