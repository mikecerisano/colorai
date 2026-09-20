"""Review scopes: deterministic waveform + vectorscope from stills.

These are *measurements for the review viewer*, computed from decoded BGR
stills — not color decisions. Both outputs are plain JSON-able dicts so the
review UI can draw them on canvas without any extra dependency:

* :func:`waveform` — per-column luma distribution (min/p25/median/p75/max)
  plus mean R/G/B traces, downsampled to ``width`` columns.
* :func:`vectorscope` — a ``size × size`` 2D histogram of chroma (Cb/Cr)
  centered on neutral, normalized so the hottest cell is 1.0.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _check_still(image_bgr: np.ndarray) -> tuple[int, int]:
    if (
        not isinstance(image_bgr, np.ndarray)
        or image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
    ):
        raise ValueError("still must be an HxWx3 BGR image")
    return int(image_bgr.shape[0]), int(image_bgr.shape[1])


def waveform(image_bgr: np.ndarray, *, width: int = 160) -> dict[str, Any]:
    """Column-wise luma distribution + RGB traces, values in ``[0, 255]``."""
    h, w = _check_still(image_bgr)
    if width < 8:
        raise ValueError("width must be >= 8")
    small = cv2.resize(image_bgr, (width, h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float64)
    bands = np.percentile(gray, [0, 25, 50, 75, 100], axis=0)
    rgb = small[..., ::-1].astype(np.float64).mean(axis=0)  # BGR -> RGB column means
    col = lambda row: [round(float(v), 1) for v in row]  # noqa: E731
    return {
        "width": width,
        "luma_min": col(bands[0]),
        "luma_p25": col(bands[1]),
        "luma_median": col(bands[2]),
        "luma_p75": col(bands[3]),
        "luma_max": col(bands[4]),
        "r_mean": col(rgb[:, 0]),
        "g_mean": col(rgb[:, 1]),
        "b_mean": col(rgb[:, 2]),
    }


def vectorscope(image_bgr: np.ndarray, *, size: int = 48) -> dict[str, Any]:
    """Chroma (Cb/Cr) 2D histogram on a ``size × size`` grid, peak = 1.0."""
    _check_still(image_bgr)
    if size < 8:
        raise ValueError("size must be >= 8")
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float64)
    cb = np.clip(ycrcb[..., 2], 16.0, 240.0) - 16.0  # 224-wide broadcast range
    cr = np.clip(ycrcb[..., 1], 16.0, 240.0) - 16.0
    gx = np.clip((cb / 224.0 * size).astype(int), 0, size - 1)
    gy = np.clip((cr / 224.0 * size).astype(int), 0, size - 1)
    grid = np.zeros((size, size), dtype=np.float64)
    np.add.at(grid, (gy.ravel(), gx.ravel()), 1.0)
    peak = float(grid.max()) or 1.0
    return {"size": size, "cells": (grid / peak).round(3).tolist()}
