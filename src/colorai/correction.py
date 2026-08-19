"""Deterministic correction transforms.

This module implements the operations behind :class:`Correction` rows: pure,
vectorized, temporally stable transforms on a single frame. They operate on
normalized RGB arrays and are intentionally *not* generative — a given
``(kind, parameters)`` always produces the same output for the same input.

Image convention: HxWx3 **RGB** order, values in ``[0, 1]`` (float) or
``[0, 255]`` (uint8). The output matches the input's numeric kind (uint8 in ->
uint8 out, float in -> float out).

Supported kinds (see :func:`validate_correction` for parameter shapes):

* ``cdl``         — ASC CDL: per-channel slope/offset/power
* ``exposure``    — global linear gain
* ``offset``      — global lift (can be negative)
* ``rgb_balance`` — per-channel gain
* ``contrast``    — contrast with a pivot
* ``saturation``  — luma-preserving saturation scale
* ``hue_rotate``  — hue rotation in HSV space (degrees)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from colorai.project.models import Correction, Shot
from colorai.project.store import ProjectStore

_LUMA = (0.2126, 0.7152, 0.0722)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

def validate_correction(kind: str, parameters: dict[str, Any]) -> None:
    """Raise ``ValueError`` if ``parameters`` are invalid for ``kind``."""
    if kind == "cdl":
        for key in ("slope", "offset", "power"):
            v = parameters.get(key, None)
            if v is None:
                continue
            if not isinstance(v, (list, tuple)) or len(v) != 3:
                raise ValueError(f"cdl {key} must be a 3-element list")
            if key == "power" and any(p <= 0 for p in v):
                raise ValueError("cdl power must be > 0")
        return

    if kind == "exposure":
        gain = parameters.get("gain", 1.0)
        if not isinstance(gain, (int, float)) or gain < 0:
            raise ValueError("exposure gain must be a non-negative number")
        return

    if kind == "offset":
        value = parameters.get("value", 0.0)
        if not isinstance(value, (int, float)):
            raise ValueError("offset value must be a number")
        return

    if kind == "rgb_balance":
        gain = parameters.get("gain", [1.0, 1.0, 1.0])
        if not isinstance(gain, (list, tuple)) or len(gain) != 3:
            raise ValueError("rgb_balance gain must be a 3-element list")
        if any(g < 0 for g in gain):
            raise ValueError("rgb_balance gain must be non-negative")
        return

    if kind == "contrast":
        amount = parameters.get("amount", 1.0)
        pivot = parameters.get("pivot", 0.5)
        if not isinstance(amount, (int, float)):
            raise ValueError("contrast amount must be a number")
        if not isinstance(pivot, (int, float)) or not (0.0 <= pivot <= 1.0):
            raise ValueError("contrast pivot must be in [0, 1]")
        return

    if kind == "saturation":
        amount = parameters.get("amount", 1.0)
        if not isinstance(amount, (int, float)) or amount < 0:
            raise ValueError("saturation amount must be a non-negative number")
        return

    if kind == "hue_rotate":
        degrees = parameters.get("degrees", 0.0)
        if not isinstance(degrees, (int, float)):
            raise ValueError("hue_rotate degrees must be a number")
        return

    raise ValueError(f"unknown correction kind: {kind!r}")


# ---------------------------------------------------------------------------
# Core transform
# ---------------------------------------------------------------------------

def _to_float_rgb(image: np.ndarray) -> tuple[np.ndarray, bool]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected an HxWx3 RGB image")
    was_uint8 = image.dtype.kind in "iu"
    f = image.astype(np.float64)
    if was_uint8:
        f /= 255.0
    return np.clip(f, 0.0, 1.0), was_uint8


def _finish(f: np.ndarray, was_uint8: bool) -> np.ndarray:
    f = np.clip(f, 0.0, 1.0)
    if was_uint8:
        return (f * 255.0).round().astype(np.uint8)
    return f.astype(np.float32)


def _as_vec3(value: Any, default: tuple[float, float, float]) -> np.ndarray:
    return np.asarray(value if value is not None else default, dtype=np.float64)


def apply_correction(image_rgb: np.ndarray, kind: str, parameters: dict[str, Any]) -> np.ndarray:
    """Apply one deterministic correction to an RGB image array."""
    validate_correction(kind, parameters)
    f, was_uint8 = _to_float_rgb(image_rgb)

    if kind == "cdl":
        slope = _as_vec3(parameters.get("slope"), (1.0, 1.0, 1.0))
        offset = _as_vec3(parameters.get("offset"), (0.0, 0.0, 0.0))
        power = _as_vec3(parameters.get("power"), (1.0, 1.0, 1.0))
        out = np.power(np.clip(f * slope + offset, 0.0, None), 1.0 / power)
    elif kind == "exposure":
        out = f * float(parameters.get("gain", 1.0))
    elif kind == "offset":
        out = f + float(parameters.get("value", 0.0))
    elif kind == "rgb_balance":
        gain = _as_vec3(parameters.get("gain"), (1.0, 1.0, 1.0))
        out = f * gain
    elif kind == "contrast":
        amount = float(parameters.get("amount", 1.0))
        pivot = float(parameters.get("pivot", 0.5))
        out = (f - pivot) * amount + pivot
    elif kind == "saturation":
        amount = float(parameters.get("amount", 1.0))
        luma = _LUMA[0] * f[..., 0] + _LUMA[1] * f[..., 1] + _LUMA[2] * f[..., 2]
        out = luma[..., None] + (f - luma[..., None]) * amount
    elif kind == "hue_rotate":
        degrees = float(parameters.get("degrees", 0.0))
        u8 = (np.clip(f, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(np.int32) + int(round(degrees / 2.0))) % 180
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float64) / 255.0
    else:  # pragma: no cover - guarded by validate_correction
        raise ValueError(f"unknown correction kind: {kind!r}")

    return _finish(out, was_uint8)


def apply_corrections(
    image_rgb: np.ndarray, corrections: Iterable[Correction | tuple[str, dict[str, Any]]]
) -> np.ndarray:
    """Apply a sequence of corrections in order (skipping disabled ones)."""
    out = image_rgb
    for c in corrections:
        if isinstance(c, Correction):
            if not c.enabled:
                continue
            kind, params = c.kind, c.parameters
        else:
            kind, params = c
        out = apply_correction(out, kind, params)
    return out


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def preview_correction(store: ProjectStore, shot: Shot, out_path: str | Path) -> Path:
    """Render ``shot``'s representative still with its enabled corrections applied.

    Loads the stored still (BGR), applies each enabled ``Correction`` in order
    (RGB), and writes the result to ``out_path``. Non-destructive: the original
    still is never overwritten.
    """
    from colorai.project.models import RepresentativeFrame

    with store.session() as session:
        rf = (
            session.query(RepresentativeFrame)
            .filter_by(shot_id=shot.id)
            .first()
        )
        if rf is None or not rf.image_path:
            raise ValueError(f"shot {shot.id} has no representative frame")
        corrections = (
            session.query(Correction)
            .filter_by(shot_id=shot.id, enabled=True)
            .order_by(Correction.id)
            .all()
        )
        still_path = rf.image_path

    bgr = cv2.imread(still_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot read still: {still_path!r}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    corrected = apply_corrections(rgb, corrections)  # uint8 RGB out for uint8 in
    out_bgr = cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)

    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), out_bgr)
    return destination
