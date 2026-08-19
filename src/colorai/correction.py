"""Deterministic correction transforms.

This module implements the operations behind :class:`Correction` rows: pure,
vectorized, temporally stable transforms on a single frame. They operate on
normalized RGB arrays and are intentionally *not* generative — a given
``(kind, parameters)`` always produces the same output for the same input.

Image convention: HxWx3 **RGB** order, values in ``[0, 1]`` (float) or
``[0, 255]`` (uint8). The output matches the input's numeric kind (uint8 in ->
uint8 out, float in -> float out).

**Working space:** grading operations (everything except ``hue_rotate``) are
applied in *linear* BT.709 light, so parameters are physically meaningful —
``exposure`` gain 2 is one stop, ``offset`` is a scene-linear lift, and ASC CDL
slope/offset/power are linear. ``hue_rotate`` is a display-referred perceptual
op and stays in gamma space.

Supported kinds (see :func:`validate_correction` for parameter shapes):

* ``cdl``         — ASC CDL: per-channel slope/offset/power
* ``exposure``    — global linear gain
* ``offset``      — global lift (can be negative)
* ``rgb_balance`` — per-channel gain
* ``contrast``    — contrast with a pivot
* ``saturation``  — luma-preserving saturation scale
* ``hue_rotate``  — hue rotation in HSV space (degrees)
* ``curve``       — tone curve from monotonic control points
                    (``rgb`` / ``per_channel`` / ``luma`` modes)
* ``lut``         — a ``.cube`` LUT (1D/3D); ``space: "linear"`` (default,
                    scene-referred) or ``"display"`` (gamma — the usual
                    Resolve Rec.709 case)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from colorai.color import bt709_to_linear, is_gradeable_transfer, linear_to_bt709
from colorai.project.models import Correction, MediaAsset, Shot
from colorai.project.store import ProjectStore

_LUMA = (0.2126, 0.7152, 0.0722)

# Identity control points for a curve with no explicit shape.
_IDENTITY_CURVE = [[0.0, 0.0], [1.0, 1.0]]


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

def _is_number(value: Any) -> bool:
    """True for finite int/float (excludes bool and NaN/inf)."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def validate_correction(kind: str, parameters: dict[str, Any]) -> None:
    """Raise ``ValueError`` if ``parameters`` are invalid for ``kind``."""
    if kind == "cdl":
        for key in ("slope", "offset", "power"):
            v = parameters.get(key, None)
            if v is None:
                continue
            if not isinstance(v, (list, tuple)) or len(v) != 3:
                raise ValueError(f"cdl {key} must be a 3-element list")
            if any(not _is_number(p) for p in v):
                raise ValueError(f"cdl {key} values must be finite numbers")
            if key == "power" and any(p <= 0 for p in v):
                raise ValueError("cdl power must be > 0")
        return

    if kind == "exposure":
        gain = parameters.get("gain", 1.0)
        if not _is_number(gain) or gain < 0:
            raise ValueError("exposure gain must be a non-negative finite number")
        return

    if kind == "offset":
        value = parameters.get("value", 0.0)
        if not _is_number(value):
            raise ValueError("offset value must be a finite number")
        return

    if kind == "rgb_balance":
        gain = parameters.get("gain", [1.0, 1.0, 1.0])
        if not isinstance(gain, (list, tuple)) or len(gain) != 3:
            raise ValueError("rgb_balance gain must be a 3-element list")
        if any(not _is_number(g) or g < 0 for g in gain):
            raise ValueError("rgb_balance gain must be non-negative finite numbers")
        return

    if kind == "contrast":
        amount = parameters.get("amount", 1.0)
        pivot = parameters.get("pivot", 0.5)
        if not _is_number(amount):
            raise ValueError("contrast amount must be a finite number")
        if not _is_number(pivot) or not (0.0 <= pivot <= 1.0):
            raise ValueError("contrast pivot must be in [0, 1]")
        return

    if kind == "saturation":
        amount = parameters.get("amount", 1.0)
        if not _is_number(amount) or amount < 0:
            raise ValueError("saturation amount must be a non-negative finite number")
        return

    if kind == "hue_rotate":
        degrees = parameters.get("degrees", 0.0)
        if not _is_number(degrees):
            raise ValueError("hue_rotate degrees must be a finite number")
        return

    if kind == "curve":
        mode = parameters.get("mode", "rgb")
        if mode not in ("rgb", "per_channel", "luma"):
            raise ValueError("curve mode must be 'rgb', 'per_channel', or 'luma'")
        if mode == "per_channel":
            points = parameters.get("points")
            if not isinstance(points, dict) or set(points) != {"r", "g", "b"}:
                raise ValueError("per_channel curve points must be {r, g, b}")
            for key in ("r", "g", "b"):
                _validate_curve_points(points[key])
        else:
            _validate_curve_points(parameters.get("points", _IDENTITY_CURVE))
        return

    if kind == "lut":
        path = parameters.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("lut path must be a non-empty string")
        if parameters.get("space", "linear") not in ("linear", "display"):
            raise ValueError("lut space must be 'linear' or 'display'")
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


def _validate_curve_points(points: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate a curve's control points and return ``(xs, ys)`` arrays.

    Points must be a list of ``[x, y]`` pairs with ``x`` strictly increasing
    (so the curve is a function), ``y`` monotonic non-decreasing, and all
    values finite within ``[0, 1]``.
    """
    if not isinstance(points, (list, tuple)) or len(points) < 2:
        raise ValueError("curve points must be a list of at least 2 [x, y] pairs")
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("curve points must be [x, y] pairs")
    if any(not _is_number(v) for v in arr.ravel()):
        raise ValueError("curve points must be finite numbers")
    xs, ys = arr[:, 0], arr[:, 1]
    if np.any(xs < 0.0) or np.any(xs > 1.0) or np.any(ys < 0.0) or np.any(ys > 1.0):
        raise ValueError("curve points must lie within [0, 1]")
    if np.any(np.diff(xs) <= 0.0):
        raise ValueError("curve x coordinates must be strictly increasing")
    if np.any(np.diff(ys) < 0.0):
        raise ValueError("curve must be monotonic (non-decreasing)")
    return xs, ys


def apply_correction(image_rgb: np.ndarray, kind: str, parameters: dict[str, Any]) -> np.ndarray:
    """Apply one deterministic correction to an RGB image array."""
    validate_correction(kind, parameters)
    f, was_uint8 = _to_float_rgb(image_rgb)

    if kind == "hue_rotate":
        # Perceptual op: rotate hue on display-referred (gamma) values.
        degrees = float(parameters.get("degrees", 0.0))
        u8 = (np.clip(f, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(np.int32) + int(round(degrees / 2.0))) % 180
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float64) / 255.0
        return _finish(out, was_uint8)

    if kind == "lut" and parameters.get("space", "linear") == "display":
        # Display-referred LUT (authored in gamma space — the usual Resolve
        # Rec.709 .cube): apply on the gamma-encoded values, not linear light.
        from colorai.lutcube import apply_cube, load_cube

        out = apply_cube(load_cube(parameters["path"]), f)
        return _finish(out, was_uint8)

    # Grade in linear (scene-referred) light.
    lin = bt709_to_linear(f)

    if kind == "cdl":
        slope = _as_vec3(parameters.get("slope"), (1.0, 1.0, 1.0))
        offset = _as_vec3(parameters.get("offset"), (0.0, 0.0, 0.0))
        power = _as_vec3(parameters.get("power"), (1.0, 1.0, 1.0))
        lin = np.power(np.clip(lin * slope + offset, 0.0, None), 1.0 / power)
    elif kind == "exposure":
        lin = lin * float(parameters.get("gain", 1.0))
    elif kind == "offset":
        lin = lin + float(parameters.get("value", 0.0))
    elif kind == "rgb_balance":
        gain = _as_vec3(parameters.get("gain"), (1.0, 1.0, 1.0))
        lin = lin * gain
    elif kind == "contrast":
        amount = float(parameters.get("amount", 1.0))
        pivot = float(parameters.get("pivot", 0.5))
        lin = (lin - pivot) * amount + pivot
    elif kind == "saturation":
        amount = float(parameters.get("amount", 1.0))
        luma = _LUMA[0] * lin[..., 0] + _LUMA[1] * lin[..., 1] + _LUMA[2] * lin[..., 2]
        lin = luma[..., None] + (lin - luma[..., None]) * amount
    elif kind == "curve":
        mode = parameters.get("mode", "rgb")
        if mode == "luma":
            xs, ys = _validate_curve_points(parameters.get("points", _IDENTITY_CURVE))
            luma = _LUMA[0] * lin[..., 0] + _LUMA[1] * lin[..., 1] + _LUMA[2] * lin[..., 2]
            curved = np.interp(luma, xs, ys)
            ratio = np.where(luma > 1e-12, curved / np.maximum(luma, 1e-12), 1.0)
            lin = lin * ratio[..., None]
        elif mode == "per_channel":
            points = parameters["points"]
            for key, channel in (("r", 0), ("g", 1), ("b", 2)):
                xs, ys = _validate_curve_points(points[key])
                lin[..., channel] = np.interp(lin[..., channel], xs, ys)
        else:  # rgb
            xs, ys = _validate_curve_points(parameters.get("points", _IDENTITY_CURVE))
            lin = np.interp(lin, xs, ys)
    elif kind == "lut":
        from colorai.lutcube import apply_cube, load_cube

        lin = apply_cube(load_cube(parameters["path"]), lin)
    else:  # pragma: no cover - guarded by validate_correction
        raise ValueError(f"unknown correction kind: {kind!r}")

    return _finish(linear_to_bt709(lin), was_uint8)


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


def normalize_parameters(kind: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Return ``parameters`` ready for persistence.

    For ``lut`` corrections this adds ``content_hash`` (a fingerprint of the
    referenced ``.cube`` file) so the persisted row records exactly which file
    version it used. Other kinds pass through unchanged.
    """
    if kind == "lut":
        path = parameters.get("path")
        if isinstance(path, str) and path.strip():
            from colorai.lutcube import cube_content_hash

            out = dict(parameters)
            out.setdefault("content_hash", cube_content_hash(path))
            return out
    return parameters


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def load_corrected_still(store: ProjectStore, shot: Shot) -> np.ndarray:
    """Return the shot's representative still (BGR uint8) with enabled corrections applied."""
    from colorai.project.models import RepresentativeFrame

    with store.session() as session:
        asset = session.get(MediaAsset, shot.asset_id)
        if asset is not None and not is_gradeable_transfer(asset.transfer):
            raise ValueError(
                "grading is defined in BT.709, but this asset's transfer is "
                f"{asset.transfer!r}; non-Rec.709 masters are not yet gradeable"
            )
        rf = session.query(RepresentativeFrame).filter_by(shot_id=shot.id).first()
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
    return cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)


def preview_correction(store: ProjectStore, shot: Shot, out_path: str | Path) -> Path:
    """Render ``shot``'s representative still with its enabled corrections applied.

    Non-destructive: the original still is never overwritten.
    """
    out_bgr = load_corrected_still(store, shot)
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), out_bgr)
    return destination
