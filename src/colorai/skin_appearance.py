"""Pure perceptual skin-appearance math (OKLab, bounded, deterministic).

ColorAI's skin-target workflow never repaints or generates pixels. It measures
a masked face in OKLab opponent chroma, derives a *bounded* chroma-only
transformation, and applies it only under a face mask in the same display-
linear working space as the rest of the correction pipeline (``bt709_to_linear``
/ ``linear_to_bt709`` — the display-referred sRGB/BT.1886 EOTF for baked
Rec.709 masters).

Version one parameters are deliberately conservative:

* ``space == "oklab"`` and ``luma_mode == "preserve"`` (no lightness change);
* ``ab_offset`` moves opponent chroma, bounded to ``[-0.04, 0.04]``;
* ``ab_scale`` compresses/expands chroma spread, bounded to ``[0.85, 1.15]``;
* ``strength`` in ``[0, 1]`` blends the whole transform toward identity.

All functions are pure (no I/O) so preview and full-master render use exactly
the same code path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# OKLab conversion matrices (Björn Ottosson). Linear sRGB <-> OKLab.
_LMS_FROM_RGB = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ],
    dtype=np.float64,
)
_OKLAB_FROM_LMS = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    dtype=np.float64,
)
_LMS_FROM_OKLAB = np.array(
    [
        [1.0, 0.3963377774, 0.2158037573],
        [1.0, -0.1055613458, -0.0638541728],
        [1.0, -0.0894841775, -1.2914855480],
    ],
    dtype=np.float64,
)
_RGB_FROM_LMS = np.array(
    [
        [4.0767416621, -3.3077115913, 0.2309699292],
        [-1.2684380046, 2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147, 1.7076147010],
    ],
    dtype=np.float64,
)

AB_OFFSET_BOUND = 0.04
AB_SCALE_MIN = 0.85
AB_SCALE_MAX = 1.15


@dataclass(frozen=True)
class SkinAppearanceParameters:
    """Validated version-one ``skin_appearance`` transform."""

    version: int
    luma_mode: str
    ab_offset: tuple[float, float]
    ab_scale: tuple[float, float]
    strength: float


def rgb_linear_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert linear sRGB float ``[0,1]`` (any ``...x3`` shape) to OKLab."""
    rgb = np.asarray(rgb, dtype=np.float64)
    lms = rgb @ _LMS_FROM_RGB.T
    lms = np.cbrt(lms)
    return lms @ _OKLAB_FROM_LMS.T


def oklab_to_rgb_linear(lab: np.ndarray) -> np.ndarray:
    """Convert OKLab back to linear sRGB float (inverse of the above)."""
    lab = np.asarray(lab, dtype=np.float64)
    lms = lab @ _LMS_FROM_OKLAB.T
    lms = lms ** 3.0
    return lms @ _RGB_FROM_LMS.T


def _finite_pair(value, name: str, lo: float, hi: float) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a 2-element list")
    out: list[float] = []
    for v in value:
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):
            raise ValueError(f"{name} values must be finite")
        if not (lo <= f <= hi):
            raise ValueError(f"{name} {f:.3f} outside bounds [{lo}, {hi}]")
        out.append(f)
    return tuple(out)


def validate_skin_appearance_parameters(params: dict) -> SkinAppearanceParameters:
    """Validate a ``skin_appearance`` parameter dict and return a typed object.

    Shared by API intake, preview, and render preflight. Raises ``ValueError``
    for any out-of-contract value (non-finite, wrong space/luma mode, or
    out-of-bounds chroma changes).
    """
    if not isinstance(params, dict):
        raise ValueError("skin_appearance parameters must be a dict")

    # Validate chroma changes first so a malformed ab_offset/ab_scale always
    # names the offending field (regardless of other missing keys).
    ab_offset = _finite_pair(
        params.get("ab_offset"), "ab_offset", -AB_OFFSET_BOUND, AB_OFFSET_BOUND
    )
    ab_scale = _finite_pair(
        params.get("ab_scale"), "ab_scale", AB_SCALE_MIN, AB_SCALE_MAX
    )

    version = params.get("version")
    if version != 1:
        raise ValueError("skin_appearance version must be 1")
    if params.get("space") != "oklab":
        raise ValueError("skin_appearance space must be 'oklab'")
    luma_mode = params.get("luma_mode")
    if luma_mode != "preserve":
        raise ValueError("skin_appearance luma_mode must be 'preserve'")

    strength = params.get("strength", 1.0)
    try:
        strength = float(strength)
    except (TypeError, ValueError):
        raise ValueError("skin_appearance strength must be a number")
    if strength != strength or strength in (float("inf"), float("-inf")):
        raise ValueError("skin_appearance strength must be finite")
    if not (0.0 <= strength <= 1.0):
        raise ValueError("skin_appearance strength must be in [0, 1]")

    return SkinAppearanceParameters(
        version=1, luma_mode="preserve", ab_offset=ab_offset, ab_scale=ab_scale, strength=strength
    )


def masked_skin_profile(linear_rgb: np.ndarray, alpha: np.ndarray) -> dict[str, list[float]]:
    """Robust OKLab chroma profile of masked skin pixels.

    ``linear_rgb`` is HxWx3 linear RGB; ``alpha`` is a same-shaped HxW float
    mask in ``[0,1]``. Pixels below ``alpha < 0.5`` are excluded. Returns a
    robust centre/spread for opponent chroma (median + MAD), plus a median
    lightness. An empty mask yields a neutral zero profile.
    """
    lab = rgb_linear_to_oklab(linear_rgb)
    mask = np.asarray(alpha, dtype=np.float64) >= 0.5
    if lab.shape[:2] != mask.shape:
        raise ValueError("alpha mask must match the image height/width")
    a = lab[..., 1][mask]
    b = lab[..., 2][mask]
    l = lab[..., 0][mask]
    if a.size == 0:
        return {"mean_ab": [0.0, 0.0], "spread_ab": [0.0, 0.0], "luma_median": 0.0}

    def _median_abs_dev(x: np.ndarray) -> float:
        med = float(np.median(x))
        return float(np.median(np.abs(x - med)) * 1.4826)

    return {
        "mean_ab": [float(np.median(a)), float(np.median(b))],
        "spread_ab": [_median_abs_dev(a), _median_abs_dev(b)],
        "luma_median": float(np.median(l)),
    }


def _clip_offset(v: float) -> float:
    return max(-AB_OFFSET_BOUND, min(AB_OFFSET_BOUND, v))


def _clip_scale(v: float) -> float:
    return max(AB_SCALE_MIN, min(AB_SCALE_MAX, v))


def derive_skin_appearance_parameters(
    candidate: dict[str, list[float]],
    target: dict[str, list[float]],
    *,
    strength: float,
) -> SkinAppearanceParameters:
    """Derive bounded chroma-only parameters moving ``candidate`` toward ``target``.

    ``candidate`` and ``target`` are masked profiles (as returned by
    :func:`masked_skin_profile`) carrying ``mean_ab`` and, optionally,
    ``spread_ab``. ``ab_offset`` shifts the candidate's opponent-chroma centre
    toward the target's; ``ab_scale`` pulls its spread toward the target's.
    Both are softened by ``strength`` and clamped to the version-one bounds.
    """
    if not (0.0 <= strength <= 1.0):
        raise ValueError("strength must be in [0, 1]")

    c_mean = candidate.get("mean_ab", [0.0, 0.0])
    t_mean = target.get("mean_ab", [0.0, 0.0])
    ab_offset = (
        _clip_offset((t_mean[0] - c_mean[0]) * strength),
        _clip_offset((t_mean[1] - c_mean[1]) * strength),
    )

    c_spread = candidate.get("spread_ab")
    t_spread = target.get("spread_ab")
    if c_spread is not None and t_spread is not None and len(c_spread) == 2 and len(t_spread) == 2:
        def _scale(c: float, t: float) -> float:
            raw = (t / c) if abs(c) > 1e-9 else 1.0
            return 1.0 + (_clip_scale(raw) - 1.0) * strength

        ab_scale = (_scale(c_spread[0], t_spread[0]), _scale(c_spread[1], t_spread[1]))
    else:
        ab_scale = (1.0, 1.0)

    return SkinAppearanceParameters(
        version=1,
        luma_mode="preserve",
        ab_offset=ab_offset,
        ab_scale=(float(ab_scale[0]), float(ab_scale[1])),
        strength=float(strength),
    )


def apply_skin_appearance(
    linear_rgb: np.ndarray,
    params: SkinAppearanceParameters,
    alpha: np.ndarray,
) -> np.ndarray:
    """Apply a chroma-only OKLab transform under ``alpha`` and return linear RGB.

    Preserves the original OKLab lightness exactly; transforms only opponent
    chroma ``a/b`` (offset then scale), blends the result with the original by
    ``alpha``, and converts back to linear RGB. ``linear_rgb`` and ``alpha``
    must share the image height/width.
    """
    lab = rgb_linear_to_oklab(linear_rgb)
    alpha = np.asarray(alpha, dtype=np.float64)
    if lab.shape[:2] != alpha.shape:
        raise ValueError("alpha mask must match the image height/width")

    a = lab[..., 1]
    b = lab[..., 2]
    new_a = (a + params.ab_offset[0]) * params.ab_scale[0]
    new_b = (b + params.ab_offset[1]) * params.ab_scale[1]

    a_out = a * (1.0 - alpha) + new_a * alpha
    b_out = b * (1.0 - alpha) + new_b * alpha

    out = lab.copy()
    out[..., 1] = a_out
    out[..., 2] = b_out
    return np.clip(oklab_to_rgb_linear(out), 0.0, 1.0)
