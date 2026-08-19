"""Color science: transfer functions and working-space handling.

Video is decoded as gamma-encoded (display-referred) Rec.709 RGB, but grading
operations are physically meaningful in *linear* (scene-referred) light. This
module provides the transfer functions to move between the two.

BT.709 and sRGB share the same OETF/EOTF piecewise curve, so one pair of
functions serves both. Everything operates on float arrays in ``[0, 1]``.

The **working space** for grading is fixed: BT.709 primaries/gamut and the
BT.709 transfer function. ffprobe reports a master's actual characteristics
(``color_space`` / ``color_transfer``); untagged H.264/MP4 is treated as
BT.709 by convention. Grading a master whose transfer is *not* BT.709 (e.g.
PQ/HDR or log) would silently mis-grade, so callers that know the asset should
check :func:`is_gradeable_transfer` first.
"""

from __future__ import annotations

import numpy as np

#: The transfer function grading is defined in (scene-referred linearization).
WORKING_TRANSFER = "bt709"
#: The color primaries / gamut grading is defined in.
WORKING_PRIMARIES = "bt709"

# ffprobe ``color_transfer`` values -> canonical short name. Values absent from
# this map are passed through unchanged (callers decide whether they're
# gradeable). sRGB shares BT.709's EOTF curve, so it aliases to ``bt709``.
_TRANSFER_ALIASES = {
    "bt709": "bt709",
    "bt.709": "bt709",
    "iec61966-2-1": "bt709",  # sRGB
    "iec61966-2-4": "bt709",  # xvYCC
    "bt470bg": "bt470bg",
    "smpte170m": "smpte170m",
    "bt2020-10": "bt2020",
    "bt2020-12": "bt2020",
    "arib-std-b67": "hlg",
    "smpte2084": "pq",
    "linear": "linear",
}

_UNTAGGED = {"", "unknown", "unspecified", "unspecified ", "n/a"}


def normalize_transfer(value: str | None) -> str:
    """Canonicalize an ffprobe ``color_transfer`` value.

    Untagged/unknown values fall back to the BT.709 working transfer (the
    standard assumption for Rec.709 HD content and baked H.264 masters).
    """
    if value is None:
        return WORKING_TRANSFER
    v = str(value).strip().lower()
    if v in _UNTAGGED:
        return WORKING_TRANSFER
    return _TRANSFER_ALIASES.get(v, v)


def normalize_color_space(value: str | None) -> str:
    """Canonicalize an ffprobe ``color_space`` value, defaulting to BT.709."""
    if value is None:
        return WORKING_PRIMARIES
    v = str(value).strip().lower()
    if v in _UNTAGGED:
        return WORKING_PRIMARIES
    aliases = {
        "bt709": "bt709",
        "bt.709": "bt709",
        "bt470bg": "bt470bg",
        "smpte170m": "smpte170m",
        "bt2020nc": "bt2020",
        "bt2020c": "bt2020",
        "smpte432": "bt2020",
        "smpte428": "bt2020",
    }
    return aliases.get(v, v)


def is_gradeable_transfer(transfer: str | None) -> bool:
    """True when the correction pipeline's linear BT.709 assumption holds.

    ``None`` (untagged) is treated as BT.709 and therefore gradeable.
    """
    return normalize_transfer(transfer) == WORKING_TRANSFER


def describe_working_space() -> str:
    """Human-readable description of the grading working space."""
    return "BT.709 (scene-referred linear light for grading, display-referred for hue)"


def bt709_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Decode gamma-encoded BT.709/sRGB values to linear light (EOTF)."""
    rgb = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        np.power((rgb + 0.055) / 1.055, 2.4),
    )


def linear_to_bt709(rgb: np.ndarray) -> np.ndarray:
    """Encode linear light to gamma-encoded BT.709/sRGB values (OETF)."""
    rgb = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    return np.where(
        rgb <= 0.0031308,
        rgb * 12.92,
        1.055 * np.power(rgb, 1.0 / 2.4) - 0.055,
    )
