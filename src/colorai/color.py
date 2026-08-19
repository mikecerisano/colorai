"""Color science: transfer functions and working-space handling.

Two transfer-function families are kept deliberately distinct:

* **Display-referred (sRGB / BT.1886 EOTF)** — how a *baked* master's code
  values map to display light. This is what the grading pipeline decodes
  with, because finished Rec.709 masters are display-referred, not
  scene-linear. At code value 0.5 this gives ~0.214 linear.
* **BT.709 camera OETF** (Rec. ITU-R BT.709) — the scene-linear -> code
  transfer used at capture (``1.099 * L**0.45 - 0.099`` with a 4.5 knee). Its
  inverse maps code back to scene linear; at code value 0.5 it gives ~0.260.

Conflating the two is a 17%+ error: the camera OETF inverse is for scene-
linear interchange, the display EOTF is for baked-master decoding.

Everything operates on float arrays in ``[0, 1]``. The **working space** for
grading is fixed: BT.709 primaries/gamut and the display-referred decode
(sRGB/BT.1886 EOTF). ffprobe reports a master's actual characteristics
(``color_space`` / ``color_transfer``); untagged H.264/MP4 is treated as
BT.709 by convention. Grading a master whose transfer is *not* BT.709 (e.g.
PQ/HDR or log) would silently mis-grade, so callers that know the asset should
check :func:`is_gradeable_transfer` first.
"""

from __future__ import annotations

import numpy as np

#: The transfer function grading is defined in (display-referred decode).
WORKING_TRANSFER = "bt709"
#: The color primaries / gamut grading is defined in.
WORKING_PRIMARIES = "bt709"

# ffprobe ``color_transfer`` values -> canonical short name. Values absent from
# this map are passed through unchanged (callers decide whether they're
# gradeable). sRGB shares BT.709's display EOTF curve, so it aliases to
# ``bt709``.
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
    """True when the correction pipeline's BT.709 working-space assumption holds.

    ``None`` (untagged) is treated as BT.709 and therefore gradeable.
    """
    return normalize_transfer(transfer) == WORKING_TRANSFER


def describe_working_space() -> str:
    """Human-readable description of the grading working space."""
    return (
        "BT.709 primaries; baked masters decoded with the display-referred "
        "sRGB/BT.1886 EOTF, graded in linear light, re-encoded with its inverse"
    )


# ---------------------------------------------------------------------------
# Display-referred transfer (sRGB / BT.1886 EOTF) — baked-master decode
# ---------------------------------------------------------------------------

def srgb_eotf(rgb: np.ndarray) -> np.ndarray:
    """Decode display-referred sRGB/BT.1886 code values to display-linear light.

    This is the correct decode for a *baked* master (its code values follow the
    display transfer, not the camera OETF). Standard value: ``srgb_eotf(0.5)``
    is ~0.2140.
    """
    rgb = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        np.power((rgb + 0.055) / 1.055, 2.4),
    )


def srgb_oetf(rgb: np.ndarray) -> np.ndarray:
    """Encode display-linear light to display-referred sRGB/BT.1886 code values."""
    rgb = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    return np.where(
        rgb <= 0.0031308,
        rgb * 12.92,
        1.055 * np.power(rgb, 1.0 / 2.4) - 0.055,
    )


# ---------------------------------------------------------------------------
# BT.709 camera transfer (Rec. ITU-R BT.709 OETF) — scene-linear interchange
# ---------------------------------------------------------------------------

def bt709_oetf(linear: np.ndarray) -> np.ndarray:
    """Encode scene-linear light with the BT.709 camera OETF.

    ``V = 1.099 * L**0.45 - 0.099`` for ``L >= 0.018``, ``V = 4.5 * L`` below.
    Standard values: ``bt709_oetf(0.018)`` is 0.081, ``bt709_oetf(0.18)`` is
    ~0.4089, ``bt709_oetf(1.0)`` is 1.0.
    """
    l = np.clip(np.asarray(linear, dtype=np.float64), 0.0, None)
    return np.where(
        l < 0.018,
        4.5 * l,
        1.099 * np.power(l, 0.45) - 0.099,
    )


def bt709_oetf_inverse(code: np.ndarray) -> np.ndarray:
    """Decode BT.709 camera code values back to scene-linear light.

    Inverse of :func:`bt709_oetf`: below the knee (``V < 0.081``) it is
    ``V / 4.5``; above, ``((V + 0.099) / 1.099) ** (1 / 0.45)``. Standard
    value: ``bt709_oetf_inverse(0.5)`` is ~0.2596.
    """
    v = np.clip(np.asarray(code, dtype=np.float64), 0.0, 1.0)
    return np.where(
        v < 0.081,
        v / 4.5,
        np.power((v + 0.099) / 1.099, 1.0 / 0.45),
    )


# Compatibility aliases used across the pipeline: the display-referred decode
# for baked masters (NOT the BT.709 camera OETF — see module docstring).
def bt709_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Alias of :func:`srgb_eotf` — display-referred decode for baked masters."""
    return srgb_eotf(rgb)


def linear_to_bt709(rgb: np.ndarray) -> np.ndarray:
    """Alias of :func:`srgb_oetf` — display-referred encode for baked masters."""
    return srgb_oetf(rgb)
