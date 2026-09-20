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

Everything operates on float arrays in ``[0, 1]``. The whole-frame grade is
**transfer-aware**: each supported transfer decodes to linear light with its
own EOTF, grades there, and re-encodes with the matching inverse — so a PQ
master round-trips through PQ, an HLG master through HLG, with no tone
mapping and no guessing. ffprobe reports a master's actual characteristics
(``color_space`` / ``color_transfer``); untagged H.264/MP4 is treated as
BT.709 by convention, and an operator may *declare* a transfer at ingest for
an untagged file (never inferred). Transfers without a known decode/encode
pair (camera log curves, exotic tags) are refused rather than silently
mis-graded, so callers that know the asset should check
:func:`is_gradeable_transfer` first.

The **face/skin subsystem stays BT.709-calibrated** (display-linear
thresholds validated there); face-local corrections on non-BT.709 assets are
refused at approve/enable/render time.
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


#: Transfers with a known decode/encode pair for the transfer-aware grade.
SUPPORTED_TRANSFERS = ("bt709", "pq", "hlg")


def is_gradeable_transfer(transfer: str | None) -> bool:
    """True when the whole-frame grade knows this transfer's EOTF pair.

    ``None`` (untagged) is treated as BT.709 and therefore gradeable.
    Camera log curves and exotic tags have no pair and are refused.
    """
    return normalize_transfer(transfer) in SUPPORTED_TRANSFERS


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


# ---------------------------------------------------------------------------
# Non-Rec.709 transfer decodes (measurement / future-path helpers)
# ---------------------------------------------------------------------------
#
# The grading pipeline stays BT.709-only (see :func:`is_gradeable_transfer`):
# these decodes exist so HDR / log-adjacent masters can be *measured* and so a
# future managed (OCIO) path has spec-standard primitives to build on. They are
# not wired into grading — nothing here changes the non-Rec.709 refusal.

# SMPTE ST 2084 (PQ) constants: peak 10 000 cd/m^2, output normalized to [0, 1].
_PQ_M1 = 2610.0 / 16384.0
_PQ_M2 = 2523.0 / 4096.0 * 128.0
_PQ_C1 = 3424.0 / 4096.0
_PQ_C2 = 2413.0 / 4096.0 * 32.0
_PQ_C3 = 2392.0 / 4096.0 * 32.0

# ARIB STD-B67 (HLG) OETF constants.
_HLG_A = 0.17883277
_HLG_B = 0.28466892
_HLG_C = 0.55991073


def pq_eotf(code: np.ndarray) -> np.ndarray:
    """Decode SMPTE ST 2084 (PQ) code values to normalized linear light.

    Output is relative to the 10 000 cd/m^2 peak (``pq_eotf(1.0)`` is 1.0).
    Standard value: ``pq_eotf(0.5)`` is ~0.0094 (~94 nits).
    """
    n = np.clip(np.asarray(code, dtype=np.float64), 0.0, 1.0)
    n_pow = np.power(n, 1.0 / _PQ_M2)
    denom = np.maximum(_PQ_C2 - _PQ_C3 * n_pow, 1e-12)
    return np.power(np.maximum(n_pow - _PQ_C1, 0.0) / denom, 1.0 / _PQ_M1)


def hlg_oetf(scene: np.ndarray) -> np.ndarray:
    """Encode scene-linear light with the HLG (ARIB STD-B67) OETF."""
    e = np.clip(np.asarray(scene, dtype=np.float64), 0.0, None)
    return np.where(
        e <= 1.0 / 12.0,
        np.sqrt(3.0 * np.maximum(e, 0.0)),
        _HLG_A * np.log(12.0 * e - _HLG_B) + _HLG_C,
    )


def hlg_oetf_inverse(code: np.ndarray) -> np.ndarray:
    """Decode HLG code values back to scene-linear light.

    Standard value: ``hlg_oetf_inverse(0.5)`` is ``1/12`` (~0.0833).
    """
    v = np.clip(np.asarray(code, dtype=np.float64), 0.0, 1.0)
    return np.where(
        v <= 0.5,
        v * v / 3.0,
        (np.exp((v - _HLG_C) / _HLG_A) + _HLG_B) / 12.0,
    )


def non_gradeable_reason(transfer: str | None) -> str | None:
    """Explain why ``transfer`` cannot be graded, or ``None`` when it can.

    Untagged masters are assumed BT.709 and gradeable, as are PQ and HLG
    (graded transfer-natively with no tone mapping). Anything without a known
    decode/encode pair — camera log curves, exotic tags — is refused rather
    than silently mis-graded.
    """
    canonical = normalize_transfer(transfer)
    if canonical in SUPPORTED_TRANSFERS:
        return None
    seen = "untagged" if transfer is None else repr(transfer)
    return (
        f"this asset's transfer is {seen} (normalized: {canonical!r}), which has "
        "no known decode/encode pair — declare PQ/HLG explicitly at ingest when "
        "that is what the file is, or deliver a Rec.709 mezzanine. Camera log "
        "is never guessed."
    )


def pq_oetf(linear: np.ndarray) -> np.ndarray:
    """Encode normalized linear light with the SMPTE ST 2084 (PQ) OETF.

    Inverse of :func:`pq_eotf` for ``[0, 1]`` (relative to the 10 000 cd/m^2
    peak): ``pq_oetf(1.0)`` is 1.0; ``pq_oetf(0.0)`` is ~7e-7 (the curve
    never quite touches zero — use an absolute tolerance near black).
    """
    y = np.clip(np.asarray(linear, dtype=np.float64), 0.0, 1.0)
    num = _PQ_C1 + _PQ_C2 * np.power(y, _PQ_M1)
    den = 1.0 + _PQ_C3 * np.power(y, _PQ_M1)
    return np.power(num / den, _PQ_M2)


def decode_transfer(code: np.ndarray, transfer: str | None) -> np.ndarray:
    """Decode code values to linear light with ``transfer``'s EOTF.

    Raises ``ValueError`` (with :func:`non_gradeable_reason`) for transfers
    without a known pair.
    """
    canonical = normalize_transfer(transfer)
    if canonical == "bt709":
        return srgb_eotf(code)
    if canonical == "pq":
        return pq_eotf(code)
    if canonical == "hlg":
        return hlg_oetf_inverse(code)
    raise ValueError(non_gradeable_reason(transfer))


def encode_transfer(linear: np.ndarray, transfer: str | None) -> np.ndarray:
    """Encode linear light to code values with ``transfer``'s inverse EOTF.

    Raises ``ValueError`` (with :func:`non_gradeable_reason`) for transfers
    without a known pair.
    """
    canonical = normalize_transfer(transfer)
    if canonical == "bt709":
        return srgb_oetf(linear)
    if canonical == "pq":
        return pq_oetf(linear)
    if canonical == "hlg":
        return hlg_oetf(linear)
    raise ValueError(non_gradeable_reason(transfer))
