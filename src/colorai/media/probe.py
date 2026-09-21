"""Media probing via ffprobe.

Extracts the stream metadata ColorAI needs to register a source master
(see :class:`colorai.project.models.MediaAsset`). Frame rate is parsed from
the exact ``avg_frame_rate`` rational (``"30000/1001"`` -> 29.97) so that
drop-frame detection stays correct.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from colorai.color import normalize_color_space, normalize_transfer

# Probe fields that map 1:1 onto MediaAsset columns (and are accepted by
# ``ProjectStore.add_asset`` as keyword arguments).
_ASSET_FIELDS = (
    "file_size_bytes",
    "width",
    "height",
    "frame_count",
    "duration_seconds",
    "pixel_format",
    "color_space",
    "transfer",
    "codec_name",
)


@dataclass(frozen=True)
class MediaProbe:
    """Probed metadata for a source master."""

    source_path: str
    file_size_bytes: int | None
    width: int | None
    height: int | None
    frame_rate: float
    frame_count: int | None
    duration_seconds: float | None
    pixel_format: str | None
    color_space: str | None
    transfer: str | None
    codec_name: str | None

    def asset_fields(self) -> dict[str, Any]:
        """Non-None fields suitable for ``ProjectStore.add_asset``."""
        return {name: getattr(self, name) for name in _ASSET_FIELDS if getattr(self, name) is not None}


def _parse_rate(rate: str) -> float:
    """Parse an ffprobe rational like ``"30000/1001"`` or ``"25/1"``.

    Raises ``ValueError`` for anything unparseable — including ffprobe's
    ``"0/0"`` and ``"N/A"`` sentinels for unknown rates — so callers can fall
    back to the next rate source instead of crashing deeper in the pipeline.
    """
    text = (rate or "").strip()
    num, sep, den = text.partition("/")
    try:
        if not sep or not den:
            return float(num)
        return float(Fraction(int(num), int(den)))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"unparseable frame rate: {rate!r}") from exc


def _int_or_none(value: Any) -> int | None:
    """``int(value)`` with ffprobe's ``"N/A"``/missing sentinels mapping to ``None``."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    """``float(value)`` with ffprobe's ``"N/A"``/missing sentinels mapping to ``None``."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def probe_media(path: str | Path) -> MediaProbe:
    """Run ffprobe on ``path`` and return its video stream metadata.

    Raises ``FileNotFoundError`` if ffprobe is unavailable, ``subprocess.CalledProcessError``
    on probe failure, and ``ValueError`` if the file has no video stream.
    """
    src = str(path)
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            src,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"ffprobe returned unparseable output for {src!r}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"ffprobe returned unexpected output for {src!r}")
    streams = data.get("streams", [])
    if not isinstance(streams, list):
        raise ValueError(f"ffprobe returned unexpected streams for {src!r}")

    video = next(
        (s for s in streams if isinstance(s, dict) and s.get("codec_type") == "video"),
        None,
    )
    if video is None:
        raise ValueError(f"no video stream found in {src!r}")

    fmt = data.get("format", {})
    if not isinstance(fmt, dict):
        fmt = {}
    duration = _float_or_none(fmt.get("duration"))
    size = _int_or_none(fmt.get("size"))

    frame_rate = None
    for key in ("avg_frame_rate", "r_frame_rate"):
        try:
            frame_rate = _parse_rate(video.get(key, ""))
            break
        except ValueError:
            continue
    if frame_rate is None:
        raise ValueError(f"no usable frame rate reported for {src!r}")

    frame_count = _int_or_none(video.get("nb_frames"))
    if frame_count is None and duration is not None:
        frame_count = round(duration * frame_rate)

    return MediaProbe(
        source_path=src,
        file_size_bytes=size,
        width=video.get("width"),
        height=video.get("height"),
        frame_rate=frame_rate,
        frame_count=frame_count,
        duration_seconds=duration,
        pixel_format=video.get("pix_fmt"),
        color_space=normalize_color_space(video.get("color_space")),
        transfer=normalize_transfer(video.get("color_transfer")),
        codec_name=video.get("codec_name"),
    )
