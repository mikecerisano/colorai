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
    """Parse an ffprobe rational like ``"30000/1001"`` or ``"25/1"``."""
    num, sep, den = rate.partition("/")
    if not sep or not den:
        return float(num)
    return float(Fraction(int(num), int(den)))


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
    data = json.loads(result.stdout)

    video = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
        None,
    )
    if video is None:
        raise ValueError(f"no video stream found in {src!r}")

    fmt = data.get("format", {})
    duration = fmt.get("duration")
    size = fmt.get("size")

    frame_rate = _parse_rate(video.get("avg_frame_rate", video.get("r_frame_rate", "0/1")))

    nb_frames = video.get("nb_frames")
    if nb_frames is not None:
        frame_count = int(nb_frames)
    elif duration is not None:
        frame_count = round(float(duration) * frame_rate)
    else:
        frame_count = None

    return MediaProbe(
        source_path=src,
        file_size_bytes=int(size) if size is not None else None,
        width=video.get("width"),
        height=video.get("height"),
        frame_rate=frame_rate,
        frame_count=frame_count,
        duration_seconds=float(duration) if duration is not None else None,
        pixel_format=video.get("pix_fmt"),
        color_space=video.get("color_space"),
        transfer=video.get("color_transfer"),
        codec_name=video.get("codec_name"),
    )
