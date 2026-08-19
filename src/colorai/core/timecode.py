"""Timecode and frame-number conversion.

SMPTE timecode is the canonical human-facing timeline coordinate in ColorAI.
Everything persisted in the project database references both a frame number
(zero-based, index into the decoded video stream) and a timecode string, so
that source masters with different frame rates remain unambiguous.

Supported conventions:

* Non-drop-frame (NDF) — used for 23.976 / 24 / 25 / 30 / 50 / 60 fps.
* Drop-frame (DF) — used for 29.97 and 59.94 fps so that timecode stays
  aligned with wall-clock time.

The frame number returned by :func:`timecode_to_frames` is the *absolute
frame index* (0-based) into the source stream, matching ffmpeg/ffprobe
semantics (``nb_frames``, ``-vf select=eq(n\\,N)``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TC_RE = re.compile(r"^(?P<h>\d{1,2}):(?P<m>[0-5]?\d):(?P<s>[0-5]?\d)[:;](?P<f>\d{1,2})$")


@dataclass(frozen=True)
class TimecodeParts:
    """Parsed timecode components."""

    hours: int
    minutes: int
    seconds: int
    frames: int

    def __str__(self) -> str:
        return f"{self.hours:02d}:{self.minutes:02d}:{self.seconds:02d}:{self.frames:02d}"


def parse_timecode(tc: str) -> TimecodeParts:
    """Parse ``HH:MM:SS:FF`` (or ``;`` separator for drop-frame).

    Raises ``ValueError`` on malformed input.
    """
    m = _TC_RE.match(tc.strip())
    if not m:
        raise ValueError(f"invalid timecode: {tc!r}")
    return TimecodeParts(
        hours=int(m.group("h")),
        minutes=int(m.group("m")),
        seconds=int(m.group("s")),
        frames=int(m.group("f")),
    )


def _round_fps(fps: float) -> int:
    """Timecode frame count is always based on the integer-rate approximation."""
    return int(round(fps))


def _drop_count(fps: float) -> int:
    """Number of frame labels dropped per minute for drop-frame rates.

    29.97 -> 2, 59.94 -> 4. Non-drop-frame rates return 0.
    """
    return round(fps * 0.066666)


def _is_drop_frame(fps: float) -> bool:
    """Infer drop-frame from the frame rate.

    Drop-frame exists only for the NTSC fractional rates (29.97 / 59.94).
    Integer 30 / 60 fps are non-drop-frame; rounding alone cannot tell them
    apart, so the inference checks the deviation from the integer rate.
    """
    r = _round_fps(fps)
    return r in (30, 60) and abs(fps - r) > 1e-6


def is_drop_frame(fps: float) -> bool:
    """Public form of :func:`_is_drop_frame`.

    True when ``fps`` denotes an NTSC fractional rate (29.97 / 59.94) whose
    timecode uses drop-frame numbering.
    """
    return _is_drop_frame(fps)


def frames_to_timecode(frame: int, fps: float, drop_frame: bool | None = None) -> str:
    """Convert a zero-based absolute frame index to SMPTE timecode.

    ``drop_frame`` may be passed explicitly; when omitted it is inferred from
    the frame rate (drop-frame for 29.97/59.94, non-drop-frame otherwise).
    Drop-frame timecode uses the ``;`` frame separator per SMPTE convention.
    """
    if frame < 0:
        raise ValueError("frame must be >= 0")
    fps_r = _round_fps(fps)
    if drop_frame is None:
        drop_frame = _is_drop_frame(fps)
    if drop_frame and fps_r not in (30, 60):
        raise ValueError(
            f"drop-frame timecode only exists for 29.97/59.94 rates, not {fps} fps"
        )

    if drop_frame:
        return _frames_to_tc_df(frame, fps_r, _drop_count(fps))

    ff = frame % fps_r
    ss = (frame // fps_r) % 60
    mm = (frame // (fps_r * 60)) % 60
    hh = frame // (fps_r * 3600)
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def _frames_to_tc_df(frame: int, fps_r: int, drop: int) -> str:
    """Drop-frame conversion using the SMPTE drop-frame label accounting."""
    # A normal minute omits `drop` frame labels (00..drop-1) from its
    # counting; every 10th minute does not.
    frames_per_min = fps_r * 60 - drop
    frames_per_10min = frames_per_min * 9 + fps_r * 60

    d = frame // frames_per_10min
    m = frame % frames_per_10min

    # Reconstruct the *displayed* label by adding back the dropped labels that
    # precede this frame.
    if m > drop:
        adjusted = frame + drop * 9 * d + drop * ((m - drop) // frames_per_min)
    else:
        adjusted = frame + drop * 9 * d

    ff = adjusted % fps_r
    ss = (adjusted // fps_r) % 60
    mm = (adjusted // (fps_r * 60)) % 60
    hh = adjusted // (fps_r * 3600)
    return f"{hh:02d}:{mm:02d}:{ss:02d};{ff:02d}"


def timecode_to_frames(tc: str, fps: float, drop_frame: bool | None = None) -> int:
    """Convert SMPTE timecode to a zero-based absolute frame index."""
    p = parse_timecode(tc)
    fps_r = _round_fps(fps)
    if drop_frame is None:
        drop_frame = _is_drop_frame(fps)
    if drop_frame and fps_r not in (30, 60):
        raise ValueError(
            f"drop-frame timecode only exists for 29.97/59.94 rates, not {fps} fps"
        )

    if p.seconds >= 60 or p.minutes >= 60 or p.frames >= fps_r:
        raise ValueError(f"timecode components out of range for {fps_r} fps: {tc!r}")

    total = ((p.hours * 60 + p.minutes) * 60 + p.seconds) * fps_r + p.frames

    if drop_frame:
        return _tc_df_to_frames(total, fps_r, _drop_count(fps))
    return total


def _tc_df_to_frames(total_labels: int, fps_r: int, drop: int) -> int:
    """Inverse of :func:`_frames_to_tc_df` (labels -> real frame count).

    ``total_labels`` is the naive count of SMPTE labels from ``00:00:00;00``,
    i.e. it includes the dropped labels as if they existed. Each 10-minute
    block is ``fps_r * 60 * 10 - drop * 9`` labels long; within a block the
    dropped labels are ``drop`` consecutive labels at the start of every
    minute except the first (minute 0 of the block, a multiple of 10, never
    drops).

    In label space, the ``k``-th drop group of a block sits at
    ``fps_r * 60 * k + drop * 9 * d`` (``d`` = block index), so the number of
    drop groups that precede the label at block offset ``m`` is
    ``(m - drop * 9 * d - 1) // (fps_r * 60)``.
    """
    frames_per_min = fps_r * 60 - drop
    frames_per_10min = frames_per_min * 9 + fps_r * 60
    dropped_per_block = drop * 9

    d = total_labels // frames_per_10min
    m = total_labels % frames_per_10min

    # First minute of a 10-minute block is a non-drop minute.
    if m < fps_r * 60 + dropped_per_block * d:
        groups = 0
    else:
        groups = (m - dropped_per_block * d - 1) // (fps_r * 60)

    return total_labels - dropped_per_block * d - drop * groups


def timecode_to_seconds(tc: str, fps: float, drop_frame: bool | None = None) -> float:
    """Timecode -> seconds (real time)."""
    return timecode_to_frames(tc, fps, drop_frame) / fps


def seconds_to_timecode(seconds: float, fps: float, drop_frame: bool | None = None) -> str:
    """Seconds -> timecode (real time)."""
    return frames_to_timecode(int(round(seconds * fps)), fps, drop_frame)


def frame_to_seconds(frame: int, fps: float) -> float:
    """Absolute frame index -> seconds."""
    return frame / fps


def seconds_to_frame(seconds: float, fps: float) -> int:
    """Seconds -> nearest absolute frame index."""
    return int(round(seconds * fps))


def format_seconds(seconds: float, fps: float, drop_frame: bool | None = None) -> str:
    """Human-friendly ``HH:MM:SS.fff`` (not SMPTE) from seconds."""
    total_ms = int(round(seconds * 1000))
    h = total_ms // 3_600_000
    m = (total_ms % 3_600_000) // 60_000
    s = (total_ms % 60_000) / 1000.0
    return f"{h:02d}:{m:02d}:{s:06.3f}"
