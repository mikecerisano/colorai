"""Tests for SMPTE timecode and frame-number conversion.

Reference anchors are hand-derived from the SMPTE drop-frame rules:
frames 00 and 01 (00..03 at 59.94) are skipped at the start of every
minute except every 10th, so that timecode tracks wall-clock time.
"""

from __future__ import annotations

import pytest

from colorai.core.timecode import (
    TimecodeParts,
    format_seconds,
    frame_to_seconds,
    frames_to_timecode,
    is_drop_frame,
    parse_timecode,
    seconds_to_frame,
    seconds_to_timecode,
    timecode_to_frames,
    timecode_to_seconds,
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_parse_ndf():
    p = parse_timecode("01:23:45:12")
    assert p == TimecodeParts(hours=1, minutes=23, seconds=45, frames=12)
    assert str(p) == "01:23:45:12"


def test_parse_df_separator():
    p = parse_timecode("01:23:45;12")
    assert p == TimecodeParts(hours=1, minutes=23, seconds=45, frames=12)


def test_parse_strips_whitespace():
    assert parse_timecode("  01:02:03:04  ") == TimecodeParts(1, 2, 3, 4)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "01:02:03",
        "1:2:3:4:5",
        "aa:bb:cc:dd",
        "01:60:00:00",
        "01:00:00:123",  # frames field must be 1-2 digits
        "-1:00:00:00",
        "01:02:03:04junk",
    ],
)
def test_parse_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_timecode(bad)


# ---------------------------------------------------------------------------
# Non-drop-frame anchors (24 / 25 fps)
# ---------------------------------------------------------------------------

def test_ndf_25fps_anchors():
    assert frames_to_timecode(0, 25.0) == "00:00:00:00"
    assert frames_to_timecode(25, 25.0) == "00:00:01:00"
    assert frames_to_timecode(25 * 60, 25.0) == "00:01:00:00"
    assert frames_to_timecode(25 * 3600, 25.0) == "01:00:00:00"
    assert frames_to_timecode(25 * 3600 - 1, 25.0) == "00:59:59:24"


def test_ndf_24fps_inferred_non_drop():
    # 23.976 rounds to 24 and must stay non-drop-frame.
    assert frames_to_timecode(24 * 60, 23.976) == "00:01:00:00"
    assert frames_to_timecode(24 * 60, 24.0) == "00:01:00:00"


def test_ndf_roundtrip_sampled():
    for fps in (23.976, 24.0, 25.0, 50.0):
        for f in (0, 1, 24, 25, 1499, 1500, 89999, 90000, 899999, 900000):
            tc = frames_to_timecode(f, fps)
            assert timecode_to_frames(tc, fps) == f


# ---------------------------------------------------------------------------
# Drop-frame anchors (29.97)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        (0, "00:00:00;00"),
        (1799, "00:00:59;29"),   # last label of minute 0
        (1800, "00:01:00;02"),   # labels 00/01 skipped
        (17982, "00:10:00;00"),  # 10th minute: no drop
        (107892, "01:00:00;00"), # exactly one hour of real time
    ],
)
def test_df_2997_anchors(frame, expected):
    assert frames_to_timecode(frame, 29.97) == expected


@pytest.mark.parametrize(
    ("tc", "expected_frame"),
    [
        ("00:00:00;00", 0),
        ("00:00:59;29", 1799),
        ("00:01:00;02", 1800),
        ("00:10:00;00", 17982),
        ("01:00:00;00", 107892),
    ],
)
def test_df_2997_inverse_anchors(tc, expected_frame):
    assert timecode_to_frames(tc, 29.97) == expected_frame


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        (0, "00:00:00;00"),
        (3599, "00:00:59;59"),
        (3600, "00:01:00;04"),   # labels 00-03 skipped at 59.94
        (35964, "00:10:00;00"),
        (215784, "01:00:00;00"), # 59.94 * 3600
    ],
)
def test_df_5994_anchors(frame, expected):
    assert frames_to_timecode(frame, 59.94) == expected


@pytest.mark.parametrize(
    ("tc", "expected_frame"),
    [
        ("00:00:59;59", 3599),
        ("00:01:00;04", 3600),
        ("00:10:00;00", 35964),
        ("01:00:00;00", 215784),
    ],
)
def test_df_5994_inverse_anchors(tc, expected_frame):
    assert timecode_to_frames(tc, 59.94) == expected_frame


def test_df_colon_separator_tolerated():
    # Many tools write drop-frame timecode with ':' — accept it.
    assert timecode_to_frames("00:01:00:02", 29.97) == 1800


# ---------------------------------------------------------------------------
# Exhaustive round-trips (drop-frame)
# ---------------------------------------------------------------------------

def test_df_roundtrip_exhaustive_2997():
    block = 17982  # real frames per 10-minute block at 29.97
    for f in range(3 * block):
        tc = frames_to_timecode(f, 29.97)
        assert timecode_to_frames(tc, 29.97) == f


def test_df_roundtrip_exhaustive_5994():
    block = 35964  # real frames per 10-minute block at 59.94
    for f in range(3 * block):
        tc = frames_to_timecode(f, 59.94)
        assert timecode_to_frames(tc, 59.94) == f


def test_df_2997_roundtrip_100k():
    # Spot-check well past the hour boundary.
    for f in (100000, 123456, 200000, 2589407 - 1):
        tc = frames_to_timecode(f, 29.97)
        assert timecode_to_frames(tc, 29.97) == f


# ---------------------------------------------------------------------------
# Rate-class inference: 29.97/59.94 are DF, integer 30/60 are NDF
# ---------------------------------------------------------------------------

def test_integer_ntsc_rates_are_non_drop():
    assert frames_to_timecode(1800, 30.0) == "00:01:00:00"
    assert frames_to_timecode(3600, 60.0) == "00:01:00:00"
    assert frames_to_timecode(30 * 3600 - 1, 30.0) == "00:59:59:29"
    assert timecode_to_frames("00:01:00:00", 30.0) == 1800
    assert timecode_to_frames("00:01:00:00", 60.0) == 3600


def test_explicit_drop_frame_override():
    # Explicit drop_frame=False at an NTSC rate gives NDF labels.
    assert frames_to_timecode(1800, 29.97, drop_frame=False) == "00:01:00:00"
    # Explicit drop_frame=True at a true 30 fps labels like 29.97.
    assert frames_to_timecode(1800, 30.0, drop_frame=True) == "00:01:00;02"
    assert timecode_to_frames("00:01:00;02", 30.0, drop_frame=True) == 1800


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_negative_frame_raises():
    with pytest.raises(ValueError):
        frames_to_timecode(-1, 24.0)


@pytest.mark.parametrize(
    "fps,expected",
    [
        (23.976, False),
        (24.0, False),
        (25.0, False),
        (29.97, True),
        (30.0, False),
        (50.0, False),
        (59.94, True),
        (60.0, False),
    ],
)
def test_is_drop_frame(fps, expected):
    assert is_drop_frame(fps) is expected


@pytest.mark.parametrize(
    ("tc", "fps"),
    [
        ("00:00:00:30", 29.97),  # frame count >= rate
        ("00:00:00:25", 25.0),
        ("00:00:60:00", 25.0),   # seconds >= 60
        ("00:60:00:00", 25.0),   # minutes >= 60
    ],
)
def test_out_of_range_components_raise(tc, fps):
    with pytest.raises(ValueError):
        timecode_to_frames(tc, fps)


def test_drop_frame_at_non_ntsc_rate_raises():
    # Drop-frame only exists for 29.97/59.94-style rates.
    with pytest.raises(ValueError):
        frames_to_timecode(0, 25.0, drop_frame=True)
    with pytest.raises(ValueError):
        timecode_to_frames("00:00:00;00", 24.0, drop_frame=True)


def test_malformed_timecode_raises_in_conversion():
    with pytest.raises(ValueError):
        timecode_to_frames("not a timecode", 24.0)


# ---------------------------------------------------------------------------
# Seconds <-> frame / timecode conversions
# ---------------------------------------------------------------------------

def test_timecode_to_seconds_ndf():
    assert timecode_to_seconds("00:00:01:00", 25.0) == pytest.approx(1.0)
    assert timecode_to_seconds("01:00:00:00", 25.0) == pytest.approx(3600.0)


def test_timecode_to_seconds_df_wall_clock():
    # Drop-frame exists so that one hour of labels == one hour of real time.
    assert timecode_to_seconds("01:00:00;00", 29.97) == pytest.approx(3600.0)
    assert timecode_to_seconds("01:00:00;00", 59.94) == pytest.approx(3600.0)


def test_seconds_to_timecode_ndf():
    assert seconds_to_timecode(1.0, 25.0) == "00:00:01:00"
    assert seconds_to_timecode(3600.0, 25.0) == "01:00:00:00"


def test_seconds_to_timecode_df():
    # 60.0 s * 29.97 = frame 1798.2 -> nearest frame 1798, one frame short
    # of the minute-1 rollover (labels 00:01:00;00/;01 are skipped).
    assert seconds_to_timecode(60.0, 29.97) == "00:00:59;28"


def test_frame_seconds_helpers():
    assert frame_to_seconds(1800, 25.0) == pytest.approx(72.0)
    assert seconds_to_frame(72.0, 25.0) == 1800
    assert seconds_to_frame(60.0, 29.97) == 1798


def test_format_seconds():
    assert format_seconds(0, 25.0) == "00:00:00.000"
    assert format_seconds(61.5, 25.0) == "00:01:01.500"
    assert format_seconds(3600.001, 25.0) == "01:00:00.001"
    assert format_seconds(359999.999, 25.0) == "99:59:59.999"  # hours can exceed 23
