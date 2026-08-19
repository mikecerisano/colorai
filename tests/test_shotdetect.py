"""Tests for shot detection."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from colorai.project import ProjectStore
from colorai.shotdetect import detect_and_store_shots, detect_shot_bounds


ffmpeg = shutil.which("ffmpeg")
ffprobe = shutil.which("ffprobe")
requires_ffmpeg = pytest.mark.skipif(
    ffmpeg is None or ffprobe is None, reason="ffmpeg/ffprobe not available"
)


@pytest.fixture(scope="module")
def hardcut_video(tmp_path_factory):
    """150 frames at 25 fps: 50 black, 50 white, 50 black (cuts at 50, 100)."""
    out = tmp_path_factory.mktemp("shots") / "hardcut.mp4"
    subprocess.run(
        [
            ffmpeg, "-v", "error",
            "-f", "lavfi", "-t", "2", "-i", "color=c=black:size=64x64:rate=25",
            "-f", "lavfi", "-t", "2", "-i", "color=c=white:size=64x64:rate=25",
            "-f", "lavfi", "-t", "2", "-i", "color=c=black:size=64x64:rate=25",
            "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p",
            "-c:v", "mpeg4", "-y", str(out),
        ],
        check=True,
    )
    return out


@requires_ffmpeg
def test_detect_shot_bounds_exact(hardcut_video):
    bounds = detect_shot_bounds(hardcut_video)
    assert bounds == [(0, 49), (50, 99), (100, 149)]


@requires_ffmpeg
def test_detect_shot_bounds_are_contiguous_and_cover(hardcut_video):
    bounds = detect_shot_bounds(hardcut_video)
    # No gaps, no overlap, full coverage of the 150 frames.
    for (a_start, a_end), (b_start, b_end) in zip(bounds, bounds[1:]):
        assert b_start == a_end + 1
    assert bounds[0][0] == 0
    assert sum(end - start + 1 for start, end in bounds) == 150


@requires_ffmpeg
def test_detect_and_store_shots(hardcut_video):
    store = ProjectStore.create(":memory:")
    project = store.create_project("shots test")
    asset = store.add_asset(
        project.id, source_path=str(hardcut_video), frame_rate=25.0
    )

    shots = detect_and_store_shots(store, asset)

    assert [s.index for s in shots] == [0, 1, 2]
    assert (shots[0].start_timecode, shots[0].end_timecode) == (
        "00:00:00:00",
        "00:00:01:24",
    )
    assert shots[1].start_timecode == "00:00:02:00"
    assert shots[2].end_timecode == "00:00:05:24"
    assert sum(s.frame_count for s in shots) == 150
    assert all(s.id is not None for s in shots)  # persisted
