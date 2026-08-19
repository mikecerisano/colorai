"""Tests for representative frame extraction."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import pytest

from colorai.frames import (
    extract_frame,
    extract_representative_frames,
    representative_frame_index,
)
from colorai.project import ProjectStore, Shot, make_shots


ffmpeg = shutil.which("ffmpeg")
requires_ffmpeg = pytest.mark.skipif(ffmpeg is None, reason="ffmpeg not available")


@pytest.fixture(scope="module")
def hardcut_video(tmp_path_factory):
    """100 frames: 50 black, 50 white at 25 fps (cut at frame 50)."""
    out = tmp_path_factory.mktemp("frames") / "hardcut.mp4"
    subprocess.run(
        [
            ffmpeg, "-v", "error",
            "-f", "lavfi", "-t", "2", "-i", "color=c=black:size=64x64:rate=25",
            "-f", "lavfi", "-t", "2", "-i", "color=c=white:size=64x64:rate=25",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0,format=yuv420p",
            "-c:v", "mpeg4", "-y", str(out),
        ],
        check=True,
    )
    return out


def _shot(start, end):
    s = Shot(index=0, start_frame=start, end_frame=end)
    s.start_timecode = s.end_timecode = "00:00:00:00"
    return s


def test_representative_frame_index_midpoint():
    assert representative_frame_index(_shot(0, 49)) == 24
    assert representative_frame_index(_shot(50, 99)) == 74
    assert representative_frame_index(_shot(0, 0)) == 0


@requires_ffmpeg
def test_extract_frame_black_and_white(hardcut_video, tmp_path):
    black = extract_frame(hardcut_video, 0, tmp_path / "black.png")
    white = extract_frame(hardcut_video, 50, tmp_path / "white.png")
    assert black.exists() and white.exists()

    assert cv2.imread(str(black)).mean() < 5
    assert cv2.imread(str(white)).mean() > 250


@requires_ffmpeg
def test_extract_representative_frames(hardcut_video, tmp_path):
    store = ProjectStore.create(":memory:")
    project = store.create_project("frames test")
    asset = store.add_asset(
        project.id, source_path=str(hardcut_video), frame_rate=25.0
    )
    shots = make_shots(asset, [(0, 49), (50, 99)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)

    stills = tmp_path / "stills"
    frames = extract_representative_frames(store, asset, shots, stills)

    assert [f.frame_index for f in frames] == [24, 74]
    assert [f.timecode for f in frames] == ["00:00:00:24", "00:00:02:24"]
    assert all(f.image_path and Path(f.image_path).exists() for f in frames)
