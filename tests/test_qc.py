"""Tests for temporal quality-control measurements."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from colorai.project import FrameMetrics, ProjectStore, make_shots
from colorai.qc import (
    blank_frames_from_lumas,
    clip_flags,
    detect_blank_frames,
    detect_flicker,
    duplicate_intervals_from_hashes,
    flicker_intervals,
    shot_clip_report,
)


ffmpeg = shutil.which("ffmpeg")
requires_ffmpeg = pytest.mark.skipif(ffmpeg is None, reason="ffmpeg not available")


def test_flicker_intervals_detects_oscillation():
    values = [0.5, 0.6, 0.5, 0.6, 0.5]
    assert flicker_intervals(values, threshold=0.05) == [(0, 4)]


def test_flicker_intervals_ignores_monotonic_ramp():
    values = [0.5, 0.6, 0.7, 0.8]
    assert flicker_intervals(values, threshold=0.05) == []


def test_flicker_intervals_requires_alternation():
    values = [0.5, 0.6, 0.7, 0.6, 0.7, 0.6]
    # The ramp at the start is excluded; only the oscillating tail is flagged.
    assert flicker_intervals(values, threshold=0.05) == [(1, 5)]


def test_clip_flags():
    assert clip_flags(0.01, 0.99) == {"clipped": True, "crushed": True}
    assert clip_flags(0.5, 0.5) == {"clipped": False, "crushed": False}


def test_shot_clip_report(tmp_path):
    store = ProjectStore.create(":memory:")
    project = store.create_project("qc")
    asset = store.add_asset(project.id, source_path="/media/m.mov", frame_rate=25.0)
    shots = make_shots(asset, [(0, 24), (25, 49)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)
        session.add(FrameMetrics(shot_id=shots[0].id, frame_index=0, luma_p5=0.01, luma_p95=0.99))
        session.add(FrameMetrics(shot_id=shots[1].id, frame_index=25, luma_p5=0.3, luma_p95=0.7))
        session.commit()

    report = shot_clip_report(store, asset.id)
    assert report[0]["shot_id"] == shots[0].id
    assert report[0]["clipped"] is True
    assert report[0]["crushed"] is True
    assert "measurement, not a defect" in report[0]["note"]
    assert report[1]["clipped"] is False and report[1]["crushed"] is False
    assert report[1]["note"] == ""


def test_blank_frames_from_lumas():
    assert [b.frame_index for b in blank_frames_from_lumas({0: 0.0, 1: 0.5, 2: 1.0})] == [0, 2]
    assert blank_frames_from_lumas({0: 0.0})[0].kind == "black"
    assert blank_frames_from_lumas({0: 1.0})[0].kind == "white"


def test_duplicate_intervals_from_hashes():
    assert duplicate_intervals_from_hashes([0, 1, 2, 3], ["a", "b", "b", "c"]) == [(1, 2)]
    assert duplicate_intervals_from_hashes([0, 1, 2], ["a", "a", "a"]) == [(0, 2)]
    assert duplicate_intervals_from_hashes([0, 1, 2], ["a", "b", "c"]) == []


@requires_ffmpeg
def test_detect_flicker_on_alternating_clip(tmp_path):
    clip = tmp_path / "flicker.mp4"
    subprocess.run(
        [
            ffmpeg, "-v", "error",
            "-f", "lavfi", "-i", "color=c=black:size=16x16:rate=25",
            "-f", "lavfi", "-i", "color=c=white:size=16x16:rate=25",
            "-filter_complex",
            "[0:v]settb=AVTB[v0];[1:v]settb=AVTB[v1];[v0][v1]interleave=nb_inputs=2,settb=1/25,format=yuv420p",
            "-r", "25", "-t", "1", "-c:v", "mpeg4", "-y", str(clip),
        ],
        check=True,
    )
    runs = detect_flicker(clip, 0, 24, 25.0, samples=24)
    assert runs  # at least one flicker interval detected


@requires_ffmpeg
def test_detect_blank_frames_on_black_clip(tmp_path):
    clip = tmp_path / "black.mp4"
    subprocess.run(
        [
            ffmpeg, "-v", "error",
            "-f", "lavfi", "-t", "1", "-i", "color=c=black:size=16x16:rate=25",
            "-pix_fmt", "yuv420p", "-c:v", "mpeg4", "-y", str(clip),
        ],
        check=True,
    )
    blanks = detect_blank_frames(clip, 0, 24, 25.0, samples=8)
    assert blanks and all(b.kind == "black" for b in blanks)
