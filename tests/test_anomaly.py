"""Tests for deterministic blur-pulse detection."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from colorai.anomaly import blur_pulses_from_scores, detect_blur_pulses

ffmpeg = shutil.which("ffmpeg")
requires_ffmpeg = pytest.mark.skipif(ffmpeg is None, reason="ffmpeg not available")


def test_blur_pulse_in_middle():
    scores = {0: 100, 1: 100, 2: 20, 3: 15, 4: 100, 5: 100}
    pulses = blur_pulses_from_scores(scores)
    assert len(pulses) == 1
    assert (pulses[0].start_frame, pulses[0].end_frame, pulses[0].num_frames) == (2, 3, 2)
    assert pulses[0].min_ratio == pytest.approx(0.15)


def test_no_pulse_when_uniform():
    scores = {i: 100 for i in range(6)}
    assert blur_pulses_from_scores(scores) == []


def test_min_run_filters_short_pulses():
    scores = {0: 100, 1: 10, 2: 100}  # single-frame dip
    assert blur_pulses_from_scores(scores, min_run=2) == []
    assert len(blur_pulses_from_scores(scores, min_run=1)) == 1


def test_trailing_pulse():
    scores = {0: 100, 1: 100, 2: 10, 3: 8}
    pulses = blur_pulses_from_scores(scores)
    assert len(pulses) == 1
    assert (pulses[0].start_frame, pulses[0].end_frame) == (2, 3)


def test_empty_scores():
    assert blur_pulses_from_scores({}) == []


@requires_ffmpeg
def test_detect_blur_pulses_on_flat_segment(tmp_path):
    # 50 frames textured (testsrc) + 50 frames flat black = a low-sharpness run.
    clip = tmp_path / "sharpflat.mp4"
    subprocess.run(
        [
            ffmpeg, "-v", "error",
            "-f", "lavfi", "-t", "2", "-i", "testsrc=size=64x64:rate=25",
            "-f", "lavfi", "-t", "2", "-i", "color=c=black:size=64x64:rate=25",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0,format=yuv420p",
            "-c:v", "mpeg4", "-y", str(clip),
        ],
        check=True,
    )
    pulses = detect_blur_pulses(clip, 0, 99, 25.0, samples=16)
    assert len(pulses) == 1
    assert pulses[0].start_frame > 45  # in the flat half
    assert pulses[0].end_frame == 99
