"""Tests for temporal face tracking and mask propagation."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from colorai.tracking import (
    FaceTrack,
    match_box,
    propagate_shot_mask,
    sample_frames,
    stable_skin_mask,
    temporal_skin_metrics,
    track_face,
)

ffmpeg = shutil.which("ffmpeg")
requires_ffmpeg = pytest.mark.skipif(ffmpeg is None, reason="ffmpeg not available")


def test_sample_frames():
    assert sample_frames(0, 99, 5) == [0, 25, 50, 74, 99]
    assert sample_frames(0, 0, 5) == [0]
    assert sample_frames(0, 10, 1) == [0]
    assert sample_frames(10, 0, 5) == []


def test_match_box_picks_best_overlap():
    seed = (10, 10, 20, 20)
    assert match_box([(0, 0, 5, 5), (10, 10, 20, 20), (12, 12, 18, 18)], seed) == (10, 10, 20, 20)
    assert match_box([(0, 0, 5, 5), (50, 50, 5, 5)], seed) is None


def test_stable_skin_mask_majority_vote():
    a = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    b = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    c = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    out = stable_skin_mask([a, b, c])
    assert out.tolist() == [[1, 0], [1, 1]]  # pixel (1,1): a=1,b=1,c=0 -> majority 1
    with pytest.raises(ValueError):
        stable_skin_mask([])


def test_temporal_skin_metrics():
    metrics = temporal_skin_metrics([(100, 120, 180), (104, 122, 184), (102, 121, 182)])
    assert metrics["median_bgr"] == pytest.approx([102, 121, 182])
    assert metrics["stability"] < 3.0
    assert metrics["samples"] == 3
    with pytest.raises(ValueError):
        temporal_skin_metrics([])


def _fake_extract(video_path, frame_index, out_path, **kwargs):
    cv2.imwrite(str(out_path), np.zeros((32, 32, 3), dtype=np.uint8))
    return Path(out_path)


def _fake_detect(img):
    return [(5, 5, 20, 20)]


def test_track_face_with_injected_detector():
    track = track_face(
        "dummy.mp4", 0, 99, (5, 5, 20, 20), 25.0,
        samples=8, extract=_fake_extract, detect=_fake_detect,
    )
    assert track.tracked_frames == 8
    assert all(box == (5, 5, 20, 20) for _, box in track.samples)


@requires_ffmpeg
def test_propagate_shot_mask_no_faces(tmp_path):
    clip = tmp_path / "flat.mp4"
    subprocess.run(
        [
            ffmpeg, "-v", "error",
            "-f", "lavfi", "-t", "1", "-i", "color=c=black:size=64x64:rate=25",
            "-c:v", "mpeg4", "-y", str(clip),
        ],
        check=True,
    )
    result = propagate_shot_mask(clip, 0, 24, 0, 25.0, samples=4)
    assert result["tracked_frames"] == 0
