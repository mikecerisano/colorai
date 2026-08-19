"""End-to-end pipeline tests."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from colorai.pipeline import analyze_master
from colorai.project import ProjectStore


ffmpeg = shutil.which("ffmpeg")
requires_ffmpeg = pytest.mark.skipif(ffmpeg is None, reason="ffmpeg not available")


@pytest.fixture(scope="module")
def hardcut_video(tmp_path_factory):
    """50 frames black, 50 frames white at 25 fps."""
    out = tmp_path_factory.mktemp("pipeline") / "master.mp4"
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


@requires_ffmpeg
def test_analyze_master_end_to_end(hardcut_video, tmp_path):
    store = ProjectStore.create(":memory:")
    project = store.create_project("pipeline test")

    result = analyze_master(
        store, project.id, hardcut_video, stills_dir=tmp_path / "stills"
    )

    assert result.asset.status == "analyzed"
    assert result.asset.frame_rate == 25.0
    assert len(result.shots) == 2
    assert len(result.representative_frames) == 2
    assert len(result.metrics) == 2

    # Shot 0 is black, shot 1 is white; luma reflects that.
    assert result.metrics[0].luma_mean == pytest.approx(0.0, abs=0.05)
    assert result.metrics[1].luma_mean == pytest.approx(1.0, abs=0.05)

    # Everything persisted with the right relationships.
    assert result.shots[0].asset_id == result.asset.id
    assert result.representative_frames[0].shot_id == result.shots[0].id
    assert result.metrics[0].shot_id == result.shots[0].id
