"""Tests for source identity and resumable analysis."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from colorai.ingest import compute_source_hash
from colorai.pipeline import analyze_master
from colorai.project import ProjectStore


ffmpeg = shutil.which("ffmpeg")
requires_ffmpeg = pytest.mark.skipif(ffmpeg is None, reason="ffmpeg not available")


def test_compute_source_hash_deterministic(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello world" * 1000)
    assert compute_source_hash(f) == compute_source_hash(f)


def test_compute_source_hash_detects_change(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello world" * 1000)
    before = compute_source_hash(f)
    f.write_bytes(b"hello world" * 1000 + b"x")
    assert compute_source_hash(f) != before


@requires_ffmpeg
def test_analyze_master_resumes(tmp_path):
    clip = tmp_path / "master.mp4"
    subprocess.run(
        [
            ffmpeg, "-v", "error",
            "-f", "lavfi", "-t", "1", "-i", "color=c=black:size=32x32:rate=25",
            "-f", "lavfi", "-t", "1", "-i", "color=c=white:size=32x32:rate=25",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0,format=yuv420p",
            "-c:v", "mpeg4", "-y", str(clip),
        ],
        check=True,
    )

    store = ProjectStore.create(":memory:")
    project = store.create_project("resume")
    stills = tmp_path / "stills"

    first = analyze_master(store, project.id, clip, stills_dir=stills)
    still_count = len(list(stills.rglob("*.png")))

    second = analyze_master(store, project.id, clip, stills_dir=stills)

    # Same asset reused (not re-ingested), same results, no new stills.
    assert second.asset.id == first.asset.id
    assert len(second.shots) == len(first.shots)
    assert len(second.representative_frames) == len(first.representative_frames)
    assert len(list(stills.rglob("*.png"))) == still_count


@requires_ffmpeg
def test_analyze_master_force_reanalyzes(tmp_path):
    clip = tmp_path / "master.mp4"
    subprocess.run(
        [
            ffmpeg, "-v", "error",
            "-f", "lavfi", "-t", "1", "-i", "color=c=black:size=32x32:rate=25",
            "-f", "lavfi", "-t", "1", "-i", "color=c=white:size=32x32:rate=25",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0,format=yuv420p",
            "-c:v", "mpeg4", "-y", str(clip),
        ],
        check=True,
    )

    store = ProjectStore.create(":memory:")
    project = store.create_project("force")
    stills = tmp_path / "stills"

    first = analyze_master(store, project.id, clip, stills_dir=stills)
    assert len(first.shots) == 2

    # A manual merge collapses the two shots into one.
    from colorai.editorial import merge_shots

    merge_shots(store, first.shots[0].id, first.shots[1].id)

    # Forcing re-detection undoes the manual edit (same asset, fresh shots).
    forced = analyze_master(store, project.id, clip, stills_dir=stills, resume=False)
    assert forced.asset.id == first.asset.id
    assert len(forced.shots) == 2
