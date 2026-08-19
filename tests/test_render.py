"""Tests for the full-master render/export path."""

from __future__ import annotations

import shutil
import subprocess

import cv2
import pytest

from colorai.frames import extract_frame
from colorai.project import Correction, ProjectStore, make_shots
from colorai.render import (
    build_shot_spans,
    corrections_for_frame,
    render_master,
)


ffmpeg = shutil.which("ffmpeg")
ffprobe = shutil.which("ffprobe")
requires_ffmpeg = pytest.mark.skipif(
    ffmpeg is None or ffprobe is None, reason="ffmpeg/ffprobe not available"
)


def _store_with_shots():
    store = ProjectStore.create(":memory:")
    project = store.create_project("render test")
    asset = store.add_asset(
        project.id, source_path="/media/m.mov", frame_rate=25.0
    )
    shots = make_shots(asset, [(0, 24), (25, 49), (50, 74)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)
    return store, asset, shots


def test_build_shot_spans_only_includes_enabled_corrections():
    store, asset, shots = _store_with_shots()
    with store.session() as session:
        session.add(Correction(shot_id=shots[0].id, kind="exposure", parameters={"gain": 2.0}))
        session.add(
            Correction(
                shot_id=shots[0].id, kind="offset", parameters={"value": 0.1}, enabled=False
            )
        )
        session.add(Correction(shot_id=shots[2].id, kind="saturation", parameters={"amount": 0.5}))
        session.commit()

    spans = build_shot_spans(store, asset.id)
    assert [s.start_frame for s in spans] == [0, 25, 50]
    assert spans[0].corrections == (("exposure", {"gain": 2.0}),)
    assert spans[1].corrections == ()
    assert spans[2].corrections == (("saturation", {"amount": 0.5}),)


def test_corrections_for_frame_respects_bounds_and_gaps():
    spans = [
        type("Span", (), {"start_frame": 0, "end_frame": 9, "corrections": (("exposure", {"gain": 2.0}),)})(),
        type("Span", (), {"start_frame": 20, "end_frame": 29, "corrections": (("saturation", {"amount": 0.5}),)})(),
    ]
    assert corrections_for_frame(0, spans) == (("exposure", {"gain": 2.0}),)
    assert corrections_for_frame(9, spans) == (("exposure", {"gain": 2.0}),)
    assert corrections_for_frame(15, spans) == ()  # gap
    assert corrections_for_frame(25, spans) == (("saturation", {"amount": 0.5}),)


def test_corrections_for_frame_empty():
    assert corrections_for_frame(0, []) == ()


@requires_ffmpeg
def test_render_master_applies_offset_to_black_shot(tmp_path):
    # 50 frames: 25 black, 25 white at 25 fps.
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
    project = store.create_project("render e2e")
    asset = store.add_asset(
        project.id,
        source_path=str(clip),
        frame_rate=25.0,
        width=32,
        height=32,
        frame_count=50,
    )
    shots = make_shots(asset, [(0, 24), (25, 49)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)
        # Lift the black shot to mid-gray; leave the white shot untouched.
        session.add(Correction(shot_id=shots[0].id, kind="offset", parameters={"value": 0.5}))
        session.commit()

    out = render_master(store, asset.id, tmp_path / "rendered.mp4")

    assert out.exists()
    lifted = extract_frame(out, 0, tmp_path / "lifted.png")
    white = extract_frame(out, 25, tmp_path / "white.png")
    assert cv2.imread(str(lifted)).mean() > 100  # black shot lifted
    assert cv2.imread(str(white)).mean() > 250  # white shot untouched


@requires_ffmpeg
def test_render_master_no_corrections_is_identity(tmp_path):
    clip = tmp_path / "master.mp4"
    subprocess.run(
        [
            ffmpeg, "-v", "error",
            "-f", "lavfi", "-t", "1", "-i", "color=c=white:size=16x16:rate=10",
            "-pix_fmt", "yuv420p", "-c:v", "mpeg4", "-y", str(clip),
        ],
        check=True,
    )
    store = ProjectStore.create(":memory:")
    project = store.create_project("render identity")
    asset = store.add_asset(
        project.id,
        source_path=str(clip),
        frame_rate=10.0,
        width=16,
        height=16,
        frame_count=10,
    )
    shots = make_shots(asset, [(0, 9)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        session.commit()

    out = render_master(store, asset.id, tmp_path / "identity.mp4")
    frame = extract_frame(out, 0, tmp_path / "f0.png")
    assert cv2.imread(str(frame)).mean() > 250
