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


def _enabled_face_correction_store(tmp_path, keyframes, *, state="approved", enabled=True):
    """A store with one enabled face correction whose track uses ``keyframes``."""
    from colorai.project import FaceCorrection, FaceTrack, SkinMetric
    from colorai.skin_analysis import create_subject

    store = ProjectStore.create(":memory:")
    project = store.create_project("render face")
    asset = store.add_asset(
        project.id, source_path="/media/m.mov", frame_rate=25.0, width=64, height=64
    )
    shots = make_shots(asset, [(0, 9)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)
    shot = shots[0]
    alice = create_subject(store, asset.id, "Alice")
    with store.session() as session:
        metric = SkinMetric(
            shot_id=shot.id, face_index=0, mean_b=0.3, mean_g=0.3, mean_r=0.5,
            sample_pixels=10, subject_id=alice.id,
            bbox_x=16, bbox_y=16, bbox_w=32, bbox_h=32,
        )
        session.add(metric)
        session.flush()
        track = FaceTrack(
            shot_id=shot.id, skin_metric_id=metric.id, subject_id=alice.id,
            source_width=64, source_height=64, analysis_scale=64,
            keyframes=keyframes,
            sample_count=2, tracked_count=2, coverage=1.0, max_gap=0.0,
            skin_stability=0.01, median_bgr=[0.3, 0.3, 0.5], state="valid",
        )
        session.add(track)
        session.flush()
        session.add(
            FaceCorrection(
                shot_id=shot.id, subject_id=alice.id, skin_metric_id=metric.id,
                face_track_id=track.id, kind="rgb_balance",
                parameters={"gain": [1.0, 1.0, 1.0]},
                reason="bad", classification="skin_mismatch",
                state=state, enabled=enabled,
            )
        )
        session.commit()
    return store, asset


def test_render_aborts_on_invalid_enabled_face_correction(tmp_path):
    from colorai.face_corrections import ValidationError

    store, asset = _enabled_face_correction_store(
        tmp_path, [[0, 0.25, 0.25, 0.5, 0.5], [9, 0.25, 0.25, 0.5, 0.5]],
        state="suggested",  # enabled but NOT approved -> abort before output
    )
    out = tmp_path / "should_not_exist.mp4"
    with pytest.raises(ValidationError):
        render_master(store, asset.id, out)
    assert not out.exists()


def test_render_preflight_rejects_zero_width_keyframe(tmp_path):
    from colorai.face_corrections import ValidationError

    store, asset = _enabled_face_correction_store(
        tmp_path, [[0, 0.25, 0.25, 0.0, 0.5], [9, 0.25, 0.25, 0.5, 0.5]],
    )
    out = tmp_path / "no_output.mp4"
    with pytest.raises(ValidationError):
        render_master(store, asset.id, out)
    assert not out.exists()


def test_render_preflight_rejects_negative_height_keyframe(tmp_path):
    from colorai.face_corrections import ValidationError

    store, asset = _enabled_face_correction_store(
        tmp_path, [[0, 0.25, 0.25, 0.5, -0.1], [9, 0.25, 0.25, 0.5, 0.5]],
    )
    out = tmp_path / "no_output.mp4"
    with pytest.raises(ValidationError):
        render_master(store, asset.id, out)
    assert not out.exists()


def test_render_preflight_rejects_fractional_frame_index(tmp_path):
    from colorai.face_corrections import ValidationError

    store, asset = _enabled_face_correction_store(
        tmp_path, [[0, 0.25, 0.25, 0.5, 0.5], [3.5, 0.25, 0.25, 0.5, 0.5]],
    )
    out = tmp_path / "no_output.mp4"
    with pytest.raises(ValidationError):
        render_master(store, asset.id, out)
    assert not out.exists()


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


@requires_ffmpeg
def test_render_preserves_audio_and_color_tags(tmp_path):
    clip = tmp_path / "av.mp4"
    subprocess.run(
        [
            ffmpeg, "-v", "error",
            "-f", "lavfi", "-t", "1", "-i", "color=c=black:size=16x16:rate=10",
            "-f", "lavfi", "-t", "1", "-i", "sine=frequency=440",
            "-shortest", "-pix_fmt", "yuv420p", "-c:v", "mpeg4", "-c:a", "aac",
            "-y", str(clip),
        ],
        check=True,
    )
    store = ProjectStore.create(":memory:")
    project = store.create_project("render av")
    asset = store.add_asset(
        project.id, source_path=str(clip), frame_rate=10.0,
        width=16, height=16, frame_count=10, color_space="bt709", transfer="bt709",
    )
    shots = make_shots(asset, [(0, 9)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        session.commit()

    out = render_master(store, asset.id, tmp_path / "av_out.mp4")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,color_transfer", "-of", "json", str(out)],
        capture_output=True, text=True, check=True,
    )
    import json

    streams = json.loads(probe.stdout)["streams"]
    types = {s["codec_type"] for s in streams}
    assert {"video", "audio"} <= types  # audio preserved
    video = next(s for s in streams if s["codec_type"] == "video")
    assert video.get("color_transfer") == "bt709"  # source color tag carried through


@requires_ffmpeg
def test_render_rejects_non_bt709_transfer(tmp_path):
    store = ProjectStore.create(":memory:")
    project = store.create_project("render hdr")
    asset = store.add_asset(
        project.id, source_path="/media/hdr.mov", frame_rate=25.0,
        width=16, height=16, transfer="smpte2084",
    )
    with pytest.raises(ValueError, match="not yet gradeable"):
        render_master(store, asset.id, tmp_path / "out.mp4")


@requires_ffmpeg
def test_render_rejects_incomplete_decode(tmp_path):
    clip = tmp_path / "short.mp4"
    subprocess.run(
        [
            ffmpeg, "-v", "error",
            "-f", "lavfi", "-t", "1", "-i", "color=c=black:size=16x16:rate=10",
            "-pix_fmt", "yuv420p", "-c:v", "mpeg4", "-y", str(clip),
        ],
        check=True,
    )
    store = ProjectStore.create(":memory:")
    project = store.create_project("render short")
    asset = store.add_asset(
        project.id, source_path=str(clip), frame_rate=10.0,
        width=16, height=16, frame_count=50,  # metadata claims 50; file has 10
    )
    with pytest.raises(RuntimeError, match="incomplete output"):
        render_master(store, asset.id, tmp_path / "out.mp4")
