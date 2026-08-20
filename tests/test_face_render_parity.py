"""Real preview/render parity for a moving tracked face (ffmpeg fixture)."""

from __future__ import annotations

import shutil
import subprocess

import cv2
import numpy as np
import pytest

from colorai.editorial import assign_shot_group, create_group
from colorai.frames import extract_frame
from colorai.project import (
    FaceCorrection,
    FaceTrack,
    ProjectStore,
    SkinMetric,
    make_representative_frame,
    make_shots,
)
from colorai.references import approve_reference, propose_reference
from colorai.skin_analysis import create_subject

ffmpeg = shutil.which("ffmpeg")
ffprobe = shutil.which("ffprobe")
requires_ffmpeg = pytest.mark.skipif(
    ffmpeg is None or ffprobe is None, reason="ffmpeg/ffprobe not available"
)

_SKIN_RGB = (148, 97, 89)  # a YCrCb skin-like value


def _make_clip(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(8):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:, :] = (0, 128, 0)  # green background (RGB)
        x = 8 + 2 * i
        img[24:40, x:x + 16] = _SKIN_RGB          # moving "face"
        img[48:56, 48:56] = _SKIN_RGB             # static second participant
        cv2.imwrite(str(frames / f"{i:03d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        [
            ffmpeg, "-v", "error", "-framerate", "25",
            "-i", str(frames / "%03d.png"),
            "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p",
            "-y", str(clip),
        ],
        check=True,
    )
    return clip


@requires_ffmpeg
def test_preview_and_render_match_on_moving_face(tmp_path):
    clip = _make_clip(tmp_path)

    store = ProjectStore.create(":memory:")
    project = store.create_project("parity")
    asset = store.add_asset(
        project.id, source_path=str(clip), frame_rate=25.0,
        width=64, height=64, frame_count=8,
    )
    shots = make_shots(asset, [(0, 7)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)
    shot = shots[0]
    alice = create_subject(store, asset.id, "Alice")
    group = create_group(store, asset.id, "interview", kind="setup")
    assign_shot_group(store, shot.id, group.id)

    # Representative still = decoded source frame 0 (same input preview and
    # render both start from).
    still = tmp_path / "still0.png"
    extract_frame(str(clip), 0, still, fps=25.0)

    with store.session() as session:
        metric = SkinMetric(
            shot_id=shot.id, face_index=0, mean_b=0.30, mean_g=0.30, mean_r=0.50,
            sample_pixels=100, subject_id=alice.id,
            bbox_x=8, bbox_y=24, bbox_w=16, bbox_h=16,
        )
        session.add(metric)
        session.flush()
        session.add(
            FaceTrack(
                shot_id=shot.id, skin_metric_id=metric.id, subject_id=alice.id,
                source_width=64, source_height=64, analysis_scale=64,
                keyframes=[[0, 0.125, 0.375, 0.25, 0.25], [7, 0.34375, 0.375, 0.25, 0.25]],
                sample_count=2, tracked_count=2, coverage=1.0, max_gap=0.0,
                skin_stability=0.01, median_bgr=[0.30, 0.30, 0.50], state="valid",
            )
        )
        session.add(make_representative_frame(shot, 0, image_path=str(still), frame_rate=25.0))
        session.commit()

    ref = propose_reference(
        store, asset_id=asset.id, shot_id=shot.id, reason="hero",
        confidence=0.9, subject_id=alice.id, group_id=group.id,
    )
    approve_reference(store, ref.id)

    from colorai.face_corrections import (
        approve_face_correction,
        enable_face_correction,
        propose_face_correction,
    )

    with store.session() as session:
        track_id = session.query(FaceTrack).filter_by(skin_metric_id=metric.id).one().id
    c = propose_face_correction(
        store, shot_id=shot.id, subject_id=alice.id, skin_metric_id=metric.id,
        face_track_id=track_id, reference_shot_id=shot.id, reference_group_id=group.id,
        reason="blue up", confidence=0.8, classification="skin_mismatch",
        gain=(1.0, 1.0, 1.10),
    )
    approve_face_correction(store, c.id)
    enable_face_correction(store, c.id)

    from colorai.correction import load_corrected_still
    from colorai.render import render_master

    preview_bgr = load_corrected_still(store, shot)

    out = render_master(store, asset.id, tmp_path / "out.mp4", crf=0, pixel_format="yuv444p")
    rendered_frame_png = tmp_path / "rendered0.png"
    extract_frame(str(out), 0, rendered_frame_png)
    rendered_bgr = cv2.imread(str(rendered_frame_png), cv2.IMREAD_COLOR)

    # Preview and real render match within a documented codec tolerance.
    assert np.allclose(preview_bgr, rendered_bgr, atol=6)

    # Intended tracked skin region's blue channel changed vs. the decoded
    # source frame (gain 1.10 is a small, conservative finishing lift).
    source0 = cv2.imread(str(still), cv2.IMREAD_COLOR)
    assert not np.allclose(preview_bgr[24:40, 8:24, 0], source0[24:40, 8:24, 0], atol=1)

    # Background and second participant are untouched within tolerance.
    assert np.allclose(preview_bgr[0:16, 0:16], source0[0:16, 0:16], atol=6)
    assert np.allclose(preview_bgr[48:56, 48:56], source0[48:56, 48:56], atol=6)


def _oval(nx, ny, nw, nh, points=24):
    import numpy as np

    cx, cy = nx + nw / 2.0, ny + nh / 2.0
    rx, ry = nw / 2.0, nh / 2.0
    return [[float(cx + rx * np.cos(2 * np.pi * i / points)), float(cy + ry * np.sin(2 * np.pi * i / points))] for i in range(points)]


def _skin_appearance_fixture(tmp_path):
    """A moving-face asset with one enabled, reviewed skin_appearance correction."""
    clip = _make_clip(tmp_path)

    from colorai.project import (
        FaceMaskTrack,
        FaceTrack,
        SkinAppearanceTarget,
        SkinMetric,
        ProjectStore,
        make_representative_frame,
        make_shots,
    )
    from colorai.skin_analysis import create_subject

    store = ProjectStore.create(":memory:")
    project = store.create_project("skin appearance parity")
    asset = store.add_asset(
        project.id, source_path=str(clip), frame_rate=25.0,
        width=64, height=64, frame_count=8,
    )
    shots = make_shots(asset, [(0, 7)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)
    shot = shots[0]
    alice = create_subject(store, asset.id, "Alice")
    group = create_group(store, asset.id, "interview", kind="setup")
    assign_shot_group(store, shot.id, group.id)

    still = tmp_path / "skin_still0.png"
    extract_frame(str(clip), 0, still, fps=25.0)

    with store.session() as session:
        metric = SkinMetric(
            shot_id=shot.id, face_index=0, mean_b=0.30, mean_g=0.30, mean_r=0.50,
            sample_pixels=100, subject_id=alice.id,
            bbox_x=8, bbox_y=24, bbox_w=16, bbox_h=16,
        )
        session.add(metric)
        session.flush()
        track = FaceTrack(
            shot_id=shot.id, skin_metric_id=metric.id, subject_id=alice.id,
            source_width=64, source_height=64, analysis_scale=64,
            keyframes=[[0, 0.125, 0.375, 0.25, 0.25], [7, 0.34375, 0.375, 0.25, 0.25]],
            sample_count=2, tracked_count=2, coverage=1.0, max_gap=0.0,
            skin_stability=0.01, median_bgr=[0.30, 0.30, 0.50], state="valid",
        )
        session.add(track)
        session.flush()
        mask = FaceMaskTrack(
            face_track_id=track.id, shot_id=shot.id, subject_id=alice.id,
            backend="fallback", backend_version="0", strategy="face_oval_skin",
            landmark_keyframes=[
                [0, {"oval": _oval(0.125, 0.375, 0.25, 0.25), "eyes": [], "brows": [], "lips": [], "hairline": []}],
                [7, {"oval": _oval(0.34375, 0.375, 0.25, 0.25), "eyes": [], "brows": [], "lips": [], "hairline": []}],
            ],
            coverage=1.0, max_gap=0.0, review_state="approved_for_proposal",
            human_approved=True,
        )
        session.add(mask)
        session.flush()
        target = SkinAppearanceTarget(
            subject_id=alice.id, group_id=group.id, reference_id=None,
            mask_track_id=mask.id,
            profile={"mean_ab": [0.0, 0.0], "spread_ab": [0.01, 0.01]},
            canonical_profile={"mean_ab": [0.0, 0.0], "spread_ab": [0.01, 0.01]},
            approved_preview_parameters={}, state="approved",
        )
        session.add(target)
        session.flush()
        session.add(
            FaceCorrection(
                shot_id=shot.id, subject_id=alice.id, skin_metric_id=metric.id,
                face_track_id=track.id, mask_track_id=mask.id, skin_target_id=target.id,
                reference_group_id=group.id, kind="skin_appearance",
                parameters={"version": 1, "space": "oklab", "luma_mode": "preserve",
                            "ab_offset": [-0.03, 0.0], "ab_scale": [1.0, 1.0], "strength": 1.0},
                reason="reduce red", confidence=0.8, classification="skin_mismatch",
                state="approved", enabled=True,
            )
        )
        session.add(make_representative_frame(shot, 0, image_path=str(still), frame_rate=25.0))
        session.commit()

    return store, asset, shot, still


@requires_ffmpeg
def test_skin_appearance_preview_and_render_match_on_moving_face(tmp_path):
    from colorai.correction import load_corrected_still
    from colorai.render import render_master

    store, asset, shot, still = _skin_appearance_fixture(tmp_path)

    preview_bgr = load_corrected_still(store, shot)
    out = render_master(store, asset.id, tmp_path / "out_skin.mp4", crf=0, pixel_format="yuv444p")
    rendered_frame_png = tmp_path / "skin_rendered0.png"
    extract_frame(str(out), 0, rendered_frame_png)
    rendered_bgr = cv2.imread(str(rendered_frame_png), cv2.IMREAD_COLOR)

    assert np.allclose(preview_bgr, rendered_bgr, atol=6)

    source0 = cv2.imread(str(still), cv2.IMREAD_COLOR)
    # The intended tracked skin region changed (chroma-only, red reduced).
    assert not np.allclose(preview_bgr[24:40, 8:24], source0[24:40, 8:24], atol=2)
    # Background and the second participant are untouched within tolerance.
    assert np.allclose(preview_bgr[0:16, 0:16], source0[0:16, 0:16], atol=6)
    assert np.allclose(preview_bgr[48:56, 48:56], source0[48:56, 48:56], atol=6)


_SKIN_RED_RGB = (148, 97, 89)  # red-tinted skin (RGB), classified as skin


def _red_two_shot_clip(tmp_path):
    """Two 4-frame shots (96x96): red skin face + large white teeth + dark eye +
    green background + a second skin participant (outside the selected box)."""
    frames = tmp_path / "red_frames"
    frames.mkdir()
    for i in range(8):
        img = np.zeros((96, 96, 3), dtype=np.uint8)
        img[:, :] = (0, 128, 0)  # green background (RGB)
        img[24:64, 24:56] = _SKIN_RED_RGB          # red-tinted skin face
        img[42:50, 32:48] = (255, 255, 255)        # teeth (non-skin, large)
        img[30:38, 28:44] = (30, 30, 30)           # eye (non-skin, large)
        img[48:64, 64:80] = (150, 100, 90)         # second participant
        cv2.imwrite(str(frames / f"{i:03d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    clip = tmp_path / "red_two.mp4"
    subprocess.run(
        [ffmpeg, "-v", "error", "-framerate", "25", "-i", str(frames / "%03d.png"),
         "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", "-y", str(clip)],
        check=True,
    )
    return clip


@requires_ffmpeg
def test_red_source_approved_neutral_target_moves_second_angle(tmp_path):
    """End-to-end: a warm/red source reference with an approved cooler target
    makes a second angle move toward that approved appearance, while
    background, teeth, eye, and a second participant stay materially unchanged."""
    from colorai.project import (
        FaceMaskTrack,
        FaceTrack,
        ProjectStore,
        SkinMetric,
        make_representative_frame,
        make_shots,
    )
    from colorai.render import render_master
    from colorai.skin_analysis import create_subject
    from colorai.skin_targets import (
        approve_skin_target,
        create_skin_reference,
        match_skin_target_to_group,
        propose_skin_appearance_correction,
        suggest_skin_target,
    )
    from colorai.face_corrections import (
        approve_face_correction,
        enable_face_correction,
    )

    clip = _red_two_shot_clip(tmp_path)

    store = ProjectStore.create(":memory:")
    project = store.create_project("red target e2e")
    asset = store.add_asset(
        project.id, source_path=str(clip), frame_rate=25.0, width=96, height=96, frame_count=8
    )
    shots = make_shots(asset, [(0, 3), (4, 7)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)
    alice = create_subject(store, asset.id, "Alice")
    group = create_group(store, asset.id, "interview", kind="setup")
    for s in shots:
        assign_shot_group(store, s.id, group.id)

    # Representative stills (frame 0 and frame 4) for both shots.
    still0 = tmp_path / "red0.png"
    still4 = tmp_path / "red4.png"
    extract_frame(str(clip), 0, still0, fps=25.0)
    extract_frame(str(clip), 4, still4, fps=25.0)

    def oval(nx, ny, nw, nh):
        import numpy as _np
        cx, cy = nx + nw / 2.0, ny + nh / 2.0
        return [[float(cx + nw / 2.0 * _np.cos(2 * _np.pi * i / 24)),
                 float(cy + nh / 2.0 * _np.sin(2 * _np.pi * i / 24))] for i in range(24)]

    with store.session() as session:
        metrics = []
        track_ids = []
        for s in shots:
            m = SkinMetric(
                shot_id=s.id, face_index=0, mean_b=0.35, mean_g=0.38, mean_r=0.58,
                sample_pixels=100, subject_id=alice.id,
                bbox_x=24, bbox_y=24, bbox_w=32, bbox_h=40,
            )
            session.add(m)
            session.flush()
            metrics.append(m)
            t = FaceTrack(
                shot_id=s.id, skin_metric_id=m.id, subject_id=alice.id,
                source_width=96, source_height=96, analysis_scale=96,
                keyframes=[[s.start_frame, 0.25, 0.25, 0.3333, 0.4167]],
                sample_count=1, tracked_count=1, coverage=1.0, max_gap=0.0,
                skin_stability=0.01, median_bgr=[0.35, 0.38, 0.58], state="valid",
            )
            session.add(t)
            session.flush()
            track_ids.append(t.id)
            session.add(FaceMaskTrack(
                face_track_id=t.id, shot_id=s.id, subject_id=alice.id,
                backend="fallback", backend_version="0", strategy="face_oval_skin",
                landmark_keyframes=[[s.start_frame, {"oval": oval(0.25, 0.25, 0.3333, 0.4167), "eyes": [], "brows": [], "lips": [], "hairline": []}]],
                coverage=1.0, max_gap=0.0, review_state="approved_for_proposal", human_approved=True,
            ))
        session.add(make_representative_frame(shots[0], 0, image_path=str(still0), frame_rate=25.0))
        session.add(make_representative_frame(shots[1], 4, image_path=str(still4), frame_rate=25.0))
        session.commit()

    ref = create_skin_reference(
        store, subject_id=alice.id, group_id=group.id,
        source_shot_id=shots[0].id, frame_index=shots[0].start_frame,
        role="accurate_skin_reference",
    )
    target = suggest_skin_target(
        store, subject_id=alice.id, group_id=group.id, reference_id=ref.id,
        face_track_id=track_ids[0],
        parameters={"version": 1, "space": "oklab", "luma_mode": "preserve",
                    "ab_offset": [-0.04, 0.0], "ab_scale": [1.0, 1.0], "strength": 1.0},
        rationale="reduce red cast", confidence=0.9,
    )
    approve_skin_target(store, target.id)

    result = match_skin_target_to_group(store, target_id=target.id, group_id=group.id)
    candidate = next(c for c in result["candidates"] if c["shot_id"] == shots[1].id)
    assert candidate["state"] == "ready"
    assert candidate["parameters"]["ab_offset"][0] < 0.0

    correction = propose_skin_appearance_correction(
        store, shot_id=shots[1].id, subject_id=alice.id, skin_metric_id=metrics[1].id,
        face_track_id=track_ids[1], skin_target_id=target.id,
        parameters=candidate["parameters"], reason="match to approved target", confidence=0.8,
    )
    approve_face_correction(store, correction.id)
    enable_face_correction(store, correction.id)

    out = render_master(store, asset.id, tmp_path / "red_out.mp4", crf=0, pixel_format="yuv444p")
    rendered4 = tmp_path / "rendered4.png"
    extract_frame(str(out), 4, rendered4)
    out_bgr = cv2.imread(str(rendered4), cv2.IMREAD_COLOR)

    # Candidate face moved toward the approved (less red) target.
    src4 = cv2.imread(str(still4), cv2.IMREAD_COLOR)
    face_src = src4[24:64, 24:56]
    face_out = out_bgr[24:64, 24:56]
    assert float(face_out[..., 2].mean()) < float(face_src[..., 2].mean()) - 2.0

    # Teeth and eye (non-skin) are materially unchanged at their centres.
    assert np.allclose(out_bgr[44:48, 36:44], src4[44:48, 36:44], atol=6)
    assert np.allclose(out_bgr[32:36, 32:40], src4[32:36, 32:40], atol=6)
    # Background and second participant are unchanged.
    assert np.allclose(out_bgr[0:8, 0:8], src4[0:8, 0:8], atol=6)
    assert np.allclose(out_bgr[48:64, 64:80], src4[48:64, 64:80], atol=6)
