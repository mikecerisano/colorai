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
