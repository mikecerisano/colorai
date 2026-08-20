"""Tests for reviewable temporal face-mask tracks."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from colorai.editorial import assign_shot_group, create_group
from colorai.face_masks import (
    FaceMaskTrack,
    FALLBACK_STRATEGY,
    LANDMARK_STRATEGY,
    build_face_mask_track,
    interpolate_mask_geometry,
    render_face_mask,
    validate_face_mask_track,
)
from colorai.project import (
    FaceTrack,
    ProjectStore,
    SkinMetric,
    make_shots,
)
from colorai.skin_analysis import create_subject


def _synthetic_landmark_geometry():
    # 96x96 normalized coordinates. Oval covers most of the face; protected
    # polygons (eyes/mouth) are large enough that their centres stay exactly 0.
    oval = [[0.156, 0.083], [0.844, 0.083], [0.875, 0.5], [0.844, 0.917], [0.156, 0.917], [0.125, 0.5]]
    left_eye = [[0.18, 0.18], [0.48, 0.18], [0.48, 0.44], [0.18, 0.44]]
    right_eye = [[0.60, 0.18], [0.86, 0.18], [0.86, 0.44], [0.60, 0.44]]
    mouth = [[0.30, 0.56], [0.70, 0.56], [0.70, 0.78], [0.30, 0.78]]
    return {
        "oval": oval,
        "eyes": [left_eye, right_eye],
        "brows": [],
        "lips": [mouth],
        "hairline": [],
    }


def test_landmark_mask_excludes_protected_eye_mouth_and_hair_pixels():
    geometry = _synthetic_landmark_geometry()
    mask = render_face_mask((96, 96), geometry, strategy=LANDMARK_STRATEGY)
    assert mask[30, 32] == 0.0  # inside left eye -> protected
    assert mask[65, 48] == 0.0  # inside mouth -> protected
    assert mask[0, 48] < 0.1  # outside oval (hair) -> excluded
    assert mask[52, 48] > 0.5  # cheek -> skin


def test_fallback_mask_is_a_feathered_oval_without_protection():
    geometry = {"oval": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]], "eyes": [], "brows": [], "lips": [], "hairline": []}
    mask = render_face_mask((96, 96), geometry, strategy=FALLBACK_STRATEGY)
    assert mask[48, 48] > 0.5
    assert mask[4, 48] < 0.1


def test_interpolate_mask_geometry_interpolates_and_clamps():
    g0 = {"oval": [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]]}
    g1 = {"oval": [[0.4, 0.4], [0.9, 0.4], [0.4, 0.9]]}
    keyframes = [(0, g0), (10, g1)]
    mid = interpolate_mask_geometry(keyframes, 5)
    assert mid["oval"][0] == pytest.approx([0.2, 0.2])
    assert interpolate_mask_geometry(keyframes, -1) is g0
    assert interpolate_mask_geometry(keyframes, 99) is g1
    assert interpolate_mask_geometry([], 0) is None


def _fixture(tmp_path):
    store = ProjectStore.create(str(tmp_path / "p.sqlite3"))
    project = store.create_project("masks")
    asset = store.add_asset(
        project.id, source_path="/media/m.mov", frame_rate=25.0, width=96, height=96
    )
    shots = make_shots(asset, [(0, 0)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)
    shot = shots[0]
    alice = create_subject(store, asset.id, "Alice")
    group = create_group(store, asset.id, "interview", kind="setup")
    assign_shot_group(store, shot.id, group.id)
    with store.session() as session:
        metric = SkinMetric(
            shot_id=shot.id, face_index=0, mean_b=0.3, mean_g=0.3, mean_r=0.5,
            sample_pixels=100, subject_id=alice.id, bbox_x=16, bbox_y=8, bbox_w=64, bbox_h=80,
        )
        session.add(metric)
        session.flush()
        track = FaceTrack(
            shot_id=shot.id, skin_metric_id=metric.id, subject_id=alice.id,
            source_width=96, source_height=96, analysis_scale=96,
            keyframes=[[0, 0.2, 0.1, 0.6, 0.8]],
            sample_count=1, tracked_count=1, coverage=1.0, max_gap=0.0,
            skin_stability=0.01, median_bgr=[0.3, 0.3, 0.5], state="valid",
        )
        session.add(track)
        session.flush()
        metric_id, track_id = metric.id, track.id
        session.commit()
    return store, asset, shot, alice, metric_id, track_id


def _fake_extract(tmp_path):
    def extract(video_path, frame_index, out_path, fps=None, scale=None):
        img = np.zeros((96, 96, 3), dtype=np.uint8)
        img[:, :] = (89, 97, 148)  # BGR skin-like
        cv2.imwrite(str(out_path), img)
        return out_path

    return extract


def test_build_mask_track_fallback_is_labelled_and_valid(tmp_path):
    store, asset, shot, alice, metric_id, track_id = _fixture(tmp_path)
    mask = build_face_mask_track(
        store, track_id, detector=None, samples=2, extract=_fake_extract(tmp_path)
    )
    assert mask.backend == "fallback"
    assert mask.strategy == FALLBACK_STRATEGY
    assert mask.state == "valid"
    assert mask.review_state == "unreviewed"
    assert len(mask.landmark_keyframes) == 1


def test_build_mask_track_landmark_subtracts_protected(tmp_path):
    store, asset, shot, alice, metric_id, track_id = _fixture(tmp_path)

    def detector(image_bgr, box):
        return _synthetic_landmark_geometry()

    mask = build_face_mask_track(
        store, track_id, detector=detector, samples=2, extract=_fake_extract(tmp_path)
    )
    assert mask.backend == "mediapipe"
    assert mask.strategy == LANDMARK_STRATEGY
    assert mask.state == "valid"
    rendered = render_face_mask((96, 96), mask.landmark_keyframes[0][1], strategy=LANDMARK_STRATEGY)
    assert rendered[30, 32] == 0.0


def test_unreviewed_mask_cannot_be_used_for_a_proposal(tmp_path):
    store, asset, shot, alice, metric_id, track_id = _fixture(tmp_path)
    mask = build_face_mask_track(
        store, track_id, detector=None, samples=2, extract=_fake_extract(tmp_path)
    )
    with pytest.raises(ValueError, match="approved_for_proposal"):
        validate_face_mask_track(store, mask.id, require_review=True)


def test_validate_face_mask_requires_valid_quality(tmp_path):
    store, asset, shot, alice, metric_id, track_id = _fixture(tmp_path)
    with store.session() as session:
        mask = FaceMaskTrack(
            face_track_id=track_id, shot_id=shot.id, subject_id=alice.id,
            backend="fallback", backend_version="0", strategy=FALLBACK_STRATEGY,
            landmark_keyframes=[], coverage=0.1, max_gap=0.0,
            review_state="approved_for_proposal",
        )
        session.add(mask)
        session.flush()
        mask_id = mask.id
        session.commit()
    with pytest.raises(ValueError, match="coverage"):
        validate_face_mask_track(store, mask_id, require_review=False)


def test_fallback_mask_requires_explicit_human_approval(tmp_path):
    store, asset, shot, alice, metric_id, track_id = _fixture(tmp_path)
    with store.session() as session:
        mask = FaceMaskTrack(
            face_track_id=track_id, shot_id=shot.id, subject_id=alice.id,
            backend="fallback", backend_version="0", strategy=FALLBACK_STRATEGY,
            landmark_keyframes=[], coverage=1.0, max_gap=0.0,
            review_state="approved_for_proposal", human_approved=False,
        )
        session.add(mask)
        session.flush()
        mask_id = mask.id
        session.commit()

    with pytest.raises(ValueError, match="human approval"):
        validate_face_mask_track(store, mask_id, require_review=True)

    # Once a human explicitly approves, the fallback is usable.
    with store.session() as session:
        m = session.get(FaceMaskTrack, mask_id)
        m.human_approved = True
        session.commit()
    assert validate_face_mask_track(store, mask_id, require_review=True).id == mask_id
