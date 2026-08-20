"""Tests for persisted face tracks and local face corrections."""

from __future__ import annotations

from colorai.project import (
    FaceCorrection,
    FaceTrack,
    ProjectStore,
    Shot,
    SkinMetric,
    Subject,
    make_shots,
)
from colorai.skin_analysis import create_subject


def _fixture():
    store = ProjectStore.create(":memory:")
    project = store.create_project("face corrections")
    asset = store.add_asset(
        project.id, source_path="/media/m.mov", frame_rate=25.0, width=1920, height=1080
    )
    shots = make_shots(asset, [(0, 49)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)
    alice = create_subject(store, asset.id, "Alice")
    with store.session() as session:
        session.add(
            SkinMetric(
                shot_id=shots[0].id, face_index=0,
                mean_b=0.30, mean_g=0.30, mean_r=0.50,
                sample_pixels=100, subject_id=alice.id,
                bbox_x=100, bbox_y=120, bbox_w=200, bbox_h=240,
            )
        )
        session.commit()
    with store.session() as session:
        metric = session.query(SkinMetric).filter_by(shot_id=shots[0].id).first()
    return store, asset, shots[0], alice, metric


def test_face_track_and_correction_persist_with_defaults():
    store, asset, shot, alice, metric = _fixture()

    with store.session() as session:
        track = FaceTrack(
            shot_id=shot.id, skin_metric_id=metric.id, subject_id=alice.id,
            source_width=1920, source_height=1080, analysis_scale=480,
            keyframes=[[0, 0.1, 0.1, 0.2, 0.2], [25, 0.1, 0.1, 0.2, 0.2]],
            sample_count=2, tracked_count=2, coverage=1.0, max_gap=0.0,
            skin_stability=0.02, median_bgr=[0.30, 0.30, 0.50], state="valid",
        )
        session.add(track)
        session.flush()
        session.refresh(track)

        correction = FaceCorrection(
            shot_id=shot.id, subject_id=alice.id, skin_metric_id=metric.id,
            face_track_id=track.id, reference_shot_id=shot.id,
            kind="rgb_balance", parameters={"gain": [1.0, 0.95, 0.90]},
            reason="slightly warm", classification="skin_mismatch", confidence=0.8,
        )
        session.add(correction)
        session.flush()
        session.refresh(correction)

        assert correction.state == "suggested"
        assert correction.enabled is False
        assert correction.kind == "rgb_balance"
        assert correction.face_track_id == track.id

    with store.session() as session:
        assert session.query(FaceTrack).count() == 1
        assert session.query(FaceCorrection).count() == 1


def test_face_correction_cannot_enable_without_approval_by_default():
    store, asset, shot, alice, metric = _fixture()
    with store.session() as session:
        correction = FaceCorrection(
            shot_id=shot.id, subject_id=alice.id, skin_metric_id=metric.id,
            kind="rgb_balance", parameters={"gain": [1.0, 1.0, 1.0]},
            reason="", classification="skin_mismatch",
        )
        session.add(correction)
        session.commit()
        session.refresh(correction)
        assert correction.enabled is False
        assert correction.state == "suggested"
