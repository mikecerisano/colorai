"""Tests for skin-appearance references, mask tracks, and targets."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from colorai.editorial import assign_shot_group, create_group
from colorai.project import (
    FaceMaskTrack,
    FaceTrack,
    ProjectStore,
    Shot,
    ShotGroup,
    SkinAppearanceReference,
    SkinAppearanceTarget,
    SkinMetric,
    Subject,
    make_shots,
)
from colorai.skin_analysis import create_subject


def _fixture(tmp_path, n_shots=2):
    """A project/asset with one subject, a setup group, shots, and skin metrics."""
    store = ProjectStore.create(str(tmp_path / "p.sqlite3"))
    project = store.create_project("skin targets")
    asset = store.add_asset(
        project.id, source_path="/media/m.mov", frame_rate=25.0, width=1920, height=1080
    )
    shots = make_shots(asset, [(0, 24), (25, 49)][:n_shots])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)
    alice = create_subject(store, asset.id, "Alice")
    group = create_group(store, asset.id, "interview", kind="setup")
    for s in shots:
        assign_shot_group(store, s.id, group.id)

    metrics = []
    with store.session() as session:
        for s in shots:
            m = SkinMetric(
                shot_id=s.id, face_index=0,
                mean_b=0.30, mean_g=0.30, mean_r=0.50,
                sample_pixels=100, subject_id=alice.id,
                bbox_x=100, bbox_y=120, bbox_w=200, bbox_h=240,
            )
            session.add(m)
            session.flush()
            metrics.append(m)
        session.commit()
    return store, asset, shots, alice, group, metrics


def test_skin_target_records_provenance_and_safe_defaults(tmp_path):
    store, asset, shots, alice, group, metrics = _fixture(tmp_path)

    with store.session() as session:
        ref = SkinAppearanceReference(
            subject_id=alice.id, asset_id=asset.id, group_id=group.id,
            source_kind="project_frame", role="accurate_skin_reference",
            content_hash="a" * 64, source_shot_id=shots[0].id,
            frame_index=shots[0].start_frame,
            crop_geometry={"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
        )
        session.add(ref)
        session.flush()

        target = SkinAppearanceTarget(
            subject_id=alice.id, group_id=group.id, reference_id=ref.id,
            profile={"mean_ab": [0.01, 0.02], "spread_ab": [0.01, 0.01]},
            approved_preview_parameters={"version": 1},
        )
        session.add(target)
        session.flush()

        assert target.state == "suggested"
        assert ref.state == "active"
        assert ref.crop_geometry["x"] == 0.1

    # FaceMaskTrack defaults: valid geometry, unreviewed, with quality fields.
    with store.session() as session:
        track = FaceTrack(
            shot_id=shots[0].id, skin_metric_id=metrics[0].id, subject_id=alice.id,
            source_width=1920, source_height=1080, analysis_scale=480,
            keyframes=[[0, 0.1, 0.1, 0.2, 0.2], [24, 0.1, 0.1, 0.2, 0.2]],
            sample_count=2, tracked_count=2, coverage=1.0, max_gap=0.0,
            skin_stability=0.01, median_bgr=[0.30, 0.30, 0.50], state="valid",
        )
        session.add(track)
        session.flush()
        mask = FaceMaskTrack(
            face_track_id=track.id, shot_id=shots[0].id, subject_id=alice.id,
            backend="fallback", backend_version="0", strategy="face_oval_skin",
            landmark_keyframes=[], coverage=1.0, max_gap=0.0,
        )
        session.add(mask)
        session.flush()
        assert mask.state == "valid"
        assert mask.review_state == "unreviewed"
        assert mask.review_reason == ""
