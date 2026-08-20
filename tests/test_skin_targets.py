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


def _reviewed_tracks(tmp_path):
    """A subject/setup with two shots, valid face tracks, and reviewed masks."""
    store, asset, shots, alice, group, metrics = _fixture(tmp_path, n_shots=2)
    track_ids = []
    mask_ids = []
    with store.session() as session:
        for m in metrics:
            track = FaceTrack(
                shot_id=m.shot_id, skin_metric_id=m.id, subject_id=alice.id,
                source_width=1920, source_height=1080, analysis_scale=480,
                keyframes=[[m.shot_id * 25, 0.1, 0.1, 0.2, 0.2]],
                sample_count=1, tracked_count=1, coverage=1.0, max_gap=0.0,
                skin_stability=0.01, median_bgr=[0.3, 0.3, 0.5], state="valid",
            )
            session.add(track)
            session.flush()
            mask = FaceMaskTrack(
                face_track_id=track.id, shot_id=m.shot_id, subject_id=alice.id,
                backend="fallback", backend_version="0", strategy="face_oval_skin",
                landmark_keyframes=[], coverage=1.0, max_gap=0.0,
                review_state="approved_for_proposal",
            )
            session.add(mask)
            session.flush()
            track_ids.append(track.id)
            mask_ids.append(mask.id)
        session.commit()
    return store, asset, shots, alice, group, metrics, track_ids, mask_ids


def test_external_reference_is_copied_by_hash_and_retains_provenance(tmp_path):
    store, asset, shots, alice, group, metrics = _fixture(tmp_path)
    reference = tmp_path / "accurate.jpg"
    reference.write_bytes(b"pixels")

    from colorai.skin_targets import create_skin_reference

    created = create_skin_reference(
        store, subject_id=alice.id, group_id=group.id,
        external_path=reference, role="accurate_skin_reference",
    )
    assert Path(created.managed_path).read_bytes() == b"pixels"
    assert created.source_path == str(reference)
    assert created.content_hash == hashlib.sha256(b"pixels").hexdigest()
    assert created.source_kind == "external_image"


def test_external_reference_requires_existing_file(tmp_path):
    store, asset, shots, alice, group, metrics = _fixture(tmp_path)
    from colorai.skin_targets import create_skin_reference

    with pytest.raises(ValueError, match="does not exist"):
        create_skin_reference(
            store, subject_id=alice.id, group_id=group.id,
            external_path=tmp_path / "missing.jpg", role="accurate_skin_reference",
        )


def test_project_frame_reference_must_be_in_group_and_have_face(tmp_path):
    store, asset, shots, alice, group, metrics = _fixture(tmp_path)
    from colorai.skin_targets import create_skin_reference

    created = create_skin_reference(
        store, subject_id=alice.id, group_id=group.id,
        source_shot_id=shots[0].id, frame_index=shots[0].start_frame,
        role="accurate_skin_reference",
    )
    assert created.source_kind == "project_frame"
    assert created.source_shot_id == shots[0].id


def test_skin_target_approval_sequence(tmp_path):
    store, asset, shots, alice, group, metrics, track_ids, mask_ids = _reviewed_tracks(tmp_path)
    from colorai.skin_targets import (
        approve_skin_target,
        create_skin_reference,
        suggest_skin_target,
    )

    ref = create_skin_reference(
        store, subject_id=alice.id, group_id=group.id,
        source_shot_id=shots[0].id, frame_index=shots[0].start_frame,
        role="accurate_skin_reference",
    )
    target = suggest_skin_target(
        store, subject_id=alice.id, group_id=group.id, reference_id=ref.id,
        face_track_id=track_ids[0],
        parameters={"version": 1, "space": "oklab", "luma_mode": "preserve",
                    "ab_offset": [-0.01, 0.0], "ab_scale": [1.0, 1.0], "strength": 0.6},
        rationale="neutralize red", confidence=0.8,
    )
    assert target.state == "suggested"
    approved = approve_skin_target(store, target.id)
    assert approved.state == "approved"


def test_suggest_skin_target_requires_reviewed_mask(tmp_path):
    store, asset, shots, alice, group, metrics = _fixture(tmp_path, n_shots=2)
    from colorai.skin_targets import suggest_skin_target

    with store.session() as session:
        track = FaceTrack(
            shot_id=shots[0].id, skin_metric_id=metrics[0].id, subject_id=alice.id,
            source_width=1920, source_height=1080,
            keyframes=[[0, 0.1, 0.1, 0.2, 0.2]],
            sample_count=1, tracked_count=1, coverage=1.0, max_gap=0.0,
            skin_stability=0.01, median_bgr=[0.3, 0.3, 0.5], state="valid",
        )
        session.add(track)
        session.flush()
        mask = FaceMaskTrack(
            face_track_id=track.id, shot_id=shots[0].id, subject_id=alice.id,
            backend="fallback", backend_version="0", strategy="face_oval_skin",
            landmark_keyframes=[], coverage=1.0, max_gap=0.0,
            review_state="unreviewed",
        )
        session.add(mask)
        session.flush()
        track_id = track.id
        session.commit()

    with pytest.raises(ValueError, match="approved_for_proposal"):
        suggest_skin_target(
            store, subject_id=alice.id, group_id=group.id, reference_id=None,
            face_track_id=track_id,
            parameters={"version": 1, "space": "oklab", "luma_mode": "preserve",
                        "ab_offset": [0.0, 0.0], "ab_scale": [1.0, 1.0], "strength": 0.5},
            rationale="x", confidence=0.5,
        )


def test_match_rejects_cross_variant(tmp_path):
    store, asset, shots, alice, group, metrics, track_ids, mask_ids = _reviewed_tracks(tmp_path)
    from colorai.editorial import create_group
    from colorai.skin_targets import (
        approve_skin_target,
        create_skin_reference,
        match_skin_target_to_group,
        suggest_skin_target,
    )

    ref = create_skin_reference(
        store, subject_id=alice.id, group_id=group.id,
        source_shot_id=shots[0].id, frame_index=shots[0].start_frame,
        role="accurate_skin_reference",
    )
    target = suggest_skin_target(
        store, subject_id=alice.id, group_id=group.id, reference_id=ref.id,
        face_track_id=track_ids[0],
        parameters={"version": 1, "space": "oklab", "luma_mode": "preserve",
                    "ab_offset": [0.0, 0.0], "ab_scale": [1.0, 1.0], "strength": 0.5},
        rationale="x", confidence=0.5,
    )
    approve_skin_target(store, target.id)

    other_variant = create_group(store, asset.id, "variant B", kind="variant", parent_id=group.id)
    with pytest.raises(ValueError, match="exact target setup or variant"):
        match_skin_target_to_group(store, target_id=target.id, group_id=other_variant.id)


def test_match_one_shot_is_qc_only(tmp_path):
    store, asset, shots, alice, group, metrics = _fixture(tmp_path, n_shots=1)
    from colorai.skin_targets import (
        approve_skin_target,
        create_skin_reference,
        match_skin_target_to_group,
        suggest_skin_target,
    )
    with store.session() as session:
        track = FaceTrack(
            shot_id=shots[0].id, skin_metric_id=metrics[0].id, subject_id=alice.id,
            source_width=1920, source_height=1080,
            keyframes=[[0, 0.1, 0.1, 0.2, 0.2]],
            sample_count=1, tracked_count=1, coverage=1.0, max_gap=0.0,
            skin_stability=0.01, median_bgr=[0.3, 0.3, 0.5], state="valid",
        )
        session.add(track)
        session.flush()
        session.add(FaceMaskTrack(
            face_track_id=track.id, shot_id=shots[0].id, subject_id=alice.id,
            backend="fallback", backend_version="0", strategy="face_oval_skin",
            landmark_keyframes=[], coverage=1.0, max_gap=0.0,
            review_state="approved_for_proposal",
        ))
        session.flush()
        track_id = track.id
        session.commit()

    ref = create_skin_reference(
        store, subject_id=alice.id, group_id=group.id,
        source_shot_id=shots[0].id, frame_index=shots[0].start_frame,
        role="accurate_skin_reference",
    )
    target = suggest_skin_target(
        store, subject_id=alice.id, group_id=group.id, reference_id=ref.id,
        face_track_id=track_id,
        parameters={"version": 1, "space": "oklab", "luma_mode": "preserve",
                    "ab_offset": [0.0, 0.0], "ab_scale": [1.0, 1.0], "strength": 0.5},
        rationale="x", confidence=0.5,
    )
    approve_skin_target(store, target.id)
    result = match_skin_target_to_group(store, target_id=target.id, group_id=group.id)
    assert result["status"] == "qc_only"


def test_match_derives_distinct_parameters_per_candidate(tmp_path):
    store, asset, shots, alice, group, metrics, track_ids, mask_ids = _reviewed_tracks(tmp_path)
    from colorai.skin_targets import (
        approve_skin_target,
        create_skin_reference,
        match_skin_target_to_group,
        suggest_skin_target,
    )

    # Give the two candidate shots visibly different skin means.
    with store.session() as session:
        m1 = session.query(SkinMetric).filter_by(id=metrics[1].id).one()
        m1.mean_b = 0.25
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
                    "ab_offset": [0.0, 0.0], "ab_scale": [1.0, 1.0], "strength": 0.6},
        rationale="x", confidence=0.5,
    )
    approve_skin_target(store, target.id)
    result = match_skin_target_to_group(store, target_id=target.id, group_id=group.id)
    candidates = result["candidates"]
    ready = [c for c in candidates if c["state"] == "ready"]
    assert len(ready) == 2
    # The reference shot's own candidate is essentially identity; the other has
    # a non-zero offset toward the target.
    offsets = {c["shot_id"]: c["parameters"]["ab_offset"] for c in ready}
    assert offsets[shots[0].id] != offsets[shots[1].id]


def test_propose_skin_appearance_correction_is_suggested_and_disabled(tmp_path):
    store, asset, shots, alice, group, metrics, track_ids, mask_ids = _reviewed_tracks(tmp_path)
    from colorai.skin_targets import (
        approve_skin_target,
        create_skin_reference,
        propose_skin_appearance_correction,
        suggest_skin_target,
    )

    ref = create_skin_reference(
        store, subject_id=alice.id, group_id=group.id,
        source_shot_id=shots[0].id, frame_index=shots[0].start_frame,
        role="accurate_skin_reference",
    )
    target = suggest_skin_target(
        store, subject_id=alice.id, group_id=group.id, reference_id=ref.id,
        face_track_id=track_ids[0],
        parameters={"version": 1, "space": "oklab", "luma_mode": "preserve",
                    "ab_offset": [0.0, 0.0], "ab_scale": [1.0, 1.0], "strength": 0.5},
        rationale="x", confidence=0.5,
    )
    approve_skin_target(store, target.id)

    correction = propose_skin_appearance_correction(
        store, shot_id=shots[1].id, subject_id=alice.id, skin_metric_id=metrics[1].id,
        face_track_id=track_ids[1], skin_target_id=target.id,
        parameters={"version": 1, "space": "oklab", "luma_mode": "preserve",
                    "ab_offset": [-0.01, 0.0], "ab_scale": [1.0, 1.0], "strength": 0.5},
        reason="warm skin", confidence=0.8,
    )
    assert correction.state == "suggested"
    assert correction.enabled is False
    assert correction.kind == "skin_appearance"
    assert correction.skin_target_id == target.id


def test_propose_skin_appearance_rejects_cross_group_candidate(tmp_path):
    store, asset, shots, alice, group, metrics, track_ids, mask_ids = _reviewed_tracks(tmp_path)
    from colorai.editorial import create_group
    from colorai.skin_targets import (
        approve_skin_target,
        create_skin_reference,
        propose_skin_appearance_correction,
        suggest_skin_target,
    )

    ref = create_skin_reference(
        store, subject_id=alice.id, group_id=group.id,
        source_shot_id=shots[0].id, frame_index=shots[0].start_frame,
        role="accurate_skin_reference",
    )
    target = suggest_skin_target(
        store, subject_id=alice.id, group_id=group.id, reference_id=ref.id,
        face_track_id=track_ids[0],
        parameters={"version": 1, "space": "oklab", "luma_mode": "preserve",
                    "ab_offset": [0.0, 0.0], "ab_scale": [1.0, 1.0], "strength": 0.5},
        rationale="x", confidence=0.5,
    )
    approve_skin_target(store, target.id)

    # Move shots[1] to a different setup; it can no longer receive this target.
    from colorai.editorial import assign_shot_group

    other_group = create_group(store, asset.id, "other setup", kind="setup")
    assign_shot_group(store, shots[1].id, other_group.id)

    with pytest.raises(ValueError, match="outside the target's exact group"):
        propose_skin_appearance_correction(
            store, shot_id=shots[1].id, subject_id=alice.id, skin_metric_id=metrics[1].id,
            face_track_id=track_ids[1], skin_target_id=target.id,
            parameters={"version": 1, "space": "oklab", "luma_mode": "preserve",
                        "ab_offset": [0.0, 0.0], "ab_scale": [1.0, 1.0], "strength": 0.5},
            reason="x", confidence=0.5,
        )
