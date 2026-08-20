"""Tests for the draft-only skin-target MCP surface."""

from __future__ import annotations

import asyncio

import pytest

from colorai import mcp_server
from colorai.editorial import assign_shot_group, create_group
from colorai.project import (
    FaceMaskTrack,
    FaceTrack,
    ProjectStore,
    SkinMetric,
    make_shots,
)
from colorai.skin_analysis import create_subject


def _store(tmp_path, n_shots=2):
    db = tmp_path / "p.sqlite3"
    store = ProjectStore.create(db)
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
    track_ids = []
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
            track = FaceTrack(
                shot_id=s.id, skin_metric_id=m.id, subject_id=alice.id,
                source_width=1920, source_height=1080, analysis_scale=480,
                keyframes=[[s.start_frame, 0.1, 0.1, 0.2, 0.2]],
                sample_count=1, tracked_count=1, coverage=1.0, max_gap=0.0,
                skin_stability=0.01, median_bgr=[0.30, 0.30, 0.50], state="valid",
            )
            session.add(track)
            session.flush()
            metrics.append(m)
            track_ids.append(track.id)
        session.commit()

    return db, asset, shots, alice, group, metrics, track_ids


def test_skin_target_mcp_surface_is_draft_only():
    names = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert {
        "skin_target_workspace",
        "request_skin_reference",
        "create_skin_appearance_reference",
        "list_skin_appearance_references",
        "build_face_mask_track",
        "get_face_mask_contact_sheet",
        "review_face_mask_evidence",
        "suggest_skin_appearance_target",
        "list_skin_appearance_targets",
        "match_skin_target_to_setup",
    } <= names
    # Human-only actions are never exposed to the agent.
    for forbidden in (
        "approve_skin_target",
        "reject_skin_target",
        "approve_face_correction",
        "enable_face_correction",
        "render_master",
    ):
        assert forbidden not in names


def test_mcp_cannot_draft_target_before_mask_review(tmp_path):
    db, asset, shots, alice, group, metrics, track_ids = _store(tmp_path)
    with ProjectStore.open(db).session() as session:
        session.add(FaceMaskTrack(
            face_track_id=track_ids[0], shot_id=shots[0].id, subject_id=alice.id,
            backend="fallback", backend_version="0", strategy="face_oval_skin",
            landmark_keyframes=[], coverage=1.0, max_gap=0.0,
            review_state="unreviewed",
        ))
        session.commit()

    out = mcp_server.suggest_skin_appearance_target(
        db, alice.id, group.id, None, track_ids[0],
        {"version": 1, "space": "oklab", "luma_mode": "preserve",
         "ab_offset": [0, 0], "ab_scale": [1, 1], "strength": 0.5},
        "test", 0.8,
    )
    assert "approved_for_proposal" in out.get("error", "")


def test_mcp_match_requires_approved_target(tmp_path):
    db, asset, shots, alice, group, metrics, track_ids = _store(tmp_path)
    # Create a suggested (not approved) target directly.
    from colorai.project import SkinAppearanceTarget

    with ProjectStore.open(db).session() as session:
        target = SkinAppearanceTarget(
            subject_id=alice.id, group_id=group.id, reference_id=None,
            profile={"mean_ab": [0, 0], "spread_ab": [0.01, 0.01]},
            approved_preview_parameters={}, state="suggested",
        )
        session.add(target)
        session.flush()
        target_id = target.id
        session.commit()

    out = mcp_server.match_skin_target_to_setup(db, target_id, group.id)
    assert "approved" in out.get("error", "")


def test_mcp_review_face_mask_evidence_records_review(tmp_path):
    db, asset, shots, alice, group, metrics, track_ids = _store(tmp_path)
    with ProjectStore.open(db).session() as session:
        mask = FaceMaskTrack(
            face_track_id=track_ids[0], shot_id=shots[0].id, subject_id=alice.id,
            backend="fallback", backend_version="0", strategy="face_oval_skin",
            landmark_keyframes=[], coverage=1.0, max_gap=0.0,
            review_state="unreviewed",
        )
        session.add(mask)
        session.flush()
        mask_id = mask.id
        session.commit()

    out = mcp_server.review_face_mask_evidence(db, mask_id, "approved_for_proposal", "looks clean")
    assert out["review_state"] == "approved_for_proposal"


def test_mcp_build_face_mask_track_uses_fallback_when_mediapipe_unavailable(tmp_path, monkeypatch):
    """When MediaPipe is unavailable the MCP path must produce the labelled
    fallback (detector=None), not fail every frame because a detector returns
    None."""
    import colorai.face_masks as fm

    db, asset, shots, alice, group, metrics, track_ids = _store(tmp_path)

    calls = {}

    def spy(store, face_track_id, *, detector=None, samples=16):
        calls["detector"] = detector
        return type(
            "Mask", (),
            {
                "id": 1, "state": "valid", "backend": "fallback",
                "strategy": "face_oval_skin", "coverage": 1.0, "max_gap": 0.0,
                "review_state": "unreviewed", "review_reason": "",
            },
        )()

    monkeypatch.setattr(fm, "landmark_backend_available", lambda: False)
    monkeypatch.setattr(fm, "build_face_mask_track", spy)

    out = mcp_server.build_face_mask_track(db, track_ids[0])
    assert calls["detector"] is None
    assert out["backend"] == "fallback"
    assert out["strategy"] == "face_oval_skin"
