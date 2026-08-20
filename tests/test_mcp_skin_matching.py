"""Tests for the skin-first matching MCP surface (draft-only permissions)."""

from __future__ import annotations

import asyncio

import pytest

from colorai import mcp_server
from colorai.editorial import assign_shot_group, create_group
from colorai.project import (
    FaceTrack,
    ProjectStore,
    Shot,
    SkinMetric,
    make_shots,
)
from colorai.references import approve_reference, propose_reference
from colorai.skin_analysis import create_subject


def _store(tmp_path, n_shots=3):
    db = tmp_path / "p.sqlite3"
    store = ProjectStore.create(db)
    project = store.create_project("skin")
    asset = store.add_asset(
        project.id, source_path="/media/m.mov", frame_rate=25.0, width=1920, height=1080
    )
    shots = make_shots(asset, [(0, 24), (25, 49), (50, 74)][:n_shots])
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
        for i, s in enumerate(shots):
            m = SkinMetric(
                shot_id=s.id, face_index=0,
                mean_b=0.30, mean_g=0.30, mean_r=0.50,
                sample_pixels=100, subject_id=alice.id,
                bbox_x=100, bbox_y=120, bbox_w=200, bbox_h=240,
            )
            session.add(m)
            session.flush()
            session.add(
                FaceTrack(
                    shot_id=s.id, skin_metric_id=m.id, subject_id=alice.id,
                    source_width=1920, source_height=1080, analysis_scale=480,
                    keyframes=[[s.start_frame, 0.1, 0.1, 0.2, 0.2], [s.end_frame, 0.1, 0.1, 0.2, 0.2]],
                    sample_count=16, tracked_count=16, coverage=1.0, max_gap=0.0,
                    skin_stability=0.01, median_bgr=[0.30, 0.30, 0.50], state="valid",
                )
            )
            metrics.append(m)
        session.commit()

    return db, asset, shots, alice, group, metrics


def _approved_reference(db, asset, alice, group, shot):
    ref = propose_reference(
        ProjectStore.open(db), asset_id=asset.id, shot_id=shot.id,
        reason="hero", confidence=0.9, subject_id=alice.id, group_id=group.id,
    )
    approve_reference(ProjectStore.open(db), ref.id)
    return shot.id


def test_mcp_surface_excludes_human_face_actions():
    names = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert "skin_matching_workspace" in names
    assert "build_face_track" in names
    assert "get_face_track_contact_sheet" in names
    assert "skin_first_match_subject_setup" in names
    assert "propose_face_correction" in names
    assert "list_face_corrections" in names
    assert "get_face_correction" in names
    assert "update_face_correction" in names
    # Human-only actions are never exposed to the agent.
    for forbidden in ("approve_face_correction", "enable_face_correction", "reject_face_correction", "delete_face_correction"):
        assert forbidden not in names


def test_skin_first_requires_approved_reference(tmp_path):
    db, asset, shots, alice, group, metrics = _store(tmp_path)
    result = mcp_server.skin_first_match_subject_setup(db, asset.id, alice.id, group.id)
    assert result["error"] is not None
    assert "approved reference" in result["error"]


def test_skin_first_qc_only_for_one_shot(tmp_path):
    db, asset, shots, alice, group, metrics = _store(tmp_path, n_shots=1)
    _approved_reference(db, asset, alice, group, shots[0])
    result = mcp_server.skin_first_match_subject_setup(db, asset.id, alice.id, group.id)
    assert result["status"] == "qc_only"


def test_skin_first_no_whole_frame_corrections_on_background_diff(tmp_path):
    from colorai.project import FrameMetrics

    db, asset, shots, alice, group, metrics = _store(tmp_path)
    _approved_reference(db, asset, alice, group, shots[0])

    # Radically different whole-frame luma (background), identical skin.
    with ProjectStore.open(db).session() as session:
        for i, s in enumerate(shots):
            session.add(
                FrameMetrics(
                    shot_id=s.id, frame_index=s.start_frame,
                    luma_mean=[0.2, 0.5, 0.9][i], luma_std=0.1,
                    r_mean=0.5, g_mean=0.5, b_mean=0.5, saturation_mean=0.1,
                )
            )
        session.commit()

    result = mcp_server.skin_first_match_subject_setup(db, asset.id, alice.id, group.id)
    assert result["status"] == "ok"
    # Skin is identical, so no candidate gain; and there is no whole-frame
    # correction field anywhere in the skin-first result.
    assert all(c["gain_rgb"] is None for c in result["candidates"])
    assert "corrections" not in result
    assert "whole_frame" not in result


def test_skin_first_proposes_capped_gain_when_skin_differs(tmp_path):
    db, asset, shots, alice, group, metrics = _store(tmp_path)
    _approved_reference(db, asset, alice, group, shots[0])

    # Make shots[1] skin warmer (lower blue) so a gain is proposed.
    with ProjectStore.open(db).session() as session:
        m = session.query(SkinMetric).filter_by(id=metrics[1].id).one()
        m.mean_b = 0.25
        session.commit()

    result = mcp_server.skin_first_match_subject_setup(db, asset.id, alice.id, group.id)
    cand = next(c for c in result["candidates"] if c["shot_id"] == shots[1].id)
    assert cand["gain_rgb"] is not None
    assert all(0.90 <= g <= 1.10 for g in cand["gain_rgb"])


def test_propose_face_correction_is_suggested_and_disabled(tmp_path):
    db, asset, shots, alice, group, metrics = _store(tmp_path, n_shots=2)
    _approved_reference(db, asset, alice, group, shots[0])
    with ProjectStore.open(db).session() as session:
        track_id = session.query(FaceTrack).filter_by(skin_metric_id=metrics[1].id).one().id

    created = mcp_server.propose_face_correction(
        db, shots[1].id, alice.id, metrics[1].id, track_id,
        shots[0].id, group.id, "warm skin", 0.8, "skin_mismatch", [1.0, 0.95, 1.05],
    )
    assert created["state"] == "suggested"
    assert created["enabled"] is False

    # Non-skin_mismatch classification must not carry a correction.
    bad = mcp_server.propose_face_correction(
        db, shots[1].id, alice.id, metrics[1].id, track_id,
        shots[0].id, group.id, "lighting", 0.8, "intentional_lighting", [1.0, 1.0, 1.0],
    )
    assert "error" in bad


def test_propose_rejects_track_subject_mismatch(tmp_path):
    db, asset, shots, alice, group, metrics = _store(tmp_path, n_shots=2)
    _approved_reference(db, asset, alice, group, shots[0])
    bob = create_subject(ProjectStore.open(db), asset.id, "Bob")
    with ProjectStore.open(db).session() as session:
        track_id = session.query(FaceTrack).filter_by(skin_metric_id=metrics[1].id).one().id

    out = mcp_server.propose_face_correction(
        db, shots[1].id, bob.id, metrics[1].id, track_id,
        shots[0].id, group.id, "x", 0.8, "skin_mismatch", [1.0, 1.0, 1.0],
    )
    assert "different subject" in out.get("error", "")


def test_propose_rejects_wrong_reference_shot(tmp_path):
    db, asset, shots, alice, group, metrics = _store(tmp_path, n_shots=2)
    _approved_reference(db, asset, alice, group, shots[0])
    with ProjectStore.open(db).session() as session:
        track_id = session.query(FaceTrack).filter_by(skin_metric_id=metrics[1].id).one().id

    out = mcp_server.propose_face_correction(
        db, shots[1].id, alice.id, metrics[1].id, track_id,
        shots[1].id, group.id, "x", 0.8, "skin_mismatch", [1.0, 1.0, 1.0],
    )
    assert "does not match the approved" in out.get("error", "")


def test_update_rejects_non_skin_mismatch_classification(tmp_path):
    db, asset, shots, alice, group, metrics = _store(tmp_path, n_shots=2)
    _approved_reference(db, asset, alice, group, shots[0])
    with ProjectStore.open(db).session() as session:
        track_id = session.query(FaceTrack).filter_by(skin_metric_id=metrics[1].id).one().id
    created = mcp_server.propose_face_correction(
        db, shots[1].id, alice.id, metrics[1].id, track_id,
        shots[0].id, group.id, "warm", 0.8, "skin_mismatch", [1.0, 1.0, 1.05],
    )
    out = mcp_server.update_face_correction(db, created["id"], classification="intentional_lighting")
    assert "skin_mismatch" in out.get("error", "")
