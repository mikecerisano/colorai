"""Tests for the skin-target review API and workspace surface."""

from __future__ import annotations

import cv2
import numpy as np
from fastapi.testclient import TestClient

from colorai.editorial import assign_shot_group, create_group
from colorai.project import (
    FaceMaskTrack,
    FaceTrack,
    ProjectStore,
    SkinAppearanceReference,
    SkinAppearanceTarget,
    SkinMetric,
    make_representative_frame,
    make_shots,
)
from colorai.skin_analysis import create_subject
from colorai.ui import create_app


def _target_fixture(tmp_path):
    store = ProjectStore.create(":memory:")
    project = store.create_project("skin target ui")
    asset = store.add_asset(
        project.id, source_path="/media/m.mov", frame_rate=25.0, width=64, height=64
    )
    shots = make_shots(asset, [(0, 24), (25, 49)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)

    stills = tmp_path / "stills"
    stills.mkdir()
    still = stills / "still0.png"
    cv2.imwrite(str(still), np.full((64, 64, 3), 128, dtype=np.uint8))
    with store.session() as session:
        session.add(make_representative_frame(shots[0], 0, image_path=str(still), frame_rate=25.0))
        session.commit()

    alice = create_subject(store, asset.id, "Alice")
    group = create_group(store, asset.id, "interview", kind="setup")
    for s in shots:
        assign_shot_group(store, s.id, group.id)

    with store.session() as session:
        m0 = SkinMetric(
            shot_id=shots[0].id, face_index=0, mean_b=0.30, mean_g=0.30, mean_r=0.50,
            sample_pixels=10, subject_id=alice.id, bbox_x=16, bbox_y=16, bbox_w=32, bbox_h=32,
        )
        session.add(m0)
        session.flush()
        track = FaceTrack(
            shot_id=shots[0].id, skin_metric_id=m0.id, subject_id=alice.id,
            source_width=64, source_height=64, analysis_scale=64,
            keyframes=[[0, 0.25, 0.25, 0.5, 0.5]],
            sample_count=1, tracked_count=1, coverage=1.0, max_gap=0.0,
            skin_stability=0.01, median_bgr=[0.3, 0.3, 0.5], state="valid",
        )
        session.add(track)
        session.flush()
        mask = FaceMaskTrack(
            face_track_id=track.id, shot_id=shots[0].id, subject_id=alice.id,
            backend="fallback", backend_version="0", strategy="face_oval_skin",
            landmark_keyframes=[], coverage=1.0, max_gap=0.0,
            review_state="approved_for_proposal", human_approved=True,
        )
        session.add(mask)
        session.flush()
        ref = SkinAppearanceReference(
            subject_id=alice.id, asset_id=asset.id, group_id=group.id,
            source_kind="project_frame", role="accurate_skin_reference",
            content_hash="a" * 64, source_shot_id=shots[0].id, frame_index=0,
        )
        session.add(ref)
        session.flush()
        target = SkinAppearanceTarget(
            subject_id=alice.id, group_id=group.id, reference_id=ref.id,
            profile={"mean_ab": [0.0, 0.0], "spread_ab": [0.01, 0.01]},
            approved_preview_parameters={"version": 1, "space": "oklab", "luma_mode": "preserve",
                                          "ab_offset": [-0.01, 0.0], "ab_scale": [1.0, 1.0], "strength": 0.5},
            state="suggested", rationale="neutralize red", confidence=0.8,
        )
        session.add(target)
        session.flush()
        ids = {"asset": asset.id, "alice": alice.id, "group": group.id,
               "mask": mask.id, "target": target.id, "shot0": shots[0].id, "shot1": shots[1].id}
        session.commit()

    client = TestClient(create_app(store, stills))
    return client, ids


def test_skin_target_workspace_shows_reference_mask_and_disabled_proposal(tmp_path):
    client, ids = _target_fixture(tmp_path)
    html = client.get("/").text
    assert "Skin target" in html
    assert "accurate reference" in html
    assert "mask reviewed" in html
    assert "Approve target" in html
    assert "Approve mask" in html


def test_target_actions_preserve_selected_setup_view(tmp_path):
    client, ids = _target_fixture(tmp_path)
    response = client.post(f"/api/skin-targets/{ids['target']}/approve")
    assert response.status_code == 200
    assert response.json() == {"id": ids["target"], "state": "approved"}
    # The setup workspace remains present after the action.
    html = client.get("/").text
    assert "interview" in html
    assert "Skin target" in html


def test_reject_target(tmp_path):
    client, ids = _target_fixture(tmp_path)
    response = client.post(f"/api/skin-targets/{ids['target']}/reject")
    assert response.json()["state"] == "rejected"


def test_project_frame_reference_intake_api(tmp_path):
    client, ids = _target_fixture(tmp_path)
    # A new reference via the API (project frame).
    r = client.post(
        f"/api/assets/{ids['asset']}/skin-references",
        json={"subject_id": ids["alice"], "group_id": ids["group"],
              "role": "creative_look_direction",
              "source_shot_id": ids["shot0"], "frame_index": 0},
    )
    assert r.status_code == 201
    assert r.json()["role"] == "creative_look_direction"


def test_mask_contact_sheet_endpoint_returns_png(tmp_path):
    client, ids = _target_fixture(tmp_path)
    r = client.get(f"/api/face-mask-tracks/{ids['mask']}/contact-sheet.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_mask_review_endpoint(tmp_path):
    client, ids = _target_fixture(tmp_path)
    r = client.post(
        f"/api/face-mask-tracks/{ids['mask']}/review",
        json={"review_state": "needs_rebuild", "reason": "too coarse"},
    )
    assert r.status_code == 200
    assert r.json()["review_state"] == "needs_rebuild"
