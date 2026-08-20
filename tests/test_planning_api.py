"""Tests for the organization-draft review UI (API + server-rendered HTML)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from colorai.planning import create_organization_plan
from colorai.project import ProjectStore, SkinMetric, make_shots
from colorai.skin_analysis import create_subject
from colorai.ui import create_app


def _asset(store):
    project = store.create_project("draft test")
    asset = store.add_asset(
        project.id, source_path="/media/m.mov", frame_rate=25.0, width=64, height=64
    )
    shots = make_shots(asset, [(0, 24), (25, 49), (50, 74), (75, 99)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)
    alice = create_subject(store, asset.id, "Alice")
    bob = create_subject(store, asset.id, "Bob")

    def add_face(shot, face_index, subject_id):
        with store.session() as session:
            session.add(
                SkinMetric(
                    shot_id=shot.id, face_index=face_index,
                    mean_b=0.3, mean_g=0.3, mean_r=0.5, sample_pixels=100,
                    subject_id=subject_id, bbox_x=0, bbox_y=0, bbox_w=10, bbox_h=10,
                )
            )
            session.commit()

    add_face(shots[0], 0, alice.id)
    add_face(shots[1], 0, alice.id)
    add_face(shots[2], 0, alice.id)
    add_face(shots[2], 1, bob.id)
    return asset, shots, alice, bob


def _complete_plan(asset, shots, alice):
    groups = [{"draft_key": "s1", "name": "Alice interview", "kind": "setup", "participant_ids": [alice.id]}]
    items = [
        {"shot_id": shots[0].id, "destination_type": "planned_group", "target_draft_key": "s1", "decision": "accepted"},
        {"shot_id": shots[1].id, "destination_type": "planned_group", "target_draft_key": "s1", "decision": "accepted"},
        {"shot_id": shots[2].id, "destination_type": "unresolved", "decision": "rejected"},
        {"shot_id": shots[3].id, "destination_type": "broll", "decision": "accepted"},
    ]
    return groups, items


def _draft_client(tmp_path):
    store = ProjectStore.create(":memory:")
    asset, shots, alice, bob = _asset(store)
    groups, items = _complete_plan(asset, shots, alice)
    plan = create_organization_plan(store, asset.id, groups, items, summary="organize")
    client = TestClient(create_app(store, tmp_path / "stills"))
    return client, asset, shots, alice, bob, plan.id


def test_organization_draft_endpoint(tmp_path):
    client, asset, shots, alice, bob, plan_id = _draft_client(tmp_path)
    draft = client.get(f"/api/assets/{asset.id}/organization-draft").json()
    assert draft["id"] == plan_id
    assert draft["state"] == "draft"
    assert len(draft["groups"]) == 1
    assert draft["validation"]["errors"] == []


def test_html_renders_draft_sections(tmp_path):
    client, asset, shots, alice, bob, plan_id = _draft_client(tmp_path)
    body = client.get("/").text

    assert "Proposed setup families" in body
    assert "B-roll" in body
    assert "Needs a decision" in body
    assert "Validation and apply" in body
    assert "Apply accepted plan" in body
    assert "Alice interview" in body  # the proposed setup


def test_stage_plan_item_persists_without_reload(tmp_path):
    client, asset, shots, alice, bob, plan_id = _draft_client(tmp_path)

    r = client.patch(
        f"/api/plans/{plan_id}/items/{shots[2].id}",
        json={"decision": "accepted", "destination_type": "broll"},
    )
    assert r.status_code == 200
    assert r.json()["decision"] == "accepted"
    assert r.json()["destination_type"] == "broll"

    draft = client.get(f"/api/assets/{asset.id}/organization-draft").json()
    needs = {n["shot_id"]: n for n in draft["needs_decision"]}
    # shot 2 is now accepted, so it left the "needs a decision" bucket.
    assert shots[2].id not in needs


def test_apply_failure_rolls_back_via_api(tmp_path):
    store = ProjectStore.create(":memory:")
    asset, shots, alice, bob = _asset(store)
    # Incomplete coverage: only one shot planned.
    plan = create_organization_plan(
        store, asset.id, [],
        [{"shot_id": shots[0].id, "destination_type": "broll", "decision": "accepted"}],
    )
    client = TestClient(create_app(store, tmp_path / "stills"))

    client.post(f"/api/plans/{plan.id}/approve")
    r = client.post(f"/api/plans/{plan.id}/apply")
    body = r.json()
    assert body["error"] == "validation failed"

    # No shot was moved or excused.
    from colorai.project import Shot

    with store.session() as session:
        for s in session.query(Shot).filter_by(asset_id=asset.id).all():
            assert s.group_id is None and s.excused is False
