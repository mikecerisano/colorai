"""Tests for the MCP organization-planning surface."""

from __future__ import annotations

from colorai import mcp_server
from colorai.project import ProjectStore, SkinMetric, make_shots
from colorai.skin_analysis import create_subject


def _planning_store(tmp_path, *, with_stills=False):
    db = tmp_path / "project.sqlite3"
    store = ProjectStore.create(db)
    project = store.create_project("film")
    asset = store.add_asset(project.id, source_path="/media/m.mov", frame_rate=25.0, width=64, height=64)
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

    if with_stills:
        import cv2
        import numpy as np

        from colorai.project import make_representative_frame

        with store.session() as session:
            for shot in shots:
                still = tmp_path / f"still_{shot.id}.png"
                cv2.imwrite(str(still), np.full((16, 16, 3), 40, dtype=np.uint8))
                session.add(
                    make_representative_frame(shot, shot.start_frame, image_path=str(still), frame_rate=25.0)
                )
            session.commit()

    return str(db), asset, shots, alice, bob


def _complete_plan(shots, alice):
    groups = [{"draft_key": "s1", "name": "Alice interview", "kind": "setup", "participant_ids": [alice.id]}]
    items = [
        {"shot_id": shots[0].id, "destination_type": "planned_group", "target_draft_key": "s1", "decision": "accepted"},
        {"shot_id": shots[1].id, "destination_type": "planned_group", "target_draft_key": "s1", "decision": "accepted"},
        {"shot_id": shots[2].id, "destination_type": "unresolved", "decision": "rejected"},
        {"shot_id": shots[3].id, "destination_type": "broll", "decision": "accepted"},
    ]
    return groups, items


def test_organization_workspace_and_draft_cycle(tmp_path):
    db, asset, shots, alice, bob = _planning_store(tmp_path)

    ws = mcp_server.organization_workspace(db, asset.id)
    assert [s["id"] for s in ws["subjects"]] == [alice.id, bob.id]
    assert ws["groups"] == []
    assert len(ws["ungrouped_shots"]) == 4
    assert ws["broll_group_id"] is None
    assert ws["active_draft"] is None

    groups, items = _complete_plan(shots, alice)
    created = mcp_server.create_organization_plan(db, asset.id, groups, items, summary="organize")
    assert created["state"] == "draft"
    plan_id = created["id"]

    ws2 = mcp_server.organization_workspace(db, asset.id)
    assert ws2["active_draft"]["id"] == plan_id
    assert ws2["validation_summary"]["errors"] == []

    got = mcp_server.get_organization_plan(db, plan_id)
    assert len(got["groups"]) == 1
    assert len(got["items"]) == 4
    assert [p["id"] for p in mcp_server.list_organization_plans(db, asset.id)] == [plan_id]


def test_mcp_draft_calls_do_not_change_shots(tmp_path):
    db, asset, shots, alice, bob = _planning_store(tmp_path)
    groups, items = _complete_plan(shots, alice)

    plan = mcp_server.create_organization_plan(db, asset.id, groups, items)
    report = mcp_server.validate_organization_plan(db, plan["id"])

    assert report["errors"] == []
    # Drafting and validating never touch the live shot assignments.
    shot_list = mcp_server.list_shots(db, asset.id)
    assert all(s["group_id"] is None and not s["excused"] for s in shot_list)


def test_validate_reports_coverage_error_without_change(tmp_path):
    db, asset, shots, alice, bob = _planning_store(tmp_path)
    plan = mcp_server.create_organization_plan(
        db, asset.id, [],
        [{"shot_id": shots[0].id, "destination_type": "broll"}],
    )
    report = mcp_server.validate_organization_plan(db, plan["id"])
    assert any("missing from the plan" in e for e in report["errors"])
    assert all(s["group_id"] is None for s in mcp_server.list_shots(db, asset.id))


def test_update_item_and_group(tmp_path):
    db, asset, shots, alice, bob = _planning_store(tmp_path)
    plan = mcp_server.create_organization_plan(
        db, asset.id, [],
        [{"shot_id": shots[0].id, "destination_type": "broll", "decision": "proposed"}],
    )
    updated = mcp_server.update_organization_plan_item(
        db, plan["id"], shots[0].id,
        decision="accepted", destination_type="intentional_exception",
        human_override_reason="not an interview shot",
    )
    assert updated["decision"] == "accepted"
    assert updated["destination_type"] == "intentional_exception"

    group = mcp_server.create_organization_plan(
        db, asset.id,
        [{"draft_key": "g1", "name": "v1", "kind": "setup"}],
        [],
    )
    renamed = mcp_server.update_organization_plan_group(db, group["id"], "g1", name="renamed")
    assert renamed["name"] == "renamed"


def test_approve_and_apply_via_mcp(tmp_path):
    db, asset, shots, alice, bob = _planning_store(tmp_path)
    groups, items = _complete_plan(shots, alice)
    plan = mcp_server.create_organization_plan(db, asset.id, groups, items)

    assert mcp_server.approve_organization_plan(db, plan["id"])["state"] == "approved"
    result = mcp_server.apply_organization_plan(db, plan["id"])
    assert result["state"] == "applied"

    shot_list = {s["id"]: s for s in mcp_server.list_shots(db, asset.id)}
    assert shot_list[shots[0].id]["group_id"] is not None
    assert shot_list[shots[1].id]["group_id"] is not None
    assert shot_list[shots[3].id]["group_id"] is not None  # b-roll group

    groups_now = mcp_server.list_shot_groups(db, asset.id)
    kinds = {g["kind"] for g in groups_now}
    assert "setup" in kinds and "broll" in kinds


def test_get_shot_contact_sheet(tmp_path):
    db, asset, shots, alice, bob = _planning_store(tmp_path, with_stills=True)
    image = mcp_server.get_shot_contact_sheet(db, [shots[0].id, shots[1].id], columns=2)

    data = getattr(image, "data", None)
    assert isinstance(data, bytes)
    assert data[:4] == b"\x89PNG"
