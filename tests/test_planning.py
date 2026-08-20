"""Tests for the durable organization-plan domain."""

from __future__ import annotations

import pytest

from colorai.editorial import GROUP_KIND_BROLL, GROUP_KIND_SETUP, create_group
from colorai.planning import (
    DEST_BROLL,
    DEST_EXISTING_GROUP,
    DEST_INTENTIONAL_EXCEPTION,
    DEST_PLANNED_GROUP,
    DEST_UNRESOLVED,
    STATE_APPLIED,
    STATE_APPROVED,
    STATE_SUPERSEDED,
    apply_organization_plan,
    approve_organization_plan,
    create_organization_plan,
    get_organization_plan,
    list_organization_plans,
    update_organization_plan_item,
    validate_organization_plan,
)
from colorai.project import ProjectStore, Shot, SkinMetric, ShotGroup, make_shots
from colorai.skin_analysis import create_subject


def _store_with_asset(tmp_path):
    store = ProjectStore.create(":memory:")
    project = store.create_project("plan test")
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
                    mean_b=0.30, mean_g=0.30, mean_r=0.50,
                    sample_pixels=100, subject_id=subject_id,
                    bbox_x=0, bbox_y=0, bbox_w=10, bbox_h=10,
                )
            )
            session.commit()

    add_face(shots[0], 0, alice.id)
    add_face(shots[1], 0, alice.id)
    add_face(shots[2], 0, alice.id)
    add_face(shots[2], 1, bob.id)
    # shots[3] has no face (b-roll material).

    return store, asset, shots, alice, bob


def _setup_item(shot_id, draft_key=None, group_id=None):
    if draft_key:
        return {
            "shot_id": shot_id,
            "destination_type": DEST_PLANNED_GROUP,
            "target_draft_key": draft_key,
            "decision": "accepted",
        }
    if group_id:
        return {
            "shot_id": shot_id,
            "destination_type": DEST_EXISTING_GROUP,
            "target_group_id": group_id,
            "decision": "accepted",
        }
    raise AssertionError("provide draft_key or group_id")


def test_create_draft_and_supersede(tmp_path):
    store, asset, shots, alice, bob = _store_with_asset(tmp_path)
    g = [{"draft_key": "s1", "name": "Alice interview", "kind": "setup", "participant_ids": [alice.id]}]
    items = [
        _setup_item(shots[0].id, draft_key="s1"),
        _setup_item(shots[1].id, draft_key="s1"),
        {"shot_id": shots[2].id, "destination_type": DEST_UNRESOLVED, "decision": "rejected"},
        {"shot_id": shots[3].id, "destination_type": DEST_BROLL, "decision": "accepted"},
    ]
    p1 = create_organization_plan(store, asset.id, g, items, summary="first")
    assert p1.state == "draft"

    p2 = create_organization_plan(store, asset.id, g, items, summary="second")
    assert p2.state == "draft"
    assert get_organization_plan(store, p1.id)["state"] == STATE_SUPERSEDED


def test_duplicate_shot_destination_rejected(tmp_path):
    store, asset, shots, alice, bob = _store_with_asset(tmp_path)
    items = [
        {"shot_id": shots[0].id, "destination_type": DEST_BROLL},
        {"shot_id": shots[0].id, "destination_type": DEST_UNRESOLVED},
    ]
    with pytest.raises(ValueError, match="more than one destination"):
        create_organization_plan(store, asset.id, [], items)


def test_variant_requires_setup_parent(tmp_path):
    store, asset, shots, alice, bob = _store_with_asset(tmp_path)
    groups = [{"draft_key": "v1", "name": "golden hour", "kind": "variant"}]
    with pytest.raises(ValueError, match="no setup parent"):
        create_organization_plan(store, asset.id, groups, [])


def test_unknown_shot_rejected(tmp_path):
    store, asset, shots, alice, bob = _store_with_asset(tmp_path)
    items = [{"shot_id": 99999, "destination_type": DEST_BROLL}]
    with pytest.raises(ValueError, match="unknown shot"):
        create_organization_plan(store, asset.id, [], items)


def test_validate_reports_incomplete_coverage(tmp_path):
    store, asset, shots, alice, bob = _store_with_asset(tmp_path)
    plan = create_organization_plan(
        store, asset.id, [],
        [{"shot_id": shots[0].id, "destination_type": DEST_BROLL}],
    )
    report = validate_organization_plan(store, plan.id)
    assert any("missing from the plan" in e for e in report["errors"])


def test_apply_requires_approved_and_rolls_back_on_error(tmp_path):
    store, asset, shots, alice, bob = _store_with_asset(tmp_path)
    # Incomplete coverage: only one of four shots is planned.
    plan = create_organization_plan(
        store, asset.id, [],
        [{"shot_id": shots[0].id, "destination_type": DEST_BROLL, "decision": "accepted"}],
    )
    approve_organization_plan(store, plan.id)

    result = apply_organization_plan(store, plan.id)
    assert result["error"] == "validation failed"
    assert any("missing from the plan" in e for e in result["errors"])

    # Nothing changed: plan still approved, shot 0 still ungrouped/non-excused.
    assert get_organization_plan(store, plan.id)["state"] == STATE_APPROVED
    with store.session() as session:
        shot = session.get(Shot, shots[0].id)
        assert shot.group_id is None
        assert shot.excused is False


def test_apply_creates_setup_variant_broll_and_exception(tmp_path):
    store, asset, shots, alice, bob = _store_with_asset(tmp_path)
    groups = [
        {"draft_key": "s1", "name": "Alice interview", "kind": "setup", "participant_ids": [alice.id]},
        {"draft_key": "v1", "name": "golden hour", "kind": "variant", "parent_draft_key": "s1", "participant_ids": [alice.id]},
    ]
    items = [
        _setup_item(shots[0].id, draft_key="s1"),
        _setup_item(shots[1].id, draft_key="v1"),
        {"shot_id": shots[2].id, "destination_type": DEST_UNRESOLVED, "decision": "rejected"},
        {"shot_id": shots[3].id, "destination_type": DEST_BROLL, "decision": "accepted"},
    ]
    plan = create_organization_plan(store, asset.id, groups, items, summary="organize")
    approve_organization_plan(store, plan.id)
    result = apply_organization_plan(store, plan.id)

    assert "error" not in result
    assert result["state"] == STATE_APPLIED

    with store.session() as session:
        s0 = session.get(Shot, shots[0].id)
        s1 = session.get(Shot, shots[1].id)
        s3 = session.get(Shot, shots[3].id)
        assert s0.excused is False and s0.group_id is not None
        assert s1.group_id is not None and s1.group_id != s0.group_id
        setup = session.query(ShotGroup).filter_by(name="Alice interview").one()
        variant = session.query(ShotGroup).filter_by(name="golden hour").one()
        assert variant.parent_id == setup.id
        broll = session.get(ShotGroup, s3.group_id)
        assert broll.kind == GROUP_KIND_BROLL

    # Idempotent: applying an applied plan is refused.
    again = apply_organization_plan(store, plan.id)
    assert again.get("error")


def test_intentional_exception_and_existing_group(tmp_path):
    store, asset, shots, alice, bob = _store_with_asset(tmp_path)
    existing = create_group(store, asset.id, "cam A", kind=GROUP_KIND_SETUP)

    items = [
        _setup_item(shots[0].id, group_id=existing.id),
        {"shot_id": shots[1].id, "destination_type": DEST_INTENTIONAL_EXCEPTION, "decision": "accepted"},
        {"shot_id": shots[2].id, "destination_type": DEST_UNRESOLVED, "decision": "rejected"},
        {"shot_id": shots[3].id, "destination_type": DEST_UNRESOLVED, "decision": "rejected"},
    ]
    plan = create_organization_plan(store, asset.id, [], items)
    approve_organization_plan(store, plan.id)

    result = apply_organization_plan(store, plan.id)
    assert result["state"] == STATE_APPLIED
    with store.session() as session:
        s0 = session.get(Shot, shots[0].id)
        s1 = session.get(Shot, shots[1].id)
        assert s0.group_id == existing.id and s0.excused is False
        assert s1.group_id is None and s1.excused is True


def test_reference_scope_violation_is_blocking(tmp_path):
    from colorai.references import approve_reference, propose_reference

    store, asset, shots, alice, bob = _store_with_asset(tmp_path)
    existing = create_group(store, asset.id, "cam A", kind=GROUP_KIND_SETUP)
    with store.session() as session:
        shot = session.get(Shot, shots[0].id)
        shot.group_id = existing.id
        session.commit()

    ref = propose_reference(
        store, asset_id=asset.id, shot_id=shots[0].id, reason="hero",
        confidence=0.9, group_id=existing.id,
    )
    approve_reference(store, ref.id)

    plan = create_organization_plan(
        store, asset.id, [],
        [
            {"shot_id": shots[0].id, "destination_type": DEST_BROLL, "decision": "accepted"},
            {"shot_id": shots[1].id, "destination_type": DEST_UNRESOLVED, "decision": "rejected"},
            {"shot_id": shots[2].id, "destination_type": DEST_UNRESOLVED, "decision": "rejected"},
            {"shot_id": shots[3].id, "destination_type": DEST_UNRESOLVED, "decision": "rejected"},
        ],
    )

    report = validate_organization_plan(store, plan.id)
    assert any("outside its group scope" in e for e in report["errors"])


def test_list_plans_and_update_item(tmp_path):
    store, asset, shots, alice, bob = _store_with_asset(tmp_path)
    plan = create_organization_plan(
        store, asset.id, [],
        [{"shot_id": shots[0].id, "destination_type": DEST_BROLL}],
    )
    assert [p["id"] for p in list_organization_plans(store, asset.id)] == [plan.id]

    updated = update_organization_plan_item(
        store, plan.id, shots[0].id,
        decision="accepted", destination_type=DEST_INTENTIONAL_EXCEPTION,
        human_override_reason="not part of the interview",
    )
    assert updated["decision"] == "accepted"
    assert updated["destination_type"] == DEST_INTENTIONAL_EXCEPTION
    assert updated["human_override_reason"] == "not part of the interview"
