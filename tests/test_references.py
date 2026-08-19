"""Tests for human-approved reference-shot proposals."""

from __future__ import annotations

import pytest

from colorai.project import ProjectStore, ReferenceProposal, make_shots
from colorai.references import (
    approve_reference,
    effective_reference_shot_id,
    list_reference_proposals,
    propose_reference,
    reject_reference,
)
from colorai.skin_analysis import create_subject


def _setup():
    store = ProjectStore.create(":memory:")
    project = store.create_project("refs")
    asset = store.add_asset(project.id, source_path="/media/m.mov", frame_rate=25.0)
    shots = make_shots(asset, [(0, 24), (25, 49), (50, 74)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)
    subject = create_subject(store, asset.id, "Alice")
    return store, asset, shots, subject


def test_propose_reference_is_suggested():
    store, asset, shots, subject = _setup()
    p = propose_reference(
        store,
        asset_id=asset.id,
        shot_id=shots[1].id,
        reason="stable framing, skin well lit",
        confidence=0.9,
        subject_id=subject.id,
    )
    assert p.state == "suggested"
    assert p.author == "agent"
    assert p.shot_id == shots[1].id
    assert [x.id for x in list_reference_proposals(store, asset.id)] == [p.id]


def test_propose_reference_validation():
    store, asset, shots, subject = _setup()
    with pytest.raises(ValueError, match="confidence"):
        propose_reference(store, asset_id=asset.id, shot_id=shots[0].id, reason="x", confidence=1.5)
    with pytest.raises(ValueError, match="reason"):
        propose_reference(store, asset_id=asset.id, shot_id=shots[0].id, reason="  ", confidence=0.5)
    with pytest.raises(ValueError, match="does not belong"):
        propose_reference(
            store, asset_id=asset.id, shot_id=shots[0].id, reason="x",
            confidence=0.5, subject_id=9999,
        )


def test_approve_reference_sets_state_and_hero_shot():
    store, asset, shots, subject = _setup()
    p = propose_reference(
        store, asset_id=asset.id, shot_id=shots[2].id, reason="hero", confidence=0.8,
        subject_id=subject.id,
    )
    approved = approve_reference(store, p.id)
    assert approved.state == "approved"
    with store.session() as session:
        subject_row = session.get(type(subject), subject.id)
        assert subject_row.reference_shot_id == shots[2].id


def test_reject_reference():
    store, asset, shots, subject = _setup()
    p = propose_reference(store, asset_id=asset.id, shot_id=shots[0].id, reason="try", confidence=0.4)
    assert reject_reference(store, p.id).state == "rejected"
    assert effective_reference_shot_id(store, asset_id=asset.id) is None


def test_effective_reference_requires_approval():
    store, asset, shots, subject = _setup()
    propose_reference(
        store, asset_id=asset.id, shot_id=shots[1].id, reason="maybe", confidence=0.7,
        subject_id=subject.id,
    )
    # Suggested (unapproved) proposals are not an effective reference.
    assert effective_reference_shot_id(store, asset_id=asset.id, subject_id=subject.id) is None


def test_effective_reference_falls_back_to_subject_hero():
    from colorai.skin_analysis import set_reference

    store, asset, shots, subject = _setup()
    set_reference(store, subject.id, shots[2].id)  # human selects the hero shot
    assert effective_reference_shot_id(store, subject_id=subject.id) == shots[2].id


def test_approved_proposal_wins_for_group_scope():
    from colorai.editorial import create_group

    store, asset, shots, subject = _setup()
    group = create_group(store, asset.id, "interview A", kind="setup", camera="A")
    propose_reference(
        store, asset_id=asset.id, shot_id=shots[0].id, reason="scope ref", confidence=0.6,
        subject_id=subject.id, group_id=group.id,
    )
    assert effective_reference_shot_id(store, subject_id=subject.id, group_id=group.id) is None
    # Approve the group-scoped proposal.
    p = list_reference_proposals(store, asset.id)[0]
    approve_reference(store, p.id)
    assert effective_reference_shot_id(store, subject_id=subject.id, group_id=group.id) == shots[0].id


def test_state_values_constrained():
    store, asset, shots, subject = _setup()
    p = propose_reference(store, asset_id=asset.id, shot_id=shots[0].id, reason="x", confidence=0.5)
    with store.session() as session:
        row = session.get(ReferenceProposal, p.id)
        assert row.state in ("suggested", "approved", "rejected")
