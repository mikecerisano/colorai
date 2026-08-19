"""Tests for group-aware subject/setup matching."""

from __future__ import annotations

from colorai.editorial import assign_shot_group, create_group
from colorai.matching import match_subject_in_group
from colorai.project import (
    Correction,
    FrameMetrics,
    ProjectStore,
    Shot,
    SkinMetric,
    make_shots,
)
from colorai.references import approve_reference, propose_reference
from colorai.skin_analysis import create_subject, set_reference


def _metric(shot_id, luma, r, g, b):
    return FrameMetrics(
        shot_id=shot_id,
        frame_index=shot_id * 25,  # whatever, per-shot unique is fine
        luma_mean=luma,
        luma_std=0.1,
        r_mean=r,
        g_mean=g,
        b_mean=b,
        saturation_mean=0.1,
    )


def _setup(group=False):
    store = ProjectStore.create(":memory:")
    project = store.create_project("matching")
    asset = store.add_asset(project.id, source_path="/media/m.mov", frame_rate=25.0)
    shots = make_shots(asset, [(0, 24), (25, 49), (50, 74)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)
    alice = create_subject(store, asset.id, "Alice")
    bob = create_subject(store, asset.id, "Bob")

    # Alice: reference-like skin on shot 0, deviating skin on shots 1 and 2.
    # Bob appears in shot 2 with his own skin (must not affect Alice).
    with store.session() as session:
        for face_index, (shot, b, g, r, subj) in enumerate([
            (shots[0], 0.35, 0.38, 0.58, alice.id),
            (shots[1], 0.35, 0.38, 0.29, alice.id),  # r half -> rgb_balance fix
            (shots[2], 0.35, 0.38, 0.20, alice.id),  # r even lower
            (shots[2], 0.20, 0.25, 0.40, bob.id),    # Bob, different person
        ]):
            session.add(
                SkinMetric(
                    shot_id=shot.id, face_index=face_index,
                    mean_b=b, mean_g=g, mean_r=r,
                    sample_pixels=100, subject_id=subj,
                )
            )
        session.add(_metric(shots[0].id, 0.5, 0.5, 0.5, 0.5))
        session.add(_metric(shots[1].id, 0.25, 0.25, 0.25, 0.25))  # 1 stop darker
        session.add(_metric(shots[2].id, 0.4, 0.4, 0.4, 0.4))      # 0.3 stops darker
        session.commit()

    if group:
        g = create_group(store, asset.id, "interview A", kind="setup", camera="A")
        assign_shot_group(store, shots[1].id, g.id)
        assign_shot_group(store, shots[2].id, g.id)
        return store, asset, shots, alice, bob, g
    return store, asset, shots, alice, bob


def test_no_approved_reference_blocks_matching():
    store, asset, shots, alice, bob = _setup()
    proposals, error = match_subject_in_group(store, asset.id, subject_id=alice.id)
    assert proposals == []
    assert "approved reference" in error


def test_matching_after_human_sets_hero_shot():
    store, asset, shots, alice, bob = _setup()
    set_reference(store, alice.id, shots[0].id)

    proposals, error = match_subject_in_group(store, asset.id, subject_id=alice.id)
    assert error is None
    by_shot = {p.shot_id: p for p in proposals}
    assert set(by_shot) == {shots[1].id, shots[2].id}
    for p in proposals:
        assert p.reference_shot_id == shots[0].id
        assert p.subject_id == alice.id
        kinds = {c.kind for c in p.corrections}
        assert "rgb_balance" in kinds  # skin fix within the subject
        assert "exposure" in kinds    # luma fix vs the reference


def test_group_scope_restricts_members():
    store, asset, shots, alice, bob, g = _setup(group=True)
    set_reference(store, alice.id, shots[0].id)

    # Alice's face in shot 0 is OUTSIDE the group; with no approved group
    # reference the scope cannot be matched.
    proposals, error = match_subject_in_group(store, asset.id, subject_id=alice.id, group_id=g.id)
    assert proposals == []
    assert "not a member" in error

    # Approve a reference inside the group -> matching works for members only.
    p = propose_reference(
        store, asset_id=asset.id, shot_id=shots[1].id, reason="good framing",
        confidence=0.8, subject_id=alice.id, group_id=g.id,
    )
    approve_reference(store, p.id)
    proposals, error = match_subject_in_group(store, asset.id, subject_id=alice.id, group_id=g.id)
    assert error is None
    assert [x.shot_id for x in proposals] == [shots[2].id]  # only the other member
    assert proposals[0].reference_shot_id == shots[1].id
    assert proposals[0].group_id == g.id


def test_skin_never_compared_across_subjects():
    store, asset, shots, alice, bob = _setup()
    set_reference(store, alice.id, shots[0].id)
    proposals, _ = match_subject_in_group(store, asset.id, subject_id=alice.id)
    # Bob's very different skin in shot 2 must not influence Alice's shot-2
    # proposal: the skin correction comes from Alice's own faces only.
    for p in proposals:
        for c in p.corrections:
            assert c.kind != "rgb_balance" or "skin" in p.reasons


def test_persist_writes_disabled_corrections():
    store, asset, shots, alice, bob = _setup()
    set_reference(store, alice.id, shots[0].id)

    proposals, _ = match_subject_in_group(store, asset.id, subject_id=alice.id, persist=True)
    assert proposals
    with store.session() as session:
        rows = (
            session.query(Correction)
            .filter(Correction.shot_id.in_([p.shot_id for p in proposals]))
            .all()
        )
    assert len(rows) == sum(len(p.corrections) for p in proposals)
    assert all(not c.enabled for c in rows)  # never auto-applied
