"""Tests for deterministic organization suggestions (needs-organization)."""

from __future__ import annotations

from colorai.organization import (
    OrganizationResult,
    ShotEvidence,
    suggest_organization,
)


def _ev(shot_id: int, index: int, *subject_ids: int | None) -> ShotEvidence:
    return ShotEvidence(
        shot_id=shot_id, index=index, face_subject_ids=tuple(subject_ids)
    )


def test_empty_queue_is_all_empty():
    res = suggest_organization([], {})
    assert res == OrganizationResult((), (), ())


def test_no_face_rows_is_broll():
    res = suggest_organization([_ev(1, 0)], {})
    assert res.broll_shot_ids == (1,)
    assert res.interview_clusters == ()
    assert res.judgment == ()


def test_single_assigned_subject_is_interview_candidate():
    res = suggest_organization([_ev(1, 0, 10)], {10: "Alice"})
    assert res.broll_shot_ids == ()
    assert res.judgment == ()
    assert len(res.interview_clusters) == 1
    cluster = res.interview_clusters[0]
    assert cluster.subject_id == 10
    assert cluster.subject_name == "Alice"
    assert cluster.member_shot_ids == (1,)


def test_adjacent_same_subject_forms_one_cluster():
    shots = [_ev(1, 0, 10), _ev(2, 1, 10), _ev(3, 2, 10)]
    res = suggest_organization(shots, {10: "Alice"})
    assert len(res.interview_clusters) == 1
    assert res.interview_clusters[0].member_shot_ids == (1, 2, 3)


def test_index_gap_splits_clusters():
    # Default adjacency_gap is 3: indices 0 and 5 are non-adjacent.
    shots = [_ev(1, 0, 10), _ev(2, 5, 10)]
    res = suggest_organization(shots, {10: "Alice"})
    assert len(res.interview_clusters) == 2
    assert res.interview_clusters[0].member_shot_ids == (1,)
    assert res.interview_clusters[1].member_shot_ids == (2,)


def test_different_subjects_do_not_merge():
    shots = [_ev(1, 0, 10), _ev(2, 1, 11)]
    res = suggest_organization(shots, {10: "Alice", 11: "Bob"})
    assert len(res.interview_clusters) == 2


def test_multi_person_shot_is_judgment():
    res = suggest_organization([_ev(1, 0, 10, 11)], {10: "Alice", 11: "Bob"})
    assert res.interview_clusters == ()
    assert res.broll_shot_ids == ()
    assert len(res.judgment) == 1
    item = res.judgment[0]
    assert item.shot_id == 1
    assert item.subject_ids == (10, 11)
    assert "multi-person" in item.reasons[0]


def test_face_without_assigned_subject_is_judgment():
    res = suggest_organization([_ev(1, 0, None)], {})
    assert len(res.judgment) == 1
    assert res.judgment[0].shot_id == 1
    assert "no subject assigned" in res.judgment[0].reasons[0]


def test_representative_is_shot_with_most_faces():
    # Shot 1 has two faces for the same subject, shot 2 has one: shot 1 wins.
    shots = [_ev(1, 0, 10, 10), _ev(2, 1, 10)]
    res = suggest_organization(shots, {10: "Alice"})
    assert res.interview_clusters[0].representative_shot_id == 1


def test_representative_tie_goes_to_earlier_shot():
    shots = [_ev(1, 0, 10), _ev(2, 1, 10)]
    res = suggest_organization(shots, {10: "Alice"})
    assert res.interview_clusters[0].representative_shot_id == 1
