"""Tests for lighting variants within interview/setup families."""

from __future__ import annotations

import pytest

from colorai.editorial import assign_shot_group, create_group
from colorai.matching import cross_variant_skin_consistency, match_subject_in_group
from colorai.project import ProjectStore, SkinMetric, make_shots
from colorai.references import (
    approve_reference,
    effective_reference_shot_id,
    propose_reference,
)
from colorai.skin_analysis import create_subject, set_reference


def _variant_fixture(*, drift_afternoon: bool = False):
    store = ProjectStore.create(":memory:")
    project = store.create_project("variants")
    asset = store.add_asset(project.id, source_path="/media/m.mov", frame_rate=25.0)
    shots = make_shots(asset, [(0, 24), (25, 49), (50, 74), (75, 99)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)

    alice = create_subject(store, asset.id, "Alice")
    family = create_group(store, asset.id, "interview", kind="setup")
    morning = create_group(store, asset.id, "morning", kind="variant", parent_id=family.id)
    afternoon = create_group(store, asset.id, "afternoon", kind="variant", parent_id=family.id)
    assign_shot_group(store, shots[0].id, morning.id)
    assign_shot_group(store, shots[1].id, morning.id)
    assign_shot_group(store, shots[2].id, afternoon.id)
    assign_shot_group(store, shots[3].id, afternoon.id)

    afternoon_r = 0.30 if drift_afternoon else 0.58
    with store.session() as session:
        for i, shot in enumerate(shots):
            session.add(
                SkinMetric(
                    shot_id=shot.id, face_index=0,
                    mean_b=0.35, mean_g=0.38, mean_r=(0.58 if i < 2 else afternoon_r),
                    sample_pixels=100, subject_id=alice.id,
                )
            )
        # Within-variant exposure difference: afternoon shot 3 is darker, so
        # variant matching has a whole-frame signal to propose (skin is equal).
        from colorai.project import FrameMetrics

        for shot, luma in ((shots[0], 0.5), (shots[1], 0.5), (shots[2], 0.5), (shots[3], 0.25)):
            session.add(
                FrameMetrics(
                    shot_id=shot.id, frame_index=shot.start_frame,
                    luma_mean=luma, luma_std=0.1, r_mean=luma, g_mean=luma, b_mean=luma,
                    saturation_mean=0.1,
                )
            )
        session.commit()

    return store, asset, shots, alice, family, morning, afternoon


def test_variant_group_requires_parent():
    store = ProjectStore.create(":memory:")
    project = store.create_project("v")
    asset = store.add_asset(project.id, source_path="/media/m.mov", frame_rate=25.0)
    family = create_group(store, asset.id, "interview", kind="setup")

    with pytest.raises(ValueError, match="parent"):
        create_group(store, asset.id, "morning", kind="variant")
    with pytest.raises(ValueError, match="setup family"):
        create_group(store, asset.id, "morning", kind="variant", parent_id=9999)

    # A non-variant group must not have a parent.
    with pytest.raises(ValueError, match="only 'variant'"):
        create_group(store, asset.id, "bogus", kind="generic", parent_id=family.id)

    variant = create_group(store, asset.id, "morning", kind="variant", parent_id=family.id)
    assert variant.parent_id == family.id


def test_variant_reference_does_not_clobber_hero_shot():
    store, asset, shots, alice, family, morning, afternoon = _variant_fixture()
    set_reference(store, alice.id, shots[0].id)  # subject hero

    p = propose_reference(
        store, asset_id=asset.id, shot_id=shots[2].id, reason="afternoon hero",
        confidence=0.8, subject_id=alice.id, group_id=afternoon.id,
    )
    approve_reference(store, p.id)

    # The subject's asset-wide hero shot is unchanged by a variant approval.
    assert effective_reference_shot_id(store, subject_id=alice.id) == shots[0].id


def test_variant_matching_requires_own_reference():
    store, asset, shots, alice, family, morning, afternoon = _variant_fixture()
    # A subject hero shot exists (in the morning variant), but the afternoon
    # variant has no approved reference of its own -> refuse, no fallback.
    set_reference(store, alice.id, shots[0].id)

    proposals, error = match_subject_in_group(
        store, asset.id, subject_id=alice.id, group_id=afternoon.id
    )
    assert proposals == []
    assert "lighting variant" in error

    # Approve a reference inside the afternoon variant -> matching works.
    p = propose_reference(
        store, asset_id=asset.id, shot_id=shots[2].id, reason="afternoon hero",
        confidence=0.8, subject_id=alice.id, group_id=afternoon.id,
    )
    approve_reference(store, p.id)
    proposals, error = match_subject_in_group(
        store, asset.id, subject_id=alice.id, group_id=afternoon.id
    )
    assert error is None
    assert [x.shot_id for x in proposals] == [shots[3].id]
    assert proposals[0].reference_shot_id == shots[2].id


def test_cross_variant_skin_consistency_consistent():
    store, asset, shots, alice, family, morning, afternoon = _variant_fixture()
    set_reference(store, alice.id, shots[0].id)  # baseline = morning skin

    deviations, error = cross_variant_skin_consistency(
        store, asset.id, subject_id=alice.id, family_group_id=family.id
    )
    assert error is None
    # Afternoon skin matches morning -> no issue.
    afternoon_dev = next(d for d in deviations if d.variant_id == afternoon.id)
    assert afternoon_dev.is_issue is False
    assert afternoon_dev.correction is None


def test_cross_variant_skin_consistency_flags_drift():
    store, asset, shots, alice, family, morning, afternoon = _variant_fixture(drift_afternoon=True)
    set_reference(store, alice.id, shots[0].id)

    deviations, error = cross_variant_skin_consistency(
        store, asset.id, subject_id=alice.id, family_group_id=family.id
    )
    assert error is None
    afternoon_dev = next(d for d in deviations if d.variant_id == afternoon.id)
    assert afternoon_dev.is_issue is True
    # Skin-only correction: rgb_balance, never a whole-frame exposure fix.
    assert afternoon_dev.correction.kind == "rgb_balance"


def test_cross_variant_requires_setup_family():
    store, asset, shots, alice, family, morning, afternoon = _variant_fixture()
    deviations, error = cross_variant_skin_consistency(
        store, asset.id, subject_id=alice.id, family_group_id=morning.id  # a variant, not a family
    )
    assert deviations == []
    assert "not a setup family" in error
