"""Tests for editorial operations: review state, exceptions, grouping, split/merge."""

from __future__ import annotations

import pytest

from colorai.analysis import load_shot_features
from colorai.editorial import (
    assign_shot_group,
    create_group,
    delete_group,
    list_groups,
    merge_shots,
    rename_group,
    set_excused,
    set_review_status,
    split_shot,
    unassign_shot_group,
)
from colorai.project import (
    Correction,
    MediaAsset,
    ProjectStore,
    Shot,
    make_shots,
)


def _asset_with_shots(store, bounds=((0, 24), (25, 49), (50, 74))):
    project = store.create_project("editorial")
    asset = store.add_asset(
        project.id, source_path="/media/m.mov", frame_rate=25.0
    )
    shots = make_shots(asset, bounds)
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)
    return asset, shots


def _shot_list(store, asset_id):
    with store.session() as session:
        return (
            session.query(Shot)
            .filter_by(asset_id=asset_id)
            .order_by(Shot.index)
            .all()
        )


def test_set_review_status_and_excused():
    store = ProjectStore.create(":memory:")
    _, shots = _asset_with_shots(store)

    updated = set_review_status(store, shots[0].id, "approved")
    assert updated.review_status == "approved"
    with pytest.raises(ValueError):
        set_review_status(store, shots[0].id, "bogus")

    excused = set_excused(store, shots[0].id, True)
    assert excused.excused is True


def test_group_crud_and_assignment():
    store = ProjectStore.create(":memory:")
    asset, shots = _asset_with_shots(store)

    group = create_group(store, asset.id, "interview cam A")
    assert [g.id for g in list_groups(store, asset.id)] == [group.id]

    renamed = rename_group(store, group.id, "cam A")
    assert renamed.name == "cam A"

    assigned = assign_shot_group(store, shots[0].id, group.id)
    assert assigned.group_id == group.id
    unassign_shot_group(store, shots[0].id)
    assert unassign_shot_group(store, shots[0].id).group_id is None

    delete_group(store, group.id)
    assert list_groups(store, asset.id) == []


def test_split_shot_copies_corrections_and_renumbers():
    store = ProjectStore.create(":memory:")
    asset, shots = _asset_with_shots(store)
    with store.session() as session:
        session.add(
            Correction(shot_id=shots[0].id, kind="exposure", parameters={"gain": 2.0})
        )
        session.commit()

    a, b = split_shot(store, shots[0].id, at_frame=10)

    remaining = _shot_list(store, asset.id)
    assert [(s.index, s.start_frame, s.end_frame) for s in remaining] == [
        (0, 0, 9),
        (1, 10, 24),
        (2, 25, 49),
        (3, 50, 74),
    ]
    assert a.start_timecode == "00:00:00:00" and a.end_timecode == "00:00:00:09"
    assert b.start_timecode == "00:00:00:10" and b.end_timecode == "00:00:00:24"

    with store.session() as session:
        for sid in (a.id, b.id):
            cs = session.query(Correction).filter_by(shot_id=sid).all()
            assert len(cs) == 1 and cs[0].kind == "exposure"
        asset_row = session.get(MediaAsset, asset.id)
        assert asset_row.status == "registered"


def test_split_shot_rejects_boundary_outside():
    store = ProjectStore.create(":memory:")
    _, shots = _asset_with_shots(store)
    with pytest.raises(ValueError):
        split_shot(store, shots[0].id, at_frame=0)  # not strictly inside
    with pytest.raises(ValueError):
        split_shot(store, shots[0].id, at_frame=25)  # == end+1


def test_merge_shots_appends_corrections_and_renumbers():
    store = ProjectStore.create(":memory:")
    asset, shots = _asset_with_shots(store)
    with store.session() as session:
        session.add(
            Correction(shot_id=shots[0].id, kind="exposure", parameters={"gain": 2.0})
        )
        session.add(
            Correction(shot_id=shots[1].id, kind="offset", parameters={"value": 0.1})
        )
        session.commit()

    merged = merge_shots(store, shots[0].id, shots[1].id)

    remaining = _shot_list(store, asset.id)
    assert [(s.index, s.start_frame, s.end_frame) for s in remaining] == [
        (0, 0, 49),
        (1, 50, 74),
    ]
    assert merged.end_timecode == "00:00:01:24"

    with store.session() as session:
        cs = session.query(Correction).filter_by(shot_id=merged.id).order_by(Correction.id).all()
        assert [c.kind for c in cs] == ["exposure", "offset"]


def test_merge_shots_rejects_non_adjacent():
    store = ProjectStore.create(":memory:")
    _, shots = _asset_with_shots(store)
    with pytest.raises(ValueError):
        merge_shots(store, shots[0].id, shots[2].id)


def test_load_shot_features_skips_excused():
    from colorai.project import FrameMetrics

    store = ProjectStore.create(":memory:")
    _, shots = _asset_with_shots(store, bounds=((0, 24), (25, 49)))
    with store.session() as session:
        for shot, luma in zip(shots, (0.2, 0.8)):
            session.add(
                FrameMetrics(
                    shot_id=shot.id,
                    frame_index=shot.start_frame,
                    luma_mean=luma,
                    r_mean=luma,
                    g_mean=luma,
                    b_mean=luma,
                    saturation_mean=0.0,
                )
            )
        session.commit()

    set_excused(store, shots[0].id, True)
    features = load_shot_features(store, shots[0].asset_id)
    assert [f.shot_id for f in features] == [shots[1].id]
