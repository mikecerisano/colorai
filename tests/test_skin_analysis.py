"""Tests for per-subject skin-tone consistency."""

from __future__ import annotations

import numpy as np
import pytest

from colorai.project import ProjectStore, SkinMetric, make_shots
from colorai.skin_analysis import (
    FaceSkin,
    cluster_by_skin,
    propose_skin_match,
    skin_consistency,
    skin_features,
)


def _face(shot_id, b, g, r, face_index=0):
    return FaceSkin(shot_id, face_index, b, g, r)


def test_cluster_by_skin_separates_subjects():
    a1 = _face(1, 0.35, 0.38, 0.58)
    a2 = _face(2, 0.36, 0.37, 0.59)
    b1 = _face(3, 0.55, 0.56, 0.78)

    clusters = cluster_by_skin([a1, a2, b1])
    assert len(clusters) == 2
    # subject A has two faces, subject B has one.
    assert sorted(len(c) for c in clusters) == [1, 2]


def test_propose_skin_match_off_and_on_target():
    ref = np.array([0.36, 0.38, 0.59])
    close = _face(1, 0.36, 0.38, 0.59)
    assert propose_skin_match(ref, close) is None

    drifted = _face(2, 0.40, 0.43, 0.63)
    correction = propose_skin_match(ref, drifted)
    assert correction is not None
    assert correction.kind == "rgb_balance"
    assert correction.parameters["gain"] == pytest.approx(
        [0.36 / 0.40, 0.38 / 0.43, 0.59 / 0.63], abs=1e-3
    )


def _build_skin_asset(store):
    project = store.create_project("skin test")
    asset = store.add_asset(project.id, source_path="/media/m.mov", frame_rate=25.0)
    shots = make_shots(asset, [(0, 24), (25, 49), (50, 74), (75, 99)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)

    # Subject A (3 faces, one drifted) + subject B (1 face).
    rows = [
        (shots[0].id, 0.35, 0.38, 0.58),
        (shots[1].id, 0.36, 0.37, 0.59),
        (shots[2].id, 0.55, 0.56, 0.78),
        (shots[3].id, 0.40, 0.43, 0.63),  # drifted A -> outlier
    ]
    with store.session() as session:
        for shot_id, b, g, r in rows:
            session.add(
                SkinMetric(
                    shot_id=shot_id,
                    face_index=0,
                    mean_b=b,
                    mean_g=g,
                    mean_r=r,
                    sample_pixels=100,
                )
            )
        session.commit()
    return asset, shots


def test_skin_features_roundtrip():
    store = ProjectStore.create(":memory:")
    asset, shots = _build_skin_asset(store)
    features = skin_features(store, asset.id)
    assert len(features) == 4
    assert all(f.b > 0 for f in features)


def test_skin_consistency_groups_and_flags():
    store = ProjectStore.create(":memory:")
    asset, shots = _build_skin_asset(store)

    deviations = skin_consistency(store, asset.id)

    assert {d.subject_id for d in deviations} == {0, 1}  # two subjects
    outliers = [d for d in deviations if d.is_outlier]
    assert len(outliers) == 1
    assert outliers[0].shot_id == shots[3].id  # the drifted face
    assert outliers[0].corrections[0].kind == "rgb_balance"
