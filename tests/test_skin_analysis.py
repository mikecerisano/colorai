"""Tests for per-subject skin-tone consistency and editable subjects."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import colorai.face as face_mod
from colorai.project import (
    ProjectStore,
    RepresentativeFrame,
    Shot,
    SkinMetric,
    Subject,
    make_representative_frame,
    make_shots,
)
from colorai.skin_analysis import (
    FaceSkin,
    assign_face,
    auto_assign_subjects,
    cluster_embeddings,
    create_subject,
    delete_subject,
    face_features,
    merge_subjects,
    propose_skin_match,
    rename_subject,
    set_reference,
    skin_consistency,
)


def _add_face(store, shot_id, b, g, r, subject_id=None):
    row = SkinMetric(
        shot_id=shot_id,
        face_index=0,
        mean_b=b,
        mean_g=g,
        mean_r=r,
        sample_pixels=100,
        subject_id=subject_id,
    )
    with store.session() as session:
        session.add(row)
        session.flush()
        session.refresh(row)
    return row


def _build_asset(store, n_shots=4):
    project = store.create_project("skin test")
    asset = store.add_asset(project.id, source_path="/media/m.mov", frame_rate=25.0)
    bounds = [(i * 25, i * 25 + 24) for i in range(n_shots)]
    shots = make_shots(asset, bounds)
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)
    return asset, shots


def test_propose_skin_match():
    ref = np.array([0.36, 0.38, 0.59])
    close = FaceSkin(1, 1, 0, None, 0.36, 0.38, 0.59)
    assert propose_skin_match(ref, close) is None

    drifted = FaceSkin(2, 2, 0, None, 0.40, 0.43, 0.63)
    correction = propose_skin_match(ref, drifted)
    assert correction is not None
    assert correction.kind == "rgb_balance"
    assert correction.parameters["gain"] == pytest.approx(
        [0.36 / 0.40, 0.38 / 0.43, 0.59 / 0.63], abs=1e-3
    )


def test_cluster_embeddings():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    clusters = cluster_embeddings([a, a, b, a])
    assert sorted(len(c) for c in clusters) == [1, 3]
    assert len(cluster_embeddings([])) == 0


def test_subject_crud_and_reassign():
    store = ProjectStore.create(":memory:")
    asset, shots = _build_asset(store, n_shots=3)

    a = create_subject(store, asset.id, "Alice")
    b = create_subject(store, asset.id, "Bob")

    f1 = _add_face(store, shots[0].id, 0.35, 0.38, 0.58, subject_id=a.id)
    f2 = _add_face(store, shots[1].id, 0.36, 0.37, 0.59, subject_id=b.id)

    # Reassign f2 from Bob to Alice.
    assign_face(store, f2.id, a.id)
    features = face_features(store, asset.id)
    assert all(f.subject_id == a.id for f in features)

    # Rename, then merge Bob into Alice (Bob has no faces now).
    rename_subject(store, a.id, "Interviewee 1")
    merge_subjects(store, a.id, b.id)

    with store.session() as session:
        names = [s.name for s in session.query(Subject).order_by(Subject.id)]
    assert names == ["Interviewee 1"]


def test_delete_subject_unassigns_faces():
    store = ProjectStore.create(":memory:")
    asset, shots = _build_asset(store, n_shots=1)
    a = create_subject(store, asset.id, "Alice")
    _add_face(store, shots[0].id, 0.35, 0.38, 0.58, subject_id=a.id)

    delete_subject(store, a.id)
    assert face_features(store, asset.id)[0].subject_id is None


def test_skin_consistency_groups_by_subject():
    store = ProjectStore.create(":memory:")
    asset, shots = _build_asset(store, n_shots=4)
    a = create_subject(store, asset.id, "A")
    b = create_subject(store, asset.id, "B")

    _add_face(store, shots[0].id, 0.35, 0.38, 0.58, subject_id=a.id)
    _add_face(store, shots[1].id, 0.36, 0.37, 0.59, subject_id=a.id)
    _add_face(store, shots[2].id, 0.55, 0.56, 0.78, subject_id=b.id)
    _add_face(store, shots[3].id, 0.40, 0.43, 0.63, subject_id=a.id)  # drifted A

    deviations = skin_consistency(store, asset.id)
    assert {d.subject_id for d in deviations} == {a.id, b.id}
    outliers = [d for d in deviations if d.is_outlier]
    assert len(outliers) == 1
    assert outliers[0].shot_id == shots[3].id
    assert outliers[0].corrections[0].kind == "rgb_balance"


def test_skin_consistency_uses_reference_shot():
    store = ProjectStore.create(":memory:")
    asset, shots = _build_asset(store, n_shots=2)
    a = create_subject(store, asset.id, "A")

    hero = _add_face(store, shots[0].id, 0.35, 0.38, 0.58, subject_id=a.id)
    other = _add_face(store, shots[1].id, 0.40, 0.43, 0.63, subject_id=a.id)
    set_reference(store, a.id, shots[0].id)

    deviations = skin_consistency(store, asset.id)
    by_shot = {d.shot_id: d for d in deviations}
    assert by_shot[shots[0].id].distance == pytest.approx(0.0)  # hero is the target
    assert by_shot[shots[1].id].is_outlier  # deviates from hero


def test_auto_assign_subjects_by_embedding(monkeypatch, tmp_path):
    store = ProjectStore.create(":memory:")
    asset, shots = _build_asset(store, n_shots=3)

    # Write tiny stills and attach representative frames.
    still = tmp_path / "still.png"
    cv2.imwrite(str(still), np.full((16, 16, 3), [128, 128, 128], dtype=np.uint8))
    for shot in shots:
        _add_face(store, shot.id, 0.36, 0.38, 0.59)
        with store.session() as session:
            session.add(
                make_representative_frame(
                    shot, shot.start_frame, image_path=str(still), frame_rate=25.0
                )
            )
            session.commit()

    # Same person for shots 0 and 1, a different person for shot 2.
    person_a = np.array([1.0, 0.0, 0.0])
    person_b = np.array([0.0, 1.0, 0.0])
    embeddings = iter([person_a, person_a, person_b])

    def fake_analyze_faces(img):
        embedding = next(embeddings, None)
        if embedding is None:
            return []
        return [{"bbox": [0, 0, 16, 16], "skin": None, "embedding": embedding}]

    monkeypatch.setattr(face_mod, "analyze_faces", fake_analyze_faces)

    subjects = auto_assign_subjects(store, asset.id)
    assert len(subjects) == 2

    features = face_features(store, asset.id)
    assert features[0].subject_id == features[1].subject_id  # same person
    assert features[2].subject_id != features[0].subject_id  # different person
