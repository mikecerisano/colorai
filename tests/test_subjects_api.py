"""Tests for subject, note, and tracking API endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from colorai.project import ProjectStore, SkinMetric, make_shots
from colorai.skin_analysis import create_subject as _create_subject
from colorai.ui import create_app


def _client(tmp_path):
    store = ProjectStore.create(":memory:")
    project = store.create_project("film")
    asset = store.add_asset(project.id, source_path="/media/m.mov", frame_rate=25.0)
    shots = make_shots(asset, [(0, 24), (25, 49)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)

    a = _create_subject(store, asset.id, "Alice")
    with store.session() as session:
        session.add(
            SkinMetric(
                shot_id=shots[0].id, face_index=0,
                mean_b=0.35, mean_g=0.38, mean_r=0.58,
                sample_pixels=100, subject_id=a.id,
            )
        )
        session.commit()

    client = TestClient(create_app(store, tmp_path / "stills"))
    return client, asset, shots, a


def test_list_subjects(tmp_path):
    client, asset, shots, a = _client(tmp_path)
    r = client.get(f"/api/assets/{asset.id}/subjects")
    assert r.status_code == 200
    subjects = r.json()
    assert subjects[0]["name"] == "Alice"
    assert subjects[0]["faces"][0]["shot_id"] == shots[0].id


def test_create_and_rename_subject(tmp_path):
    client, asset, shots, a = _client(tmp_path)
    created = client.post(f"/api/assets/{asset.id}/subjects", json={"name": "Bob"})
    assert created.status_code == 201

    r = client.patch(f"/api/subjects/{created.json()['id']}", json={"name": "Robert"})
    assert r.json()["name"] == "Robert"


def test_set_reference(tmp_path):
    client, asset, shots, a = _client(tmp_path)
    r = client.post(f"/api/subjects/{a.id}/reference", json={"shot_id": shots[0].id})
    assert r.json()["reference_shot_id"] == shots[0].id
    r = client.post(f"/api/subjects/{a.id}/reference", json={"shot_id": None})
    assert r.json()["reference_shot_id"] is None


def test_assign_and_unassign_face(tmp_path):
    client, asset, shots, a = _client(tmp_path)
    # The only face is skin_metric id 1.
    r = client.patch("/api/skin_metrics/1", json={"subject_id": None})
    assert r.status_code == 200

    subjects = client.get(f"/api/assets/{asset.id}/subjects").json()
    assert subjects[0]["faces"] == []

    r = client.patch("/api/skin_metrics/1", json={"subject_id": a.id})
    subjects = client.get(f"/api/assets/{asset.id}/subjects").json()
    assert len(subjects[0]["faces"]) == 1


def test_merge_and_delete(tmp_path):
    client, asset, shots, a = _client(tmp_path)
    b = client.post(f"/api/assets/{asset.id}/subjects", json={"name": "Bob"}).json()

    r = client.post("/api/subjects/merge", json={"keep_id": a.id, "drop_id": b["id"]})
    assert r.status_code == 200

    r = client.delete(f"/api/subjects/{a.id}")
    assert r.status_code == 204
    assert client.get(f"/api/assets/{asset.id}/subjects").json() == []


def test_skin_consistency_endpoint(tmp_path):
    client, asset, shots, a = _client(tmp_path)
    r = client.get(f"/api/assets/{asset.id}/skin-consistency")
    assert r.status_code == 200
    assert len(r.json()) == 1  # one face, its own reference


def test_notes(tmp_path):
    client, asset, shots, a = _client(tmp_path)
    created = client.post(
        f"/api/assets/{asset.id}/notes",
        json={"text": "check this", "author": "me", "shot_id": shots[0].id},
    )
    assert created.status_code == 201

    notes = client.get(f"/api/assets/{asset.id}/notes").json()
    assert notes[0]["text"] == "check this"
    assert notes[0]["author"] == "me"


def test_track_unknown_shot_404(tmp_path):
    client, asset, shots, a = _client(tmp_path)
    assert client.get("/api/shots/9999/track").status_code == 404
