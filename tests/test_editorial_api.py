"""Tests for the editorial review-state, grouping, and split/merge API."""

from __future__ import annotations

from fastapi.testclient import TestClient

from colorai.project import ProjectStore, make_shots
from colorai.ui import create_app


def _client(tmp_path):
    store = ProjectStore.create(":memory:")
    project = store.create_project("film")
    asset = store.add_asset(project.id, source_path="/media/m.mov", frame_rate=25.0)
    shots = make_shots(asset, [(0, 24), (25, 49), (50, 74)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)
    client = TestClient(create_app(store, tmp_path / "stills"))
    return client, asset, shots


def test_update_shot_review_state(tmp_path):
    client, asset, shots = _client(tmp_path)
    r = client.patch(f"/api/shots/{shots[0].id}", json={"review_status": "approved", "excused": True})
    assert r.status_code == 200
    assert r.json() == {"id": shots[0].id, "review_status": "approved", "excused": True}

    detail = client.get(f"/api/shots/{shots[0].id}").json()
    assert detail["review_status"] == "approved"
    assert detail["excused"] is True


def test_update_shot_invalid_review_status(tmp_path):
    client, asset, shots = _client(tmp_path)
    r = client.patch(f"/api/shots/{shots[0].id}", json={"review_status": "bogus"})
    assert r.status_code == 400


def test_split_and_merge_api(tmp_path):
    client, asset, shots = _client(tmp_path)
    r = client.post(f"/api/shots/{shots[0].id}/split", json={"at_frame": 10})
    assert r.status_code == 201
    first, second = r.json()["first"], r.json()["second"]

    detail = client.get(f"/api/shots/{first}").json()
    assert (detail["start_frame"], detail["end_frame"]) == (0, 9)

    r = client.post("/api/shots/merge", json={"shot_id_a": first, "shot_id_b": second})
    assert r.status_code == 200
    assert (r.json()["start_frame"], r.json()["end_frame"]) == (0, 24)


def test_group_api(tmp_path):
    client, asset, shots = _client(tmp_path)
    created = client.post(f"/api/assets/{asset.id}/groups", json={"name": "cam A"})
    assert created.status_code == 201
    group_id = created.json()["id"]

    assert client.get(f"/api/assets/{asset.id}/groups").json()[0]["name"] == "cam A"

    r = client.put(f"/api/shots/{shots[0].id}/group", json={"group_id": group_id})
    assert r.json()["group_id"] == group_id

    groups = client.get(f"/api/assets/{asset.id}/groups").json()
    assert groups[0]["shot_ids"] == [shots[0].id]

    assert client.delete(f"/api/shots/{shots[0].id}/group").status_code == 200
    assert client.delete(f"/api/groups/{group_id}").status_code == 204
    assert client.get(f"/api/assets/{asset.id}/groups").json() == []


def test_reference_proposal_api(tmp_path):
    client, asset, shots = _client(tmp_path)
    created = client.post(
        f"/api/assets/{asset.id}/reference-proposals",
        json={"shot_id": shots[0].id, "reason": "hero: stable and well lit", "confidence": 0.8},
    )
    assert created.status_code == 201
    proposal_id = created.json()["id"]

    listing = client.get(f"/api/assets/{asset.id}/reference-proposals").json()
    assert listing[0]["state"] == "suggested"

    approved = client.post(f"/api/reference-proposals/{proposal_id}/approve")
    assert approved.json()["state"] == "approved"
    assert client.get(f"/api/assets/{asset.id}/reference-proposals").json()[0]["state"] == "approved"


def test_reference_proposal_reject_and_invalid(tmp_path):
    client, asset, shots = _client(tmp_path)
    r = client.post(
        f"/api/assets/{asset.id}/reference-proposals",
        json={"shot_id": shots[0].id, "reason": "", "confidence": 0.5},
    )
    assert r.status_code == 400

    created = client.post(
        f"/api/assets/{asset.id}/reference-proposals",
        json={"shot_id": shots[0].id, "reason": "maybe", "confidence": 0.3},
    ).json()
    rejected = client.post(f"/api/reference-proposals/{created['id']}/reject")
    assert rejected.json()["state"] == "rejected"
    assert client.post(f"/api/reference-proposals/9999/approve").status_code == 404
