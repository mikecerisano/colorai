"""Tests for the review UI: assignment dropdown, face crops, setup membership,
and multi-face display data."""

from __future__ import annotations

import cv2
import numpy as np
from fastapi.testclient import TestClient

from colorai.project import (
    ProjectStore,
    SkinMetric,
    make_representative_frame,
    make_shots,
)
from colorai.skin_analysis import create_subject
from colorai.ui import create_app


def _client(tmp_path):
    store = ProjectStore.create(":memory:")
    project = store.create_project("ui test")
    asset = store.add_asset(
        project.id, source_path="/media/m.mov", frame_rate=25.0, width=64, height=64
    )
    shots = make_shots(asset, [(0, 24), (25, 49)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)

    stills = tmp_path / "stills"
    stills.mkdir()
    still = stills / "still0.png"
    cv2.imwrite(str(still), np.full((64, 64, 3), 128, dtype=np.uint8))
    with store.session() as session:
        session.add(
            make_representative_frame(shots[0], 0, image_path=str(still), frame_rate=25.0)
        )
        session.commit()

    alice = create_subject(store, asset.id, "Alice")
    bob = create_subject(store, asset.id, "Bob")
    with store.session() as session:
        session.add(
            SkinMetric(
                shot_id=shots[0].id, face_index=0,
                mean_b=0.30, mean_g=0.30, mean_r=0.50,
                sample_pixels=100, subject_id=alice.id,
                bbox_x=10, bbox_y=12, bbox_w=20, bbox_h=24,
            )
        )
        session.add(
            SkinMetric(
                shot_id=shots[0].id, face_index=1,
                mean_b=0.20, mean_g=0.25, mean_r=0.40,
                sample_pixels=100, subject_id=bob.id,
                bbox_x=32, bbox_y=14, bbox_w=18, bbox_h=22,
            )
        )
        session.commit()

    client = TestClient(create_app(store, stills))
    return client, asset, shots, alice, bob


def test_assigned_face_shows_subject_selected_not_unassign(tmp_path):
    client, asset, shots, alice, bob = _client(tmp_path)
    body = client.get("/").text

    # The currently-assigned subject must be the selected value…
    assert "selected>Alice</option>" in body
    assert "selected>Bob</option>" in body
    # …and "Unassigned" is a real option (not a default "unassign").
    assert ">Unassigned</option>" in body


def test_assigned_face_can_be_unassigned_and_moved(tmp_path):
    client, asset, shots, alice, bob = _client(tmp_path)

    ws = client.get(f"/api/assets/{asset.id}/workspace").json()
    face = ws["subjects"][0]["faces"][0]
    skin_metric_id = face["skin_metric_id"]

    # Unassign via the API (subject_id null).
    r = client.patch(f"/api/skin_metrics/{skin_metric_id}", json={"subject_id": None})
    assert r.status_code == 200
    ws = client.get(f"/api/assets/{asset.id}/workspace").json()
    unassigned_ids = [f["skin_metric_id"] for f in ws["unassigned_faces"]]
    assert skin_metric_id in unassigned_ids

    # Move to another subject (Bob).
    client.patch(f"/api/skin_metrics/{skin_metric_id}", json={"subject_id": bob.id})
    ws = client.get(f"/api/assets/{asset.id}/workspace").json()
    bob_faces = next(s for s in ws["subjects"] if s["id"] == bob.id)["faces"]
    assert skin_metric_id in [f["skin_metric_id"] for f in bob_faces]


def test_face_crop_endpoint_serves_image(tmp_path):
    client, asset, shots, alice, bob = _client(tmp_path)
    ws = client.get(f"/api/assets/{asset.id}/workspace").json()
    face_id = ws["subjects"][0]["faces"][0]["skin_metric_id"]

    r = client.get(f"/api/skin_metrics/{face_id}/crop.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:4] == b"\x89PNG"


def test_multi_face_bbox_data_present(tmp_path):
    client, asset, shots, alice, bob = _client(tmp_path)
    ws = client.get(f"/api/assets/{asset.id}/workspace").json()

    all_faces = [f for s in ws["subjects"] for f in s["faces"]]
    assert len(all_faces) == 2
    boxes = [tuple(f["bbox"]) for f in all_faces]
    assert boxes[0] != boxes[1]  # distinct boxes for distinct faces
    assert all(all(v is not None for v in b) for b in boxes)


def test_setup_membership_in_workspace(tmp_path):
    client, asset, shots, alice, bob = _client(tmp_path)
    # Create a setup and assign shot 0.
    g = client.post(
        f"/api/assets/{asset.id}/groups",
        json={"name": "interview A", "kind": "setup", "camera": "A"},
    ).json()
    client.put(f"/api/shots/{shots[0].id}/group", json={"group_id": g["id"]})

    ws = client.get(f"/api/assets/{asset.id}/workspace").json()
    setup = next(s for s in ws["setups"] if s["id"] == g["id"])
    assert setup["kind"] == "setup"
    assert setup["camera"] == "A"
    assert setup["shot_ids"] == [shots[0].id]

    # Shot 1 is still unassigned to any setup.
    unassigned_shot_ids = [s["id"] for s in ws["unassigned_shots"]]
    assert shots[1].id in unassigned_shot_ids


def test_reference_approval_reflected_in_setup_badge(tmp_path):
    client, asset, shots, alice, bob = _client(tmp_path)
    g = client.post(
        f"/api/assets/{asset.id}/groups",
        json={"name": "interview A", "kind": "setup"},
    ).json()
    client.put(f"/api/shots/{shots[0].id}/group", json={"group_id": g["id"]})
    proposal = client.post(
        f"/api/assets/{asset.id}/reference-proposals",
        json={"shot_id": shots[0].id, "reason": "hero", "confidence": 0.9, "group_id": g["id"]},
    ).json()

    ws = client.get(f"/api/assets/{asset.id}/workspace").json()
    setup = next(s for s in ws["setups"] if s["id"] == g["id"])
    assert setup["reference_state"] == "suggested"

    client.post(f"/api/reference-proposals/{proposal['id']}/approve")
    ws = client.get(f"/api/assets/{asset.id}/workspace").json()
    setup = next(s for s in ws["setups"] if s["id"] == g["id"])
    assert setup["reference_state"] == "approved"
    assert setup["approved_reference_shot_id"] == shots[0].id
