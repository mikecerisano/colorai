"""Tests for the correction API and preview endpoint."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from colorai.color import bt709_to_linear, linear_to_bt709
from colorai.project import ProjectStore, make_representative_frame, make_shots
from colorai.ui import create_app


def _shot_with_midgray_still(tmp_path):
    store = ProjectStore.create(":memory:")
    project = store.create_project("correction api test")
    asset = store.add_asset(project.id, source_path="/media/m.mov", frame_rate=25.0)
    shot = make_shots(asset, [(0, 24)])[0]
    with store.session() as session:
        session.add(shot)
        session.flush()
        session.refresh(shot)

    stills = tmp_path / "stills"
    stills.mkdir()
    still = stills / "still.png"
    # Mid-gray BGR still.
    cv2.imwrite(str(still), np.full((8, 8, 3), [128, 128, 128], dtype=np.uint8))
    with store.session() as session:
        session.add(
            make_representative_frame(shot, 0, image_path=str(still), frame_rate=25.0)
        )
        session.commit()

    return store, stills, shot.id


def _client(tmp_path):
    store, stills, shot_id = _shot_with_midgray_still(tmp_path)
    return TestClient(create_app(store, stills)), shot_id


def test_add_correction(tmp_path):
    client, shot_id = _client(tmp_path)
    r = client.post(
        f"/api/shots/{shot_id}/corrections",
        json={"kind": "exposure", "parameters": {"gain": 2.0}},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["kind"] == "exposure"
    assert body["parameters"] == {"gain": 2.0}
    assert body["enabled"] is True
    assert body["id"] is not None


def test_add_correction_rejects_invalid(tmp_path):
    client, shot_id = _client(tmp_path)
    r = client.post(
        f"/api/shots/{shot_id}/corrections",
        json={"kind": "bogus", "parameters": {}},
    )
    assert r.status_code == 400


def test_get_shot_lists_corrections(tmp_path):
    client, shot_id = _client(tmp_path)
    client.post(
        f"/api/shots/{shot_id}/corrections",
        json={"kind": "offset", "parameters": {"value": 0.1}},
    )
    r = client.get(f"/api/shots/{shot_id}")
    assert r.status_code == 200
    assert r.json()["corrections"][0]["kind"] == "offset"


def test_toggle_and_delete_correction(tmp_path):
    client, shot_id = _client(tmp_path)
    created = client.post(
        f"/api/shots/{shot_id}/corrections",
        json={"kind": "saturation", "parameters": {"amount": 0.0}},
    ).json()

    # Toggle off.
    r = client.patch(f"/api/corrections/{created['id']}", json={"enabled": False})
    assert r.status_code == 200
    assert r.json()["enabled"] is False

    # Delete.
    r = client.delete(f"/api/corrections/{created['id']}")
    assert r.status_code == 204
    assert client.get(f"/api/shots/{shot_id}").json()["corrections"] == []


def test_update_correction_rejects_invalid_params(tmp_path):
    client, shot_id = _client(tmp_path)
    created = client.post(
        f"/api/shots/{shot_id}/corrections", json={"kind": "exposure", "parameters": {}}
    ).json()
    r = client.patch(
        f"/api/corrections/{created['id']}", json={"parameters": {"gain": -1.0}}
    )
    assert r.status_code == 400


def test_preview_without_corrections_is_original(tmp_path):
    client, shot_id = _client(tmp_path)
    r = client.get(f"/shots/{shot_id}/preview.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    arr = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert img.mean() == pytest.approx(128.0, abs=3)


def test_preview_applies_correction(tmp_path):
    client, shot_id = _client(tmp_path)
    client.post(
        f"/api/shots/{shot_id}/corrections",
        json={"kind": "exposure", "parameters": {"gain": 2.0}},
    )
    r = client.get(f"/shots/{shot_id}/preview.png")
    img = cv2.imdecode(np.frombuffer(r.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    # Mid-gray decoded to linear, doubled, then re-encoded.
    expected = float(linear_to_bt709(bt709_to_linear(128 / 255) * 2.0))
    assert img.mean() == pytest.approx(expected * 255, abs=5)


def test_preview_404_for_unknown_shot(tmp_path):
    store, stills, _ = _shot_with_midgray_still(tmp_path)
    client = TestClient(create_app(store, stills))
    assert client.get("/shots/9999/preview.png").status_code == 404
