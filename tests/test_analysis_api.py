"""Tests for the consistency-analysis API endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from colorai.project import FrameMetrics, ProjectStore, make_shots
from colorai.ui import create_app


def _asset_with_metrics(store):
    project = store.create_project("outlier api test")
    asset = store.add_asset(project.id, source_path="/media/m.mov", frame_rate=25.0)
    shots = make_shots(asset, [(0, 24), (25, 49), (50, 74)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)

    rows = [
        (shots[0].id, 0.5, 0.5, 0.5, 0.5),   # median reference
        (shots[1].id, 0.25, 0.25, 0.25, 0.25),  # darker -> exposure
        (shots[2].id, 0.5, 0.5, 0.4, 0.4),   # channel imbalance -> rgb_balance
    ]
    with store.session() as session:
        for shot_id, luma, r, g, b in rows:
            session.add(
                FrameMetrics(
                    shot_id=shot_id,
                    frame_index=0,
                    luma_mean=luma,
                    r_mean=r,
                    g_mean=g,
                    b_mean=b,
                    saturation_mean=0.0,
                )
            )
        session.commit()
    return asset, shots


def _client(tmp_path):
    store = ProjectStore.create(":memory:")
    asset, shots = _asset_with_metrics(store)
    client = TestClient(create_app(store, tmp_path / "stills"))
    return client, asset, shots


def test_outliers_endpoint(tmp_path):
    client, asset, shots = _client(tmp_path)
    r = client.get(f"/api/assets/{asset.id}/outliers")
    assert r.status_code == 200
    outliers = r.json()["outliers"]
    assert len(outliers) == 2
    kinds = {c["kind"] for o in outliers for c in o["corrections"]}
    assert kinds == {"exposure", "rgb_balance"}
    assert all(o["is_outlier"] for o in outliers)


def test_outliers_endpoint_explicit_reference(tmp_path):
    client, asset, shots = _client(tmp_path)
    r = client.get(f"/api/assets/{asset.id}/outliers", params={"reference_shot_id": shots[1].id})
    assert r.status_code == 200
    assert shots[1].id not in {o["shot_id"] for o in r.json()["outliers"]}


def test_apply_proposals_endpoint(tmp_path):
    client, asset, shots = _client(tmp_path)
    r = client.post(f"/api/assets/{asset.id}/apply-proposals")
    assert r.status_code == 201
    created = r.json()["created"]
    assert len(created) == 2
    assert {c["kind"] for c in created} == {"exposure", "rgb_balance"}


def test_outliers_unknown_asset_404(tmp_path):
    store = ProjectStore.create(":memory:")
    client = TestClient(create_app(store, tmp_path / "stills"))
    assert client.get("/api/assets/9999/outliers").status_code == 404
    assert client.post("/api/assets/9999/apply-proposals").status_code == 404


def test_outliers_unknown_reference_404(tmp_path):
    client, asset, shots = _client(tmp_path)
    r = client.get(f"/api/assets/{asset.id}/outliers", params={"reference_shot_id": 9999})
    assert r.status_code == 404


def test_propose_for_shot(tmp_path):
    client, asset, shots = _client(tmp_path)
    r = client.post(f"/api/shots/{shots[1].id}/propose")
    assert r.status_code == 201
    created = r.json()["created"]
    assert any(c["kind"] == "exposure" for c in created)

    # Proposals should now show up on the shot.
    shot = client.get(f"/api/shots/{shots[1].id}").json()
    assert shot["corrections"]
