"""Tests for the human skin-matching review API and HTML."""

from __future__ import annotations

from fastapi.testclient import TestClient

from colorai.editorial import assign_shot_group, create_group
from colorai.project import (
    FaceCorrection,
    FaceTrack,
    ProjectStore,
    SkinMetric,
    make_shots,
)
from colorai.skin_analysis import create_subject
from colorai.ui import create_app


def _client(tmp_path, n_shots=2):
    store = ProjectStore.create(":memory:")
    project = store.create_project("skin ui")
    asset = store.add_asset(
        project.id, source_path="/media/m.mov", frame_rate=25.0, width=64, height=64
    )
    shots = make_shots(asset, [(0, 24), (25, 49)][:n_shots])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)
    alice = create_subject(store, asset.id, "Alice")
    group = create_group(store, asset.id, "interview", kind="setup")
    for s in shots:
        assign_shot_group(store, s.id, group.id)

    metrics = []
    with store.session() as session:
        for i, s in enumerate(shots):
            m = SkinMetric(
                shot_id=s.id, face_index=0, mean_b=0.3, mean_g=0.3, mean_r=0.5,
                sample_pixels=10, subject_id=alice.id,
                bbox_x=16, bbox_y=16, bbox_w=32, bbox_h=32,
            )
            session.add(m)
            session.flush()
            session.add(
                FaceTrack(
                    shot_id=s.id, skin_metric_id=m.id, subject_id=alice.id,
                    source_width=64, source_height=64, analysis_scale=64,
                    keyframes=[[s.start_frame, 0.25, 0.25, 0.5, 0.5], [s.end_frame, 0.25, 0.25, 0.5, 0.5]],
                    sample_count=2, tracked_count=2, coverage=1.0, max_gap=0.0,
                    skin_stability=0.01, median_bgr=[0.3, 0.3, 0.5], state="valid",
                )
            )
            metrics.append(m)
        session.commit()

    client = TestClient(create_app(store, tmp_path / "stills"))
    return client, store, asset, shots, alice, group, metrics


def _add_suggested_correction(store, shot, alice, metric):
    with store.session() as session:
        track_id = session.query(FaceTrack).filter_by(skin_metric_id=metric.id).one().id
    with store.session() as session:
        c = FaceCorrection(
            shot_id=shot.id, subject_id=alice.id, skin_metric_id=metric.id,
            face_track_id=track_id, kind="rgb_balance",
            parameters={"gain": [1.0, 0.95, 1.05]},
            reason="warm", confidence=0.8, classification="skin_mismatch",
            state="suggested", enabled=False,
        )
        session.add(c)
        session.flush()
        session.refresh(c)
        return c.id


def test_qc_only_status_for_single_shot(tmp_path):
    client, store, asset, shots, alice, group, metrics = _client(tmp_path, n_shots=1)
    body = client.get("/").text
    assert "qc_only" in body
    assert "No matching target" in body


def test_suggestion_cannot_enable_before_approval(tmp_path):
    client, store, asset, shots, alice, group, metrics = _client(tmp_path)
    cid = _add_suggested_correction(store, shots[1], alice, metrics[1])

    r = client.post(f"/api/face-corrections/{cid}/enable")
    assert r.json()["error"] == "must be approved before enabling"

    # Still disabled.
    ws = client.get(f"/api/assets/{asset.id}/skin-matching/{group.id}").json()
    props = [p for part in ws for p in part["proposals"]]
    assert all(p["enabled"] is False for p in props)


def test_approve_then_enable_sequence(tmp_path):
    client, store, asset, shots, alice, group, metrics = _client(tmp_path)
    cid = _add_suggested_correction(store, shots[1], alice, metrics[1])

    assert client.post(f"/api/face-corrections/{cid}/approve").json()["state"] == "approved"
    assert client.post(f"/api/face-corrections/{cid}/enable").json()["enabled"] is True

    ws = client.get(f"/api/assets/{asset.id}/skin-matching/{group.id}").json()
    props = [p for part in ws for p in part["proposals"]]
    assert props[0]["enabled"] is True


def test_reject_and_mark_intentional(tmp_path):
    client, store, asset, shots, alice, group, metrics = _client(tmp_path)
    cid = _add_suggested_correction(store, shots[1], alice, metrics[1])

    assert client.post(f"/api/face-corrections/{cid}/reject").json()["state"] == "rejected"

    cid2 = _add_suggested_correction(store, shots[1], alice, metrics[1])
    out = client.post(f"/api/face-corrections/{cid2}/mark-intentional").json()
    assert out["state"] == "rejected"
    assert out["classification"] == "intentional_lighting"


def test_html_renders_skin_matching_actions(tmp_path):
    client, store, asset, shots, alice, group, metrics = _client(tmp_path)
    _add_suggested_correction(store, shots[1], alice, metrics[1])

    body = client.get("/").text
    assert "Skin matching" in body
    assert "Approve correction" in body
    assert "Reject" in body
    assert "Mark intentional" in body
    # The proposal shows temporal evidence, RGB gain, crops, context, and
    # a local before/after comparison.
    assert "RGB gain" in body
    assert "cov" in body
    assert "stab" in body
    assert "candidate face" in body
    assert "corrected" in body
    assert "show box" in body
