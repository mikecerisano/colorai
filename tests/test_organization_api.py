"""Tests for the needs-organization workspace payload, bulk API, and HTML."""

from __future__ import annotations

from fastapi.testclient import TestClient

from colorai.project import ProjectStore, SkinMetric, make_shots
from colorai.skin_analysis import create_subject
from colorai.ui import create_app


def _org_client(tmp_path):
    store = ProjectStore.create(":memory:")
    project = store.create_project("org test")
    asset = store.add_asset(
        project.id, source_path="/media/m.mov", frame_rate=25.0, width=64, height=64
    )
    shots = make_shots(asset, [(0, 24), (25, 49), (50, 74), (75, 99)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)

    alice = create_subject(store, asset.id, "Alice")
    bob = create_subject(store, asset.id, "Bob")

    def add_face(shot, face_index, subject_id):
        with store.session() as session:
            session.add(
                SkinMetric(
                    shot_id=shot.id, face_index=face_index,
                    mean_b=0.30, mean_g=0.30, mean_r=0.50,
                    sample_pixels=100, subject_id=subject_id,
                    bbox_x=10, bbox_y=12, bbox_w=20, bbox_h=24,
                )
            )
            session.commit()

    add_face(shots[0], 0, alice.id)                      # interview (Alice)
    add_face(shots[1], 0, alice.id)                      # interview (Alice)
    add_face(shots[2], 0, alice.id)                      # multi-person
    add_face(shots[2], 1, bob.id)
    # shots[3] has no face -> b-roll

    client = TestClient(create_app(store, tmp_path / "stills"))
    return client, asset, shots, alice, bob


def _workspace(client, asset_id):
    return client.get(f"/api/assets/{asset_id}/workspace").json()


def test_workspace_classifies_three_buckets(tmp_path):
    client, asset, shots, alice, bob = _org_client(tmp_path)
    org = _workspace(client, asset.id)["organization"]

    assert [m["id"] for m in org["interview_clusters"][0]["members"]] == [
        shots[0].id, shots[1].id,
    ]
    assert org["interview_clusters"][0]["subject_id"] == alice.id

    assert [j["shot_id"] for j in org["judgment"]] == [shots[2].id]

    assert [s["id"] for s in org["broll_shots"]] == [shots[3].id]

    # Queue contains all group-less, non-excused shots.
    assert set(org["queue_shot_ids"]) == {s.id for s in shots}
    assert org["dismissed_shots"] == []


def test_create_setup_action_assigns_members(tmp_path):
    client, asset, shots, alice, bob = _org_client(tmp_path)
    r = client.post(
        f"/api/assets/{asset.id}/organize",
        json={"action": "create_setup", "name": "Alice interview", "shot_ids": [shots[0].id, shots[1].id]},
    )
    assert r.status_code == 200
    assert r.json()["action"] == "create_setup"
    assert r.json()["shots"] == 2

    ws = _workspace(client, asset.id)
    by_id = {s["id"]: s for s in ws["shots"]}
    group_id = r.json()["group_id"]
    assert by_id[shots[0].id]["group_id"] == group_id
    assert by_id[shots[1].id]["group_id"] == group_id
    # The other shots were not moved.
    assert by_id[shots[2].id]["group_id"] is None
    assert by_id[shots[3].id]["group_id"] is None


def test_assign_action_adds_to_existing_setup(tmp_path):
    client, asset, shots, alice, bob = _org_client(tmp_path)
    setup = client.post(
        f"/api/assets/{asset.id}/groups", json={"name": "existing", "kind": "setup"}
    ).json()

    r = client.post(
        f"/api/assets/{asset.id}/organize",
        json={"action": "assign", "group_id": setup["id"], "shot_ids": [shots[0].id]},
    )
    assert r.status_code == 200

    ws = _workspace(client, asset.id)
    shot0 = next(s for s in ws["shots"] if s["id"] == shots[0].id)
    assert shot0["group_id"] == setup["id"]


def test_dismiss_and_restore_broll(tmp_path):
    client, asset, shots, alice, bob = _org_client(tmp_path)
    r = client.post(
        f"/api/assets/{asset.id}/organize",
        json={"action": "dismiss", "shot_ids": [shots[3].id]},
    )
    assert r.status_code == 200

    org = _workspace(client, asset.id)["organization"]
    assert [s["id"] for s in org["dismissed_shots"]] == [shots[3].id]
    assert shots[3].id not in org["queue_shot_ids"]

    # Restore returns it to the queue.
    client.post(
        f"/api/assets/{asset.id}/organize",
        json={"action": "restore", "shot_ids": [shots[3].id]},
    )
    org = _workspace(client, asset.id)["organization"]
    assert org["dismissed_shots"] == []
    assert shots[3].id in org["queue_shot_ids"]


def test_send_to_broll_pile_and_restore(tmp_path):
    client, asset, shots, alice, bob = _org_client(tmp_path)

    r = client.post(
        f"/api/assets/{asset.id}/organize",
        json={"action": "send_to_broll", "shot_ids": [shots[3].id]},
    )
    assert r.status_code == 200
    assert r.json()["action"] == "send_to_broll"

    ws = _workspace(client, asset.id)
    org = ws["organization"]
    assert [s["id"] for s in org["broll_pile"]] == [shots[3].id]
    assert shots[3].id not in org["queue_shot_ids"]
    assert all(setup["name"] != "B-roll" for setup in ws["setups"])

    r = client.post(
        f"/api/assets/{asset.id}/organize",
        json={"action": "restore_broll", "shot_ids": [shots[3].id]},
    )
    assert r.status_code == 200
    org = _workspace(client, asset.id)["organization"]
    assert org["broll_pile"] == []
    assert shots[3].id in org["queue_shot_ids"]


def test_workspace_does_not_move_existing_assignments(tmp_path):
    client, asset, shots, alice, bob = _org_client(tmp_path)
    setup = client.post(
        f"/api/assets/{asset.id}/groups", json={"name": "manual", "kind": "setup"}
    ).json()
    client.put(f"/api/shots/{shots[0].id}/group", json={"group_id": setup["id"]})

    # Reading the workspace (which computes suggestions) must be a pure read:
    # no shot's group assignment changes.
    before = {s["id"]: s["group_id"] for s in _workspace(client, asset.id)["shots"]}
    after = {s["id"]: s["group_id"] for s in _workspace(client, asset.id)["shots"]}
    assert before == after
    assert before[shots[0].id] == setup["id"]
    assert before[shots[1].id] is None


def test_needs_organization_html_buckets(tmp_path):
    client, asset, shots, alice, bob = _org_client(tmp_path)
    body = client.get("/").text

    assert "Needs organization" in body
    assert "Interview / setup candidates" in body
    assert "B-roll / non-interview" in body
    assert "Needs judgment" in body
    assert "Create setup from this" in body
    assert "Add to setup" in body
    assert "Dismiss / keep unorganized" in body
    assert "Mark intentional" in body
    assert "Send to B-roll" in body
    assert "B-roll pile" in body
    # The raw timecode checklist is gone; the visual buckets replace it.
    assert "Unassigned shots" not in body
