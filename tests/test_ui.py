"""Tests for the review UI."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from colorai.metrics import store_frame_metrics
from colorai.project import ProjectStore, make_representative_frame, make_shots
from colorai.ui import create_app


def _build_reviewable_project(store, stills_dir: Path) -> tuple[str, str]:
    """Create one project with two shots, stills, and metrics.

    Returns (stills_base, first_still_relpath) for request assertions.
    """
    project = store.create_project("review film")
    asset = store.add_asset(
        project.id, source_path="/media/m.mov", frame_rate=25.0, width=16, height=16
    )
    shots = make_shots(asset, [(0, 24), (25, 49)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)

    rfs = []
    for shot in shots:
        idx = (shot.start_frame + shot.end_frame) // 2
        path = stills_dir / f"asset_{asset.id:04d}" / f"shot_{shot.index:04d}_frame_{idx:06d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), (0, 0, 0)).save(path)
        rfs.append(
            make_representative_frame(
                shot, idx, image_path=str(path), frame_rate=asset.frame_rate
            )
        )

    with store.session() as session:
        session.add_all(rfs)
        session.flush()
        for rf in rfs:
            session.refresh(rf)

    # Metrics are added via the store helper (separate session).
    for shot, rf in zip(shots, rfs):
        store_frame_metrics(
            store,
            shot,
            rf.frame_index,
            {
                "luma_mean": 0.5,
                "luma_std": 0.1,
                "luma_min": 0.0,
                "luma_p5": 0.0,
                "luma_median": 0.5,
                "luma_p95": 1.0,
                "luma_max": 1.0,
                "r_mean": 0.5,
                "g_mean": 0.5,
                "b_mean": 0.5,
                "saturation_mean": 0.0,
            },
        )

    rel = rfs[0].image_path.replace(str(stills_dir.resolve()) + "/", "")
    return str(stills_dir), rel


def test_index_renders_shots(tmp_path):
    stills_dir = tmp_path / "stills"
    stills_dir.mkdir()
    store = ProjectStore.create(":memory:")
    stills_base, first_rel = _build_reviewable_project(store, stills_dir)

    client = TestClient(create_app(store, stills_base))
    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    assert "review film" in body
    assert "shot 0" in body
    assert "shot 1" in body
    assert "00:00:00:00" in body  # first shot start timecode
    assert "Unassigned" in body  # unassigned faces land in the face-review inbox


def test_stills_are_served(tmp_path):
    stills_dir = tmp_path / "stills"
    stills_dir.mkdir()
    store = ProjectStore.create(":memory:")
    stills_base, first_rel = _build_reviewable_project(store, stills_dir)

    client = TestClient(create_app(store, stills_base))
    response = client.get(f"/stills/{first_rel}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")


def test_asset_export_endpoint(tmp_path):
    from colorai.project.models import Correction

    stills_dir = tmp_path / "stills"
    stills_dir.mkdir()
    store = ProjectStore.create(":memory:")
    _stills_base, _rel = _build_reviewable_project(store, stills_dir)
    with store.session() as session:
        from colorai.project.models import Shot

        shot = session.query(Shot).order_by(Shot.index).first()
        shot_id, asset_id = shot.id, shot.asset_id
        session.add(
            Correction(shot_id=shot_id, kind="exposure",
                       parameters={"gain": 1.2}, enabled=True)
        )
        session.commit()

    client = TestClient(create_app(store, str(stills_dir)))
    response = client.post(f"/api/assets/{asset_id}/export", json={})
    assert response.status_code == 201
    body = response.json()
    assert len(body["shots"]) == 2
    assert body["shots"][0]["format"] == "cdl"
    assert body["shots"][1]["format"] == "none"
    out_dir = Path(body["out_dir"])
    assert (out_dir / "timeline.edl").exists()
    assert (out_dir / "manifest.json").exists()

    missing = client.post("/api/assets/9999/export", json={})
    assert missing.status_code == 404


def test_index_switches_between_masters(tmp_path):
    stills_dir = tmp_path / "stills"
    stills_dir.mkdir()
    store = ProjectStore.create(":memory:")
    first = store.create_project("film one")
    a1 = store.add_asset(first.id, source_path="/media/one.mov", frame_rate=25.0)
    second = store.create_project("film two")
    a2 = store.add_asset(second.id, source_path="/media/two.mov", frame_rate=25.0)
    from colorai.project import make_shots

    with store.session() as session:
        session.add_all(make_shots(a1, [(0, 9)]))
        session.add_all(make_shots(a2, [(0, 19)]))
        session.commit()

    client = TestClient(create_app(store, str(stills_dir)))
    home = client.get("/")
    assert home.status_code == 200
    assert "film one" in home.text and "film two" in home.text
    assert "one.mov" in home.text and "two.mov" in home.text

    other = client.get(f"/?asset_id={a2.id}")
    assert f"value=\"{a2.id}\" selected" in other.text

    fallback = client.get("/?asset_id=9999")
    assert f"value=\"{a1.id}\" selected" in fallback.text

    by_project = client.get(f"/?project_id={second.id}")
    assert f"value=\"{a2.id}\" selected" in by_project.text


def test_shot_scopes_endpoint(tmp_path):
    stills_dir = tmp_path / "stills"
    stills_dir.mkdir()
    store = ProjectStore.create(":memory:")
    _stills_base, _rel = _build_reviewable_project(store, stills_dir)
    with store.session() as session:
        from colorai.project.models import Shot

        shot_id = session.query(Shot).order_by(Shot.index).first().id

    client = TestClient(create_app(store, str(stills_dir)))
    response = client.get(f"/shots/{shot_id}/scopes.json")
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "corrected"
    assert body["waveform"]["width"] == 160
    assert len(body["waveform"]["luma_median"]) == 160
    assert body["vectorscope"]["size"] == 48

    original = client.get(f"/shots/{shot_id}/scopes.json?source=original")
    assert original.status_code == 200
    assert original.json()["source"] == "original"

    assert client.get(f"/shots/{shot_id}/scopes.json?source=bogus").status_code == 400
    assert client.get("/shots/9999/scopes.json").status_code == 404


def test_index_contains_lightbox_viewer(tmp_path):
    stills_dir = tmp_path / "stills"
    stills_dir.mkdir()
    store = ProjectStore.create(":memory:")
    _stills_base, _rel = _build_reviewable_project(store, stills_dir)

    body = TestClient(create_app(store, str(stills_dir))).get("/").text
    assert "openLightbox" in body
    assert "lb-wipe" in body
    assert "scopes.json" in body
    assert "keydown" in body
    assert "lb-wave" in body and "lb-vec" in body


def test_index_shows_analysis_tab_and_summaries(tmp_path):
    from colorai.project.models import Correction, ShotGroup

    stills_dir = tmp_path / "stills"
    stills_dir.mkdir()
    store = ProjectStore.create(":memory:")
    _stills_base, _rel = _build_reviewable_project(store, stills_dir)
    with store.session() as session:
        from colorai.project.models import Shot

        shot = session.query(Shot).order_by(Shot.index).first()
        group = ShotGroup(asset_id=shot.asset_id, name="interview", kind="setup")
        session.add(group)
        session.flush()
        shot.group_id = group.id
        session.add(
            Correction(shot_id=shot.id, kind="exposure",
                       parameters={"gain": 2.0}, enabled=True)
        )
        session.commit()

    body = TestClient(create_app(store, str(stills_dir))).get("/").text
    assert "tab-analysis" in body
    assert "Shot consistency" in body
    assert "analysis-ref" in body
    assert "exposure 2.00× (+1.00 stops)" in body
