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
    assert "Unassigned" in body  # shots without a setup land in the inbox


def test_stills_are_served(tmp_path):
    stills_dir = tmp_path / "stills"
    stills_dir.mkdir()
    store = ProjectStore.create(":memory:")
    stills_base, first_rel = _build_reviewable_project(store, stills_dir)

    client = TestClient(create_app(store, stills_base))
    response = client.get(f"/stills/{first_rel}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
