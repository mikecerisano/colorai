"""Tests for the MCP server tools and agent notes."""

from __future__ import annotations

import pytest

from colorai import mcp_server
from colorai.project import ProjectStore, SkinMetric, make_shots
from colorai.skin_analysis import create_subject


def _make_store(tmp_path):
    db = tmp_path / "project.sqlite3"
    store = ProjectStore.create(db)
    project = store.create_project("film")
    asset = store.add_asset(project.id, source_path="/media/m.mov", frame_rate=25.0)
    shots = make_shots(asset, [(0, 24), (25, 49)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)
    subject = create_subject(store, asset.id, "Alice")
    with store.session() as session:
        session.add(
            SkinMetric(
                shot_id=shots[0].id,
                face_index=0,
                mean_b=0.35,
                mean_g=0.38,
                mean_r=0.58,
                sample_pixels=100,
                subject_id=subject.id,
            )
        )
        session.commit()
    return str(db), asset, shots, subject


def test_list_and_get(tmp_path):
    db, asset, shots, subject = _make_store(tmp_path)

    projects = mcp_server.list_projects(db)
    assert projects == [{"id": 1, "name": "film"}]

    assets = mcp_server.list_assets(db)
    assert assets[0]["source_path"] == "/media/m.mov"

    shot_list = mcp_server.list_shots(db, asset.id)
    assert [s["index"] for s in shot_list] == [0, 1]

    detail = mcp_server.get_shot(db, shots[0].id)
    assert detail["start_timecode"] == "00:00:00:00"
    assert detail["skin_faces"][0]["subject_id"] == subject.id


def test_subject_refine_tools(tmp_path):
    db, asset, shots, subject = _make_store(tmp_path)

    mcp_server.rename_subject(db, subject.id, "Interviewee 1")
    subjects = mcp_server.list_subjects(db, asset.id)
    assert subjects[0]["name"] == "Interviewee 1"

    mcp_server.set_reference(db, subject.id, shots[0].id)
    subjects = mcp_server.list_subjects(db, asset.id)
    assert subjects[0]["reference_shot_id"] == shots[0].id


def test_merge_subjects(tmp_path):
    db, asset, shots, subject = _make_store(tmp_path)
    other = create_subject(ProjectStore.open(db), asset.id, "Bob")
    mcp_server.merge_subjects(db, subject.id, other.id)
    subjects = mcp_server.list_subjects(db, asset.id)
    assert [s["name"] for s in subjects] == ["Alice"]


def test_correction_tools(tmp_path):
    db, asset, shots, subject = _make_store(tmp_path)

    created = mcp_server.add_correction(
        db, shots[0].id, "exposure", {"gain": 2.0}
    )
    assert created["kind"] == "exposure"
    assert mcp_server.toggle_correction(db, created["id"], False) == "ok"
    assert mcp_server.get_shot(db, shots[0].id)["corrections"][0]["enabled"] is False
    assert mcp_server.delete_correction(db, created["id"]) == "ok"
    assert mcp_server.get_shot(db, shots[0].id)["corrections"] == []


def test_add_correction_rejects_invalid_kind(tmp_path):
    db, asset, shots, subject = _make_store(tmp_path)
    with pytest.raises(ValueError):
        mcp_server.add_correction(db, shots[0].id, "bogus", {})


def test_notes(tmp_path):
    db, asset, shots, subject = _make_store(tmp_path)

    note = mcp_server.add_note(
        db,
        asset.id,
        "shot 0 skin is the reference for this subject",
        author="claude",
        shot_id=shots[0].id,
    )
    assert note["author"] == "claude"

    notes = mcp_server.list_notes(db, asset.id)
    assert len(notes) == 1
    assert notes[0]["shot_id"] == shots[0].id


def test_skin_consistency_tool(tmp_path):
    db, asset, shots, subject = _make_store(tmp_path)
    result = mcp_server.skin_consistency(db, asset.id)
    # Only one face assigned; it is its own reference, so no outlier.
    assert len(result) == 1
    assert result[0]["is_outlier"] is False


def test_skin_metric_tools(tmp_path):
    db, asset, shots, subject = _make_store(tmp_path)

    # The existing face is shots[0] skin_metric id 1.
    result = mcp_server.set_skin_metric(db, 1, 0.40, 0.43, 0.63)
    assert result["mean_bgr"] == [0.4, 0.43, 0.63]

    added = mcp_server.add_skin_metric(db, shots[1].id, 0, 0.5, 0.5, 0.5, subject_id=subject.id)
    assert added["shot_id"] == shots[1].id

    assert mcp_server.unassign_face(db, 1) == "ok"
    assert mcp_server.delete_skin_metric(db, 1) == "ok"

    detail = mcp_server.get_shot(db, shots[0].id)
    assert detail["skin_faces"] == []


def test_get_shot_still_returns_image(tmp_path):
    import cv2
    import numpy as np

    from colorai.project import make_representative_frame

    db, asset, shots, subject = _make_store(tmp_path)
    still = tmp_path / "still.png"
    cv2.imwrite(str(still), np.zeros((8, 8, 3), dtype=np.uint8))
    with ProjectStore.open(db).session() as session:
        session.add(
            make_representative_frame(shots[0], 0, image_path=str(still), frame_rate=25.0)
        )
        session.commit()

    image = mcp_server.get_shot_still(db, shots[0].id)
    assert str(image.path) == str(still)


def test_get_shot_still_unknown_shot_raises(tmp_path):
    db, asset, shots, subject = _make_store(tmp_path)
    with pytest.raises(ValueError):
        mcp_server.get_shot_still(db, 9999)


def test_get_shot_frame_unknown_shot_raises(tmp_path):
    db, asset, shots, subject = _make_store(tmp_path)
    with pytest.raises(ValueError):
        mcp_server.get_shot_frame(db, 9999, 0)


def test_detect_blur_pulses_unknown_shot(tmp_path):
    db, asset, shots, subject = _make_store(tmp_path)
    with pytest.raises(ValueError):
        mcp_server.detect_blur_pulses(db, 9999)


def test_mcp_server_lists_tools():
    import asyncio

    tools = asyncio.run(mcp_server.mcp.list_tools())
    names = {t.name for t in tools}
    assert "get_shot" in names
    assert "assign_face" in names
    assert "add_note" in names
    assert "split_shot" in names
    assert "merge_shots" in names
    assert "set_shot_review_status" in names
    assert "create_shot_group" in names


def test_editorial_review_and_group_tools(tmp_path):
    db, asset, shots, subject = _make_store(tmp_path)

    assert mcp_server.set_shot_review_status(db, shots[0].id, "approved") == "ok"
    assert mcp_server.set_shot_excused(db, shots[0].id, True) == "ok"

    detail = mcp_server.get_shot(db, shots[0].id)
    assert detail["review_status"] == "approved"
    assert detail["excused"] is True

    group = mcp_server.create_shot_group(db, asset.id, "cam A")
    assert mcp_server.list_shot_groups(db, asset.id)[0]["name"] == "cam A"
    assert mcp_server.assign_shot_group(db, shots[0].id, group["id"]) == "ok"
    assert mcp_server.get_shot(db, shots[0].id)["group_id"] == group["id"]
    assert mcp_server.unassign_shot_group(db, shots[0].id) == "ok"
    assert mcp_server.delete_shot_group(db, group["id"]) == "ok"
    assert mcp_server.list_shot_groups(db, asset.id) == []


def test_split_and_merge_tools(tmp_path):
    db, asset, shots, subject = _make_store(tmp_path)

    result = mcp_server.split_shot(db, shots[0].id, at_frame=10)
    first, second = result["first"], result["second"]
    assert (first["start_frame"], first["end_frame"]) == (0, 9)
    assert (second["start_frame"], second["end_frame"]) == (10, 24)

    merged = mcp_server.merge_shots(db, first["id"], second["id"])
    assert (merged["start_frame"], merged["end_frame"]) == (0, 24)
