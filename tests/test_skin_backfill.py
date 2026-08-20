"""Tests for bbox-only legacy skin-metric backfill."""

from __future__ import annotations

import cv2
import numpy as np

from colorai.pipeline import analyze_master
from colorai.project import (
    ProjectStore,
    SkinMetric,
    make_representative_frame,
    make_shots,
)
from colorai.skin_analysis import backfill_missing_skin_metric_bboxes, create_subject


def _store_with_legacy_metrics(tmp_path):
    store = ProjectStore.create(":memory:")
    project = store.create_project("backfill")
    asset = store.add_asset(
        project.id, source_path="/media/m.mov", frame_rate=25.0, width=64, height=64
    )
    shots = make_shots(asset, [(0, 9), (10, 19)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)
    alice = create_subject(store, asset.id, "Alice")

    still0 = tmp_path / "s0.png"
    still1 = tmp_path / "s1.png"
    cv2.imwrite(str(still0), np.zeros((64, 64, 3), dtype=np.uint8))
    cv2.imwrite(str(still1), np.zeros((64, 64, 3), dtype=np.uint8))

    with store.session() as session:
        session.add(make_representative_frame(shots[0], 0, image_path=str(still0), frame_rate=25.0))
        session.add(make_representative_frame(shots[1], 0, image_path=str(still1), frame_rate=25.0))
        # shot 0: two legacy metrics, null bboxes, one subject-assigned.
        session.add(
            SkinMetric(shot_id=shots[0].id, face_index=0, mean_b=0.30, mean_g=0.30, mean_r=0.50,
                       sample_pixels=100, subject_id=alice.id)
        )
        session.add(
            SkinMetric(shot_id=shots[0].id, face_index=1, mean_b=0.20, mean_g=0.25, mean_r=0.40,
                       sample_pixels=100, subject_id=None)
        )
        # shot 1: one legacy metric, null bboxes.
        session.add(
            SkinMetric(shot_id=shots[1].id, face_index=0, mean_b=0.35, mean_g=0.38, mean_r=0.58,
                       sample_pixels=100, subject_id=None)
        )
        session.commit()

    with store.session() as session:
        metrics = session.query(SkinMetric).order_by(SkinMetric.id).all()
    return store, asset, shots, alice, metrics


def _detect(image):
    return [{"bbox": [10, 10, 20, 20]}, {"bbox": [30, 30, 18, 18]}]


def test_backfill_fills_only_null_bboxes_and_preserves_skin(tmp_path):
    store, asset, shots, alice, metrics = _store_with_legacy_metrics(tmp_path)

    # Give one metric an existing bbox that must be left untouched.
    with store.session() as session:
        session.query(SkinMetric).filter_by(id=metrics[0].id).update(
            {"bbox_x": 1, "bbox_y": 2, "bbox_w": 3, "bbox_h": 4}, synchronize_session=False
        )
        session.commit()

    result = backfill_missing_skin_metric_bboxes(store, asset.id, detect=_detect)
    assert result["scanned"] == 2  # metrics[1] and metrics[2] were null
    assert metrics[1].id in result["backfilled"]
    assert metrics[2].id in result["backfilled"]

    with store.session() as session:
        m0 = session.get(SkinMetric, metrics[0].id)
        m1 = session.get(SkinMetric, metrics[1].id)
        m2 = session.get(SkinMetric, metrics[2].id)

    # Existing box untouched.
    assert (m0.bbox_x, m0.bbox_y, m0.bbox_w, m0.bbox_h) == (1, 2, 3, 4)
    # Skin means, sample count, subject, and face_index are byte-for-byte stable.
    for before, after in ((metrics[0], m0), (metrics[1], m1), (metrics[2], m2)):
        assert after.mean_b == before.mean_b
        assert after.mean_g == before.mean_g
        assert after.mean_r == before.mean_r
        assert after.sample_pixels == before.sample_pixels
        assert after.subject_id == before.subject_id
        assert after.face_index == before.face_index

    assert (m1.bbox_x, m1.bbox_y, m1.bbox_w, m1.bbox_h) == (30, 30, 18, 18)
    assert (m2.bbox_x, m2.bbox_y, m2.bbox_w, m2.bbox_h) == (10, 10, 20, 20)


def test_backfill_unrecoverable_face_index_remains_unresolved(tmp_path):
    store, asset, shots, alice, metrics = _store_with_legacy_metrics(tmp_path)

    def single_face(image):
        return [{"bbox": [10, 10, 20, 20]}]

    result = backfill_missing_skin_metric_bboxes(store, asset.id, detect=single_face)
    # metrics[1] is face_index 1 but only one face is detected -> unresolved.
    unresolved_ids = [u["skin_metric_id"] for u in result["unresolved"]]
    assert metrics[1].id in unresolved_ids
    assert metrics[1].id not in result["backfilled"]
    with store.session() as session:
        m1 = session.get(SkinMetric, metrics[1].id)
    assert m1.bbox_x is None and m1.bbox_w is None


def test_resume_analysis_triggers_backfill(tmp_path, monkeypatch):
    from colorai.ingest import compute_source_hash

    master = tmp_path / "m.mov"
    master.write_bytes(b"fake master")
    source_hash = compute_source_hash(master)

    store = ProjectStore.create(":memory:")
    project = store.create_project("resume")
    asset = store.add_asset(
        project.id, source_path=str(master), frame_rate=25.0,
        width=64, height=64, source_hash=source_hash,
        analyze_params={"threshold": 27.0, "min_scene_len": 15},
    )
    with store.session() as session:
        session.query(type(asset)).filter_by(id=asset.id).update(
            {"status": "analyzed"}, synchronize_session=False
        )
        session.commit()

    shots = make_shots(asset, [(0, 9)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)
    alice = create_subject(store, asset.id, "Alice")
    still = tmp_path / "still.png"
    cv2.imwrite(str(still), np.zeros((64, 64, 3), dtype=np.uint8))
    with store.session() as session:
        session.add(make_representative_frame(shots[0], 0, image_path=str(still), frame_rate=25.0))
        session.add(
            SkinMetric(shot_id=shots[0].id, face_index=0, mean_b=0.3, mean_g=0.3, mean_r=0.5,
                       sample_pixels=10, subject_id=alice.id)
        )
        session.commit()

    calls: list[int] = []

    def spy_backfill(store_arg, asset_id):
        calls.append(asset_id)
        return {"scanned": 1, "backfilled": [], "unresolved": []}

    monkeypatch.setattr("colorai.pipeline.backfill_missing_skin_metric_bboxes", spy_backfill)

    result = analyze_master(store, project.id, str(master), stills_dir=tmp_path / "stills", resume=True)
    assert calls == [asset.id]
    # Shots were not redetected/replaced; identity/organization untouched.
    assert [s.id for s in result.shots] == [shots[0].id]
    with store.session() as session:
        metric = session.query(SkinMetric).filter_by(shot_id=shots[0].id).one()
        assert metric.subject_id == alice.id
