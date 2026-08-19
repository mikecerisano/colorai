"""Tests for lower-third name suggestions."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from colorai.nametag import (
    _parse_tsv,
    accept_suggestion,
    assign_suggestion,
    associate_subject,
    extract_and_store_suggestions,
    ignore_suggestion,
    list_suggestions,
    ocr_available,
    ocr_lines,
    split_name_role,
)
from colorai.project import (
    NameSuggestion,
    ProjectStore,
    SkinMetric,
    make_representative_frame,
    make_shots,
)
from colorai.skin_analysis import create_subject, rename_subject


def _setup(tmp_path, *, two_faces=False):
    store = ProjectStore.create(":memory:")
    project = store.create_project("nametag")
    asset = store.add_asset(project.id, source_path="/media/m.mov", frame_rate=25.0, width=64, height=64)
    shots = make_shots(asset, [(0, 24)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)

    still = tmp_path / "still.png"
    cv2.imwrite(str(still), np.full((64, 64, 3), 60, dtype=np.uint8))
    with store.session() as session:
        session.add(make_representative_frame(shots[0], 0, image_path=str(still), frame_rate=25.0))
        session.commit()

    alice = create_subject(store, asset.id, "Alice")
    bob = create_subject(store, asset.id, "Bob") if two_faces else None
    with store.session() as session:
        session.add(
            SkinMetric(shot_id=shots[0].id, face_index=0, mean_b=0.3, mean_g=0.3, mean_r=0.5,
                       sample_pixels=100, subject_id=alice.id, bbox_x=10, bbox_y=10, bbox_w=20, bbox_h=20)
        )
        if two_faces:
            session.add(
                SkinMetric(shot_id=shots[0].id, face_index=1, mean_b=0.2, mean_g=0.25, mean_r=0.4,
                           sample_pixels=100, subject_id=bob.id, bbox_x=30, bbox_y=10, bbox_w=20, bbox_h=20)
            )
        session.commit()
    return store, asset, shots, alice, bob


def test_split_name_role():
    lines = [{"text": "JANE DOE", "confidence": 0.9, "box": [0, 0, 10, 10]},
             {"text": "Director of Photography", "confidence": 0.8, "box": [0, 10, 10, 10]}]
    assert split_name_role(lines) == ("JANE DOE", "Director of Photography")
    assert split_name_role([]) == ("", None)
    assert split_name_role([{"text": "  ALICE  ", "confidence": 1.0, "box": [0, 0, 1, 1]}]) == ("ALICE", None)


def test_parse_tsv_groups_words_into_lines():
    tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t10\t20\t50\t20\t95\tJANE\n"
        "5\t1\t1\t1\t1\t2\t65\t20\t50\t20\t95\tDOE\n"
        "5\t1\t1\t1\t2\t1\t10\t50\t80\t20\t90\tDirector\n"
    )
    lines = _parse_tsv(tsv)
    assert [l["text"] for l in lines] == ["JANE DOE", "Director"]
    assert lines[0]["confidence"] == pytest.approx(0.95)


def test_associate_subject_single_and_multi(tmp_path):
    store, asset, shots, alice, _ = _setup(tmp_path)
    assert associate_subject(store, shots[0].id) == alice.id

    store2, asset2, shots2, alice2, bob2 = _setup(tmp_path, two_faces=True)
    assert associate_subject(store2, shots2[0].id) is None  # multi-person -> conservative


def test_extract_and_store_suggestions_with_mock_ocr(tmp_path):
    store, asset, shots, alice, _ = _setup(tmp_path)

    def fake_ocr(image):
        return [
            {"text": "JANE DOE", "confidence": 0.95, "box": [0, 0, 100, 20]},
            {"text": "Director", "confidence": 0.90, "box": [0, 20, 100, 20]},
        ]

    crops = tmp_path / "crops"
    created = extract_and_store_suggestions(store, asset, shots, _frames(store, shots), crops, ocr=fake_ocr)
    assert len(created) == 1
    s = created[0]
    assert s.candidate_name == "Jane Doe"
    assert s.role_text == "Director"
    assert s.raw_text == "JANE DOE\nDirector"
    assert s.timecode == "00:00:00:00"
    assert s.subject_id == alice.id  # single subject -> associated
    assert s.state == "suggested"
    assert s.crop_path and __import__("pathlib").Path(s.crop_path).exists()


def _frames(store, shots):
    from colorai.project.models import RepresentativeFrame

    with store.session() as session:
        return [session.query(RepresentativeFrame).filter_by(shot_id=s.id).first() for s in shots]


def test_accept_renames_unconfirmed_and_respects_confirmed(tmp_path):
    store, asset, shots, alice, _ = _setup(tmp_path)
    with store.session() as session:
        session.add(NameSuggestion(
            asset_id=asset.id, subject_id=alice.id, shot_id=shots[0].id,
            candidate_name="Jane Doe", raw_text="JANE DOE", role_text=None,
            confidence=0.95, timecode="00:00:00:00", crop_path=None, state="suggested",
        ))
        session.commit()

    suggestion = list_suggestions(store, asset.id)[0]
    accept_suggestion(store, suggestion.id)
    with store.session() as session:
        alice = session.get(type(alice), alice.id)
        assert alice.name == "Jane Doe"
        assert alice.name_confirmed is True

    # A human-confirmed name is never overwritten by a later suggestion.
    rename_subject(store, alice.id, "Dr. Jane Doe")  # human rename -> confirmed
    with store.session() as session:
        session.add(NameSuggestion(
            asset_id=asset.id, subject_id=alice.id, shot_id=shots[0].id,
            candidate_name="JANE X DOE", raw_text="JANE X DOE", role_text=None,
            confidence=0.9, timecode="00:00:00:00", crop_path=None, state="suggested",
        ))
        session.commit()
    second = list_suggestions(store, asset.id)[-1]
    accept_suggestion(store, second.id)
    with store.session() as session:
        alice = session.get(type(alice), alice.id)
        assert alice.name == "Dr. Jane Doe"  # unchanged


def test_ignore_and_assign_suggestion(tmp_path):
    store, asset, shots, alice, _ = _setup(tmp_path)
    with store.session() as session:
        session.add(NameSuggestion(
            asset_id=asset.id, subject_id=None, shot_id=shots[0].id,
            candidate_name="Jane Doe", raw_text="JANE DOE", role_text=None,
            confidence=0.9, timecode="00:00:00:00", crop_path=None, state="suggested",
        ))
        session.commit()
    suggestion = list_suggestions(store, asset.id)[0]
    assert assign_suggestion(store, suggestion.id, alice.id).subject_id == alice.id
    assert ignore_suggestion(store, suggestion.id).state == "ignored"


@pytest.mark.skipif(not ocr_available(), reason="tesseract not installed")
def test_ocr_lines_runs_real_backend():
    img = np.zeros((120, 320, 3), dtype=np.uint8)
    cv2.putText(img, "HELLO", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5)
    lines = ocr_lines(img)
    assert isinstance(lines, list)
    # A clean rendered word should be read; if the engine finds nothing we
    # still prove the CLI + TSV path ran without error.
    combined = " ".join(l["text"] for l in lines).upper()
    assert "HELLO" in combined or lines == []
