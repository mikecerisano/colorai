"""Tests for the persistent project model and store."""

from __future__ import annotations

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from colorai.project import (
    Correction,
    FrameMetrics,
    MediaAsset,
    Project,
    ProjectStore,
    RepresentativeFrame,
    Shot,
    make_representative_frame,
    make_shots,
)


@pytest.fixture
def store():
    return ProjectStore.create(":memory:")


@pytest.fixture
def project(store):
    return store.create_project("test film")


@pytest.fixture
def asset_ndf(store, project):
    return store.add_asset(
        project.id,
        source_path="/media/master.mov",
        frame_rate=25.0,
        width=1920,
        height=1080,
        frame_count=3750,
        duration_seconds=150.0,
        pixel_format="yuv420p",
        codec_name="prores",
    )


@pytest.fixture
def asset_df(store, project):
    return store.add_asset(project.id, source_path="/media/master_df.mov", frame_rate=29.97)


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

def test_create_and_list_projects(store):
    assert store.list_projects() == []
    p1 = store.create_project("first")
    p2 = store.create_project("second")
    names = [p.name for p in store.list_projects()]
    assert names == ["first", "second"]
    assert p1.id != p2.id


def test_get_project(store, project):
    assert store.get_project(project.id).name == "test film"
    assert store.get_project(9999) is None


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------

def test_add_asset_derives_timecode_format(asset_ndf, asset_df):
    assert asset_ndf.timecode_format == "NDF"
    assert asset_df.timecode_format == "DF"


def test_add_asset_stores_probe_fields(asset_ndf):
    assert asset_ndf.width == 1920
    assert asset_ndf.height == 1080
    assert asset_ndf.frame_rate == 25.0
    assert asset_ndf.frame_count == 3750
    assert asset_ndf.duration_seconds == 150.0
    assert asset_ndf.codec_name == "prores"


def test_add_asset_rejects_unknown_probe_fields(store, project):
    with pytest.raises(TypeError):
        store.add_asset(project.id, source_path="/x.mov", frame_rate=25.0, bogus=1)


def test_add_asset_requires_existing_project(store):
    with pytest.raises(IntegrityError):
        store.add_asset(9999, source_path="/x.mov", frame_rate=25.0)


# ---------------------------------------------------------------------------
# Shots
# ---------------------------------------------------------------------------

def test_make_shots_derives_timecodes(asset_ndf):
    shots = make_shots(asset_ndf, [(0, 24), (25, 49), (50, 100)])
    assert [s.index for s in shots] == [0, 1, 2]
    assert (shots[0].start_timecode, shots[0].end_timecode) == ("00:00:00:00", "00:00:00:24")
    assert shots[1].start_timecode == "00:00:01:00"
    assert shots[2].end_timecode == "00:00:04:00"
    assert shots[0].frame_count == 25
    assert shots[2].frame_count == 51


def test_make_shots_drop_frame_timecodes(asset_df):
    shots = make_shots(asset_df, [(1798, 1800)])
    # Frame 1798 -> 00:00:59;28 (last frame before the minute-1 rollover).
    assert shots[0].start_timecode == "00:00:59;28"
    assert shots[0].end_timecode == "00:01:00;02"


def test_make_shots_rejects_inverted_bounds(asset_ndf):
    with pytest.raises(ValueError):
        make_shots(asset_ndf, [(50, 25)])


def test_shots_unique_per_asset_index(store, asset_ndf):
    shots = make_shots(asset_ndf, [(0, 24), (25, 49)])
    with store.session() as session:
        session.add_all(shots)
        session.commit()
    with pytest.raises(IntegrityError):
        duplicate = make_shots(asset_ndf, [(0, 24)])
        with store.session() as session:
            session.add_all(duplicate)
            session.commit()


def test_asset_has_ordered_shots(store, asset_ndf):
    with store.session() as session:
        session.add_all(make_shots(asset_ndf, [(0, 24), (25, 49)]))
        session.commit()

    with store.session() as session:
        asset = session.get(MediaAsset, asset_ndf.id)
        assert [s.index for s in sorted(asset.shots, key=lambda s: s.index)] == [0, 1]


# ---------------------------------------------------------------------------
# Representative frames
# ---------------------------------------------------------------------------

def test_representative_frame_derives_timecode(store, asset_ndf):
    with store.session() as session:
        session.add_all(make_shots(asset_ndf, [(0, 24)]))
        session.flush()
        shot = session.query(Shot).one()
        rf = make_representative_frame(shot, 12, image_path="/still/000012.jpg")
        session.add(rf)
        session.commit()
        rf_id = rf.id

    with store.session() as session:
        rf = session.get(RepresentativeFrame, rf_id)
        assert rf.frame_index == 12
        assert rf.timecode == "00:00:00:12"
        assert rf.image_path == "/still/000012.jpg"


def test_representative_frame_one_per_shot(store, asset_ndf):
    with store.session() as session:
        session.add_all(make_shots(asset_ndf, [(0, 24)]))
        session.commit()

    with pytest.raises(IntegrityError):
        with store.session() as session:
            shot = session.query(Shot).one()
            session.add(make_representative_frame(shot, 1))
            session.add(make_representative_frame(shot, 2))
            session.commit()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_frame_metrics_nullable_fields(store, asset_ndf):
    with store.session() as session:
        session.add_all(make_shots(asset_ndf, [(0, 24)]))
        session.flush()
        shot = session.query(Shot).one()
        session.add(
            FrameMetrics(
                shot_id=shot.id,
                frame_index=0,
                luma_mean=0.5,
                luma_std=0.1,
                r_mean=0.4,
                g_mean=0.5,
                b_mean=0.6,
                saturation_mean=0.3,
            )
        )
        session.commit()

    with store.session() as session:
        m = session.query(FrameMetrics).one()
        assert m.luma_mean == pytest.approx(0.5)
        assert m.luma_p95 is None  # unset percentile stays NULL
        assert m.saturation_mean == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Corrections
# ---------------------------------------------------------------------------

def test_correction_json_roundtrip(store, asset_ndf):
    with store.session() as session:
        session.add_all(make_shots(asset_ndf, [(0, 24)]))
        session.flush()
        shot = session.query(Shot).one()
        session.add(
            Correction(
                shot_id=shot.id,
                kind="cdl",
                parameters={"slope": [1.0, 1.0, 1.0], "offset": [0.0, 0.0, 0.0]},
            )
        )
        session.commit()

    with store.session() as session:
        c = session.query(Correction).one()
        assert c.kind == "cdl"
        assert c.parameters["slope"] == [1.0, 1.0, 1.0]
        assert c.enabled is True


# ---------------------------------------------------------------------------
# Cascades + persistence
# ---------------------------------------------------------------------------

def test_deleting_asset_cascades_to_shots(store, asset_ndf):
    with store.session() as session:
        session.add_all(make_shots(asset_ndf, [(0, 24), (25, 49)]))
        session.commit()

    with store.session() as session:
        session.delete(session.get(MediaAsset, asset_ndf.id))
        session.commit()

    with store.session() as session:
        assert session.scalar(select(func.count()).select_from(Shot)) == 0


def test_file_store_roundtrip(tmp_path):
    path = tmp_path / "data" / "project.sqlite3"
    s = ProjectStore.create(path)
    p = s.create_project("persisted")
    s.add_asset(p.id, source_path="/media/m.mov", frame_rate=24.0)
    s.add_asset(p.id, source_path="/media/m2.mov", frame_rate=23.976)

    # Reopen from disk; data must survive.
    s2 = ProjectStore.open(path)
    projects = s2.list_projects()
    assert [p.name for p in projects] == ["persisted"]
    with s2.session() as session:
        rates = sorted(a.frame_rate for a in session.query(MediaAsset).all())
        assert rates == [23.976, 24.0]


def test_session_rolls_back_on_error(store):
    with pytest.raises(RuntimeError):
        with store.session() as session:
            session.add(Project(name="doomed"))
            raise RuntimeError("boom")
    assert store.list_projects() == []
