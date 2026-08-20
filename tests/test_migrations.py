"""Tests for Alembic migrations and the ``db migrate`` command."""

from __future__ import annotations

from pathlib import Path

import colorai
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

from colorai.project.models import Base


def _alembic_config(db_path: Path) -> tuple[Config, str]:
    pkg_dir = Path(colorai.__file__).resolve().parent
    cfg = Config(str(pkg_dir / "alembic.ini"))
    cfg.set_main_option("script_location", str(pkg_dir / "migrations"))
    return cfg, f"sqlite+pysqlite:///{db_path}"


def test_upgrade_head_creates_full_schema(tmp_path, monkeypatch):
    db = tmp_path / "migrated.sqlite3"
    cfg, url = _alembic_config(db)
    monkeypatch.setenv("COLORAI_DB_URL", url)

    command.upgrade(cfg, "head")

    names = set(inspect(create_engine(url)).get_table_names())
    expected = {
        "projects",
        "media_assets",
        "shots",
        "representative_frames",
        "frame_metrics",
        "corrections",
        "alembic_version",
        "organization_plans",
        "organization_plan_groups",
        "organization_plan_items",
        "face_tracks",
        "face_corrections",
        "skin_appearance_references",
        "face_mask_tracks",
        "skin_appearance_targets",
    }
    assert expected <= names


def test_downgrade_base_removes_tables(tmp_path, monkeypatch):
    db = tmp_path / "migrated.sqlite3"
    cfg, url = _alembic_config(db)
    monkeypatch.setenv("COLORAI_DB_URL", url)

    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")

    names = set(inspect(create_engine(url)).get_table_names())
    assert "shots" not in names
    assert "projects" not in names


def test_db_migrate_cli(tmp_path):
    from colorai.cli import main

    db = tmp_path / "proj.sqlite3"
    assert main(["db", "migrate", "--project", str(db)]) == 0

    names = set(inspect(create_engine(f"sqlite+pysqlite:///{db}")).get_table_names())
    assert "shots" in names
    assert "corrections" in names


def test_open_legacy_db_stamps_and_preserves_data(tmp_path):
    """A pre-Alembic database (full schema, populated, no version table) opens
    without table-creation errors and keeps its data."""
    from colorai.project import ProjectStore

    db = tmp_path / "legacy.sqlite3"
    engine = create_engine(f"sqlite+pysqlite:///{db}")
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO projects (name, created_at, updated_at) "
                "VALUES ('legacy', '2026-01-01 00:00:00', '2026-01-01 00:00:00')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO media_assets (project_id, source_path, frame_rate, "
                "timecode_format, status, created_at, updated_at) VALUES "
                "(1, '/media/m.mov', 25.0, 'NDF', 'analyzed', '2026-01-01 00:00:00', '2026-01-01 00:00:00')"
            )
        )
    engine.dispose()

    store = ProjectStore.open(db)
    assert store.list_projects()[0].name == "legacy"
    with store.session() as session:
        from colorai.project.models import MediaAsset

        assert session.query(MediaAsset).first().source_path == "/media/m.mov"

    engine2 = create_engine(f"sqlite+pysqlite:///{db}")
    assert inspect(engine2).has_table("alembic_version")
    engine2.dispose()

    # Re-opening is a no-op and still preserves data.
    assert ProjectStore.open(db).list_projects()[0].name == "legacy"


def test_open_legacy_db_at_older_revision_migrates_forward(tmp_path, monkeypatch):
    """A legacy database at an older schema revision is stamped at that
    revision and upgraded forward, without losing its data."""
    from colorai.project import ProjectStore

    db = tmp_path / "old_legacy.sqlite3"
    cfg, url = _alembic_config(db)
    monkeypatch.setenv("COLORAI_DB_URL", url)

    # Schema as of the subjects revision (before notes/source-hash/editorial).
    command.upgrade(cfg, "b494e149e8b0")
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO projects (name, created_at, updated_at) "
                "VALUES ('old', '2026-01-01 00:00:00', '2026-01-01 00:00:00')"
            )
        )
    engine.dispose()

    # Strip the version stamp to simulate a pre-Alembic database.
    engine2 = create_engine(url)
    with engine2.begin() as conn:
        conn.execute(text("DROP TABLE alembic_version"))
    engine2.dispose()

    store = ProjectStore.open(db)
    assert store.list_projects()[0].name == "old"

    engine3 = create_engine(url)
    inspector = inspect(engine3)
    assert inspector.has_table("reference_proposals")
    assert inspector.has_table("organization_plans")
    assert inspector.has_table("organization_plan_groups")
    assert inspector.has_table("organization_plan_items")
    assert inspector.has_table("face_tracks")
    assert inspector.has_table("face_corrections")
    assert "review_status" in {c["name"] for c in inspector.get_columns("shots")}
    assert "source_hash" in {c["name"] for c in inspector.get_columns("media_assets")}
    engine3.dispose()


def test_previous_head_upgrades_and_face_rows_remain_readable(tmp_path, monkeypatch):
    """A database at the previous head (d3e4f5a6b7c8) upgrades to the new head
    with existing ``FaceTrack``/``FaceCorrection`` rows intact."""
    from sqlalchemy.orm import Session

    db = tmp_path / "prev_head.sqlite3"
    cfg, url = _alembic_config(db)
    monkeypatch.setenv("COLORAI_DB_URL", url)
    command.upgrade(cfg, "d3e4f5a6b7c8")

    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO projects (name, created_at, updated_at) VALUES "
                "('prev', '2026-01-01 00:00:00', '2026-01-01 00:00:00')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO media_assets (project_id, source_path, frame_rate, "
                "timecode_format, status, created_at, updated_at) VALUES "
                "(1, '/m.mov', 25.0, 'NDF', 'analyzed', '2026-01-01 00:00:00', '2026-01-01 00:00:00')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO shots (asset_id, \"index\", start_frame, end_frame, "
                "start_timecode, end_timecode, review_status, excused, created_at) "
                "VALUES (1, 0, 0, 24, '00:00:00:00', '00:00:00:24', 'pending', 0, '2026-01-01 00:00:00')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO subjects (asset_id, name, name_confirmed, created_at) "
                "VALUES (1, 'Alice', 0, '2026-01-01 00:00:00')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO skin_metrics (shot_id, subject_id, face_index, mean_b, mean_g, mean_r, "
                "sample_pixels, created_at) VALUES (1, 1, 0, 0.3, 0.3, 0.5, 10, '2026-01-01 00:00:00')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO face_tracks (shot_id, skin_metric_id, subject_id, source_width, "
                "source_height, keyframes, sample_count, tracked_count, coverage, max_gap, "
                "skin_stability, median_bgr, state, created_at) VALUES "
                "(1, 1, 1, 1920, 1080, '[ [0, 0.1, 0.1, 0.2, 0.2] ]', 1, 1, 1.0, 0.0, 0.01, '[0.3, 0.3, 0.5]', 'valid', '2026-01-01 00:00:00')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO face_corrections (shot_id, subject_id, skin_metric_id, face_track_id, "
                "kind, parameters, reason, confidence, classification, state, enabled, created_at) VALUES "
                "(1, 1, 1, 1, 'rgb_balance', '{\"gain\": [1.0, 1.0, 1.0]}', '', 0.8, 'skin_mismatch', 'suggested', 0, '2026-01-01 00:00:00')"
            )
        )
    engine.dispose()

    # Upgrade to the new head (adds skin targets + mask tracks + skin_target_id).
    command.upgrade(cfg, "head")

    inspector = inspect(create_engine(url))
    assert inspector.has_table("skin_appearance_references")
    assert inspector.has_table("face_mask_tracks")
    assert inspector.has_table("skin_appearance_targets")
    assert "skin_target_id" in {c["name"] for c in inspector.get_columns("face_corrections")}

    # Existing face rows remain readable through the ORM.
    from colorai.project.models import FaceCorrection, FaceTrack
    from colorai.project.store import ProjectStore

    with ProjectStore.open(db).session() as session:
        assert session.query(FaceTrack).count() == 1
        assert session.query(FaceCorrection).count() == 1
        assert session.query(FaceCorrection).first().kind == "rgb_balance"
        assert session.query(FaceCorrection).first().skin_target_id is None



def test_open_legacy_db_with_empty_version_table(tmp_path):
    """A populated legacy database whose failed migration attempt left an
    *empty* ``alembic_version`` table must be treated as unversioned: stamped
    at the recognized revision and upgraded — never table-recreated.

    Regression: Alembic treats an empty version table as a blank database and
    tries the initial migration again ("table projects already exists").
    """
    from colorai.project import ProjectStore

    db = tmp_path / "legacy_empty_version.sqlite3"
    engine = create_engine(f"sqlite+pysqlite:///{db}")
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO projects (name, created_at, updated_at) "
                "VALUES ('legacy', '2026-01-01 00:00:00', '2026-01-01 00:00:00')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO media_assets (project_id, source_path, frame_rate, "
                "timecode_format, status, created_at, updated_at) VALUES "
                "(1, '/media/m.mov', 25.0, 'NDF', 'analyzed', '2026-01-01 00:00:00', '2026-01-01 00:00:00')"
            )
        )
        # The leftover from a failed migration attempt: an empty version table.
        conn.execute(
            text(
                "CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL, "
                "CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num))"
            )
        )
    engine.dispose()

    store = ProjectStore.open(db)
    assert store.list_projects()[0].name == "legacy"
    with store.session() as session:
        from colorai.project.models import MediaAsset

        assert session.query(MediaAsset).first().source_path == "/media/m.mov"

    # The version table is now stamped (exactly one row), not blank.
    engine2 = create_engine(f"sqlite+pysqlite:///{db}")
    with engine2.connect() as conn:
        rows = conn.execute(text("SELECT COUNT(*) FROM alembic_version")).scalar()
    assert rows == 1
    engine2.dispose()

    # Re-opening is still a no-op with data intact.
    assert ProjectStore.open(db).list_projects()[0].name == "legacy"
