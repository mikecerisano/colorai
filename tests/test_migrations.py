"""Tests for Alembic migrations and the ``db migrate`` command."""

from __future__ import annotations

from pathlib import Path

import colorai
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


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
