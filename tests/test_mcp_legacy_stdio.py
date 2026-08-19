"""End-to-end MCP stdio smoke for the legacy-database bootstrap.

Spawns the real ``colorai mcp`` server as a subprocess and drives it over the
stdio JSON-RPC transport, exactly as Codex does. Regression: a populated
legacy database with an *empty* ``alembic_version`` table used to make
``list_projects`` fail with "table projects already exists".
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from colorai.project.models import Base

mcp = pytest.importorskip("mcp", reason="mcp client not installed")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402


def _colorai_bin() -> Path:
    """The installed ``colorai`` entry point next to the running interpreter."""
    bin_dir = Path(shutil.which("python") or "").parent
    candidate = bin_dir / "colorai"
    if candidate.exists():
        return candidate
    # Fallback: repo-local venv layout.
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / ".venv" / "bin" / "colorai"


def _legacy_empty_version_db(tmp_path: Path) -> Path:
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
        # The leftover from a failed migration attempt: an empty version table.
        conn.execute(
            text(
                "CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL, "
                "CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num))"
            )
        )
    engine.dispose()
    return db


async def _call_list_projects(project_db: Path, colorai_bin: Path) -> list[dict]:
    params = StdioServerParameters(command=str(colorai_bin), args=["mcp"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await asyncio.wait_for(
                session.call_tool("list_projects", {"project": str(project_db)}),
                timeout=60,
            )
            projects: list[dict] = []
            for block in result.content:
                if getattr(block, "type", None) != "text":
                    continue
                value = json.loads(block.text)
                # FastMCP emits one JSON object per list item.
                if isinstance(value, list):
                    projects.extend(value)
                else:
                    projects.append(value)
            return projects


def test_list_projects_over_stdio_mcp_on_legacy_db(tmp_path):
    db = _legacy_empty_version_db(tmp_path)
    projects = asyncio.run(_call_list_projects(db, _colorai_bin()))
    assert projects == [{"id": 1, "name": "legacy"}]

    # The version table is now stamped at head, and data is intact.
    engine = create_engine(f"sqlite+pysqlite:///{db}")
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT COUNT(*) FROM alembic_version")).scalar()
        names = conn.execute(text("SELECT name FROM projects")).scalars().all()
    assert rows == 1
    assert names == ["legacy"]
    engine.dispose()
