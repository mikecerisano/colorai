"""Project database access.

A :class:`ProjectStore` wraps a SQLAlchemy engine for a single SQLite project
file and owns schema creation. The store is deliberately thin: it exposes a
transactional ``session()`` context manager plus a few high-level helpers for
the operations that appear early in the pipeline. Richer domain logic (ingest,
shot detection, metrics) lives in their own modules and uses ``session()``.

Timeline construction helpers centralize the frame -> timecode derivation so
callers never hand-write an inconsistent timecode string.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from colorai.color import WORKING_PRIMARIES, WORKING_TRANSFER
from colorai.core.timecode import frames_to_timecode, is_drop_frame
from colorai.project.models import Base, MediaAsset, Project, RepresentativeFrame, Shot


def _sqlite_engine(path: str | Path) -> Engine:
    if str(path) == ":memory:":
        # A single in-memory database shared across all connections in tests.
        return create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite+pysqlite:///{db_path.as_posix()}")


def _enable_foreign_keys(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _record):  # noqa: ANN001
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def _legacy_chain(inspector) -> list[tuple[str, bool]]:
    """Marker predicates per schema revision (oldest to newest).

    Used to recognize a pre-Alembic database's schema version so it can be
    stamped at the right point before upgrading forward. Each predicate checks
    for the tables/columns that revision introduced.
    """
    tables = set(inspector.get_table_names())

    def columns(table: str) -> set[str]:
        if table not in tables:
            return set()
        return {c["name"] for c in inspector.get_columns(table)}

    has_initial = {
        "projects", "media_assets", "shots", "corrections",
        "frame_metrics", "representative_frames",
    } <= tables
    return [
        ("17b6ba1a84c7", has_initial),
        ("3336e58df071", "skin_metrics" in tables),
        ("b494e149e8b0", "subjects" in tables and "subject_id" in columns("skin_metrics")),
        ("0026a3722cec", "notes" in tables),
        ("c7d3e8a1f2b4", "source_hash" in columns("media_assets")),
        ("b5e2d9c3a4f6", "shot_groups" in tables and "review_status" in columns("shots")),
        ("d8f4e6a1c2b3", "reference_proposals" in tables and "kind" in columns("shot_groups")),
        ("e7a2b4c5d6f8", "parent_id" in columns("shot_groups")),
        ("f3c5d7e9a1b2", "bbox_x" in columns("skin_metrics")),
        ("a6b8c9d1e2f3", "name_suggestions" in tables and "name_confirmed" in columns("subjects")),
        ("c1d2e3f4a5b6", "organization_plans" in tables and "organization_plan_groups" in tables and "organization_plan_items" in tables),
        ("d3e4f5a6b7c8", "face_tracks" in tables and "face_corrections" in tables),
    ]


def _migrate(path: Path) -> None:
    """Apply Alembic migrations to a file-based project database.

    Idempotent: ``upgrade head`` is a no-op when already current, and applies
    any pending revisions when opening an older database.

    Pre-Alembic databases (populated tables, no ``alembic_version`` row) are
    recognized and stamped at the matching revision first, so opening them
    never tries to recreate existing tables or touches their data.
    """
    import os

    import colorai
    from alembic import command
    from alembic.config import Config

    path.parent.mkdir(parents=True, exist_ok=True)
    pkg_dir = Path(colorai.__file__).resolve().parent
    cfg = Config(str(pkg_dir / "alembic.ini"))
    cfg.set_main_option("script_location", str(pkg_dir / "migrations"))
    os.environ["COLORAI_DB_URL"] = f"sqlite+pysqlite:///{path.resolve().as_posix()}"

    _bootstrap_legacy(path, cfg)
    command.upgrade(cfg, "head")


def _bootstrap_legacy(path: Path, cfg: object) -> None:
    """Stamp a legacy (pre-Alembic) database at its matching revision.

    A legacy database is one with populated tables but no *current* version:
    either no ``alembic_version`` table at all, or an empty one left behind by
    a failed migration attempt (which Alembic would misread as a blank
    database). We only read the schema, stamp the version table, and let the
    normal upgrade path apply whatever is genuinely missing. Never drops or
    rewrites data.
    """
    from alembic import command
    from sqlalchemy import create_engine, inspect, text

    if not path.exists() or path.stat().st_size == 0:
        return

    engine = create_engine(f"sqlite+pysqlite:///{path.resolve().as_posix()}")
    try:
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        if not tables:
            return  # fresh empty file; upgrade head creates everything

        version_rows = 0
        if "alembic_version" in tables:
            with engine.connect() as conn:
                version_rows = conn.execute(
                    text("SELECT COUNT(*) FROM alembic_version")
                ).scalar()
        if version_rows > 0:
            return  # already managed; upgrade head applies pending revisions

        # Legacy: unversioned, or an empty version table from a failed run.
        chain = _legacy_chain(inspector)
        if not chain[0][1]:
            raise RuntimeError(
                f"{path} has tables but does not look like a ColorAI database; "
                "refusing to guess its schema version"
            )
        stamp: str | None = None
        for revision, present in chain:
            if present:
                stamp = revision
            else:
                break
        command.stamp(cfg, stamp)
    finally:
        engine.dispose()


class ProjectStore:
    """Thin wrapper around a SQLite project database."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    # -- construction ---------------------------------------------------------

    @classmethod
    def create(cls, path: str | Path = ":memory:") -> "ProjectStore":
        """Create (or migrate) a project database.

        In-memory databases use ``create_all`` (test convenience); file
        databases are created via Alembic migrations so the schema is
        versioned and existing databases are upgraded in place.
        """
        if str(path) == ":memory:":
            engine = _sqlite_engine(path)
            _enable_foreign_keys(engine)
            Base.metadata.create_all(engine)
            return cls(engine)
        db_path = Path(path)
        _migrate(db_path)
        engine = _sqlite_engine(db_path)
        _enable_foreign_keys(engine)
        return cls(engine)

    @classmethod
    def open(cls, path: str | Path) -> "ProjectStore":
        """Open an existing project database, applying pending migrations."""
        db_path = Path(path)
        _migrate(db_path)
        engine = _sqlite_engine(db_path)
        _enable_foreign_keys(engine)
        return cls(engine)

    # -- transactions ---------------------------------------------------------

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Transactional session; commits on success, rolls back on error."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    # -- projects -------------------------------------------------------------

    def create_project(self, name: str) -> Project:
        project = Project(name=name)
        with self.session() as session:
            session.add(project)
            session.flush()
            session.refresh(project)
        return project

    def list_projects(self) -> list[Project]:
        with self.session() as session:
            return list(session.query(Project).order_by(Project.id).all())

    def get_project(self, project_id: int) -> Project | None:
        with self.session() as session:
            return session.get(Project, project_id)

    # -- assets ---------------------------------------------------------------

    def add_asset(
        self,
        project_id: int,
        *,
        source_path: str,
        frame_rate: float,
        **probe_fields: object,
    ) -> MediaAsset:
        """Register a source master against a project.

        ``timecode_format`` is derived from ``frame_rate``. Any probed stream
        metadata may be passed as keyword arguments (``width``, ``height``,
        ``frame_count``, ``duration_seconds``, ``pixel_format``, ...).
        """
        known = {
            "file_size_bytes",
            "width",
            "height",
            "frame_count",
            "duration_seconds",
            "pixel_format",
            "color_space",
            "transfer",
            "codec_name",
            "source_hash",
            "analyze_params",
        }
        unexpected = set(probe_fields) - known
        if unexpected:
            raise TypeError(f"unknown probe fields: {sorted(unexpected)}")

        # Default the working color space/transfer so the grading assumption is
        # explicit in the row, not implicit in the code. ffprobe values are
        # canonicalized by the probe, but direct callers may omit them.
        probe_fields.setdefault("color_space", WORKING_PRIMARIES)
        probe_fields.setdefault("transfer", WORKING_TRANSFER)

        asset = MediaAsset(
            project_id=project_id,
            source_path=source_path,
            frame_rate=frame_rate,
            timecode_format="DF" if is_drop_frame(frame_rate) else "NDF",
            **probe_fields,
        )
        with self.session() as session:
            session.add(asset)
            session.flush()
            session.refresh(asset)
        return asset


def make_shots(asset: MediaAsset, boundaries: Iterable[tuple[int, int]]) -> list[Shot]:
    """Build ordered :class:`Shot` rows for ``(start_frame, end_frame)`` bounds.

    Bounds are inclusive. Timecodes are derived from ``asset.frame_rate``, so a
    caller cannot accidentally store a frame/timecode mismatch.
    """
    shots: list[Shot] = []
    for index, (start, end) in enumerate(boundaries):
        if end < start:
            raise ValueError(f"shot boundary end < start: {(start, end)}")
        shots.append(
            Shot(
                asset_id=asset.id,
                index=index,
                start_frame=start,
                end_frame=end,
                start_timecode=frames_to_timecode(start, asset.frame_rate),
                end_timecode=frames_to_timecode(end, asset.frame_rate),
            )
        )
    return shots


def make_representative_frame(
    shot: Shot,
    frame_index: int,
    image_path: str | None = None,
    *,
    frame_rate: float | None = None,
) -> RepresentativeFrame:
    """Build the representative still for a shot.

    ``frame_rate`` defaults to the shot's asset rate; pass it explicitly when
    the shot is detached from its session (avoids a lazy relationship load).
    """
    fps = frame_rate if frame_rate is not None else shot.asset.frame_rate
    return RepresentativeFrame(
        shot_id=shot.id,
        frame_index=frame_index,
        timecode=frames_to_timecode(frame_index, fps),
        image_path=image_path,
    )
