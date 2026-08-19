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


class ProjectStore:
    """Thin wrapper around a SQLite project database."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    # -- construction ---------------------------------------------------------

    @classmethod
    def create(cls, path: str | Path = ":memory:") -> "ProjectStore":
        """Create a new project database (tables created from the schema)."""
        engine = _sqlite_engine(path)
        _enable_foreign_keys(engine)
        Base.metadata.create_all(engine)
        return cls(engine)

    @classmethod
    def open(cls, path: str | Path) -> "ProjectStore":
        """Open an existing project database without creating tables."""
        engine = _sqlite_engine(path)
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
        }
        unexpected = set(probe_fields) - known
        if unexpected:
            raise TypeError(f"unknown probe fields: {sorted(unexpected)}")

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
    shot: Shot, frame_index: int, image_path: str | None = None
) -> RepresentativeFrame:
    """Build the representative still for a shot."""
    return RepresentativeFrame(
        shot_id=shot.id,
        frame_index=frame_index,
        timecode=frames_to_timecode(frame_index, shot.asset.frame_rate),
        image_path=image_path,
    )
