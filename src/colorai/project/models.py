"""Persistent project model for ColorAI.

Every analysis and correction is represented explicitly in this schema; the
source master is never modified. Timeline coordinates are stored as *both* a
zero-based absolute frame number (canonical) and a SMPTE timecode string
(denormalized for humans and for rate-ambiguity safety), matching the
convention in :mod:`colorai.core.timecode`.

The schema is intentionally small and explicit:

* :class:`Project`      — top-level container for one finishing session.
* :class:`MediaAsset`   — an ingested source master and its probed metadata.
* :class:`Shot`         — a detected shot, a contiguous ``[start, end]`` frame
                          interval covering part of one asset.
* :class:`RepresentativeFrame` — one still chosen to represent a shot.
* :class:`FrameMetrics` — image statistics measured on a representative frame.
* :class:`Correction`   — a deterministic, temporally stable grade, stored as
                          a ``kind`` discriminator plus a JSON parameter blob.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    """Naive UTC timestamp.

    Stored tz-naive (UTC) to keep SQLite storage and comparisons simple and
    unambiguous; document any future change to aware datetimes here.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    """Declarative base shared by all project models."""


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )

    assets: Mapped[list["MediaAsset"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class MediaAsset(Base):
    __tablename__ = "media_assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Identity of the source master.
    source_path: Mapped[str] = mapped_column(String(4096), nullable=False)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer)

    # Probed stream metadata (from ffprobe). ``frame_rate`` is the nominal
    # rate used to derive timecode; the rest are descriptive.
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    frame_rate: Mapped[float] = mapped_column(Float, nullable=False)
    frame_count: Mapped[int | None] = mapped_column(Integer)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    pixel_format: Mapped[str | None] = mapped_column(String(32))
    color_space: Mapped[str | None] = mapped_column(String(32))
    transfer: Mapped[str | None] = mapped_column(String(32))
    codec_name: Mapped[str | None] = mapped_column(String(64))

    # "NDF" or "DF", derived from ``frame_rate``.
    timecode_format: Mapped[str] = mapped_column(String(3), nullable=False, default="NDF")

    # Lifecycle: registered -> ingested -> analyzed.
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="registered")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )

    project: Mapped[Project] = relationship(back_populates="assets")
    shots: Mapped[list["Shot"]] = relationship(
        back_populates="asset", cascade="all, delete-orphan"
    )


class Shot(Base):
    __tablename__ = "shots"
    __table_args__ = (
        UniqueConstraint("asset_id", "index", name="uq_shot_asset_index"),
        Index("ix_shots_asset_frames", "asset_id", "start_frame", "end_frame"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    asset_id: Mapped[int] = mapped_column(
        ForeignKey("media_assets.id", ondelete="CASCADE"), nullable=False, index=True
    )

    index: Mapped[int] = mapped_column(Integer, nullable=False)  # 0-based ordinal
    start_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    end_frame: Mapped[int] = mapped_column(Integer, nullable=False)  # inclusive
    start_timecode: Mapped[str] = mapped_column(String(16), nullable=False)
    end_timecode: Mapped[str] = mapped_column(String(16), nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    asset: Mapped[MediaAsset] = relationship(back_populates="shots")
    representative_frame: Mapped["RepresentativeFrame | None"] = relationship(
        back_populates="shot", cascade="all, delete-orphan", uselist=False
    )

    @property
    def frame_count(self) -> int:
        """Number of frames in the shot (inclusive bounds)."""
        return self.end_frame - self.start_frame + 1


class RepresentativeFrame(Base):
    __tablename__ = "representative_frames"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    shot_id: Mapped[int] = mapped_column(
        ForeignKey("shots.id", ondelete="CASCADE"), nullable=False, unique=True
    )

    frame_index: Mapped[int] = mapped_column(Integer, nullable=False)
    timecode: Mapped[str] = mapped_column(String(16), nullable=False)
    image_path: Mapped[str | None] = mapped_column(String(4096))

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    shot: Mapped[Shot] = relationship(back_populates="representative_frame")


class FrameMetrics(Base):
    __tablename__ = "frame_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    shot_id: Mapped[int] = mapped_column(
        ForeignKey("shots.id", ondelete="CASCADE"), nullable=False, index=True
    )
    frame_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # Luminance statistics (0..1, or native code values if normalized=False).
    luma_min: Mapped[float | None] = mapped_column(Float)
    luma_p5: Mapped[float | None] = mapped_column(Float)
    luma_mean: Mapped[float | None] = mapped_column(Float)
    luma_median: Mapped[float | None] = mapped_column(Float)
    luma_p95: Mapped[float | None] = mapped_column(Float)
    luma_max: Mapped[float | None] = mapped_column(Float)
    luma_std: Mapped[float | None] = mapped_column(Float)

    # Per-channel means (0..1).
    r_mean: Mapped[float | None] = mapped_column(Float)
    g_mean: Mapped[float | None] = mapped_column(Float)
    b_mean: Mapped[float | None] = mapped_column(Float)

    # Simple chroma/saturation proxy (mean chroma magnitude, 0..1).
    saturation_mean: Mapped[float | None] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    shot: Mapped[Shot] = relationship()


class Correction(Base):
    __tablename__ = "corrections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    shot_id: Mapped[int] = mapped_column(
        ForeignKey("shots.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Discriminator for the deterministic operation: "cdl", "exposure",
    # "offset", "white_balance", "contrast", "saturation", "hue_rotate", ...
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )

    shot: Mapped[Shot] = relationship()


class Subject(Base):
    """A human-editable person/group within an asset.

    Auto-assignment (face identity embeddings) produces the initial grouping;
    the human owns it from there — renaming, merging, splitting, and picking a
    ``reference_shot_id`` whose skin tone is the target for that subject.
    """

    __tablename__ = "subjects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    asset_id: Mapped[int] = mapped_column(
        ForeignKey("media_assets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    reference_shot_id: Mapped[int | None] = mapped_column(
        ForeignKey("shots.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    asset: Mapped[MediaAsset] = relationship()
    faces: Mapped[list["SkinMetric"]] = relationship(back_populates="subject")


class SkinMetric(Base):
    """Skin-tone signature of one face in a shot's representative frame.

    ``mean_b/g/r`` are the mean skin-pixel channel values normalized to
    ``[0, 1]`` (BGR order). ``face_index`` is the 0-based ordinal of the face
    within the still (a two-person shot has rows 0 and 1). ``subject_id`` is
    the human-editable grouping.
    """

    __tablename__ = "skin_metrics"
    __table_args__ = (
        UniqueConstraint("shot_id", "face_index", name="uq_skin_shot_face"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    shot_id: Mapped[int] = mapped_column(
        ForeignKey("shots.id", ondelete="CASCADE"), nullable=False, index=True
    )
    subject_id: Mapped[int | None] = mapped_column(
        ForeignKey("subjects.id", ondelete="SET NULL"), nullable=True, index=True
    )
    face_index: Mapped[int] = mapped_column(Integer, nullable=False)
    mean_b: Mapped[float] = mapped_column(Float, nullable=False)
    mean_g: Mapped[float] = mapped_column(Float, nullable=False)
    mean_r: Mapped[float] = mapped_column(Float, nullable=False)
    sample_pixels: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    shot: Mapped[Shot] = relationship()
    subject: Mapped["Subject | None"] = relationship(back_populates="faces")


class Note(Base):
    """Agent (or human) annotation attached to an asset, shot, or subject.

    This is where an LLM/agent's reasoning lives: it can explain *why* it
    regrouped faces, flagged a shot, or proposed a correction, so the
    filmmaker reviews judgments, not just numbers.
    """

    __tablename__ = "notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    asset_id: Mapped[int] = mapped_column(
        ForeignKey("media_assets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    shot_id: Mapped[int | None] = mapped_column(
        ForeignKey("shots.id", ondelete="CASCADE"), nullable=True
    )
    subject_id: Mapped[int | None] = mapped_column(
        ForeignKey("subjects.id", ondelete="CASCADE"), nullable=True
    )
    author: Mapped[str] = mapped_column(String(64), nullable=False, default="agent")
    text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


# Silence unused-import lint for re-exported names.
__all__ = [
    "Base",
    "Project",
    "MediaAsset",
    "Shot",
    "RepresentativeFrame",
    "FrameMetrics",
    "Correction",
    "SkinMetric",
    "Subject",
    "Note",
    "utcnow",
]
