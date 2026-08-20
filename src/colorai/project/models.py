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

    # Robust source identity (fast content fingerprint) so re-analysis can
    # recognize the same master and resume instead of reprocessing.
    source_hash: Mapped[str | None] = mapped_column(String(128))
    # The shot-detection parameters used when this asset was last analyzed;
    # resume only reuses results when they match.
    analyze_params: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

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

    # Editorial state: human review/approval and intentional-exception marker.
    review_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending"
    )
    excused: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Scene / camera-family grouping (human-editable).
    group_id: Mapped[int | None] = mapped_column(
        ForeignKey("shot_groups.id", ondelete="SET NULL"), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    asset: Mapped[MediaAsset] = relationship(back_populates="shots")
    group: Mapped["ShotGroup | None"] = relationship(back_populates="shots")
    representative_frame: Mapped["RepresentativeFrame | None"] = relationship(
        back_populates="shot", cascade="all, delete-orphan", uselist=False
    )

    @property
    def frame_count(self) -> int:
        """Number of frames in the shot (inclusive bounds)."""
        return self.end_frame - self.start_frame + 1


class ShotGroup(Base):
    """A human-editable scene / camera-family grouping of shots.

    Lets the reviewer cluster shots that share a scene or a camera setup (e.g.
    "interview cam A"), independent of the automatic shot detection and the
    per-person ``Subject`` grouping.

    ``kind`` is ``"setup"`` for an interview/setup family (the unit for
    group-aware matching) or ``"generic"`` for any other grouping. ``camera``
    is an optional angle label ("A", "wide", "closeup", ...) assigned by the
    human/agent — never inferred from pixels.
    """

    __tablename__ = "shot_groups"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    asset_id: Mapped[int] = mapped_column(
        ForeignKey("media_assets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False, default="generic")
    camera: Mapped[str | None] = mapped_column(String(64))
    # For ``kind="variant"``: the parent setup family this lighting variant
    # belongs to. ``NULL`` for top-level groups (``generic``/``setup``).
    parent_id: Mapped[int | None] = mapped_column(
        ForeignKey("shot_groups.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    asset: Mapped[MediaAsset] = relationship()
    shots: Mapped[list["Shot"]] = relationship(back_populates="group")


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
    # True once a human (or an explicitly accepted suggestion) set the name;
    # a lower-third suggestion never overwrites a confirmed name.
    name_confirmed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
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
    # Detected face box in the representative still (pixels, ``(x, y, w, h)``),
    # so the review UI can render a crop/overlay for multi-person frames.
    bbox_x: Mapped[int | None] = mapped_column(Integer)
    bbox_y: Mapped[int | None] = mapped_column(Integer)
    bbox_w: Mapped[int | None] = mapped_column(Integer)
    bbox_h: Mapped[int | None] = mapped_column(Integer)
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


class ReferenceProposal(Base):
    """A human-approvable reference-shot proposal for a matching scope.

    A vision agent proposes a hero shot for a subject and/or setup group with
    a reason and confidence; the proposal stays ``suggested`` until a human
    explicitly approves or rejects it. An ``approved`` proposal is the
    effective reference for group-aware matching in that scope.

    State machine: ``suggested`` -> ``approved`` | ``rejected`` (no automatic
    transitions). Nothing else is applied from a proposal.
    """

    __tablename__ = "reference_proposals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    asset_id: Mapped[int] = mapped_column(
        ForeignKey("media_assets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    subject_id: Mapped[int | None] = mapped_column(
        ForeignKey("subjects.id", ondelete="SET NULL"), nullable=True
    )
    group_id: Mapped[int | None] = mapped_column(
        ForeignKey("shot_groups.id", ondelete="SET NULL"), nullable=True
    )
    shot_id: Mapped[int] = mapped_column(
        ForeignKey("shots.id", ondelete="CASCADE"), nullable=False, index=True
    )
    author: Mapped[str] = mapped_column(String(64), nullable=False, default="agent")
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="suggested")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )

    asset: Mapped[MediaAsset] = relationship()
    subject: Mapped["Subject | None"] = relationship()
    group: Mapped["ShotGroup | None"] = relationship()
    shot: Mapped[Shot] = relationship()


class NameSuggestion(Base):
    """A lower-third OCR name candidate for a subject.

    Lower-thirds are **evidence, not identity truth**: a suggestion is
    ``suggested`` until a human accepts or ignores it. Accepting never
    overwrites a human-confirmed subject name, and multi-person shots are
    associated conservatively (``subject_id`` stays ``NULL`` until a human
    assigns it). Role/affiliation text is stored separately from the name.
    """

    __tablename__ = "name_suggestions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    asset_id: Mapped[int] = mapped_column(
        ForeignKey("media_assets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    subject_id: Mapped[int | None] = mapped_column(
        ForeignKey("subjects.id", ondelete="SET NULL"), nullable=True, index=True
    )
    shot_id: Mapped[int] = mapped_column(
        ForeignKey("shots.id", ondelete="CASCADE"), nullable=False, index=True
    )
    candidate_name: Mapped[str] = mapped_column(String(255), nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    role_text: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    timecode: Mapped[str] = mapped_column(String(16), nullable=False)
    crop_path: Mapped[str | None] = mapped_column(String(4096))
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="suggested")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    asset: Mapped[MediaAsset] = relationship()
    subject: Mapped["Subject | None"] = relationship()
    shot: Mapped[Shot] = relationship()


class OrganizationPlan(Base):
    """A durable, reviewable editorial-organization proposal for an asset.

    An agent drafts a plan; a human reviews and approves it; applying the
    approved plan is an atomic transaction that creates groups/variants and
    moves shots. States: ``draft`` -> ``approved`` -> ``applied``, with a
    later un-applied draft marking earlier ones ``superseded``. Applied plans
    are immutable audit records.
    """

    __tablename__ = "organization_plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    asset_id: Mapped[int] = mapped_column(
        ForeignKey("media_assets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="draft")
    author: Mapped[str] = mapped_column(String(64), nullable=False, default="agent")
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    approved_by: Mapped[str | None] = mapped_column(String(64))
    approved_at: Mapped[datetime | None] = mapped_column(DateTime)
    applied_at: Mapped[datetime | None] = mapped_column(DateTime)

    asset: Mapped[MediaAsset] = relationship()
    groups: Mapped[list["OrganizationPlanGroup"]] = relationship(
        back_populates="plan", cascade="all, delete-orphan"
    )
    items: Mapped[list["OrganizationPlanItem"]] = relationship(
        back_populates="plan", cascade="all, delete-orphan"
    )


class OrganizationPlanGroup(Base):
    """A planned setup or lighting variant inside an organization plan.

    ``draft_key`` is a stable within-plan identifier; ``parent_draft_key``
    points at another planned group for a variant. ``existing_group_id`` links
    a planned destination to a current group (for reusing/renaming).
    ``participant_ids`` are the subjects present in that setup/variant.
    """

    __tablename__ = "organization_plan_groups"
    __table_args__ = (
        UniqueConstraint("plan_id", "draft_key", name="uq_org_plan_group_draft_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    plan_id: Mapped[int] = mapped_column(
        ForeignKey("organization_plans.id", ondelete="CASCADE"), nullable=False, index=True
    )
    draft_key: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False, default="setup")
    camera: Mapped[str | None] = mapped_column(String(64))
    parent_draft_key: Mapped[str | None] = mapped_column(String(64))
    existing_group_id: Mapped[int | None] = mapped_column(
        ForeignKey("shot_groups.id", ondelete="SET NULL")
    )
    participant_ids: Mapped[list[int] | None] = mapped_column(JSON, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)

    plan: Mapped[OrganizationPlan] = relationship(back_populates="groups")


class OrganizationPlanItem(Base):
    """One shot's proposed destination inside an organization plan.

    ``decision`` is ``proposed`` / ``accepted`` / ``rejected``; the unique
    ``(plan_id, shot_id)`` key guarantees a single destination per draft.
    ``destination_type`` and the target group/draft-key fields record what
    apply should do for an accepted item.
    """

    __tablename__ = "organization_plan_items"
    __table_args__ = (
        UniqueConstraint("plan_id", "shot_id", name="uq_org_plan_item_shot"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    plan_id: Mapped[int] = mapped_column(
        ForeignKey("organization_plans.id", ondelete="CASCADE"), nullable=False, index=True
    )
    shot_id: Mapped[int] = mapped_column(
        ForeignKey("shots.id", ondelete="CASCADE"), nullable=False, index=True
    )
    decision: Mapped[str] = mapped_column(String(16), nullable=False, default="proposed")
    destination_type: Mapped[str] = mapped_column(String(32), nullable=False)
    target_group_id: Mapped[int | None] = mapped_column(
        ForeignKey("shot_groups.id", ondelete="SET NULL")
    )
    target_draft_key: Mapped[str | None] = mapped_column(String(64))
    reason: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    evidence: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    human_override_reason: Mapped[str | None] = mapped_column(Text)

    plan: Mapped[OrganizationPlan] = relationship(back_populates="items")
    shot: Mapped[Shot] = relationship(foreign_keys=[shot_id])


class FaceTrack(Base):
    """A persisted temporal face track for one detected face in a shot.

    Records sampled keyframes (normalized ``[x, y, w, h]`` boxes relative to
    the source frame), quality metrics, and state. Built from the same
    subject/face selected on the representative still; used by the shared mask
    compositor for preview and render. The source asset stays read-only.
    """

    __tablename__ = "face_tracks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    shot_id: Mapped[int] = mapped_column(
        ForeignKey("shots.id", ondelete="CASCADE"), nullable=False, index=True
    )
    skin_metric_id: Mapped[int] = mapped_column(
        ForeignKey("skin_metrics.id", ondelete="CASCADE"), nullable=False, index=True
    )
    subject_id: Mapped[int | None] = mapped_column(
        ForeignKey("subjects.id", ondelete="SET NULL"), nullable=True, index=True
    )
    source_width: Mapped[int] = mapped_column(Integer, nullable=False)
    source_height: Mapped[int] = mapped_column(Integer, nullable=False)
    analysis_scale: Mapped[int | None] = mapped_column(Integer)
    keyframes: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tracked_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    coverage: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    max_gap: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    skin_stability: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    median_bgr: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="valid")
    failure_reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    shot: Mapped[Shot] = relationship()


class FaceCorrection(Base):
    """A human-reviewable, face-local correction for one subject in a shot.

    Separate from whole-frame ``Correction`` rows. Version one supports only
    ``kind="rgb_balance"`` with per-channel linear gains clamped to
    ``[0.90, 1.10]``. ``enabled`` can become true only after approval in the
    review UI; agents may only draft/revise suggestions.
    """

    __tablename__ = "face_corrections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    shot_id: Mapped[int] = mapped_column(
        ForeignKey("shots.id", ondelete="CASCADE"), nullable=False, index=True
    )
    subject_id: Mapped[int | None] = mapped_column(
        ForeignKey("subjects.id", ondelete="SET NULL"), nullable=True, index=True
    )
    skin_metric_id: Mapped[int | None] = mapped_column(
        ForeignKey("skin_metrics.id", ondelete="SET NULL"), nullable=True
    )
    face_track_id: Mapped[int | None] = mapped_column(
        ForeignKey("face_tracks.id", ondelete="SET NULL"), nullable=True
    )
    reference_shot_id: Mapped[int | None] = mapped_column(
        ForeignKey("shots.id", ondelete="SET NULL"), nullable=True
    )
    reference_group_id: Mapped[int | None] = mapped_column(
        ForeignKey("shot_groups.id", ondelete="SET NULL"), nullable=True
    )
    kind: Mapped[str] = mapped_column(String(64), nullable=False, default="rgb_balance")
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    evidence: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    classification: Mapped[str] = mapped_column(String(32), nullable=False, default="skin_mismatch")
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="suggested")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    shot: Mapped[Shot] = relationship(foreign_keys=[shot_id])


# Silence unused-import lint for re-exported names.
__all__ = [
    "Base",
    "Project",
    "MediaAsset",
    "Shot",
    "ShotGroup",
    "RepresentativeFrame",
    "FrameMetrics",
    "Correction",
    "SkinMetric",
    "Subject",
    "Note",
    "ReferenceProposal",
    "NameSuggestion",
    "OrganizationPlan",
    "OrganizationPlanGroup",
    "OrganizationPlanItem",
    "FaceTrack",
    "FaceCorrection",
    "utcnow",
]
