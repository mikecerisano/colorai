"""MCP server exposing ColorAI's deterministic engine to external agents.

This is the seam between the deterministic "body" (measure + execute) and an
LLM/agent "brain" (judge + explain). Tools split into two groups:

* **read** — inspect projects, shots, metrics, skin subjects, deviations, notes.
* **refine** — regroup faces, merge/split subjects, set references, add/toggle
  corrections, and annotate with reasoning.

Every tool is stateless over a project database path, so Claude Code, Codex, or
any MCP client can drive it with no vendor-specific integration.

Run with: ``colorai mcp`` (stdio transport).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP, Image

from colorai.project.store import ProjectStore

mcp = FastMCP(
    "colorai",
    instructions=(
        "ColorAI deterministic finishing/QC engine. Read measurements, then "
        "refine grouping and corrections with clear reasoning in notes. "
        "Never modify source media; all edits go through these tools. "
        "Agents may draft and revise organization plans, but must NOT approve "
        "or apply a plan — those are human-only decisions."
    ),
)


def _open(project: str) -> ProjectStore:
    return ProjectStore.open(Path(project))


# ---------------------------------------------------------------------------
# Read tools
# ---------------------------------------------------------------------------

@mcp.tool()
def list_projects(project: str) -> list[dict]:
    """List projects in the database at ``project`` (a .sqlite3 path)."""
    store = _open(project)
    return [{"id": p.id, "name": p.name} for p in store.list_projects()]


@mcp.tool()
def list_assets(project: str) -> list[dict]:
    """List ingested media assets."""
    from colorai.project.models import MediaAsset

    store = _open(project)
    with store.session() as session:
        return [
            {
                "id": a.id,
                "project_id": a.project_id,
                "source_path": a.source_path,
                "frame_rate": a.frame_rate,
                "frame_count": a.frame_count,
                "duration_seconds": a.duration_seconds,
                "status": a.status,
                "color_space": a.color_space,
                "transfer": a.transfer,
                "source_hash": a.source_hash,
            }
            for a in session.query(MediaAsset).order_by(MediaAsset.id).all()
        ]


@mcp.tool()
def list_shots(project: str, asset_id: int) -> list[dict]:
    """List shots for an asset with timecode bounds."""
    from colorai.project.models import Shot

    store = _open(project)
    with store.session() as session:
        return [
            {
                "id": s.id,
                "index": s.index,
                "start_frame": s.start_frame,
                "end_frame": s.end_frame,
                "start_timecode": s.start_timecode,
                "end_timecode": s.end_timecode,
                "review_status": s.review_status,
                "excused": s.excused,
                "group_id": s.group_id,
            }
            for s in session.query(Shot).filter_by(asset_id=asset_id).order_by(Shot.index).all()
        ]


@mcp.tool()
def get_shot(project: str, shot_id: int) -> dict:
    """Full detail for one shot: metrics, skin faces, corrections, notes."""
    from colorai.project.models import Correction, FrameMetrics, Note, Shot, SkinMetric

    store = _open(project)
    with store.session() as session:
        shot = session.get(Shot, shot_id)
        if shot is None:
            return {}
        metrics = session.query(FrameMetrics).filter_by(shot_id=shot_id).first()
        skin = session.query(SkinMetric).filter_by(shot_id=shot_id).order_by(SkinMetric.face_index).all()
        corrections = session.query(Correction).filter_by(shot_id=shot_id).order_by(Correction.id).all()
        notes = session.query(Note).filter_by(shot_id=shot_id).order_by(Note.id).all()
        return {
            "id": shot.id,
            "asset_id": shot.asset_id,
            "index": shot.index,
            "start_frame": shot.start_frame,
            "end_frame": shot.end_frame,
            "start_timecode": shot.start_timecode,
            "end_timecode": shot.end_timecode,
            "review_status": shot.review_status,
            "excused": shot.excused,
            "group_id": shot.group_id,
            "metrics": {
                "luma_mean": metrics.luma_mean if metrics else None,
                "luma_std": metrics.luma_std if metrics else None,
                "r_mean": metrics.r_mean if metrics else None,
                "g_mean": metrics.g_mean if metrics else None,
                "b_mean": metrics.b_mean if metrics else None,
                "saturation_mean": metrics.saturation_mean if metrics else None,
            },
            "skin_faces": [
                {
                    "id": m.id,
                    "face_index": m.face_index,
                    "subject_id": m.subject_id,
                    "mean_bgr": [round(m.mean_b, 4), round(m.mean_g, 4), round(m.mean_r, 4)],
                }
                for m in skin
            ],
            "corrections": [
                {"id": c.id, "kind": c.kind, "parameters": c.parameters, "enabled": c.enabled}
                for c in corrections
            ],
            "notes": [{"id": n.id, "author": n.author, "text": n.text} for n in notes],
        }


@mcp.tool()
def list_subjects(project: str, asset_id: int) -> list[dict]:
    """List human-editable subjects and their face counts."""
    from colorai.project.models import SkinMetric, Subject

    store = _open(project)
    with store.session() as session:
        subjects = session.query(Subject).filter_by(asset_id=asset_id).order_by(Subject.id).all()
        out = []
        for s in subjects:
            count = session.query(SkinMetric).filter_by(subject_id=s.id).count()
            out.append(
                {
                    "id": s.id,
                    "name": s.name,
                    "reference_shot_id": s.reference_shot_id,
                    "face_count": count,
                }
            )
        return out


@mcp.tool()
def skin_consistency(project: str, asset_id: int) -> list[dict]:
    """Per-subject skin-tone deviations with proposed corrections."""
    from colorai.skin_analysis import skin_consistency as _skin

    store = _open(project)
    return [
        {
            "shot_id": d.shot_id,
            "face_index": d.face_index,
            "subject_id": d.subject_id,
            "distance": round(d.distance, 4),
            "is_outlier": d.is_outlier,
            "corrections": [
                {"kind": c.kind, "parameters": c.parameters} for c in d.corrections
            ],
        }
        for d in _skin(store, asset_id)
    ]


@mcp.tool()
def propose_shot_corrections(
    project: str, asset_id: int, reference_shot_id: int | None = None
) -> list[dict]:
    """Propose luma/balance/saturation corrections for outlier shots."""
    from colorai.analysis import find_outliers

    store = _open(project)
    return [
        {
            "shot_id": o.shot_id,
            "is_outlier": o.is_outlier,
            "reasons": list(o.reasons),
            "corrections": [
                {"kind": c.kind, "parameters": c.parameters} for c in o.corrections
            ],
        }
        for o in find_outliers(store, asset_id, reference_shot_id=reference_shot_id)
    ]


@mcp.tool()
def track_shot_face(project: str, shot_id: int, face_index: int = 0, samples: int = 8) -> dict:
    """Temporally track a face across a shot and return its robust skin signature.

    Produces a temporally-stable skin sample (median over tracked frames) and a
    stability score, instead of the single representative frame's point sample.
    """
    from colorai.project.models import MediaAsset, Shot
    from colorai.tracking import propagate_shot_mask

    store = _open(project)
    with store.session() as session:
        shot = session.get(Shot, shot_id)
        if shot is None:
            return {"error": "shot not found"}
        asset = session.get(MediaAsset, shot.asset_id)

    result = propagate_shot_mask(
        asset.source_path,
        shot.start_frame,
        shot.end_frame,
        face_index,
        asset.frame_rate,
        samples=samples,
    )
    return {
        "tracked_frames": result["tracked_frames"],
        "median_bgr": result.get("median_bgr"),
        "stability": result.get("stability"),
        "mask_coverage": float(result["mask"].mean()) if "mask" in result else None,
        "error": result.get("error"),
    }


@mcp.tool()
def detect_blur_pulses(project: str, shot_id: int, samples: int = 16) -> list[dict]:
    """Detect blur pulses (short low-sharpness intervals) in a shot.

    The signature of Gyroflow-style stabilization: a few motion-blurred frames
    between sharp frames. Returns inclusive frame intervals.
    """
    from colorai.anomaly import detect_blur_pulses as _detect
    from colorai.project.models import MediaAsset, Shot

    store = _open(project)
    with store.session() as session:
        shot = session.get(Shot, shot_id)
        if shot is None:
            raise ValueError("shot not found")
        asset = session.get(MediaAsset, shot.asset_id)

    pulses = _detect(
        asset.source_path, shot.start_frame, shot.end_frame, asset.frame_rate,
        samples=samples,
    )
    return [
        {
            "start_frame": p.start_frame,
            "end_frame": p.end_frame,
            "num_frames": p.num_frames,
            "min_ratio": round(p.min_ratio, 3),
        }
        for p in pulses
    ]


@mcp.tool()
def detect_flicker(project: str, shot_id: int, samples: int = 24) -> list[dict]:
    """Detect frame-to-frame luma flicker in a shot (inclusive frame intervals)."""
    from colorai.project.models import MediaAsset, Shot
    from colorai.qc import detect_flicker as _detect

    store = _open(project)
    with store.session() as session:
        shot = session.get(Shot, shot_id)
        if shot is None:
            raise ValueError("shot not found")
        asset = session.get(MediaAsset, shot.asset_id)

    runs = _detect(
        asset.source_path, shot.start_frame, shot.end_frame, asset.frame_rate,
        samples=samples,
    )
    return [{"start_frame": a, "end_frame": b} for a, b in runs]


@mcp.tool()
def shot_clip_report(project: str, asset_id: int) -> list[dict]:
    """Per-shot highlight/shadow measurements (evidence, not defects).

    Bright windows/practicals/speculars and deep shadows are normal in a
    nearly finished master; use these signals to compare similar shots, not
    to trigger automatic fixes.
    """
    from colorai.qc import shot_clip_report as _report

    return _report(_open(project), asset_id)


@mcp.tool()
def detect_blank_frames(project: str, shot_id: int, samples: int = 24) -> list[dict]:
    """Flag near-black / near-white frames (a damaged-frame signature) in a shot."""
    from colorai.project.models import MediaAsset, Shot
    from colorai.qc import detect_blank_frames as _detect

    store = _open(project)
    with store.session() as session:
        shot = session.get(Shot, shot_id)
        if shot is None:
            raise ValueError("shot not found")
        asset = session.get(MediaAsset, shot.asset_id)

    blanks = _detect(
        asset.source_path, shot.start_frame, shot.end_frame, asset.frame_rate,
        samples=samples,
    )
    return [{"frame_index": b.frame_index, "kind": b.kind} for b in blanks]


@mcp.tool()
def generative_status() -> dict:
    """Report whether the generative restoration tier (RIFE + LaMa) is ready."""
    from colorai.generative import generative_models_status

    return generative_models_status()


@mcp.tool()
def get_shot_still(project: str, shot_id: int) -> Image:
    """Return a shot's representative frame as an image (for vision agents)."""
    from colorai.project.models import RepresentativeFrame, Shot

    store = _open(project)
    with store.session() as session:
        shot = session.get(Shot, shot_id)
        if shot is None:
            raise ValueError("shot not found")
        rf = session.query(RepresentativeFrame).filter_by(shot_id=shot_id).first()
        if rf is None or not rf.image_path:
            raise ValueError("shot has no representative frame")
        return Image(path=rf.image_path)


@mcp.tool()
def get_shot_frame(project: str, shot_id: int, frame_index: int, scale: int | None = None) -> Image:
    """Extract an arbitrary frame from a shot and return it as an image."""
    import shutil
    import tempfile

    from colorai.frames import extract_frame
    from colorai.project.models import MediaAsset, Shot

    store = _open(project)
    with store.session() as session:
        shot = session.get(Shot, shot_id)
        if shot is None:
            raise ValueError("shot not found")
        asset = session.get(MediaAsset, shot.asset_id)

    probe_dir = Path(tempfile.mkdtemp(prefix="colorai_vision_"))
    try:
        out = extract_frame(
            asset.source_path, frame_index, probe_dir / "frame.png",
            fps=asset.frame_rate, scale=scale,
        )
        return Image(data=out.read_bytes(), format="png")
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)


@mcp.tool()
def list_notes(project: str, asset_id: int) -> list[dict]:
    """List agent/human annotations for an asset."""
    from colorai.project.models import Note

    store = _open(project)
    with store.session() as session:
        return [
            {
                "id": n.id,
                "shot_id": n.shot_id,
                "subject_id": n.subject_id,
                "author": n.author,
                "text": n.text,
            }
            for n in session.query(Note).filter_by(asset_id=asset_id).order_by(Note.id).all()
        ]


# ---------------------------------------------------------------------------
# Refine tools
# ---------------------------------------------------------------------------

@mcp.tool()
def assign_face(project: str, skin_metric_id: int, subject_id: int) -> str:
    """Move a face into a subject (fix a mis-grouping)."""
    from colorai.skin_analysis import assign_face as _assign

    _assign(_open(project), skin_metric_id, subject_id)
    return "ok"


@mcp.tool()
def unassign_face(project: str, skin_metric_id: int) -> str:
    from colorai.skin_analysis import unassign_face as _unassign

    _unassign(_open(project), skin_metric_id)
    return "ok"


@mcp.tool()
def set_skin_metric(
    project: str, skin_metric_id: int, mean_b: float, mean_g: float, mean_r: float
) -> dict:
    """Override a face's sampled skin signature (fix a bad skin sample)."""
    from colorai.skin_analysis import set_skin_metric as _set

    m = _set(
        _open(project),
        skin_metric_id,
        mean_b=mean_b,
        mean_g=mean_g,
        mean_r=mean_r,
    )
    return (
        {"id": m.id, "mean_bgr": [round(m.mean_b, 4), round(m.mean_g, 4), round(m.mean_r, 4)]}
        if m
        else {"error": "skin metric not found"}
    )


@mcp.tool()
def add_skin_metric(
    project: str,
    shot_id: int,
    face_index: int,
    mean_b: float,
    mean_g: float,
    mean_r: float,
    subject_id: int | None = None,
) -> dict:
    """Add a face the detector missed, with an explicit skin signature."""
    from colorai.skin_analysis import add_skin_metric as _add

    m = _add(
        _open(project),
        shot_id,
        face_index,
        mean_b=mean_b,
        mean_g=mean_g,
        mean_r=mean_r,
        subject_id=subject_id,
    )
    return {"id": m.id, "shot_id": m.shot_id, "face_index": m.face_index, "subject_id": m.subject_id}


@mcp.tool()
def delete_skin_metric(project: str, skin_metric_id: int) -> str:
    """Remove a false-positive face."""
    from colorai.skin_analysis import delete_skin_metric as _delete

    _delete(_open(project), skin_metric_id)
    return "ok"


@mcp.tool()
def merge_subjects(project: str, keep_id: int, drop_id: int) -> str:
    """Merge two subjects (faces of ``drop_id`` move to ``keep_id``)."""
    from colorai.skin_analysis import merge_subjects as _merge

    _merge(_open(project), keep_id, drop_id)
    return "ok"


@mcp.tool()
def rename_subject(project: str, subject_id: int, name: str) -> str:
    from colorai.skin_analysis import rename_subject as _rename

    return "ok" if _rename(_open(project), subject_id, name) else "not found"


@mcp.tool()
def set_reference(project: str, subject_id: int, shot_id: int) -> str:
    """Set a subject's hero shot (its skin becomes the match target)."""
    from colorai.skin_analysis import set_reference as _set_ref

    return "ok" if _set_ref(_open(project), subject_id, shot_id) else "not found"


# ---------------------------------------------------------------------------
# Editorial tools: review state, exceptions, grouping, split/merge
# ---------------------------------------------------------------------------

@mcp.tool()
def set_shot_review_status(project: str, shot_id: int, status: str) -> str:
    """Set a shot's review status: pending / approved / rejected."""
    from colorai.editorial import set_review_status as _set

    shot = _set(_open(project), shot_id, status)
    return "ok" if shot else "not found"


@mcp.tool()
def set_shot_excused(project: str, shot_id: int, excused: bool) -> str:
    """Mark a shot as an intentional exception (outlier detection skips it)."""
    from colorai.editorial import set_excused as _set

    shot = _set(_open(project), shot_id, excused)
    return "ok" if shot else "not found"


@mcp.tool()
def split_shot(project: str, shot_id: int, at_frame: int) -> dict:
    """Split a shot at ``at_frame`` (strictly inside); returns the two new shots."""
    from colorai.editorial import split_shot as _split

    a, b = _split(_open(project), shot_id, at_frame)
    return {
        "first": {"id": a.id, "start_frame": a.start_frame, "end_frame": a.end_frame},
        "second": {"id": b.id, "start_frame": b.start_frame, "end_frame": b.end_frame},
    }


@mcp.tool()
def merge_shots(project: str, shot_id_a: int, shot_id_b: int) -> dict:
    """Merge two adjacent shots into one (lower-index shot survives)."""
    from colorai.editorial import merge_shots as _merge

    m = _merge(_open(project), shot_id_a, shot_id_b)
    return {"id": m.id, "start_frame": m.start_frame, "end_frame": m.end_frame}


@mcp.tool()
def create_shot_group(
    project: str,
    asset_id: int,
    name: str,
    kind: str = "generic",
    camera: str | None = None,
    parent_id: int | None = None,
) -> dict:
    """Create a scene/camera-family, interview/setup, or lighting-variant group.

    ``kind="setup"`` marks an interview/setup family (the matching unit);
    ``kind="variant"`` marks a lighting variant inside a family (``parent_id``
    must be the family's id). ``camera`` is an optional human-assigned angle
    label ("A", "wide", ...).
    """
    from colorai.editorial import create_group as _create

    g = _create(_open(project), asset_id, name, kind=kind, camera=camera, parent_id=parent_id)
    return {"id": g.id, "name": g.name, "kind": g.kind, "camera": g.camera, "parent_id": g.parent_id}


@mcp.tool()
def list_shot_groups(project: str, asset_id: int) -> list[dict]:
    """List shot groups (scene/camera families / setups / variants) for an asset."""
    from colorai.editorial import list_groups as _list

    return [
        {"id": g.id, "name": g.name, "kind": g.kind, "camera": g.camera, "parent_id": g.parent_id}
        for g in _list(_open(project), asset_id)
    ]


@mcp.tool()
def update_shot_group(
    project: str,
    group_id: int,
    name: str | None = None,
    camera: str | None = None,
    kind: str | None = None,
) -> dict:
    """Update a group's name, camera label, and/or kind."""
    from colorai.editorial import update_group as _update

    g = _update(_open(project), group_id, name=name, camera=camera, kind=kind)
    return (
        {"id": g.id, "name": g.name, "kind": g.kind, "camera": g.camera}
        if g
        else {"error": "group not found"}
    )


@mcp.tool()
def rename_shot_group(project: str, group_id: int, name: str) -> str:
    from colorai.editorial import rename_group as _rename

    return "ok" if _rename(_open(project), group_id, name) else "not found"


@mcp.tool()
def delete_shot_group(project: str, group_id: int) -> str:
    from colorai.editorial import delete_group as _delete

    _delete(_open(project), group_id)
    return "ok"


@mcp.tool()
def assign_shot_group(project: str, shot_id: int, group_id: int) -> str:
    """Add a shot to a group."""
    from colorai.editorial import assign_shot_group as _assign

    return "ok" if _assign(_open(project), shot_id, group_id) else "not found"


@mcp.tool()
def unassign_shot_group(project: str, shot_id: int) -> str:
    from colorai.editorial import unassign_shot_group as _unassign

    return "ok" if _unassign(_open(project), shot_id) else "not found"


# ---------------------------------------------------------------------------
# Reference proposals (human-approved) + group-aware matching
# ---------------------------------------------------------------------------

@mcp.tool()
def propose_reference(
    project: str,
    asset_id: int,
    shot_id: int,
    reason: str,
    confidence: float,
    subject_id: int | None = None,
    group_id: int | None = None,
    author: str = "agent",
) -> dict:
    """Propose a reference (hero) shot for a subject and/or setup group.

    Stays ``suggested`` until a human approves or rejects it. ``reason``
    should address framing/lighting, skin visibility, exposure stability, and
    fit with the intended setup — not clipping/crush thresholds.
    """
    from colorai.references import propose_reference as _propose

    p = _propose(
        _open(project),
        asset_id=asset_id,
        shot_id=shot_id,
        reason=reason,
        confidence=confidence,
        subject_id=subject_id,
        group_id=group_id,
        author=author,
    )
    return {"id": p.id, "state": p.state, "shot_id": p.shot_id, "subject_id": p.subject_id, "group_id": p.group_id}


@mcp.tool()
def list_reference_proposals(project: str, asset_id: int) -> list[dict]:
    """List reference proposals (suggested/approved/rejected) for an asset."""
    from colorai.references import list_reference_proposals as _list

    return [
        {
            "id": p.id,
            "subject_id": p.subject_id,
            "group_id": p.group_id,
            "shot_id": p.shot_id,
            "author": p.author,
            "reason": p.reason,
            "confidence": p.confidence,
            "state": p.state,
        }
        for p in _list(_open(project), asset_id)
    ]


@mcp.tool()
def approve_reference(project: str, proposal_id: int) -> dict:
    """Human approval of a reference proposal (makes it the effective reference)."""
    from colorai.references import approve_reference as _approve

    p = _approve(_open(project), proposal_id)
    return {"id": p.id, "state": p.state} if p else {"error": "proposal not found"}


@mcp.tool()
def reject_reference(project: str, proposal_id: int) -> dict:
    from colorai.references import reject_reference as _reject

    p = _reject(_open(project), proposal_id)
    return {"id": p.id, "state": p.state} if p else {"error": "proposal not found"}


@mcp.tool()
def match_subject_setup(
    project: str, asset_id: int, subject_id: int, group_id: int | None = None, persist: bool = False
) -> dict:
    """Compare a subject's shots to an approved reference within a scope.

    Scope is ``subject × setup group`` (optionally the group's camera label).
    Requires an approved reference; returns an explanatory error otherwise.
    Whole-frame proposals are deterministic, include reference + group context,
    and are persisted **disabled** (never auto-applied) when ``persist=True``.
    Face-derived ``skin_corrections`` are **report-only** — they need a
    tracked, feathered face mask before they can be applied, and are never
    persisted as whole-frame grades.
    """
    from colorai.matching import match_subject_in_group

    proposals, error = match_subject_in_group(
        _open(project), asset_id, subject_id=subject_id, group_id=group_id, persist=persist
    )
    if error:
        return {"error": error, "proposals": []}
    return {
        "reference_shot_id": proposals[0].reference_shot_id if proposals else None,
        "subject_id": subject_id,
        "group_id": group_id,
        "proposals": [
            {
                "shot_id": p.shot_id,
                "reference_shot_id": p.reference_shot_id,
                "group_id": p.group_id,
                "reasons": list(p.reasons),
                "corrections": [
                    {"kind": c.kind, "parameters": c.parameters} for c in p.corrections
                ],
                "skin_corrections": [
                    {"kind": c.kind, "parameters": c.parameters} for c in p.skin_corrections
                ],
            }
            for p in proposals
        ],
    }


@mcp.tool()
def cross_variant_skin_consistency(
    project: str, asset_id: int, subject_id: int, family_group_id: int
) -> dict:
    """Check a subject's skin across a setup family's lighting variants.

    Whole-frame cross-variant differences (window light, background) are
    expected and not corrected; only face/skin consistency is reported, with a
    face-region ``rgb_balance`` proposal (report-only — never persisted as a
    whole-frame grade) when a variant genuinely drifts.
    """
    from colorai.matching import cross_variant_skin_consistency as _check

    deviations, error = _check(
        _open(project), asset_id, subject_id=subject_id, family_group_id=family_group_id
    )
    if error:
        return {"error": error, "variants": []}
    return {
        "subject_id": subject_id,
        "family_group_id": family_group_id,
        "variants": [
            {
                "variant_id": d.variant_id,
                "distance": round(d.distance, 4),
                "is_issue": d.is_issue,
                "correction": {"kind": d.correction.kind, "parameters": d.correction.parameters}
                if d.correction
                else None,
            }
            for d in deviations
        ],
    }


@mcp.tool()
def matching_workspace(project: str, asset_id: int) -> dict:
    """Structured read for matching: subjects, setup groups, member shots,
    metrics, skin samples, references, and review state."""
    from colorai.project.models import (
        Correction,
        FrameMetrics,
        MediaAsset,
        ReferenceProposal,
        Shot,
        ShotGroup,
        SkinMetric,
        Subject,
    )

    store = _open(project)
    with store.session() as session:
        if session.get(MediaAsset, asset_id) is None:
            return {"error": "asset not found"}
        subjects = session.query(Subject).filter_by(asset_id=asset_id).order_by(Subject.id).all()
        groups = session.query(ShotGroup).filter_by(asset_id=asset_id).order_by(ShotGroup.id).all()
        shots = session.query(Shot).filter_by(asset_id=asset_id).order_by(Shot.index).all()
        proposals = session.query(ReferenceProposal).filter_by(asset_id=asset_id).order_by(ReferenceProposal.id).all()

        skin_by_shot: dict[int, list[dict]] = {}
        for m in session.query(SkinMetric).filter(SkinMetric.shot_id.in_([s.id for s in shots])).order_by(SkinMetric.shot_id, SkinMetric.face_index).all():
            skin_by_shot.setdefault(m.shot_id, []).append(
                {
                    "id": m.id,
                    "face_index": m.face_index,
                    "subject_id": m.subject_id,
                    "mean_bgr": [round(m.mean_b, 4), round(m.mean_g, 4), round(m.mean_r, 4)],
                }
            )
        metrics_by_shot = {
            m.shot_id: m
            for m in session.query(FrameMetrics).filter(FrameMetrics.shot_id.in_([s.id for s in shots])).all()
        }
        corrections_by_shot: dict[int, list[dict]] = {}
        for c in session.query(Correction).filter(Correction.shot_id.in_([s.id for s in shots])).order_by(Correction.id).all():
            corrections_by_shot.setdefault(c.shot_id, []).append(
                {"kind": c.kind, "parameters": c.parameters, "enabled": c.enabled}
            )

        def shot_view(s: Shot) -> dict:
            m = metrics_by_shot.get(s.id)
            return {
                "id": s.id,
                "index": s.index,
                "start_tc": s.start_timecode,
                "end_tc": s.end_timecode,
                "review_status": s.review_status,
                "excused": s.excused,
                "group_id": s.group_id,
                "metrics": {
                    "luma_mean": m.luma_mean,
                    "luma_std": m.luma_std,
                    "r_mean": m.r_mean,
                    "g_mean": m.g_mean,
                    "b_mean": m.b_mean,
                    "saturation_mean": m.saturation_mean,
                }
                if m
                else None,
                "skin": skin_by_shot.get(s.id, []),
                "corrections": corrections_by_shot.get(s.id, []),
            }

        approved_by_scope: dict[tuple, int] = {}
        for p in proposals:
            if p.state == "approved":
                approved_by_scope[(p.subject_id, p.group_id)] = p.shot_id

        return {
            "asset_id": asset_id,
            "subjects": [
                {
                    "id": s.id,
                    "name": s.name,
                    "reference_shot_id": s.reference_shot_id,
                    "approved_reference": approved_by_scope.get((s.id, None)),
                }
                for s in subjects
            ],
            "groups": [
                {
                    "id": g.id,
                    "name": g.name,
                    "kind": g.kind,
                    "camera": g.camera,
                    "parent_id": g.parent_id,
                    "member_shots": [shot_view(s) for s in shots if s.group_id == g.id],
                }
                for g in groups
            ],
            "ungrouped_shots": [shot_view(s) for s in shots if s.group_id is None],
            "reference_proposals": [
                {
                    "id": p.id,
                    "subject_id": p.subject_id,
                    "group_id": p.group_id,
                    "shot_id": p.shot_id,
                    "author": p.author,
                    "reason": p.reason,
                    "confidence": p.confidence,
                    "state": p.state,
                }
                for p in proposals
            ],
        }


# ---------------------------------------------------------------------------
# Organization planning (draft / validate / approve / apply)
# ---------------------------------------------------------------------------

@mcp.tool()
def organization_workspace(project: str, asset_id: int) -> dict:
    """Structured read for drafting an organization plan.

    Returns current groups, ungrouped/intentional/B-roll states, subjects,
    per-shot face assignments, existing approved references, and the active
    draft plus its validation summary (if any). Read-only.
    """
    from colorai.planning import (
        find_broll_group,
        list_organization_plans,
        validate_organization_plan,
    )
    from colorai.project.models import (
        MediaAsset,
        ReferenceProposal,
        Shot,
        ShotGroup,
        SkinMetric,
        Subject,
    )

    store = _open(project)
    with store.session() as session:
        if session.get(MediaAsset, asset_id) is None:
            return {"error": "asset not found"}
        subjects = session.query(Subject).filter_by(asset_id=asset_id).order_by(Subject.id).all()
        groups = session.query(ShotGroup).filter_by(asset_id=asset_id).order_by(ShotGroup.id).all()
        shots = session.query(Shot).filter_by(asset_id=asset_id).order_by(Shot.index).all()
        refs = session.query(ReferenceProposal).filter_by(asset_id=asset_id, state="approved").all()
        faces = (
            session.query(SkinMetric)
            .filter(SkinMetric.shot_id.in_([s.id for s in shots]))
            .order_by(SkinMetric.shot_id, SkinMetric.face_index)
            .all()
        )

        shot_faces: dict[int, list[dict]] = {}
        for m in faces:
            shot_faces.setdefault(m.shot_id, []).append(
                {"skin_metric_id": m.id, "subject_id": m.subject_id, "face_index": m.face_index}
            )

        broll = find_broll_group(session, asset_id)
        result = {
            "asset_id": asset_id,
            "subjects": [
                {"id": s.id, "name": s.name, "reference_shot_id": s.reference_shot_id}
                for s in subjects
            ],
            "groups": [
                {
                    "id": g.id,
                    "name": g.name,
                    "kind": g.kind,
                    "camera": g.camera,
                    "parent_id": g.parent_id,
                    "member_shot_ids": [s.id for s in shots if s.group_id == g.id],
                }
                for g in groups
            ],
            "ungrouped_shots": [
                {
                    "id": s.id,
                    "index": s.index,
                    "start_tc": s.start_timecode,
                    "end_tc": s.end_timecode,
                    "excused": s.excused,
                    "faces": shot_faces.get(s.id, []),
                }
                for s in shots if s.group_id is None
            ],
            "broll_group_id": broll.id if broll else None,
            "intentional_exception_shot_ids": [
                s.id for s in shots if s.group_id is None and s.excused
            ],
            "shot_faces": shot_faces,
            "references": [
                {"id": r.id, "subject_id": r.subject_id, "group_id": r.group_id, "shot_id": r.shot_id}
                for r in refs
            ],
        }

    plans = list_organization_plans(store, asset_id)
    active = next((p for p in plans if p["state"] in ("draft", "approved")), None)
    result["active_draft"] = active
    if active:
        result["validation_summary"] = validate_organization_plan(store, active["id"])
    return result


@mcp.tool()
def get_shot_contact_sheet(project: str, shot_ids: list[int], columns: int = 5) -> Image:
    """Return a labelled contact sheet of representative frames for visual comparison."""
    import io

    from PIL import Image as PILImage
    from PIL import ImageDraw, ImageFont

    from colorai.project.models import RepresentativeFrame, Shot

    store = _open(project)
    thumb = 256
    label_h = 22
    rows = (len(shot_ids) + columns - 1) // columns
    sheet = PILImage.new("RGB", (columns * thumb, rows * (thumb + label_h)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    with store.session() as session:
        for i, sid in enumerate(shot_ids):
            shot = session.get(Shot, sid)
            rf = session.query(RepresentativeFrame).filter_by(shot_id=sid).first()
            x = (i % columns) * thumb
            y = (i // columns) * (thumb + label_h)
            if rf and rf.image_path:
                try:
                    img = PILImage.open(rf.image_path).convert("RGB")
                    img.thumbnail((thumb, thumb))
                    sheet.paste(img, (x, y))
                except Exception:
                    pass
            label = f"shot {shot.index if shot else sid} · {shot.start_timecode if shot else ''}"
            if font:
                draw.text((x + 4, y + thumb + 4), label, fill=(255, 255, 255), font=font)

    buf = io.BytesIO()
    sheet.save(buf, format="PNG")
    return Image(data=buf.getvalue(), format="png")


@mcp.tool()
def create_organization_plan(
    project: str, asset_id: int, groups: list[dict], items: list[dict],
    summary: str = "", author: str = "agent",
) -> dict:
    """Store a draft organization plan (structurally validated; never changes the asset)."""
    from colorai.planning import create_organization_plan as _create

    try:
        plan = _create(_open(project), asset_id, groups, items, summary=summary, author=author)
        return {"id": plan.id, "state": plan.state, "asset_id": plan.asset_id}
    except ValueError as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_organization_plan(project: str, plan_id: int) -> dict:
    from colorai.planning import get_organization_plan as _get

    return _get(_open(project), plan_id) or {"error": "plan not found"}


@mcp.tool()
def list_organization_plans(project: str, asset_id: int) -> list[dict]:
    from colorai.planning import list_organization_plans as _list

    return _list(_open(project), asset_id)


@mcp.tool()
def update_organization_plan_item(
    project: str,
    plan_id: int,
    shot_id: int,
    decision: str | None = None,
    destination_type: str | None = None,
    target_group_id: int | None = None,
    target_draft_key: str | None = None,
    human_override_reason: str | None = None,
) -> dict:
    from colorai.planning import update_organization_plan_item as _update

    try:
        return _update(
            _open(project), plan_id, shot_id,
            decision=decision, destination_type=destination_type,
            target_group_id=target_group_id, target_draft_key=target_draft_key,
            human_override_reason=human_override_reason,
        ) or {"error": "not found"}
    except ValueError as exc:
        return {"error": str(exc)}


@mcp.tool()
def update_organization_plan_group(
    project: str,
    plan_id: int,
    draft_key: str,
    name: str | None = None,
    camera: str | None = None,
    kind: str | None = None,
    parent_draft_key: str | None = None,
) -> dict:
    from colorai.planning import update_organization_plan_group as _update

    try:
        return _update(
            _open(project), plan_id, draft_key,
            name=name, camera=camera, kind=kind, parent_draft_key=parent_draft_key,
        ) or {"error": "not found"}
    except ValueError as exc:
        return {"error": str(exc)}


@mcp.tool()
def validate_organization_plan(project: str, plan_id: int) -> dict:
    from colorai.planning import validate_organization_plan as _validate

    return _validate(_open(project), plan_id)


# ---------------------------------------------------------------------------
# Skin-first multicamera matching (draft-only agent surface)
# ---------------------------------------------------------------------------

@mcp.tool()
def skin_matching_workspace(
    project: str, asset_id: int, subject_id: int, group_id: int
) -> dict:
    """Structured read for skin-first matching: reference, member shots, face
    boxes/crops, stored tracks, temporal skin summaries, and local proposals."""
    from colorai.project.models import (
        FaceCorrection,
        FaceTrack,
        ReferenceProposal,
        Shot,
        ShotGroup,
        SkinMetric,
    )
    from colorai.references import approved_reference_for_scope

    store = _open(project)
    ref_shot_id = approved_reference_for_scope(
        store, asset_id=asset_id, subject_id=subject_id, group_id=group_id
    )
    with store.session() as session:
        group = session.get(ShotGroup, group_id)
        if group is None or group.asset_id != asset_id:
            return {"error": "group not found for this asset"}
        member_ids = [
            s.id for s in session.query(Shot).filter_by(group_id=group_id).order_by(Shot.index).all()
        ]
        faces = (
            session.query(SkinMetric)
            .filter(SkinMetric.subject_id == subject_id, SkinMetric.shot_id.in_(member_ids))
            .order_by(SkinMetric.shot_id, SkinMetric.face_index)
            .all()
        )
        tracks = {
            f.id: session.query(FaceTrack)
            .filter_by(skin_metric_id=f.id)
            .order_by(FaceTrack.id.desc())
            .first()
            for f in faces
        }
        proposals = (
            session.query(FaceCorrection)
            .filter(FaceCorrection.subject_id == subject_id, FaceCorrection.shot_id.in_(member_ids))
            .order_by(FaceCorrection.id)
            .all()
        )
        return {
            "asset_id": asset_id,
            "subject_id": subject_id,
            "group_id": group_id,
            "reference_shot_id": ref_shot_id,
            "member_shots": [
                {"id": s.id, "index": s.index, "start_tc": s.start_timecode, "end_tc": s.end_timecode}
                for s in session.query(Shot).filter(Shot.id.in_(member_ids)).order_by(Shot.index).all()
            ],
            "faces": [
                {
                    "skin_metric_id": f.id,
                    "shot_id": f.shot_id,
                    "face_index": f.face_index,
                    "mean_bgr": [round(f.mean_b, 4), round(f.mean_g, 4), round(f.mean_r, 4)],
                    "bbox": [f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h],
                    "track": _track_brief(tracks.get(f.id)),
                }
                for f in faces
            ],
            "proposals": [_fc_brief(p) for p in proposals],
        }


def _track_brief(t) -> dict | None:
    if t is None:
        return None
    return {
        "id": t.id,
        "state": t.state,
        "coverage": t.coverage,
        "max_gap": t.max_gap,
        "skin_stability": t.skin_stability,
        "median_bgr": t.median_bgr,
        "tracked_count": t.tracked_count,
        "sample_count": t.sample_count,
        "failure_reason": t.failure_reason,
    }


def _fc_brief(c) -> dict:
    return {
        "id": c.id,
        "shot_id": c.shot_id,
        "subject_id": c.subject_id,
        "skin_metric_id": c.skin_metric_id,
        "face_track_id": c.face_track_id,
        "reference_shot_id": c.reference_shot_id,
        "reference_group_id": c.reference_group_id,
        "classification": c.classification,
        "reason": c.reason,
        "confidence": c.confidence,
        "parameters": c.parameters,
        "state": c.state,
        "enabled": c.enabled,
    }


@mcp.tool()
def build_face_track(project: str, skin_metric_id: int, samples: int = 16) -> dict:
    """Derive and persist a temporal face track (does not enable any grade)."""
    from colorai.face_corrections import build_face_track as _build

    try:
        t = _build(_open(project), skin_metric_id, samples=samples)
        return {
            "id": t.id,
            "state": t.state,
            "coverage": t.coverage,
            "max_gap": t.max_gap,
            "skin_stability": t.skin_stability,
            "tracked_count": t.tracked_count,
            "failure_reason": t.failure_reason,
        }
    except ValueError as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_face_track_contact_sheet(project: str, face_track_id: int) -> Image:
    """Return labelled tracked samples with face boxes for visual inspection."""
    import io

    from PIL import Image as PILImage
    from PIL import ImageDraw

    from colorai.frames import extract_frame
    from colorai.project.models import FaceTrack, MediaAsset, Shot

    store = _open(project)
    with store.session() as session:
        track = session.get(FaceTrack, face_track_id)
        if track is None:
            raise ValueError("face track not found")
        shot = session.get(Shot, track.shot_id)
        asset = session.get(MediaAsset, shot.asset_id)
        keyframes = list(track.keyframes or [])[:8]

    import tempfile

    probe_dir = Path(tempfile.mkdtemp(prefix="colorai_track_sheet_"))
    try:
        thumbs = []
        for fi, nx, ny, nw, nh in keyframes:
            still = extract_frame(asset.source_path, fi, probe_dir / f"{fi}.png", fps=asset.frame_rate, scale=320)
            img = PILImage.open(still).convert("RGB")
            draw = ImageDraw.Draw(img)
            w, h = img.size
            box = (nx * w, ny * h, (nx + nw) * w, (ny + nh) * h)
            draw.rectangle(box, outline=(255, 255, 0), width=2)
            thumbs.append(img)
        cols = min(4, len(thumbs))
        rows = (len(thumbs) + cols - 1) // cols if cols else 0
        tw, th = (thumbs[0].size if thumbs else (320, 180))
        sheet = PILImage.new("RGB", (cols * tw, rows * th), (24, 24, 24))
        for i, img in enumerate(thumbs):
            sheet.paste(img, ((i % cols) * tw, (i // cols) * th))
        buf = io.BytesIO()
        sheet.save(buf, format="PNG")
        return Image(data=buf.getvalue(), format="png")
    finally:
        import shutil

        shutil.rmtree(probe_dir, ignore_errors=True)


@mcp.tool()
def skin_first_match_subject_setup(
    project: str, asset_id: int, subject_id: int, group_id: int
) -> dict:
    """Deterministic face-local skin evidence only (never persists a correction)."""
    from colorai.matching import skin_first_match_subject_setup as _match

    result, error = _match(_open(project), asset_id, subject_id, group_id)
    return {"error": error, **result} if error else result


@mcp.tool()
def propose_face_correction(
    project: str,
    shot_id: int,
    subject_id: int,
    skin_metric_id: int | None,
    face_track_id: int,
    reference_shot_id: int | None,
    reference_group_id: int | None,
    reason: str,
    confidence: float,
    classification: str,
    gain: list[float],
) -> dict:
    """Draft a suggested, disabled face correction (only ``skin_mismatch``)."""
    from colorai.face_corrections import propose_face_correction as _propose

    try:
        c = _propose(
            _open(project),
            shot_id=shot_id,
            subject_id=subject_id,
            skin_metric_id=skin_metric_id,
            face_track_id=face_track_id,
            reference_shot_id=reference_shot_id,
            reference_group_id=reference_group_id,
            reason=reason,
            confidence=confidence,
            classification=classification,
            gain=tuple(gain),
        )
        return {"id": c.id, "state": c.state, "enabled": c.enabled}
    except ValueError as exc:
        return {"error": str(exc)}


@mcp.tool()
def list_face_corrections(project: str, shot_id: int | None = None) -> list[dict]:
    from colorai.face_corrections import list_face_corrections as _list

    return _list(_open(project), shot_id=shot_id)


@mcp.tool()
def get_face_correction(project: str, correction_id: int) -> dict:
    from colorai.face_corrections import get_face_correction as _get

    return _get(_open(project), correction_id) or {"error": "not found"}


@mcp.tool()
def update_face_correction(
    project: str,
    correction_id: int,
    reason: str | None = None,
    confidence: float | None = None,
    classification: str | None = None,
    gain: list[float] | None = None,
) -> dict:
    from colorai.face_corrections import update_face_correction as _update

    try:
        return _update(
            _open(project), correction_id,
            reason=reason, confidence=confidence, classification=classification,
            gain=tuple(gain) if gain else None,
        ) or {"error": "not found or not suggested"}
    except ValueError as exc:
        return {"error": str(exc)}


@mcp.tool()
def add_correction(project: str, shot_id: int, kind: str, parameters: dict) -> dict:
    """Add a deterministic correction to a shot.

    For ``lut`` corrections, the referenced ``.cube`` file's content hash is
    recorded in the parameters.
    """
    from colorai.correction import normalize_parameters, validate_correction
    from colorai.project.models import Correction

    validate_correction(kind, parameters)
    try:
        parameters = normalize_parameters(kind, parameters)
    except OSError as exc:
        return {"error": str(exc)}
    store = _open(project)
    with store.session() as session:
        from colorai.project.models import Shot

        if session.get(Shot, shot_id) is None:
            return {"error": "shot not found"}
        correction = Correction(shot_id=shot_id, kind=kind, parameters=parameters)
        session.add(correction)
        session.flush()
        session.refresh(correction)
        return {"id": correction.id, "kind": correction.kind, "enabled": correction.enabled}


@mcp.tool()
def toggle_correction(project: str, correction_id: int, enabled: bool) -> str:
    from colorai.project.models import Correction

    with _open(project).session() as session:
        correction = session.get(Correction, correction_id)
        if correction is None:
            return "not found"
        correction.enabled = enabled
    return "ok"


@mcp.tool()
def delete_correction(project: str, correction_id: int) -> str:
    from colorai.project.models import Correction

    with _open(project).session() as session:
        correction = session.get(Correction, correction_id)
        if correction is None:
            return "not found"
        session.delete(correction)
    return "ok"


@mcp.tool()
def add_note(
    project: str,
    asset_id: int,
    text: str,
    author: str = "agent",
    shot_id: int | None = None,
    subject_id: int | None = None,
) -> dict:
    """Record reasoning attached to an asset/shot/subject."""
    from colorai.project.models import Note

    note = Note(
        asset_id=asset_id, shot_id=shot_id, subject_id=subject_id, author=author, text=text
    )
    with _open(project).session() as session:
        session.add(note)
        session.flush()
        session.refresh(note)
        return {"id": note.id, "author": note.author, "text": note.text}


@mcp.tool()
def ocr_status() -> dict:
    """Report whether lower-third OCR (Tesseract) is available."""
    from colorai.nametag import ocr_status as _status

    return _status()


@mcp.tool()
def list_name_suggestions(project: str, asset_id: int) -> list[dict]:
    """List lower-third name suggestions (evidence, not identity truth)."""
    from colorai.nametag import list_suggestions as _list

    return [
        {
            "id": s.id,
            "subject_id": s.subject_id,
            "shot_id": s.shot_id,
            "candidate_name": s.candidate_name,
            "raw_text": s.raw_text,
            "role_text": s.role_text,
            "confidence": s.confidence,
            "timecode": s.timecode,
            "crop_path": s.crop_path,
            "state": s.state,
        }
        for s in _list(_open(project), asset_id)
    ]


@mcp.tool()
def accept_name_suggestion(
    project: str, suggestion_id: int, name: str | None = None
) -> dict:
    """Accept a name suggestion (optionally with a human-edited name).

    Never overwrites a human-confirmed subject name.
    """
    from colorai.nametag import accept_suggestion as _accept

    s = _accept(_open(project), suggestion_id, name=name)
    return {"id": s.id, "state": s.state, "candidate_name": s.candidate_name} if s else {"error": "not found"}


@mcp.tool()
def ignore_name_suggestion(project: str, suggestion_id: int) -> dict:
    from colorai.nametag import ignore_suggestion as _ignore

    s = _ignore(_open(project), suggestion_id)
    return {"id": s.id, "state": s.state} if s else {"error": "not found"}


@mcp.tool()
def assign_name_suggestion(project: str, suggestion_id: int, subject_id: int) -> dict:
    """Attach an unassigned (multi-person) suggestion to a subject for review."""
    from colorai.nametag import assign_suggestion as _assign

    s = _assign(_open(project), suggestion_id, subject_id)
    return {"id": s.id, "subject_id": s.subject_id} if s else {"error": "not found"}


@mcp.tool()
def generate_name_suggestions(project: str, asset_id: int) -> dict:
    """Re-run lower-third OCR over an asset's shots and persist suggestions."""
    from colorai.nametag import extract_and_store_suggestions, ocr_available
    from colorai.project.models import MediaAsset, RepresentativeFrame, Shot

    if not ocr_available():
        return {"error": "tesseract not installed", "created": 0}
    store = _open(project)
    with store.session() as session:
        asset = session.get(MediaAsset, asset_id)
        if asset is None:
            return {"error": "asset not found", "created": 0}
        shots = session.query(Shot).filter_by(asset_id=asset_id).order_by(Shot.index).all()
        frames = (
            session.query(RepresentativeFrame)
            .join(Shot, RepresentativeFrame.shot_id == Shot.id)
            .filter(Shot.asset_id == asset_id)
            .order_by(Shot.index)
            .all()
        )
    crops_dir = Path(project).parent / "crops" / f"asset_{asset_id:04d}"
    created = extract_and_store_suggestions(store, asset, shots, frames, crops_dir)
    return {"created": len(created)}


@mcp.tool()
def analyze_master(
    project: str, master: str, project_name: str | None = None, resume: bool = True
) -> dict:
    """Run the full pipeline on a master and persist results.

    ``resume=True`` reuses a previous analysis when the master is unchanged.
    """
    from colorai.pipeline import analyze_master as run_analysis

    path = Path(project)
    store = ProjectStore.create(path)
    projects = store.list_projects()
    if projects:
        project_id = projects[0].id
    else:
        project_id = store.create_project(project_name or Path(master).stem).id

    result = run_analysis(
        store, project_id, master, stills_dir=path.parent / "stills", resume=resume
    )
    return {
        "asset_id": result.asset.id,
        "shots": len(result.shots),
        "representative_frames": len(result.representative_frames),
        "metrics": len(result.metrics),
        "skin_faces": len(result.skin_metrics),
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
