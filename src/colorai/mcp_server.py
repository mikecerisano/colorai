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

from mcp.server.fastmcp import FastMCP

from colorai.project.store import ProjectStore

mcp = FastMCP(
    "colorai",
    instructions=(
        "ColorAI deterministic finishing/QC engine. Read measurements, then "
        "refine grouping and corrections with clear reasoning in notes. "
        "Never modify source media; all edits go through these tools."
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


@mcp.tool()
def add_correction(project: str, shot_id: int, kind: str, parameters: dict) -> dict:
    """Add a deterministic correction to a shot."""
    from colorai.correction import validate_correction
    from colorai.project.models import Correction

    validate_correction(kind, parameters)
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
def analyze_master(project: str, master: str, project_name: str | None = None) -> dict:
    """Run the full pipeline on a master and persist results."""
    from colorai.pipeline import analyze_master as run_analysis

    path = Path(project)
    store = ProjectStore.create(path)
    projects = store.list_projects()
    if projects:
        project_id = projects[0].id
    else:
        project_id = store.create_project(project_name or Path(master).stem).id

    result = run_analysis(store, project_id, master, stills_dir=path.parent / "stills")
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
