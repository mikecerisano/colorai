"""Local review UI.

A server-rendered review screen: one card per detected shot, showing the
representative still, its timecode range, the stored image metrics, and any
corrections with a live corrected preview.

A small JSON API backs the approval workflow:

* ``GET  /api/shots/{id}``
* ``POST /api/shots/{id}/corrections``
* ``PATCH /api/corrections/{id}``  (toggle ``enabled`` or change parameters)
* ``DELETE /api/corrections/{id}``
* ``GET  /shots/{id}/preview.png`` (corrected still, rendered on the fly)

``httpx`` is required for the FastAPI test client (declared in the ``dev``
extra), not for running the server.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from colorai.correction import load_corrected_still, normalize_parameters, validate_correction
from colorai.project.models import (
    Correction,
    FrameMetrics,
    MediaAsset,
    Note,
    Project,
    Shot,
    SkinMetric,
    Subject,
)
from colorai.project.store import ProjectStore

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _correction_dict(c: Correction) -> dict[str, Any]:
    return {
        "id": c.id,
        "shot_id": c.shot_id,
        "kind": c.kind,
        "parameters": c.parameters,
        "enabled": c.enabled,
    }


def _validate_or_400(kind: str, parameters: dict[str, Any]) -> None:
    try:
        validate_correction(kind, parameters)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class CorrectionIn(BaseModel):
    kind: str
    parameters: dict[str, Any] = {}


class CorrectionUpdate(BaseModel):
    enabled: bool | None = None
    parameters: dict[str, Any] | None = None


class SubjectIn(BaseModel):
    name: str


class SubjectRename(BaseModel):
    name: str


class ReferenceIn(BaseModel):
    shot_id: int | None = None


class MergeIn(BaseModel):
    keep_id: int
    drop_id: int


class SkinAssign(BaseModel):
    subject_id: int | None = None


class ShotUpdate(BaseModel):
    review_status: str | None = None
    excused: bool | None = None


class SplitIn(BaseModel):
    at_frame: int


class ShotMergeIn(BaseModel):
    shot_id_a: int
    shot_id_b: int


class GroupIn(BaseModel):
    name: str


class GroupAssign(BaseModel):
    group_id: int


class ReferenceProposalIn(BaseModel):
    shot_id: int
    reason: str
    confidence: float = 1.0
    subject_id: int | None = None
    group_id: int | None = None
    author: str = "human"


class NoteIn(BaseModel):
    text: str
    author: str = "human"
    shot_id: int | None = None
    subject_id: int | None = None


def _deviation_dict(d) -> dict[str, Any]:
    return {
        "shot_id": d.shot_id,
        "luma_delta_stops": d.luma_delta_stops if math.isfinite(d.luma_delta_stops) else None,
        "is_outlier": d.is_outlier,
        "reasons": list(d.reasons),
        "corrections": [{"kind": c.kind, "parameters": c.parameters} for c in d.corrections],
    }


def create_app(store: ProjectStore, stills_dir: str | Path) -> FastAPI:
    """Build the review app backed by ``store``, serving stills from ``stills_dir``."""
    stills = Path(stills_dir).resolve()
    stills.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="ColorAI")
    app.mount("/stills", StaticFiles(directory=str(stills)), name="stills")
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    # -- pages ---------------------------------------------------------------

    @app.get("/")
    def index(request: Request):
        shots_view: list[dict] = []
        project_names: list[str] = []
        asset_id: int | None = None
        with store.session() as session:
            project_names = [p.name for p in session.query(Project).order_by(Project.id)]
            first_asset = session.query(MediaAsset).order_by(MediaAsset.id).first()
            if first_asset is not None:
                asset_id = first_asset.id
            reference_view: list[dict] = []
            if asset_id is not None:
                from colorai.project.models import ReferenceProposal

                reference_view = [
                    {
                        "id": p.id,
                        "shot_id": p.shot_id,
                        "subject_id": p.subject_id,
                        "group_id": p.group_id,
                        "author": p.author,
                        "reason": p.reason,
                        "confidence": round(p.confidence, 2),
                        "state": p.state,
                    }
                    for p in session.query(ReferenceProposal)
                    .filter_by(asset_id=asset_id)
                    .order_by(ReferenceProposal.id)
                    .all()
                ]
            corrections_by_shot: dict[int, list[Correction]] = {}
            for c in session.query(Correction).order_by(Correction.id).all():
                corrections_by_shot.setdefault(c.shot_id, []).append(c)

            for shot in session.query(Shot).order_by(Shot.asset_id, Shot.index).all():
                rf = shot.representative_frame
                if rf is None:
                    continue
                metrics = (
                    session.query(FrameMetrics)
                    .filter_by(shot_id=shot.id, frame_index=rf.frame_index)
                    .first()
                )
                still_url = "/stills/" + Path(rf.image_path).resolve().relative_to(stills).as_posix()
                corrections = corrections_by_shot.get(shot.id, [])
                shots_view.append(
                    {
                        "id": shot.id,
                        "index": shot.index,
                        "start_tc": shot.start_timecode,
                        "end_tc": shot.end_timecode,
                        "frame_count": shot.frame_count,
                        "still_url": still_url,
                        "corrections": [
                            {"kind": c.kind, "enabled": c.enabled} for c in corrections
                        ],
                        "has_corrections": any(c.enabled for c in corrections),
                        "luma_mean": _fmt(metrics.luma_mean if metrics else None),
                        "luma_std": _fmt(metrics.luma_std if metrics else None),
                        "r_mean": _fmt(metrics.r_mean if metrics else None),
                        "g_mean": _fmt(metrics.g_mean if metrics else None),
                        "b_mean": _fmt(metrics.b_mean if metrics else None),
                        "saturation_mean": _fmt(metrics.saturation_mean if metrics else None),
                    }
                )
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "projects": ", ".join(project_names) or "(none)",
                "shots": shots_view,
                "asset_id": asset_id,
                "references": reference_view,
            },
        )

    # -- correction API ------------------------------------------------------

    @app.get("/api/shots/{shot_id}")
    def get_shot(shot_id: int):
        with store.session() as session:
            shot = session.get(Shot, shot_id)
            if shot is None:
                raise HTTPException(status_code=404, detail="shot not found")
            corrections = (
                session.query(Correction)
                .filter_by(shot_id=shot_id)
                .order_by(Correction.id)
                .all()
            )
        return {
            "id": shot.id,
            "index": shot.index,
            "start_frame": shot.start_frame,
            "end_frame": shot.end_frame,
            "start_timecode": shot.start_timecode,
            "end_timecode": shot.end_timecode,
            "review_status": shot.review_status,
            "excused": shot.excused,
            "group_id": shot.group_id,
            "corrections": [_correction_dict(c) for c in corrections],
        }

    @app.patch("/api/shots/{shot_id}")
    def update_shot(shot_id: int, payload: ShotUpdate):
        from colorai.editorial import set_excused, set_review_status

        try:
            if payload.review_status is not None:
                set_review_status(store, shot_id, payload.review_status)
            if payload.excused is not None:
                set_excused(store, shot_id, payload.excused)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        with store.session() as session:
            shot = session.get(Shot, shot_id)
            if shot is None:
                raise HTTPException(status_code=404, detail="shot not found")
            return {"id": shot.id, "review_status": shot.review_status, "excused": shot.excused}

    @app.post("/api/shots/{shot_id}/split", status_code=201)
    def split_shot_endpoint(shot_id: int, payload: SplitIn):
        from colorai.editorial import split_shot as _split

        try:
            a, b = _split(store, shot_id, payload.at_frame)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"first": a.id, "second": b.id}

    @app.post("/api/shots/merge", status_code=200)
    def merge_shots_endpoint(payload: ShotMergeIn):
        from colorai.editorial import merge_shots as _merge

        try:
            merged = _merge(store, payload.shot_id_a, payload.shot_id_b)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"id": merged.id, "start_frame": merged.start_frame, "end_frame": merged.end_frame}

    # -- shot groups ---------------------------------------------------------

    @app.get("/api/assets/{asset_id}/groups")
    def list_groups_endpoint(asset_id: int):
        from colorai.editorial import list_groups as _list
        from colorai.project.models import ShotGroup

        with store.session() as session:
            if session.get(MediaAsset, asset_id) is None:
                raise HTTPException(status_code=404, detail="asset not found")
            groups = _list(store, asset_id)
            return [
                {
                    "id": g.id,
                    "name": g.name,
                    "shot_ids": [
                        s.id for s in session.query(Shot).filter_by(group_id=g.id).order_by(Shot.index).all()
                    ],
                }
                for g in groups
            ]

    @app.post("/api/assets/{asset_id}/groups", status_code=201)
    def create_group_endpoint(asset_id: int, payload: GroupIn):
        from colorai.editorial import create_group as _create

        try:
            group = _create(store, asset_id, payload.name)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"id": group.id, "name": group.name, "shot_ids": []}

    @app.patch("/api/groups/{group_id}")
    def rename_group_endpoint(group_id: int, payload: GroupIn):
        from colorai.editorial import rename_group as _rename

        group = _rename(store, group_id, payload.name)
        if group is None:
            raise HTTPException(status_code=404, detail="group not found")
        return {"id": group.id, "name": group.name}

    @app.delete("/api/groups/{group_id}", status_code=204)
    def delete_group_endpoint(group_id: int):
        from colorai.editorial import delete_group as _delete

        _delete(store, group_id)

    @app.put("/api/shots/{shot_id}/group")
    def assign_shot_group_endpoint(shot_id: int, payload: GroupAssign):
        from colorai.editorial import assign_shot_group as _assign

        shot = _assign(store, shot_id, payload.group_id)
        if shot is None:
            raise HTTPException(status_code=404, detail="shot or group not found")
        return {"id": shot.id, "group_id": shot.group_id}

    @app.delete("/api/shots/{shot_id}/group")
    def unassign_shot_group_endpoint(shot_id: int):
        from colorai.editorial import unassign_shot_group as _unassign

        shot = _unassign(store, shot_id)
        if shot is None:
            raise HTTPException(status_code=404, detail="shot not found")
        return {"id": shot.id, "group_id": shot.group_id}

    # -- reference proposals ------------------------------------------------

    @app.get("/api/assets/{asset_id}/reference-proposals")
    def list_reference_proposals_endpoint(asset_id: int):
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
            for p in _list(store, asset_id)
        ]

    @app.post("/api/assets/{asset_id}/reference-proposals", status_code=201)
    def propose_reference_endpoint(asset_id: int, payload: ReferenceProposalIn):
        from colorai.references import propose_reference as _propose

        try:
            proposal = _propose(
                store,
                asset_id=asset_id,
                shot_id=payload.shot_id,
                reason=payload.reason,
                confidence=payload.confidence,
                subject_id=payload.subject_id,
                group_id=payload.group_id,
                author=payload.author,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"id": proposal.id, "state": proposal.state, "shot_id": proposal.shot_id}

    @app.post("/api/reference-proposals/{proposal_id}/approve")
    def approve_reference_endpoint(proposal_id: int):
        from colorai.references import approve_reference as _approve

        proposal = _approve(store, proposal_id)
        if proposal is None:
            raise HTTPException(status_code=404, detail="proposal not found")
        return {"id": proposal.id, "state": proposal.state}

    @app.post("/api/reference-proposals/{proposal_id}/reject")
    def reject_reference_endpoint(proposal_id: int):
        from colorai.references import reject_reference as _reject

        proposal = _reject(store, proposal_id)
        if proposal is None:
            raise HTTPException(status_code=404, detail="proposal not found")
        return {"id": proposal.id, "state": proposal.state}

    @app.post("/api/shots/{shot_id}/corrections", status_code=201)
    def add_correction(shot_id: int, payload: CorrectionIn):
        _validate_or_400(payload.kind, payload.parameters)
        try:
            parameters = normalize_parameters(payload.kind, payload.parameters)
        except OSError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        with store.session() as session:
            if session.get(Shot, shot_id) is None:
                raise HTTPException(status_code=404, detail="shot not found")
            correction = Correction(
                shot_id=shot_id, kind=payload.kind, parameters=parameters
            )
            session.add(correction)
            session.flush()
            session.refresh(correction)
        return _correction_dict(correction)

    @app.post("/api/shots/{shot_id}/propose", status_code=201)
    def propose_for_shot(shot_id: int):
        from colorai.analysis import find_outliers, persist_proposals

        with store.session() as session:
            shot = session.get(Shot, shot_id)
            if shot is None:
                raise HTTPException(status_code=404, detail="shot not found")
            asset_id = shot.asset_id
        outliers = find_outliers(store, asset_id)
        mine = [d for d in outliers if d.shot_id == shot_id]
        created = persist_proposals(store, mine)
        return {"created": [_correction_dict(c) for c in created]}

    @app.patch("/api/corrections/{correction_id}")
    def update_correction(correction_id: int, payload: CorrectionUpdate):
        with store.session() as session:
            correction = session.get(Correction, correction_id)
            if correction is None:
                raise HTTPException(status_code=404, detail="correction not found")
            if payload.enabled is not None:
                correction.enabled = payload.enabled
            if payload.parameters is not None:
                _validate_or_400(correction.kind, payload.parameters)
                correction.parameters = payload.parameters
            session.flush()
            session.refresh(correction)
        return _correction_dict(correction)

    @app.delete("/api/corrections/{correction_id}", status_code=204)
    def delete_correction(correction_id: int):
        with store.session() as session:
            correction = session.get(Correction, correction_id)
            if correction is None:
                raise HTTPException(status_code=404, detail="correction not found")
            session.delete(correction)

    # -- consistency analysis --------------------------------------------------

    @app.get("/api/assets/{asset_id}/clip-report")
    def asset_clip_report(asset_id: int):
        from colorai.qc import shot_clip_report as _report

        with store.session() as session:
            if session.get(MediaAsset, asset_id) is None:
                raise HTTPException(status_code=404, detail="asset not found")
        return _report(store, asset_id)

    @app.get("/api/assets/{asset_id}/outliers")
    def asset_outliers(asset_id: int, reference_shot_id: int | None = None):
        from colorai.analysis import find_outliers

        with store.session() as session:
            if session.get(MediaAsset, asset_id) is None:
                raise HTTPException(status_code=404, detail="asset not found")
        try:
            outliers = find_outliers(store, asset_id, reference_shot_id=reference_shot_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"outliers": [_deviation_dict(d) for d in outliers]}

    @app.post("/api/assets/{asset_id}/apply-proposals", status_code=201)
    def apply_proposals(asset_id: int, reference_shot_id: int | None = None):
        from colorai.analysis import find_outliers, persist_proposals

        with store.session() as session:
            if session.get(MediaAsset, asset_id) is None:
                raise HTTPException(status_code=404, detail="asset not found")
        try:
            outliers = find_outliers(store, asset_id, reference_shot_id=reference_shot_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        created = persist_proposals(store, outliers)
        return {"created": [_correction_dict(c) for c in created]}

    # -- subjects, notes, tracking -------------------------------------------

    def _subject_dict(session, subject: Subject) -> dict:
        faces = (
            session.query(SkinMetric)
            .filter_by(subject_id=subject.id)
            .order_by(SkinMetric.shot_id, SkinMetric.face_index)
            .all()
        )
        timecodes = {s.id: s.start_timecode for s in session.query(Shot).all()}
        return {
            "id": subject.id,
            "name": subject.name,
            "reference_shot_id": subject.reference_shot_id,
            "faces": [
                {
                    "skin_metric_id": m.id,
                    "shot_id": m.shot_id,
                    "face_index": m.face_index,
                    "timecode": timecodes.get(m.shot_id),
                    "mean_bgr": [round(m.mean_b, 3), round(m.mean_g, 3), round(m.mean_r, 3)],
                }
                for m in faces
            ],
        }

    @app.get("/api/assets/{asset_id}/subjects")
    def list_subjects(asset_id: int):
        with store.session() as session:
            if session.get(MediaAsset, asset_id) is None:
                raise HTTPException(status_code=404, detail="asset not found")
            subjects = (
                session.query(Subject)
                .filter_by(asset_id=asset_id)
                .order_by(Subject.id)
                .all()
            )
            return [_subject_dict(session, s) for s in subjects]

    @app.post("/api/assets/{asset_id}/subjects", status_code=201)
    def create_subject(asset_id: int, payload: SubjectIn):
        from colorai.skin_analysis import create_subject as _create

        subject = _create(store, asset_id, payload.name)
        return {"id": subject.id, "name": subject.name, "reference_shot_id": None, "faces": []}

    @app.patch("/api/subjects/{subject_id}")
    def rename_subject(subject_id: int, payload: SubjectRename):
        from colorai.skin_analysis import rename_subject as _rename

        subject = _rename(store, subject_id, payload.name)
        if subject is None:
            raise HTTPException(status_code=404, detail="subject not found")
        return {"id": subject.id, "name": subject.name}

    @app.post("/api/subjects/{subject_id}/reference")
    def set_subject_reference(subject_id: int, payload: ReferenceIn):
        from colorai.skin_analysis import set_reference as _set_ref

        subject = _set_ref(store, subject_id, payload.shot_id)
        if subject is None:
            raise HTTPException(status_code=404, detail="subject not found")
        return {"id": subject.id, "reference_shot_id": subject.reference_shot_id}

    @app.post("/api/subjects/merge")
    def merge_subjects(payload: MergeIn):
        from colorai.skin_analysis import merge_subjects as _merge

        _merge(store, payload.keep_id, payload.drop_id)
        return {"ok": True}

    @app.delete("/api/subjects/{subject_id}", status_code=204)
    def delete_subject(subject_id: int):
        from colorai.skin_analysis import delete_subject as _delete

        _delete(store, subject_id)

    @app.patch("/api/skin_metrics/{skin_metric_id}")
    def assign_skin_metric(skin_metric_id: int, payload: SkinAssign):
        from colorai.skin_analysis import assign_face, unassign_face

        if payload.subject_id is None:
            unassign_face(store, skin_metric_id)
        else:
            assign_face(store, skin_metric_id, payload.subject_id)
        return {"ok": True}

    @app.get("/api/assets/{asset_id}/skin-consistency")
    def skin_consistency(asset_id: int):
        from colorai.skin_analysis import skin_consistency as _skin

        with store.session() as session:
            if session.get(MediaAsset, asset_id) is None:
                raise HTTPException(status_code=404, detail="asset not found")
        return [
            {
                "shot_id": d.shot_id,
                "face_index": d.face_index,
                "subject_id": d.subject_id,
                "distance": round(d.distance, 4),
                "is_outlier": d.is_outlier,
                "corrections": [{"kind": c.kind, "parameters": c.parameters} for c in d.corrections],
            }
            for d in _skin(store, asset_id)
        ]

    @app.get("/api/assets/{asset_id}/notes")
    def list_notes(asset_id: int):
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

    @app.post("/api/assets/{asset_id}/notes", status_code=201)
    def add_note(asset_id: int, payload: NoteIn):
        note = Note(
            asset_id=asset_id,
            shot_id=payload.shot_id,
            subject_id=payload.subject_id,
            author=payload.author,
            text=payload.text,
        )
        with store.session() as session:
            session.add(note)
            session.flush()
            session.refresh(note)
        return {"id": note.id, "author": note.author, "text": note.text}

    @app.get("/api/shots/{shot_id}/track")
    def track_shot(shot_id: int, face_index: int = 0, samples: int = 8):
        from colorai.tracking import propagate_shot_mask

        with store.session() as session:
            shot = session.get(Shot, shot_id)
            if shot is None:
                raise HTTPException(status_code=404, detail="shot not found")
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

    # -- preview -------------------------------------------------------------

    @app.get("/shots/{shot_id}/preview.png")
    def preview_image(shot_id: int):
        with store.session() as session:
            shot = session.get(Shot, shot_id)
        if shot is None:
            raise HTTPException(status_code=404, detail="shot not found")
        image = load_corrected_still(store, shot)
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise HTTPException(status_code=500, detail="failed to encode preview")
        return Response(content=encoded.tobytes(), media_type="image/png")

    return app
