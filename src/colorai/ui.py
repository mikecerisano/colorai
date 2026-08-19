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
    NameSuggestion,
    Note,
    Project,
    ReferenceProposal,
    RepresentativeFrame,
    Shot,
    ShotGroup,
    SkinMetric,
    Subject,
)
from colorai.project.store import ProjectStore

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _workspace(store: ProjectStore, asset_id: int) -> dict[str, Any]:
    """Structured data for the review UI: setups, faces, inbox, references."""
    with store.session() as session:
        subjects = (
            session.query(Subject).filter_by(asset_id=asset_id).order_by(Subject.id).all()
        )
        groups = (
            session.query(ShotGroup).filter_by(asset_id=asset_id).order_by(ShotGroup.id).all()
        )
        shots = (
            session.query(Shot).filter_by(asset_id=asset_id).order_by(Shot.index).all()
        )
        proposals = (
            session.query(ReferenceProposal).filter_by(asset_id=asset_id)
            .order_by(ReferenceProposal.id).all()
        )
        face_rows = (
            session.query(SkinMetric)
            .join(Shot, SkinMetric.shot_id == Shot.id)
            .filter(Shot.asset_id == asset_id)
            .order_by(Shot.index, SkinMetric.face_index)
            .all()
        )
        metric_rows = (
            session.query(FrameMetrics)
            .join(Shot, FrameMetrics.shot_id == Shot.id)
            .filter(Shot.asset_id == asset_id).all()
        )
        correction_rows = (
            session.query(Correction)
            .join(Shot, Correction.shot_id == Shot.id)
            .filter(Shot.asset_id == asset_id).order_by(Correction.id).all()
        )

        timecodes = {s.id: s.start_timecode for s in shots}
        metrics_by_shot = {m.shot_id: m for m in metric_rows}
        corrections_by_shot: dict[int, list[dict]] = {}
        for c in correction_rows:
            corrections_by_shot.setdefault(c.shot_id, []).append(
                {"id": c.id, "kind": c.kind, "parameters": c.parameters, "enabled": c.enabled}
            )

        def face_brief(m: SkinMetric) -> dict:
            return {
                "skin_metric_id": m.id,
                "shot_id": m.shot_id,
                "face_index": m.face_index,
                "subject_id": m.subject_id,
                "timecode": timecodes.get(m.shot_id),
                "mean_bgr": [round(m.mean_b, 3), round(m.mean_g, 3), round(m.mean_r, 3)],
                "bbox": [m.bbox_x, m.bbox_y, m.bbox_w, m.bbox_h],
            }

        def shot_brief(s: Shot) -> dict:
            m = metrics_by_shot.get(s.id)
            return {
                "id": s.id,
                "index": s.index,
                "start_tc": s.start_timecode,
                "end_tc": s.end_timecode,
                "group_id": s.group_id,
                "review_status": s.review_status,
                "excused": s.excused,
                "luma_mean": m.luma_mean if m else None,
                "saturation_mean": m.saturation_mean if m else None,
                "corrections": corrections_by_shot.get(s.id, []),
            }

        proposal_state: dict[int, str] = {}
        proposal_ref: dict[int, int] = {}
        for p in proposals:
            if p.group_id is not None:
                if p.state == "approved":
                    proposal_state[p.group_id] = "approved"
                    proposal_ref[p.group_id] = p.shot_id
                elif proposal_state.get(p.group_id) != "approved" and p.state == "suggested":
                    proposal_state[p.group_id] = "suggested"

        # A setup family's members include its direct shots plus every
        # descendant variant group's shots, so the parent Corrections view is
        # complete. Computed here (not in Jinja) because Jinja loop scope
        # discards in-loop reassignment.
        children_by_parent: dict[int, list[int]] = {}
        for g in groups:
            if g.parent_id is not None:
                children_by_parent.setdefault(g.parent_id, []).append(g.id)

        def descendant_group_ids(group_id: int) -> list[int]:
            ids = [group_id]
            for child_id in children_by_parent.get(group_id, []):
                ids.extend(descendant_group_ids(child_id))
            return ids

        member_group_ids = {g.id: descendant_group_ids(g.id) for g in groups}
        members_by_group = {
            g.id: [shot_brief(s) for s in shots if s.group_id in member_group_ids[g.id]]
            for g in groups
        }

        proposal_dicts = [
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
        ]
        proposals_by_group: dict[int, list[dict]] = {}
        for p in proposal_dicts:
            if p["group_id"] is not None:
                proposals_by_group.setdefault(p["group_id"], []).append(p)

        def active_proposal_for(group_id: int) -> dict | None:
            ps = proposals_by_group.get(group_id, [])
            approved = [p for p in ps if p["state"] == "approved"]
            if approved:
                return approved[-1]  # newest approved (consistent with references.py)
            suggested = [p for p in ps if p["state"] == "suggested"]
            return suggested[-1] if suggested else None

        active_by_group = {g.id: active_proposal_for(g.id) for g in groups}

        suggestion_rows = (
            session.query(NameSuggestion).filter_by(asset_id=asset_id).order_by(NameSuggestion.id).all()
        )

        return {
            "asset_id": asset_id,
            "subjects": [
                {
                    "id": s.id,
                    "name": s.name,
                    "name_confirmed": s.name_confirmed,
                    "reference_shot_id": s.reference_shot_id,
                    "faces": [face_brief(m) for m in face_rows if m.subject_id == s.id],
                }
                for s in subjects
            ],
            "setups": [
                {
                    "id": g.id,
                    "name": g.name,
                    "kind": g.kind,
                    "camera": g.camera,
                    "parent_id": g.parent_id,
                    "shot_ids": [s.id for s in shots if s.group_id == g.id],
                    "subject_ids": sorted(
                        {
                            m.subject_id
                            for m in face_rows
                            if m.subject_id is not None and m.shot_id in (s.id for s in shots if s.group_id == g.id)
                        }
                    ),
                    "reference_state": proposal_state.get(g.id, "none"),
                    "approved_reference_shot_id": proposal_ref.get(g.id),
                    "all_members": members_by_group[g.id],
                    "active_proposal": active_by_group[g.id],
                    "reference_history": [
                        p
                        for p in proposals_by_group.get(g.id, [])
                        if active_by_group[g.id] is None or p["id"] != active_by_group[g.id]["id"]
                    ],
                }
                for g in groups
            ],
            "unassigned_faces": [
                face_brief(m) for m in face_rows if m.subject_id is None
            ],
            "unassigned_shots": [
                shot_brief(s) for s in shots if s.group_id is None
            ],
            "shots": [shot_brief(s) for s in shots],
            "reference_proposals": proposal_dicts,
            "name_suggestions": [
                {
                    "id": s.id,
                    "subject_id": s.subject_id,
                    "shot_id": s.shot_id,
                    "candidate_name": s.candidate_name,
                    "raw_text": s.raw_text,
                    "role_text": s.role_text,
                    "confidence": round(s.confidence, 2),
                    "timecode": s.timecode,
                    "state": s.state,
                }
                for s in suggestion_rows
            ],
        }


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


class GroupCreate(BaseModel):
    name: str
    kind: str = "generic"
    camera: str | None = None
    parent_id: int | None = None


class GroupUpdate(BaseModel):
    name: str | None = None
    camera: str | None = None
    kind: str | None = None


class GroupAssign(BaseModel):
    group_id: int


class ReferenceProposalIn(BaseModel):
    shot_id: int
    reason: str
    confidence: float = 1.0
    subject_id: int | None = None
    group_id: int | None = None
    author: str = "human"


class NameSuggestionAccept(BaseModel):
    name: str | None = None


class NameSuggestionAssign(BaseModel):
    subject_id: int


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
        project_names: list[str] = []
        asset_id: int | None = None
        with store.session() as session:
            project_names = [p.name for p in session.query(Project).order_by(Project.id)]
            first_asset = session.query(MediaAsset).order_by(MediaAsset.id).first()
            if first_asset is not None:
                asset_id = first_asset.id

        workspace = _workspace(store, asset_id) if asset_id is not None else {}
        subject_names = {s["id"]: s["name"] for s in workspace.get("subjects", [])}
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "projects": ", ".join(project_names) or "(none)",
                "asset_id": asset_id,
                "ws": workspace,
                "subject_names": subject_names,
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

        with store.session() as session:
            if session.get(MediaAsset, asset_id) is None:
                raise HTTPException(status_code=404, detail="asset not found")
            groups = _list(store, asset_id)
            return [
                {
                    "id": g.id,
                    "name": g.name,
                    "kind": g.kind,
                    "camera": g.camera,
                    "parent_id": g.parent_id,
                    "shot_ids": [
                        s.id for s in session.query(Shot).filter_by(group_id=g.id).order_by(Shot.index).all()
                    ],
                }
                for g in groups
            ]

    @app.post("/api/assets/{asset_id}/groups", status_code=201)
    def create_group_endpoint(asset_id: int, payload: GroupCreate):
        from colorai.editorial import create_group as _create

        try:
            group = _create(
                store, asset_id, payload.name,
                kind=payload.kind, camera=payload.camera, parent_id=payload.parent_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "id": group.id,
            "name": group.name,
            "kind": group.kind,
            "camera": group.camera,
            "parent_id": group.parent_id,
            "shot_ids": [],
        }

    @app.patch("/api/groups/{group_id}")
    def rename_group_endpoint(group_id: int, payload: GroupUpdate):
        from colorai.editorial import update_group as _update

        try:
            group = _update(
                store, group_id,
                name=payload.name, camera=payload.camera, kind=payload.kind,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if group is None:
            raise HTTPException(status_code=404, detail="group not found")
        return {"id": group.id, "name": group.name, "kind": group.kind, "camera": group.camera, "parent_id": group.parent_id}

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

    # -- lower-third name suggestions ---------------------------------------

    @app.get("/api/name-suggestions/{suggestion_id}/crop.png")
    def name_suggestion_crop(suggestion_id: int):
        from colorai.nametag import list_suggestions

        with store.session() as session:
            suggestion = session.get(NameSuggestion, suggestion_id)
            if suggestion is None or not suggestion.crop_path:
                raise HTTPException(status_code=404, detail="suggestion crop not found")
            crop_path = suggestion.crop_path
        image = cv2.imread(crop_path, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=500, detail="cannot read crop")
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise HTTPException(status_code=500, detail="failed to encode crop")
        return Response(content=encoded.tobytes(), media_type="image/png")

    @app.post("/api/name-suggestions/{suggestion_id}/accept")
    def accept_name_suggestion_endpoint(suggestion_id: int, payload: NameSuggestionAccept):
        from colorai.nametag import accept_suggestion as _accept

        suggestion = _accept(store, suggestion_id, name=payload.name)
        if suggestion is None:
            raise HTTPException(status_code=404, detail="suggestion not found")
        return {"id": suggestion.id, "state": suggestion.state, "subject_id": suggestion.subject_id}

    @app.post("/api/name-suggestions/{suggestion_id}/ignore")
    def ignore_name_suggestion_endpoint(suggestion_id: int):
        from colorai.nametag import ignore_suggestion as _ignore

        suggestion = _ignore(store, suggestion_id)
        if suggestion is None:
            raise HTTPException(status_code=404, detail="suggestion not found")
        return {"id": suggestion.id, "state": suggestion.state}

    @app.post("/api/name-suggestions/{suggestion_id}/assign")
    def assign_name_suggestion_endpoint(suggestion_id: int, payload: NameSuggestionAssign):
        from colorai.nametag import assign_suggestion as _assign

        try:
            suggestion = _assign(store, suggestion_id, payload.subject_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if suggestion is None:
            raise HTTPException(status_code=404, detail="suggestion or subject not found")
        return {"id": suggestion.id, "subject_id": suggestion.subject_id}

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
                    "bbox": [m.bbox_x, m.bbox_y, m.bbox_w, m.bbox_h],
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

    @app.get("/shots/{shot_id}/original.png")
    def original_image(shot_id: int):
        """The uncorrected representative still (for before/after comparison)."""
        with store.session() as session:
            rf = session.query(RepresentativeFrame).filter_by(shot_id=shot_id).first()
            if rf is None or not rf.image_path:
                raise HTTPException(status_code=404, detail="no still for this shot")
            still_path = rf.image_path
        bgr = cv2.imread(still_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(status_code=500, detail="cannot read still")
        ok, encoded = cv2.imencode(".png", bgr)
        if not ok:
            raise HTTPException(status_code=500, detail="failed to encode still")
        return Response(content=encoded.tobytes(), media_type="image/png")

    # -- face crop + workspace ----------------------------------------------

    @app.get("/api/skin_metrics/{skin_metric_id}/crop.png")
    def face_crop(skin_metric_id: int):
        with store.session() as session:
            m = session.get(SkinMetric, skin_metric_id)
            if m is None:
                raise HTTPException(status_code=404, detail="face not found")
            rf = session.query(RepresentativeFrame).filter_by(shot_id=m.shot_id).first()
            if rf is None or not rf.image_path:
                raise HTTPException(status_code=404, detail="no still for this face")
            still_path = rf.image_path
            bbox = (m.bbox_x, m.bbox_y, m.bbox_w, m.bbox_h)

        image = cv2.imread(still_path, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=500, detail="cannot read still")
        h, w = image.shape[:2]
        if all(v is not None for v in bbox):
            x, y, bw, bh = bbox
            pad = int(max(bw, bh) * 0.35)
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
            if x1 > x0 and y1 > y0:
                image = image[y0:y1, x0:x1]
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise HTTPException(status_code=500, detail="failed to encode crop")
        return Response(content=encoded.tobytes(), media_type="image/png")

    @app.get("/api/assets/{asset_id}/workspace")
    def asset_workspace(asset_id: int):
        with store.session() as session:
            if session.get(MediaAsset, asset_id) is None:
                raise HTTPException(status_code=404, detail="asset not found")
        return _workspace(store, asset_id)

    return app
