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

from colorai.correction import load_corrected_still, validate_correction
from colorai.project.models import (
    Correction,
    FrameMetrics,
    MediaAsset,
    Project,
    Shot,
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
        with store.session() as session:
            project_names = [p.name for p in session.query(Project).order_by(Project.id)]
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
            {"projects": ", ".join(project_names) or "(none)", "shots": shots_view},
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
            "corrections": [_correction_dict(c) for c in corrections],
        }

    @app.post("/api/shots/{shot_id}/corrections", status_code=201)
    def add_correction(shot_id: int, payload: CorrectionIn):
        _validate_or_400(payload.kind, payload.parameters)
        with store.session() as session:
            if session.get(Shot, shot_id) is None:
                raise HTTPException(status_code=404, detail="shot not found")
            correction = Correction(
                shot_id=shot_id, kind=payload.kind, parameters=payload.parameters
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
