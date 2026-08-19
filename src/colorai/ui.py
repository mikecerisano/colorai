"""Local review UI.

A minimal server-rendered review screen: one card per detected shot, showing
the representative still, its timecode range, and the stored image metrics.
The filmmaker uses this to review detections and (later) approve/reject
corrections. Still images are served straight from the stills directory.

``httpx`` is required for the FastAPI test client (declared in the ``dev``
extra), not for running the server.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from colorai.project.models import FrameMetrics, MediaAsset, Project, Shot
from colorai.project.store import ProjectStore

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def create_app(store: ProjectStore, stills_dir: str | Path) -> FastAPI:
    """Build the review app backed by ``store``, serving stills from ``stills_dir``."""
    stills = Path(stills_dir).resolve()
    stills.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="ColorAI")
    app.mount("/stills", StaticFiles(directory=str(stills)), name="stills")
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    @app.get("/")
    def index(request: Request):
        shots_view: list[dict] = []
        project_names: list[str] = []
        with store.session() as session:
            project_names = [p.name for p in session.query(Project).order_by(Project.id)]
            for shot in (
                session.query(Shot)
                .order_by(Shot.asset_id, Shot.index)
                .all()
            ):
                rf = shot.representative_frame
                if rf is None:
                    continue
                metrics = (
                    session.query(FrameMetrics)
                    .filter_by(shot_id=shot.id, frame_index=rf.frame_index)
                    .first()
                )
                still_url = "/stills/" + Path(rf.image_path).resolve().relative_to(stills).as_posix()
                shots_view.append(
                    {
                        "index": shot.index,
                        "start_tc": shot.start_timecode,
                        "end_tc": shot.end_timecode,
                        "frame_count": shot.frame_count,
                        "still_url": still_url,
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

    return app
