"""End-to-end analysis pipeline.

Ties together the vertical slices: ingest -> shot detection -> representative
frame extraction -> image metrics. Everything is written to the project
database; the source master is never modified.

The pipeline is **resumable and editing-aware**:

* Re-running on an unchanged, already-analyzed master returns cached results.
* Shot detection only runs when the asset has no shots yet (fresh) or when the
  caller forces re-detection — so manual split/merge edits survive a re-run.
* Representative frames, metrics, and skin are re-derived only for shots that
  are missing them, so an edit can be repaired without re-doing everything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from colorai.frames import extract_frame, representative_frame_index
from colorai.face import store_skin_metrics
from colorai.ingest import compute_source_hash, ingest_media
from colorai.metrics import metrics_from_path
from colorai.project.models import (
    FrameMetrics,
    MediaAsset,
    RepresentativeFrame,
    Shot,
    SkinMetric,
)
from colorai.project.store import ProjectStore, make_representative_frame
from colorai.shotdetect import (
    DEFAULT_MIN_SCENE_LEN,
    DEFAULT_THRESHOLD,
    detect_and_store_shots,
)
from colorai.skin_analysis import auto_assign_subjects


@dataclass(frozen=True)
class AnalysisResult:
    """Outcome of one ``analyze_master`` run."""

    asset: MediaAsset
    shots: list[Shot]
    representative_frames: list[RepresentativeFrame]
    metrics: list[FrameMetrics]
    skin_metrics: list[SkinMetric]


def _find_asset(
    store: ProjectStore, project_id: int, source_path: str, source_hash: str
) -> MediaAsset | None:
    with store.session() as session:
        return (
            session.query(MediaAsset)
            .filter_by(project_id=project_id, source_path=source_path)
            .filter(MediaAsset.source_hash == source_hash)
            .order_by(MediaAsset.id.desc())
            .first()
        )


def _load_shots(store: ProjectStore, asset: MediaAsset) -> list[Shot]:
    with store.session() as session:
        return (
            session.query(Shot)
            .filter_by(asset_id=asset.id)
            .order_by(Shot.index)
            .all()
        )


def _load_analysis(store: ProjectStore, asset: MediaAsset) -> AnalysisResult:
    """Reload a previously persisted analysis for ``asset``."""
    with store.session() as session:
        shots = (
            session.query(Shot).filter_by(asset_id=asset.id).order_by(Shot.index).all()
        )
        frames = (
            session.query(RepresentativeFrame)
            .join(Shot, RepresentativeFrame.shot_id == Shot.id)
            .filter(Shot.asset_id == asset.id)
            .order_by(Shot.index)
            .all()
        )
        metrics = (
            session.query(FrameMetrics)
            .join(Shot, FrameMetrics.shot_id == Shot.id)
            .filter(Shot.asset_id == asset.id)
            .order_by(Shot.index)
            .all()
        )
        skin_metrics = (
            session.query(SkinMetric)
            .join(Shot, SkinMetric.shot_id == Shot.id)
            .filter(Shot.asset_id == asset.id)
            .order_by(Shot.index, SkinMetric.face_index)
            .all()
        )
    return AnalysisResult(asset, shots, frames, metrics, skin_metrics)


def _delete_shots(store: ProjectStore, asset_id: int) -> None:
    """Delete every shot of an asset (DB cascades their dependent rows)."""
    with store.session() as session:
        session.query(Shot).filter_by(asset_id=asset_id).delete(
            synchronize_session=False
        )


def _refresh(
    store: ProjectStore,
    asset: MediaAsset,
    shots: list[Shot],
    stills_dir: Path,
) -> tuple[list[RepresentativeFrame], list[FrameMetrics], list[SkinMetric]]:
    """Fill in missing frames/metrics/skin for ``shots`` (idempotent)."""
    stills_dir.mkdir(parents=True, exist_ok=True)
    shot_ids = [s.id for s in shots]

    with store.session() as session:
        existing_rf = {
            rf.shot_id: rf
            for rf in session.query(RepresentativeFrame)
            .filter(RepresentativeFrame.shot_id.in_(shot_ids))
            .all()
        }
        have_metrics = {
            m.shot_id
            for m in session.query(FrameMetrics)
            .filter(FrameMetrics.shot_id.in_(shot_ids))
            .all()
        }
        have_skin = {
            s.shot_id
            for s in session.query(SkinMetric)
            .filter(SkinMetric.shot_id.in_(shot_ids))
            .all()
        }

    # Extract + persist missing representative frames.
    frames: list[RepresentativeFrame] = []
    new_frames: list[RepresentativeFrame] = []
    for shot in shots:
        rf = existing_rf.get(shot.id)
        if rf is not None and rf.image_path and Path(rf.image_path).exists():
            frames.append(rf)
        else:
            index = representative_frame_index(shot)
            out = stills_dir / f"shot_{shot.index:04d}_frame_{index:06d}.png"
            extract_frame(asset.source_path, index, out, fps=asset.frame_rate)
            rf = make_representative_frame(
                shot, index, image_path=str(out), frame_rate=asset.frame_rate
            )
            new_frames.append(rf)
            frames.append(rf)

    if new_frames:
        with store.session() as session:
            session.add_all(new_frames)
            session.flush()
            for rf in new_frames:
                session.refresh(rf)

    # Metrics (compute only where missing).
    metrics: list[FrameMetrics] = []
    with store.session() as session:
        for shot, rf in zip(shots, frames):
            if shot.id in have_metrics:
                m = session.query(FrameMetrics).filter_by(shot_id=shot.id).first()
            else:
                stats = metrics_from_path(rf.image_path)
                m = FrameMetrics(shot_id=shot.id, frame_index=rf.frame_index, **stats)
                session.add(m)
            metrics.append(m)
        session.flush()
        for m in metrics:
            if m.id is None:
                session.refresh(m)

    # Skin metrics (compute only where missing).
    skin_metrics: list[SkinMetric] = []
    for shot, rf in zip(shots, frames):
        if shot.id in have_skin:
            with store.session() as session:
                skin_metrics.extend(
                    session.query(SkinMetric)
                    .filter_by(shot_id=shot.id)
                    .order_by(SkinMetric.face_index)
                    .all()
                )
        else:
            skin_metrics.extend(store_skin_metrics(store, shot, rf.image_path))

    return frames, metrics, skin_metrics


def analyze_master(
    store: ProjectStore,
    project_id: int,
    master_path: str | Path,
    *,
    stills_dir: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
    min_scene_len: int = DEFAULT_MIN_SCENE_LEN,
    resume: bool = True,
) -> AnalysisResult:
    """Analyze one source master end-to-end and persist all results.

    ``stills_dir`` is the base directory for extracted stills; a per-asset
    subdirectory is created under it so re-runs on different masters do not
    collide.

    ``resume`` reuses a cached analysis when the master is unchanged (same path,
    fingerprint, and shot-detection parameters). When the asset already has
    shots (e.g. after a manual split/merge), detection is skipped so the edit
    is preserved and only missing stills/metrics/skin are re-derived. Set
    ``resume=False`` to force shot detection from scratch.
    """
    params = {"threshold": threshold, "min_scene_len": min_scene_len}
    source_hash = compute_source_hash(master_path)

    existing = _find_asset(store, project_id, str(master_path), source_hash)
    if (
        resume
        and existing is not None
        and existing.status == "analyzed"
        and existing.analyze_params == params
    ):
        return _load_analysis(store, existing)

    asset = existing if existing is not None else ingest_media(store, project_id, master_path)

    has_shots = len(_load_shots(store, asset)) > 0
    if not resume or not has_shots:
        if has_shots:
            _delete_shots(store, asset.id)
        shots = detect_and_store_shots(
            store, asset, threshold=threshold, min_scene_len=min_scene_len
        )
    else:
        shots = _load_shots(store, asset)

    asset_stills = Path(stills_dir) / f"asset_{asset.id:04d}"
    frames, metrics, skin_metrics = _refresh(store, asset, shots, asset_stills)

    # Identity-based subject grouping on fresh analyses. Skipped when any face
    # already has a subject (a prior run or a manual/agent edit), so resumable
    # runs never clobber human regroupings.
    with store.session() as session:
        assigned = (
            session.query(SkinMetric)
            .filter(SkinMetric.subject_id.isnot(None))
            .join(Shot, SkinMetric.shot_id == Shot.id)
            .filter(Shot.asset_id == asset.id)
            .count()
        )
    if assigned == 0:
        auto_assign_subjects(store, asset.id)

    with store.session() as session:
        session.query(MediaAsset).filter(MediaAsset.id == asset.id).update(
            {"status": "analyzed", "analyze_params": params}
        )
        asset = session.get(MediaAsset, asset.id)

    return AnalysisResult(asset, shots, frames, metrics, skin_metrics)
