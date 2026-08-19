"""End-to-end analysis pipeline.

Ties together the vertical slices: ingest -> shot detection -> representative
frame extraction -> image metrics. Everything is written to the project
database; the source master is never modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from colorai.frames import extract_representative_frames
from colorai.face import store_skin_metrics
from colorai.ingest import ingest_media
from colorai.metrics import metrics_from_path, store_frame_metrics
from colorai.project.models import (
    FrameMetrics,
    MediaAsset,
    RepresentativeFrame,
    Shot,
    SkinMetric,
)
from colorai.project.store import ProjectStore
from colorai.shotdetect import (
    DEFAULT_MIN_SCENE_LEN,
    DEFAULT_THRESHOLD,
    detect_and_store_shots,
)


@dataclass(frozen=True)
class AnalysisResult:
    """Outcome of one ``analyze_master`` run."""

    asset: MediaAsset
    shots: list[Shot]
    representative_frames: list[RepresentativeFrame]
    metrics: list[FrameMetrics]
    skin_metrics: list[SkinMetric]


def analyze_master(
    store: ProjectStore,
    project_id: int,
    master_path: str | Path,
    *,
    stills_dir: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
    min_scene_len: int = DEFAULT_MIN_SCENE_LEN,
) -> AnalysisResult:
    """Analyze one source master end-to-end and persist all results.

    ``stills_dir`` is the base directory for extracted stills; a per-asset
    subdirectory is created under it so re-runs on different masters do not
    collide.
    """
    asset = ingest_media(store, project_id, master_path)
    shots = detect_and_store_shots(
        store, asset, threshold=threshold, min_scene_len=min_scene_len
    )

    asset_stills = Path(stills_dir) / f"asset_{asset.id:04d}"
    frames = extract_representative_frames(store, asset, shots, asset_stills)

    metrics: list[FrameMetrics] = []
    skin_metrics: list[SkinMetric] = []
    for shot, frame in zip(shots, frames):
        stats = metrics_from_path(frame.image_path)
        metrics.append(store_frame_metrics(store, shot, frame.frame_index, stats))
        skin_metrics.extend(store_skin_metrics(store, shot, frame.image_path))

    with store.session() as session:
        session.query(MediaAsset).filter(MediaAsset.id == asset.id).update(
            {"status": "analyzed"}
        )
        asset = session.get(MediaAsset, asset.id)  # refresh lifecycle status

    return AnalysisResult(asset, shots, frames, metrics, skin_metrics)
