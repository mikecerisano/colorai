"""Shot detection via PySceneDetect.

ColorAI stores *inclusive, zero-based* frame bounds (matching ffmpeg/ffprobe
and :mod:`colorai.core.timecode`). PySceneDetect 0.7 reports scenes as
half-open ``[start, end)`` intervals with zero-based ``frame_num`` (a breaking
change from the 1-based numbering of 0.6.x), so each scene end is decremented
by one here. ``pyproject.toml`` pins ``scenedetect>=0.7`` for this reason.
"""

from __future__ import annotations

from pathlib import Path

from scenedetect import ContentDetector, SceneManager, open_video

from colorai.project.models import MediaAsset, Shot
from colorai.project.store import ProjectStore, make_shots

# Default ContentDetector threshold. Higher = fewer, more confident cuts.
DEFAULT_THRESHOLD = 27.0
# Minimum shot length in frames; suppresses flash-frame false positives.
DEFAULT_MIN_SCENE_LEN = 15


def detect_shot_bounds(
    path: str | Path,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_scene_len: int = DEFAULT_MIN_SCENE_LEN,
) -> list[tuple[int, int]]:
    """Return inclusive zero-based ``(start_frame, end_frame)`` shot bounds.

    Raises ``FileNotFoundError``/``OSError`` if ``path`` cannot be opened or
    contains no decodable video.
    """
    video = open_video(str(path))
    manager = SceneManager()
    manager.add_detector(
        ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
    )
    manager.detect_scenes(video=video)

    bounds: list[tuple[int, int]] = []
    for start, end in manager.get_scene_list():
        start_frame = start.frame_num
        end_frame = end.frame_num - 1  # half-open -> inclusive
        if end_frame >= start_frame:
            bounds.append((start_frame, end_frame))
    return bounds


def detect_and_store_shots(
    store: ProjectStore,
    asset: MediaAsset,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_scene_len: int = DEFAULT_MIN_SCENE_LEN,
) -> list[Shot]:
    """Detect shots on ``asset`` and persist them (ordered, timecoded)."""
    bounds = detect_shot_bounds(
        asset.source_path, threshold=threshold, min_scene_len=min_scene_len
    )
    shots = make_shots(asset, bounds)
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for shot in shots:
            session.refresh(shot)
    return shots
