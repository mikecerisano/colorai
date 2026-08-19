"""Representative frame selection and extraction.

For each shot a single still is chosen and extracted frame-accurately with
ffmpeg. The default selector is the middle frame; a content-aware ``sharpest``
selector samples several frames and keeps the one with the highest
Laplacian-variance sharpness (see :func:`colorai.metrics.frame_sharpness`).

Frame-accurate extraction uses ``select=eq(n\\,N)`` which decodes from the
start of the stream; that is exact but not seek-optimized. A keyframe-seek +
``select`` fast path is a documented future optimization for long-form media.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2

from colorai.metrics import frame_sharpness
from colorai.project.models import MediaAsset, RepresentativeFrame, Shot
from colorai.project.store import ProjectStore, make_representative_frame

SELECTOR_MIDDLE = "middle"
SELECTOR_SHARPEST = "sharpest"
DEFAULT_SAMPLES = 5


def representative_frame_index(shot: Shot) -> int:
    """Pick the middle frame of a shot as its representative still."""
    return (shot.start_frame + shot.end_frame) // 2


def select_sharpest(scores: dict[int, float]) -> int:
    """Pick the frame index with the highest sharpness (ties -> lowest index)."""
    if not scores:
        raise ValueError("no candidate scores")
    return max(scores, key=lambda idx: (scores[idx], -idx))


def _candidate_indices(shot: Shot, samples: int) -> list[int]:
    """Evenly spaced candidate frames within the shot (inclusive bounds)."""
    start, end = shot.start_frame, shot.end_frame
    if samples <= 1 or end == start:
        return [representative_frame_index(shot)]
    return sorted({round(start + (end - start) * i / (samples - 1)) for i in range(samples)})


def extract_frame(
    video_path: str | Path,
    frame_index: int,
    out_path: str | Path,
    *,
    fps: float | None = None,
    scale: int | None = None,
) -> Path:
    """Extract a single still from ``video_path``.

    When ``fps`` is given, uses input seek (``-ss <t>``) to jump to the target
    timestamp and decode only a short window — fast and frame-accurate enough
    for representative stills on long masters. Without ``fps``, falls back to
    the exact but slow ``select=eq(n\\,N)`` path (decodes from frame 0).
    ``scale`` optionally downscales the output to a target width (faster
    sampling).

    ``out_path`` extension determines the still format (e.g. ``.png``).
    """
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    vf = f"scale={scale}:-2" if scale else None
    if fps:
        timestamp = frame_index / fps
        cmd = [
            "ffmpeg", "-v", "error",
            "-ss", f"{timestamp:.6f}",
            "-i", str(video_path),
        ]
        if vf:
            cmd += ["-vf", vf]
        cmd += ["-frames:v", "1", "-y", str(destination)]
    else:
        sel = f"select=eq(n\\,{frame_index})"
        filter_str = f"{sel},{vf}" if vf else sel
        cmd = [
            "ffmpeg", "-v", "error",
            "-i", str(video_path),
            "-vf", filter_str,
            "-frames:v", "1",
            "-y", str(destination),
        ]
    subprocess.run(cmd, check=True)
    return destination


def _choose_index(asset: MediaAsset, shot: Shot, selector: str, samples: int) -> int:
    if selector != SELECTOR_SHARPEST:
        return representative_frame_index(shot)

    probe_dir = Path(tempfile.mkdtemp(prefix="colorai_probe_"))
    try:
        scores: dict[int, float] = {}
        for idx in _candidate_indices(shot, samples):
            still = extract_frame(
                asset.source_path, idx, probe_dir / f"{idx}.png", fps=asset.frame_rate
            )
            image = cv2.imread(str(still), cv2.IMREAD_COLOR)
            scores[idx] = frame_sharpness(image) if image is not None else 0.0
        return select_sharpest(scores)
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)


def extract_representative_frames(
    store: ProjectStore,
    asset: MediaAsset,
    shots: list[Shot],
    stills_dir: str | Path,
    *,
    selector: str = SELECTOR_MIDDLE,
    samples: int = DEFAULT_SAMPLES,
) -> list[RepresentativeFrame]:
    """Extract and persist one representative still per shot.

    ``selector`` is ``"middle"`` (default) or ``"sharpest"`` (content-aware).
    Still filenames are deterministic (``shot_0001_frame_000050.png``) so the
    operation is idempotent and reproducible.
    """
    stills = Path(stills_dir)
    frames: list[RepresentativeFrame] = []
    with store.session() as session:
        for shot in shots:
            index = _choose_index(asset, shot, selector, samples)
            out = stills / f"shot_{shot.index:04d}_frame_{index:06d}.png"
            extract_frame(asset.source_path, index, out, fps=asset.frame_rate)
            rf = make_representative_frame(
                shot, index, image_path=str(out), frame_rate=asset.frame_rate
            )
            session.add(rf)
            frames.append(rf)
        session.flush()
        for rf in frames:
            session.refresh(rf)
    return frames
