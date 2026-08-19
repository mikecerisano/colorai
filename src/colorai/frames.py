"""Representative frame selection and extraction.

For each shot a single still is chosen (currently the middle frame) and
extracted frame-accurately with ffmpeg. The still is recorded as a
:class:`~colorai.project.models.RepresentativeFrame` and used later for image
metrics and the review UI.

Frame-accurate extraction uses ``select=eq(n\\,N)`` which decodes from the
start of the stream; that is exact but not seek-optimized. A keyframe-seek +
``select`` fast path is a documented future optimization for long-form media.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from colorai.project.models import MediaAsset, RepresentativeFrame, Shot
from colorai.project.store import ProjectStore, make_representative_frame


def representative_frame_index(shot: Shot) -> int:
    """Pick the middle frame of a shot as its representative still."""
    return (shot.start_frame + shot.end_frame) // 2


def extract_frame(
    video_path: str | Path, frame_index: int, out_path: str | Path
) -> Path:
    """Extract a single, frame-accurate still from ``video_path``.

    ``out_path`` extension determines the still format (e.g. ``.png``).
    """
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"select=eq(n\\,{frame_index})",
            "-frames:v",
            "1",
            "-y",
            str(destination),
        ],
        check=True,
    )
    return destination


def extract_representative_frames(
    store: ProjectStore,
    asset: MediaAsset,
    shots: list[Shot],
    stills_dir: str | Path,
) -> list[RepresentativeFrame]:
    """Extract and persist one representative still per shot.

    Still filenames are deterministic (``shot_0001_frame_000050.png``) so the
    operation is idempotent and reproducible.
    """
    stills = Path(stills_dir)
    frames: list[RepresentativeFrame] = []
    with store.session() as session:
        for shot in shots:
            index = representative_frame_index(shot)
            out = stills / f"shot_{shot.index:04d}_frame_{index:06d}.png"
            extract_frame(asset.source_path, index, out)
            rf = make_representative_frame(
                shot, index, image_path=str(out), frame_rate=asset.frame_rate
            )
            session.add(rf)
            frames.append(rf)
        session.flush()
        for rf in frames:
            session.refresh(rf)
    return frames
