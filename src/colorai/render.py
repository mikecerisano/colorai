"""Full-master output: apply approved shot corrections across real frames.

The review workflow proposes and approves *deterministic, temporally stable*
corrections per shot (:mod:`colorai.correction`). This module is the export
path: it decodes the source master frame-by-frame, applies each shot's enabled
corrections (the exact same transforms the preview shows), and encodes a new
master. The source file is never opened for writing — output goes to a new
path.

Implementation is a correctness-first streaming pipeline: ffmpeg decodes the
master to raw RGB24 on stdout, Python applies the shot's transform, and a
second ffmpeg encodes the result. This guarantees the export matches the
preview pixel-for-pixel (modulo the encoder) at the cost of CPU/throughput —
moving raw RGB through Python is slow for long 4K masters. A documented future
optimization is compiling the deterministic transform to an ffmpeg/GPU path.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from colorai.correction import apply_corrections
from colorai.project.models import Correction, MediaAsset, Shot
from colorai.project.store import ProjectStore

# Raw frame transport pixel format shared by decoder and encoder.
_PIX_FMT = "rgb24"


@dataclass(frozen=True)
class ShotSpan:
    """One contiguous shot interval and its enabled corrections (in order)."""

    start_frame: int
    end_frame: int  # inclusive
    corrections: tuple[tuple[str, dict], ...]


def build_shot_spans(store: ProjectStore, asset_id: int) -> list[ShotSpan]:
    """Build ordered, non-overlapping ``(start, end, corrections)`` spans for an asset.

    Only *enabled* corrections are included (the same rule the preview uses).
    Shots are returned in frame order; gaps between shots are left uncorrected
    by :func:`corrections_for_frame`.
    """
    with store.session() as session:
        asset = session.get(MediaAsset, asset_id)
        if asset is None:
            raise ValueError(f"asset {asset_id} not found")
        shots = (
            session.query(Shot).filter_by(asset_id=asset_id).order_by(Shot.start_frame).all()
        )
        corrections_by_shot: dict[int, list[Correction]] = {}
        for c in (
            session.query(Correction)
            .filter(Correction.enabled.is_(True))
            .order_by(Correction.id)
            .all()
        ):
            corrections_by_shot.setdefault(c.shot_id, []).append(c)

    spans: list[ShotSpan] = []
    for shot in shots:
        spans.append(
            ShotSpan(
                shot.start_frame,
                shot.end_frame,
                tuple(
                    (c.kind, c.parameters)
                    for c in corrections_by_shot.get(shot.id, ())
                ),
            )
        )
    return spans


def corrections_for_frame(
    frame_index: int, spans: list[ShotSpan]
) -> tuple[tuple[str, dict], ...]:
    """Return the corrections to apply at ``frame_index`` (empty if none)."""
    for span in spans:
        if span.start_frame <= frame_index <= span.end_frame:
            return span.corrections
        if span.start_frame > frame_index:
            break
    return ()


def _decoder_cmd(source: str) -> list[str]:
    return [
        "ffmpeg", "-v", "error",
        "-i", source,
        "-f", "rawvideo",
        "-pix_fmt", _PIX_FMT,
        "pipe:1",
    ]


def _encoder_cmd(
    out: str, width: int, height: int, fps: float,
    *,
    codec: str, crf: int, preset: str, pixel_format: str,
) -> list[str]:
    return [
        "ffmpeg", "-v", "error",
        "-f", "rawvideo",
        "-pix_fmt", _PIX_FMT,
        "-s", f"{width}x{height}",
        "-r", f"{fps:.6f}",
        "-i", "pipe:0",
        "-c:v", codec,
        "-crf", str(crf),
        "-preset", preset,
        "-pix_fmt", pixel_format,
        "-y",
        out,
    ]


def render_master(
    store: ProjectStore,
    asset_id: int,
    out_path: str | Path,
    *,
    codec: str = "libx264",
    crf: int = 18,
    preset: str = "medium",
    pixel_format: str = "yuv420p",
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Render ``asset_id`` to ``out_path`` with its approved corrections applied.

    Raises ``ValueError`` if the asset has no probed dimensions, and lets
    ``subprocess.CalledProcessError`` propagate if ffmpeg fails.
    """
    with store.session() as session:
        asset = session.get(MediaAsset, asset_id)
        if asset is None:
            raise ValueError(f"asset {asset_id} not found")
        if not asset.width or not asset.height:
            raise ValueError(
                f"asset {asset_id} has no probed dimensions; ingest it first"
            )
        width, height = asset.width, asset.height
        fps = asset.frame_rate
        source = asset.source_path

    spans = build_shot_spans(store, asset_id)
    frame_bytes = width * height * 3
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    decoder = subprocess.Popen(
        _decoder_cmd(source), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    encoder = subprocess.Popen(
        _encoder_cmd(
            str(destination), width, height, fps,
            codec=codec, crf=crf, preset=preset, pixel_format=pixel_format,
        ),
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    import numpy as np

    frame_index = 0
    total = asset.frame_count
    try:
        while True:
            raw = decoder.stdout.read(frame_bytes)
            if not raw:
                break
            # Partial trailing frame from a truncated stream: stop cleanly.
            if len(raw) != frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
            corrections = corrections_for_frame(frame_index, spans)
            if corrections:
                frame = apply_corrections(frame, corrections)
            encoder.stdin.write(frame.tobytes())
            frame_index += 1
            if progress is not None:
                progress(frame_index, total or 0)
    finally:
        if encoder.stdin is not None:
            try:
                encoder.stdin.close()
            except BrokenPipeError:
                pass
        decoder.stdout.close() if decoder.stdout is not None else None
        decoder.wait()
        encoder.wait()

    if encoder.returncode != 0:
        err = encoder.stderr.read().decode(errors="replace")
        raise RuntimeError(f"encoder failed: {err.strip()}")

    return destination
