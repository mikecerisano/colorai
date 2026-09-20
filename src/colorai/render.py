"""Full-master output: apply approved shot corrections across real frames.

The review workflow proposes and approves *deterministic, temporally stable*
corrections per shot (:mod:`colorai.correction`). This module is the export
path: it decodes the source master frame-by-frame, applies each shot's enabled
corrections (the exact same transforms the preview shows), and encodes a new
master. The source file is never opened for writing — output goes to a new
path.

The export is **fail-safe and delivery-preserving**:

* the same non-Rec.709 transfer guard as the preview is enforced up front;
* the corrected video is encoded with the source's color tags (primaries /
  transfer / matrix);
* the source's audio, subtitles, chapters, and global metadata are preserved
  by a stream-copy mux pass;
* a decoder failure or a truncated/incomplete decode raises instead of
  silently producing a broken master.

Implementation is a correctness-first streaming pipeline: ffmpeg decodes the
master to raw RGB24 on stdout, Python applies the shot's transform, and a
second ffmpeg encodes the result. That guarantees the export matches the
preview pixel-for-pixel (modulo the encoder) at the cost of CPU/throughput —
moving raw RGB through Python is slow for long 4K masters. Timing is preserved
as CFR at the asset's exact frame rate; VFR retiming is a documented future
optimization. GPU/ffmpeg-native acceleration is likewise future work.

``jobs`` parallelizes the Python transform stage over a thread pool (the numpy
/ OpenCV ops release the GIL, so this scales with cores on graded footage)
while decode, encode, and frame order stay exactly as in the serial path —
``jobs=1`` and ``jobs=N`` receive byte-identical encoder input.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from colorai.color import is_gradeable_transfer, normalize_transfer
from colorai.project.models import Correction, MediaAsset, Shot
from colorai.project.store import ProjectStore

# Raw frame transport pixel format shared by decoder and encoder.
_PIX_FMT = "rgb24"

# Canonical color values -> ffmpeg tag names for -color_primaries/-color_trc.
_TAG_ALIASES = {
    "bt709": "bt709",
    "bt470bg": "bt470bg",
    "smpte170m": "smpte170m",
    "bt2020": "bt2020",
    "linear": "linear",
    "pq": "smpte2084",
    "hlg": "arib-std-b67",
}


@dataclass(frozen=True)
class ShotSpan:
    """One contiguous shot interval and its enabled corrections (in order)."""

    start_frame: int
    end_frame: int  # inclusive
    corrections: tuple[tuple[str, dict], ...]
    shot_id: int


def build_shot_spans(store: ProjectStore, asset_id: int) -> list[ShotSpan]:
    """Build ordered, non-overlapping ``(start, end, corrections, shot_id)`` spans.

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
                shot.id,
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


def shot_for_frame(frame_index: int, spans: list[ShotSpan]) -> int | None:
    """Return the shot id covering ``frame_index``, or ``None`` in gaps."""
    for span in spans:
        if span.start_frame <= frame_index <= span.end_frame:
            return span.shot_id
        if span.start_frame > frame_index:
            break
    return None


def _tag(value: str | None) -> str:
    from colorai.color import normalize_transfer

    return _TAG_ALIASES.get(normalize_transfer(value), "bt709")


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
    *, codec: str, crf: int, preset: str, pixel_format: str,
    color_space: str | None, transfer: str | None,
) -> list[str]:
    primaries = _tag(color_space)
    trc = _tag(transfer)
    cmd = [
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
        # Preserve the source's color tags so downstream color management
        # sees the same characteristics as the master.
        "-color_primaries", primaries,
        "-color_trc", trc,
        "-colorspace", primaries,
    ]
    if codec == "libx264":
        # x264 signals primaries/transfer via its own VUI params.
        cmd += [
            "-x264-params",
            f"colorprim={primaries}:transfer={trc}:colormatrix={primaries}",
        ]
    cmd += ["-y", out]
    return cmd


def _mux_with_source(video_path: str | Path, source: str, out_path: str | Path) -> None:
    """Stream-copy the corrected video with the source's audio, subtitles,
    chapters, and global metadata."""
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", str(video_path),
        "-i", str(source),
        "-map", "0:v:0",
        "-map", "1:a?",
        "-map", "1:s?",
        "-map_chapters", "1",
        "-map_metadata", "1",
        "-c", "copy",
        "-c:s", "mov_text",  # mp4-native subtitle codec
        "-movflags", "+faststart",
        "-y", str(out_path),
    ]
    subprocess.run(cmd, check=True)


def _transform_frame(
    frame, frame_index: int, spans: list[ShotSpan], face_specs_by_shot: dict,
    *, transfer: str | None = "bt709",
):
    """Apply one frame's whole-frame + face-local corrections (pure per frame)."""
    import numpy as np

    from colorai.correction import apply_corrections
    from colorai.face_corrections import apply_face_corrections

    corrections = corrections_for_frame(frame_index, spans)
    if corrections:
        frame = apply_corrections(frame, corrections, transfer=transfer)
    shot_id = shot_for_frame(frame_index, spans)
    face_specs = face_specs_by_shot.get(shot_id)
    if face_specs:
        frame = apply_face_corrections(frame, face_specs, frame_index)
    return np.ascontiguousarray(frame).tobytes()


def render_master(
    store: ProjectStore,
    asset_id: int,
    out_path: str | Path,
    *,
    codec: str = "libx264",
    crf: int = 18,
    preset: str = "medium",
    pixel_format: str = "yuv420p",
    jobs: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Render ``asset_id`` to ``out_path`` with its approved corrections applied.

    Raises ``ValueError`` for missing dimensions, a non-gradeable transfer, or
    ``jobs < 1``; ``RuntimeError`` on decoder/encoder failure or an incomplete
    decode; and lets ``subprocess.CalledProcessError`` propagate if the mux
    pass fails. ``jobs > 1`` threads the Python transform stage (same
    transforms, same order — byte-identical encoder input).
    """
    if jobs < 1:
        raise ValueError("jobs must be >= 1")
    with store.session() as session:
        asset = session.get(MediaAsset, asset_id)
        if asset is None:
            raise ValueError(f"asset {asset_id} not found")
        if not asset.width or not asset.height:
            raise ValueError(
                f"asset {asset_id} has no probed dimensions; ingest it first"
            )
        # Same guard as the preview: refuse transfers without a known EOTF pair.
        if not is_gradeable_transfer(asset.transfer):
            from colorai.color import non_gradeable_reason

            raise ValueError(non_gradeable_reason(asset.transfer))
        width, height = asset.width, asset.height
        fps = asset.frame_rate
        source = asset.source_path
        color_space = asset.color_space
        transfer = asset.transfer
        expected_frames = asset.frame_count

    spans = build_shot_spans(store, asset_id)

    # Preflight every enabled face correction before any output is produced.
    # Raises ValidationError (aborting render) on an invalid enabled grade.
    from colorai.face_corrections import (
        apply_face_corrections,
        load_face_correction_specs_by_asset,
    )

    face_specs_by_shot = load_face_correction_specs_by_asset(store, asset_id)

    # The face/skin compositor is BT.709-calibrated; whole-frame grades may be
    # transfer-native, but face-local grades on PQ/HLG are refused outright.
    if face_specs_by_shot and normalize_transfer(transfer) != "bt709":
        raise ValueError(
            "face-local corrections are calibrated in BT.709 and cannot render on "
            f"a {transfer!r} master; disable them or deliver a Rec.709 mezzanine"
        )

    frame_bytes = width * height * 3
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Encode to a temporary video-only file, then mux the source's streams.
    tmp_video = destination.with_suffix(destination.suffix + ".video.mp4")

    decoder = subprocess.Popen(
        _decoder_cmd(source), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    encoder = subprocess.Popen(
        _encoder_cmd(
            str(tmp_video), width, height, fps,
            codec=codec, crf=crf, preset=preset, pixel_format=pixel_format,
            color_space=color_space, transfer=transfer,
        ),
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    import numpy as np

    def _read_frame():
        raw = decoder.stdout.read(frame_bytes)
        if not raw:
            return None
        if len(raw) != frame_bytes:
            raise RuntimeError(
                "decoder produced a partial frame — the source stream is "
                "truncated or corrupt; refusing to emit an incomplete master"
            )
        return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)

    frame_index = 0
    total = expected_frames
    if jobs == 1:
        try:
            while True:
                frame = _read_frame()
                if frame is None:
                    break
                encoder.stdin.write(
                    _transform_frame(
                        frame, frame_index, spans, face_specs_by_shot,
                        transfer=transfer,
                    )
                )
                frame_index += 1
                if progress is not None:
                    progress(frame_index, total or 0)
        finally:
            if encoder.stdin is not None:
                try:
                    encoder.stdin.close()
                except BrokenPipeError:
                    pass
            if decoder.stdout is not None:
                decoder.stdout.close()
            decoder.wait()
            encoder.wait()
    else:
        from concurrent.futures import Future, ThreadPoolExecutor

        window = jobs * 8
        pending: dict[int, Future[bytes]] = {}
        next_write = 0
        worker_error: BaseException | None = None
        try:
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                while True:
                    while len(pending) < window:
                        frame = _read_frame()
                        if frame is None:
                            break
                        pending[frame_index] = pool.submit(
                            _transform_frame, frame.copy(),
                            frame_index, spans, face_specs_by_shot,
                            transfer=transfer,
                        )
                        frame_index += 1
                    while next_write in pending and (
                        len(pending) >= window or frame is None
                    ):
                        encoder.stdin.write(pending.pop(next_write).result())
                        next_write += 1
                        if progress is not None:
                            progress(next_write, total or 0)
                    if frame is None:
                        break
        except BaseException as exc:  # noqa: BLE001 — must kill the pipes first
            worker_error = exc
            for future in pending.values():
                future.cancel()
            decoder.kill()
            try:
                if encoder.stdin is not None:
                    encoder.stdin.close()
            except BrokenPipeError:
                pass
            decoder.wait()
            encoder.wait()
            tmp_video.unlink(missing_ok=True)
            if isinstance(worker_error, BrokenPipeError):
                err = encoder.stderr.read().decode(errors="replace").strip()
                raise RuntimeError(f"encoder failed: {err}") from worker_error
            raise
        else:
            if encoder.stdin is not None:
                try:
                    encoder.stdin.close()
                except BrokenPipeError:
                    pass
            if decoder.stdout is not None:
                decoder.stdout.close()
            decoder.wait()
            encoder.wait()

    if decoder.returncode != 0:
        err = decoder.stderr.read().decode(errors="replace").strip()
        raise RuntimeError(f"decoder failed: {err}")

    if encoder.returncode != 0:
        err = encoder.stderr.read().decode(errors="replace")
        raise RuntimeError(f"encoder failed: {err.strip()}")

    # Reject incomplete decoder output (one frame of slack for metadata
    # rounding on duration-derived frame counts).
    if expected_frames is not None and frame_index < expected_frames - 1:
        tmp_video.unlink(missing_ok=True)
        raise RuntimeError(
            f"incomplete output: decoded {frame_index} frames, expected "
            f"{expected_frames}; the source may be truncated"
        )

    _mux_with_source(tmp_video, source, destination)
    tmp_video.unlink(missing_ok=True)
    return destination
