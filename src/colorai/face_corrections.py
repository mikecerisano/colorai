"""Face-local skin corrections: track building and a pure mask compositor.

This is ColorAI's conservative finishing adjustment, not a retouch tool:

* version one supports only ``rgb_balance`` with per-channel linear-light gains
  clamped to ``[0.90, 1.10]``;
* the correction applies only under a tracked, feathered face-skin mask — never
  to a whole frame or a second participant;
* ``apply_face_corrections`` is pure (no database/filesystem I/O) so preview and
  full-master render use exactly the same compositor.

Image convention for the compositor: HxWx3 **RGB** uint8 or float in ``[0,1]``.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from colorai.color import bt709_to_linear, linear_to_bt709
from colorai.face import detect_faces, skin_metrics_in_region
from colorai.frames import extract_frame
from colorai.project.models import FaceTrack, MediaAsset, Shot, SkinMetric
from colorai.project.store import ProjectStore
from colorai.skin import skin_mask
from colorai.tracking import match_box, sample_frames, temporal_skin_metrics

# Conservative finishing bounds.
GAIN_MIN = 0.90
GAIN_MAX = 1.10
MIN_COVERAGE = 0.75
MAX_GAP_RATIO = 0.20
SKIN_STABILITY_THRESHOLD = 0.05


@dataclass(frozen=True)
class FaceCorrectionSpec:
    """A persisted face correction ready for the pure compositor."""

    id: int
    gain: tuple[float, float, float]
    keyframes: tuple[tuple[int, float, float, float, float], ...]
    source_width: int
    source_height: int


def validate_gain(gain: Sequence[float]) -> tuple[float, float, float]:
    """Validate and clamp a 3-channel ``rgb_balance`` gain to ``[0.90, 1.10]``."""
    if not isinstance(gain, (list, tuple)) or len(gain) != 3:
        raise ValueError("rgb_balance gain must be a 3-element list")
    out: list[float] = []
    for v in gain:
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):  # NaN/inf
            raise ValueError("gain values must be finite")
        if not (GAIN_MIN <= f <= GAIN_MAX):
            raise ValueError(
                f"gain {f:.3f} outside conservative range [{GAIN_MIN}, {GAIN_MAX}]"
            )
        out.append(f)
    return tuple(out)


def _max_gap_ratio(keyframes: Sequence[tuple[int, float, float, float, float]], start: int, end: int) -> float:
    """Largest untracked frame gap as a fraction of the shot duration."""
    if not keyframes:
        return 1.0
    duration = end - start + 1
    if duration <= 0:
        return 0.0
    frames = sorted(k[0] for k in keyframes)
    gaps = [frames[0] - start, end - frames[-1]]
    for a, b in zip(frames, frames[1:]):
        gaps.append(b - a - 1)
    return max(gaps) / duration


def _interpolate_box(
    keyframes: Sequence[tuple[int, float, float, float, float]], frame_index: int
) -> tuple[float, float, float, float]:
    """Interpolate the normalized ``(x, y, w, h)`` box at ``frame_index``.

    Clamps to the nearest keyframe outside the sampled range; never invents a
    box where the tracker had none.
    """
    if not keyframes:
        raise ValueError("no keyframes to interpolate")
    frames = sorted(keyframes, key=lambda k: k[0])
    if frame_index <= frames[0][0]:
        return (frames[0][1], frames[0][2], frames[0][3], frames[0][4])
    if frame_index >= frames[-1][0]:
        return (frames[-1][1], frames[-1][2], frames[-1][3], frames[-1][4])
    for (f0, x0, y0, w0, h0), (f1, x1, y1, w1, h1) in zip(frames, frames[1:]):
        if f0 <= frame_index <= f1:
            t = 0.0 if f1 == f0 else (frame_index - f0) / (f1 - f0)
            return (
                x0 + (x1 - x0) * t,
                y0 + (y1 - y0) * t,
                w0 + (w1 - w0) * t,
                h0 + (h1 - h0) * t,
            )
    return (frames[-1][1], frames[-1][2], frames[-1][3], frames[-1][4])


def _soft_mask(region_bgr: np.ndarray) -> np.ndarray:
    """Feathered face-skin alpha for a box crop (float ``[0,1]``, HxW)."""
    skin = skin_mask(region_bgr).astype(np.float32)
    h, w = skin.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    rx, ry = w / 2.0, h / 2.0
    oval = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
    combined = skin * oval.astype(np.float32)
    sigma = max(1.0, min(h, w) / 16.0)
    alpha = cv2.GaussianBlur(combined, (0, 0), sigmaX=sigma)
    return np.clip(alpha, 0.0, 1.0)


def _to_float_rgb(image: np.ndarray) -> tuple[np.ndarray, bool]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected an HxWx3 RGB image")
    was_uint8 = image.dtype.kind in "iu"
    f = image.astype(np.float64)
    if was_uint8:
        f /= 255.0
    return np.clip(f, 0.0, 1.0), was_uint8


def apply_face_corrections(
    image_rgb: np.ndarray,
    corrections: Sequence[FaceCorrectionSpec],
    frame_index: int,
) -> np.ndarray:
    """Apply masked ``rgb_balance`` corrections to one frame (pure).

    ``corrections`` are applied in stable ``id`` order. Later masks only fill
    pixels their alpha has not already covered.
    """
    if not corrections:
        return image_rgb

    base, was_uint8 = _to_float_rgb(image_rgb)
    h, w = base.shape[:2]
    covered = np.zeros((h, w), dtype=np.float32)

    for spec in sorted(corrections, key=lambda c: c.id):
        nx, ny, nw, nh = _interpolate_box(spec.keyframes, frame_index)
        x0 = max(0, int(round(nx * spec.source_width)))
        y0 = max(0, int(round(ny * spec.source_height)))
        x1 = min(w, int(round((nx + nw) * spec.source_width)))
        y1 = min(h, int(round((ny + nh) * spec.source_height)))
        if x1 <= x0 or y1 <= y0:
            continue

        region = base[y0:y1, x0:x1]
        # Skin mask is color-based and expects BGR uint8.
        region_bgr = cv2.cvtColor(
            (np.clip(region, 0, 1) * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR
        )
        alpha = _soft_mask(region_bgr)  # HxW float
        # Alpha not already claimed by an earlier (lower id) correction.
        remaining = np.clip(alpha - covered[y0:y1, x0:x1], 0.0, 1.0)
        if not remaining.any():
            continue

        gain = np.asarray(spec.gain, dtype=np.float64)
        linear = bt709_to_linear(region)
        corrected = linear_to_bt709(np.clip(linear * gain, 0.0, None))
        blended = region * (1.0 - remaining[..., None]) + corrected * remaining[..., None]
        base[y0:y1, x0:x1] = blended
        covered[y0:y1, x0:x1] = np.maximum(covered[y0:y1, x0:x1], alpha)

    if was_uint8:
        return (np.clip(base, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return base.astype(np.float32)


def build_face_track(
    store: ProjectStore,
    skin_metric_id: int,
    samples: int = 16,
    scale: int | None = 480,
    extract: Callable[..., Path] = extract_frame,
    detect: Callable[..., list[tuple[int, int, int, int]]] = detect_faces,
) -> FaceTrack:
    """Derive and persist a temporal face track for one ``SkinMetric``.

    Re-detects and IoU-associates the selected face across ``samples`` frames,
    preserving every successful box as a normalized keyframe. Accepts a track
    only when coverage >= 75%, the largest gap <= 20% of the shot duration,
    and temporal skin stability is below the threshold; otherwise persists a
    ``failed`` track (QC evidence only).
    """
    with store.session() as session:
        metric = session.get(SkinMetric, skin_metric_id)
        if metric is None:
            raise ValueError(f"skin metric {skin_metric_id} not found")
        shot = session.get(Shot, metric.shot_id)
        if shot is None:
            raise ValueError("skin metric has no shot")
        asset = session.get(MediaAsset, shot.asset_id)
        if asset is None or not asset.width or not asset.height:
            raise ValueError("asset dimensions are missing; ingest first")
        source_w, source_h = asset.width, asset.height
        fps = asset.frame_rate
        source = asset.source_path
        subject_id = metric.subject_id
        seed = (metric.bbox_x, metric.bbox_y, metric.bbox_w, metric.bbox_h)
        if any(v is None for v in seed):
            raise ValueError("skin metric has no bounding box")
        shot_start, shot_end = shot.start_frame, shot.end_frame

    frames = sample_frames(shot_start, shot_end, samples)
    probe_dir = Path(tempfile.mkdtemp(prefix="colorai_face_track_"))
    try:
        previous: tuple[int, int, int, int] | None = None
        keyframes: list[tuple[int, float, float, float, float]] = []
        signatures: list[tuple[float, float, float]] = []
        for fi in frames:
            still = extract(source, fi, probe_dir / f"{fi}.png", fps=fps, scale=scale)
            image = cv2.imread(str(still), cv2.IMREAD_COLOR)
            if image is None:
                continue
            ah, aw = image.shape[:2]
            if previous is None:
                sx, sy, sw, sh = seed
                previous = (
                    sx * aw / source_w,
                    sy * ah / source_h,
                    sw * aw / source_w,
                    sh * ah / source_h,
                )
            box = match_box(detect(image), previous)
            if box is None:
                continue
            previous = box
            nx, ny, nw, nh = box[0] / aw, box[1] / ah, box[2] / aw, box[3] / ah
            keyframes.append((fi, nx, ny, nw, nh))
            skin = skin_metrics_in_region(image, box)
            if skin is not None:
                signatures.append(tuple(skin["mean_bgr"]))
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)

    tracked = len(keyframes)
    coverage = tracked / samples if samples else 0.0
    max_gap = _max_gap_ratio(keyframes, shot_start, shot_end)
    metrics = temporal_skin_metrics(signatures) if signatures else None
    stability = metrics["stability"] if metrics else 1.0
    median = metrics["median_bgr"] if metrics else None

    state = "valid"
    reason: str | None = None
    if coverage < MIN_COVERAGE:
        state, reason = "failed", f"coverage {coverage:.2f} below {MIN_COVERAGE}"
    elif max_gap > MAX_GAP_RATIO:
        state, reason = "failed", f"max gap {max_gap:.2f} exceeds {MAX_GAP_RATIO}"
    elif stability > SKIN_STABILITY_THRESHOLD:
        state, reason = "failed", f"skin stability {stability:.4f} exceeds threshold"

    with store.session() as session:
        track = FaceTrack(
            shot_id=shot.id,
            skin_metric_id=skin_metric_id,
            subject_id=subject_id,
            source_width=source_w,
            source_height=source_h,
            analysis_scale=scale,
            keyframes=[list(k) for k in keyframes],
            sample_count=samples,
            tracked_count=tracked,
            coverage=coverage,
            max_gap=max_gap,
            skin_stability=stability,
            median_bgr=median,
            state=state,
            failure_reason=reason,
        )
        session.add(track)
        session.flush()
        session.refresh(track)
        return track
