"""Reviewable temporal face-mask tracks.

A :class:`~colorai.project.models.FaceTrack` records where a face *is*; a
``FaceMaskTrack`` records *which pixels of that face are safe skin* to grade.
It stores normalized landmark keyframes (face oval, eyes, brows, lips,
hairline) plus backend provenance and quality metrics, and is the gate for any
skin-target proposal: ``review_state == "approved_for_proposal"``.

Two backends exist:

* a dense-landmark backend (optional MediaPipe Face Landmarker, behind the
  existing ``face`` extra) that subtracts protected eye/brow/lip/hairline
  regions from the feathered face oval;
* a labelled **fallback** (``backend="fallback"``, ``strategy="face_oval_skin"``)
  that keeps the tracked box oval + conservative colour skin mask but does *not*
  pretend to protect facial features it never detected.

No raster mask is persisted per frame: only normalized landmark geometry is
stored, and the shared compositor renders it at the current input size so
preview and render stay identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np

from colorai.frames import extract_frame
from colorai.project.models import FaceMaskTrack, FaceTrack, MediaAsset, Shot
from colorai.project.store import ProjectStore

# These mirror face_corrections (kept local to avoid a circular import).
MIN_COVERAGE = 0.75
MAX_GAP_RATIO = 0.20

LANDMARK_STRATEGY = "landmark_skin"
FALLBACK_STRATEGY = "face_oval_skin"

REVIEW_UNREVIEWED = "unreviewed"
REVIEW_APPROVED = "approved_for_proposal"
REVIEW_NEEDS_REBUILD = "needs_rebuild"
REVIEW_UNSAFE = "unsafe"
REVIEW_STATES = (REVIEW_APPROVED, REVIEW_NEEDS_REBUILD, REVIEW_UNSAFE)


class LandmarkDetector(Protocol):
    """Return normalized landmark geometry for one face box, or ``None``."""

    def __call__(
        self, image_bgr: np.ndarray, box_xywh: tuple[int, int, int, int]
    ) -> dict[str, np.ndarray] | None: ...


# MediaPipe Face Landmarker indices for the facial-region polygons. The
# adapter is optional (only active when the ``face`` extra is installed) and
# never downloads a model at import time.
_FACE_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)
_LEFT_EYE = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
_RIGHT_EYE = (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466)
_LEFT_BROW = (70, 63, 105, 66, 107, 55, 65, 52, 53, 46)
_RIGHT_BROW = (300, 293, 334, 296, 336, 285, 295, 282, 283, 276)
_LIPS_OUTER = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146)


def landmarks_to_geometry(landmarks: np.ndarray) -> dict[str, list]:
    """Map MediaPipe ``Nx3`` normalized landmarks to mask geometry polygons."""

    def poly(indices: tuple[int, ...]) -> list[list[float]]:
        return [[float(landmarks[i, 0]), float(landmarks[i, 1])] for i in indices if i < len(landmarks)]

    return {
        "oval": poly(_FACE_OVAL),
        "eyes": [poly(_LEFT_EYE), poly(_RIGHT_EYE)],
        "brows": [poly(_LEFT_BROW), poly(_RIGHT_BROW)],
        "lips": [poly(_LIPS_OUTER)],
        "hairline": [],
    }


def mediapipe_landmark_detector(
    image_bgr: np.ndarray, box_xywh: tuple[int, int, int, int]
) -> dict | None:
    """Optional ``LandmarkDetector`` backed by MediaPipe Face Landmarker.

    Returns ``None`` when the optional dependency is absent or no face matches
    the tracked box; callers then keep the labelled fallback.
    """
    from colorai.face import face_landmarks

    landmarks_list = face_landmarks(image_bgr)
    if not landmarks_list:
        return None
    x, y, w, h = (int(v) for v in box_xywh)
    ah, aw = image_bgr.shape[:2]
    target = np.array([(x + w / 2.0) / aw, (y + h / 2.0) / ah], dtype=np.float64)
    best = min(
        landmarks_list,
        key=lambda lm: float(np.linalg.norm(lm[:, :2].mean(axis=0) - target)),
    )
    return landmarks_to_geometry(best)


mediapipe_landmark_detector.backend_version = "face_landmarker_468"  # type: ignore[attr-defined]


def landmark_backend_available() -> bool:
    """True when the optional MediaPipe landmark backend can be imported.

    Importability is the availability signal: MediaPipe Face Landmarker bundles
    its model, so a missing import means "backend unavailable" (use the labelled
    fallback), not "detection failed for this frame" (a low-coverage failure).
    """
    import importlib.util

    return importlib.util.find_spec("mediapipe") is not None


def _oval_polygon(nx: float, ny: float, nw: float, nh: float, points: int = 48) -> list[list[float]]:
    """A normalized ellipse polygon from a normalized ``(x, y, w, h)`` box."""
    cx, cy = nx + nw / 2.0, ny + nh / 2.0
    rx, ry = nw / 2.0, nh / 2.0
    out: list[list[float]] = []
    for i in range(points):
        theta = 2.0 * np.pi * i / points
        out.append([float(cx + rx * np.cos(theta)), float(cy + ry * np.sin(theta))])
    return out


def _fallback_geometry(nx: float, ny: float, nw: float, nh: float) -> dict:
    return {"oval": _oval_polygon(nx, ny, nw, nh), "eyes": [], "brows": [], "lips": [], "hairline": []}


def _polygon_mask(shape_hw: tuple[int, int], points_normalized: list[list[float]]) -> np.ndarray:
    """Boolean mask (HxW) for a normalized polygon (points inside are True)."""
    h, w = shape_hw
    pts = np.array(
        [[round(x * w), round(y * h)] for x, y in points_normalized], dtype=np.int32
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def render_face_mask(
    shape_hw: tuple[int, int],
    geometry: dict | None,
    *,
    strategy: str = LANDMARK_STRATEGY,
) -> np.ndarray:
    """Render a feathered face-skin alpha (float ``[0,1]``, HxW) from geometry.

    Fills the face oval, feathers it, then hard-subtracts protected polygons
    (eyes, brows, lips, hairline) so those pixels stay exactly untouched. For
    the labelled fallback strategy there are no protected regions (the coarse
    oval is the whole mask).
    """
    h, w = shape_hw
    geometry = geometry or {}
    oval = geometry.get("oval")
    if oval and len(oval) >= 3:
        oval_bool = _polygon_mask(shape_hw, oval).astype(np.float32)
    else:
        # A centred coarse oval when no geometry is available at all.
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        rx, ry = w / 2.0, h / 2.0
        oval_bool = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0).astype(np.float32)

    # Feather the oval boundary, then hard-subtract protected polygons so eye/
    # lip/brow/hairline pixels stay exactly untouched.
    sigma = max(1.0, min(h, w) / 24.0)
    alpha = cv2.GaussianBlur(oval_bool, (0, 0), sigmaX=sigma)
    if strategy == LANDMARK_STRATEGY:
        for key in ("eyes", "brows", "lips", "hairline"):
            for poly in geometry.get(key) or []:
                if poly and len(poly) >= 3:
                    alpha *= (1.0 - _polygon_mask(shape_hw, poly).astype(np.float32))

    return np.clip(alpha, 0.0, 1.0)


def interpolate_mask_geometry(
    keyframes: list[tuple[int, dict]], frame_index: int
) -> dict | None:
    """Linearly interpolate normalized landmark geometry at ``frame_index``.

    ``keyframes`` is ``[(frame_index, geometry), ...]`` sorted by frame. Clamps
    to the nearest keyframe outside the sampled range (never invents geometry
    the backend did not produce).
    """
    if not keyframes:
        return None
    frames = sorted(keyframes, key=lambda k: k[0])
    if frame_index <= frames[0][0]:
        return frames[0][1]
    if frame_index >= frames[-1][0]:
        return frames[-1][1]
    for (f0, g0), (f1, g1) in zip(frames, frames[1:]):
        if f0 <= frame_index <= f1:
            t = 0.0 if f1 == f0 else (frame_index - f0) / (f1 - f0)
            return _interp_geometry(g0, g1, t)
    return frames[-1][1]


def _interp_geometry(g0, g1, t: float):
    """Recursively interpolate nested geometry structures of numbers."""
    if isinstance(g0, dict) and isinstance(g1, dict):
        keys = sorted(set(g0) & set(g1))
        return {k: _interp_geometry(g0.get(k), g1.get(k), t) for k in keys}
    if isinstance(g0, (list, tuple)) and isinstance(g1, (list, tuple)):
        if len(g0) != len(g1):
            raise ValueError("cannot interpolate geometry with mismatched lengths")
        return [_interp_geometry(a, b, t) for a, b in zip(g0, g1)]
    if isinstance(g0, (int, float)) and isinstance(g1, (int, float)):
        return float(g0) + (float(g1) - float(g0)) * t
    return g1


def _max_gap_ratio(
    frames: list[int], start: int, end: int
) -> float:
    """Largest untracked frame gap as a fraction of shot duration."""
    if not frames:
        return 1.0
    duration = end - start + 1
    if duration <= 0:
        return 0.0
    ordered = sorted(frames)
    gaps = [ordered[0] - start, end - ordered[-1]]
    for a, b in zip(ordered, ordered[1:]):
        gaps.append(b - a - 1)
    return max(gaps) / duration


def build_face_mask_track(
    store: ProjectStore,
    face_track_id: int,
    *,
    detector: LandmarkDetector | None = None,
    strategy: str = LANDMARK_STRATEGY,
    samples: int = 16,
    extract: Callable[..., Path] = extract_frame,
) -> FaceMaskTrack:
    """Build and persist a reviewable ``FaceMaskTrack`` for one ``FaceTrack``.

    When ``detector`` is supplied, stores normalized landmark geometry and
    subtracts protected regions. Otherwise it stores the labelled fallback
    (``face_oval_skin``) — a coarse tracked-box oval with no feature protection.
    ``extract``/``detector`` are injectable so tests can use synthetic landmarks
    without downloading a model. Never silently substitutes a different model.
    """
    with store.session() as session:
        track = session.get(FaceTrack, face_track_id)
        if track is None:
            raise ValueError(f"face track {face_track_id} not found")
        if track.state != "valid":
            raise ValueError(f"face track {face_track_id} is not valid")
        shot = session.get(Shot, track.shot_id)
        asset = session.get(MediaAsset, shot.asset_id)
        keyframes = list(track.keyframes or [])
        subject_id = track.subject_id
        source_w, source_h = track.source_width, track.source_height
        source = asset.source_path
        fps = asset.frame_rate
        shot_start, shot_end = shot.start_frame, shot.end_frame

    if not keyframes:
        raise ValueError("face track has no keyframes")

    if detector is None:
        strategy = FALLBACK_STRATEGY
        backend, backend_version = "fallback", "0"
    else:
        backend = "mediapipe"
        backend_version = getattr(detector, "backend_version", "unknown")

    # Sample a bounded subset of the track's keyframes, evenly spaced.
    selected = _evenly_spaced(keyframes, samples)

    landmark_keyframes: list[list] = []
    probe_dir = Path(__import__("tempfile").mkdtemp(prefix="colorai_mask_track_"))
    try:
        for frame_index, nx, ny, nw, nh in selected:
            still = extract(source, int(frame_index), probe_dir / f"{int(frame_index)}.png", fps=fps, scale=None)
            image = cv2.imread(str(still), cv2.IMREAD_COLOR)
            if image is None:
                continue
            ah, aw = image.shape[:2]
            box = (
                int(round(nx * aw)), int(round(ny * ah)),
                int(round(nw * aw)), int(round(nh * ah)),
            )
            if detector is not None:
                geometry = detector(image, box)
                if geometry is None:
                    continue
            else:
                geometry = _fallback_geometry(nx, ny, nw, nh)
            landmark_keyframes.append([int(frame_index), _normalize_geometry(geometry)])
    finally:
        import shutil

        shutil.rmtree(probe_dir, ignore_errors=True)

    sampled = len(selected)
    tracked = len(landmark_keyframes)
    coverage = tracked / sampled if sampled else 0.0
    frames = [k[0] for k in landmark_keyframes]
    max_gap = _max_gap_ratio(frames, shot_start, shot_end)

    state = "valid"
    failure_reason = ""
    if coverage < MIN_COVERAGE:
        state, failure_reason = "failed", f"coverage {coverage:.2f} below {MIN_COVERAGE}"
    elif max_gap > MAX_GAP_RATIO:
        state, failure_reason = "failed", f"max gap {max_gap:.2f} exceeds {MAX_GAP_RATIO}"
    elif detector is not None and strategy == LANDMARK_STRATEGY and not _has_protected_regions(landmark_keyframes):
        state, failure_reason = "unsafe", "landmark backend produced no protected eye/lip regions"

    with store.session() as session:
        mask = FaceMaskTrack(
            face_track_id=face_track_id,
            shot_id=track.shot_id,
            subject_id=subject_id,
            backend=backend,
            backend_version=backend_version,
            strategy=strategy,
            landmark_keyframes=landmark_keyframes,
            coverage=coverage,
            max_gap=max_gap,
            state=state,
            review_state=REVIEW_UNREVIEWED,
            review_reason=failure_reason,
        )
        session.add(mask)
        session.flush()
        session.refresh(mask)
        return mask


def _evenly_spaced(keyframes: list, n: int) -> list:
    if len(keyframes) <= n:
        return list(keyframes)
    idx = sorted({round(i * (len(keyframes) - 1) / (n - 1)) for i in range(n)})
    return [keyframes[i] for i in idx]


def _normalize_geometry(geometry: dict) -> dict:
    """Ensure geometry values are JSON-friendly (nested lists of floats)."""
    def clean(v):
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, (np.integer, np.floating)):
            return float(v)
        return v

    return {k: clean(v) for k, v in (geometry or {}).items()}


def _has_protected_regions(keyframes: list) -> bool:
    for _, geometry in keyframes:
        for key in ("eyes", "lips"):
            polys = (geometry or {}).get(key) or []
            if any(p and len(p) >= 3 for p in polys):
                return True
    return False


def validate_face_mask_track(
    store: ProjectStore, mask_track_id: int, *, require_review: bool = True
) -> FaceMaskTrack:
    """Return a mask track after validating it is usable for a proposal.

    ``require_review=True`` demands ``review_state == "approved_for_proposal"``
    (the sole proposal gate). Also rejects non-valid tracks and those outside
    the coverage/gap quality bounds.
    """
    with store.session() as session:
        mask = session.get(FaceMaskTrack, mask_track_id)
        if mask is None:
            raise ValueError(f"face mask track {mask_track_id} not found")
        if mask.state != "valid":
            raise ValueError(f"face mask track {mask_track_id} is not valid ({mask.state})")
        if mask.coverage < MIN_COVERAGE:
            raise ValueError(f"face mask track {mask_track_id} coverage below threshold")
        if mask.max_gap > MAX_GAP_RATIO:
            raise ValueError(f"face mask track {mask_track_id} gap exceeds threshold")
        if require_review and mask.review_state != REVIEW_APPROVED:
            raise ValueError(
                f"face mask track {mask_track_id} must be reviewed "
                f"'approved_for_proposal' (currently {mask.review_state!r})"
            )
        if require_review and mask.backend == "fallback" and not mask.human_approved:
            raise ValueError(
                f"face mask track {mask_track_id} is a lower-confidence fallback "
                "mask and requires explicit human approval before use"
            )
        return mask


def make_face_mask_contact_sheet(
    store: ProjectStore,
    mask_track_id: int,
    *,
    extract: Callable[..., Path] = extract_frame,
    scale: int = 320,
    max_frames: int = 8,
):
    """Render the actual mask alpha as a cyan overlay on sampled source frames.

    Returns a PIL ``Image.Image`` with a yellow face box and frame index label.
    """
    from PIL import Image as PILImage
    from PIL import ImageDraw

    with store.session() as session:
        mask = session.get(FaceMaskTrack, mask_track_id)
        if mask is None:
            raise ValueError("face mask track not found")
        track = session.get(FaceTrack, mask.face_track_id)
        shot = session.get(Shot, mask.shot_id)
        asset = session.get(MediaAsset, shot.asset_id)
        keyframes = list(mask.landmark_keyframes or [])[:max_frames]
        source = asset.source_path
        fps = asset.frame_rate

    import shutil
    import tempfile

    probe_dir = Path(tempfile.mkdtemp(prefix="colorai_mask_sheet_"))
    try:
        thumbs = []
        for frame_index, geometry in keyframes:
            still = extract(source, int(frame_index), probe_dir / f"{frame_index}.png", fps=fps, scale=scale)
            img = PILImage.open(still).convert("RGB")
            w, h = img.size
            alpha = render_face_mask((h, w), geometry, strategy=mask.strategy)
            overlay = PILImage.new("RGB", img.size, (0, 255, 255))
            img = PILImage.blend(img, overlay, alpha=0.0)
            arr = np.asarray(img, dtype=np.float64)
            cyan = np.array([0, 255, 255], dtype=np.float64)
            arr = arr * (1.0 - alpha[..., None]) + cyan[None, None, :] * (alpha[..., None] * 0.45)
            img = PILImage.fromarray(arr.round().astype(np.uint8))
            draw = ImageDraw.Draw(img)
            # Box for the fallback oval / landmark oval.
            oval = geometry.get("oval")
            if oval:
                xs = [p[0] * w for p in oval]
                ys = [p[1] * h for p in oval]
                draw.rectangle((min(xs), min(ys), max(xs), max(ys)), outline=(255, 255, 0), width=2)
            draw.text((4, 4), str(int(frame_index)), fill=(255, 255, 255))
            thumbs.append(img)

        cols = min(4, len(thumbs))
        rows = (len(thumbs) + cols - 1) // cols if cols else 0
        if not thumbs:
            return PILImage.new("RGB", (320, 180), (24, 24, 24))
        tw, th = (thumbs[0].size if thumbs else (320, 180))
        sheet = PILImage.new("RGB", (cols * tw, rows * th), (24, 24, 24))
        for i, img in enumerate(thumbs):
            sheet.paste(img, ((i % cols) * tw, (i // cols) * th))
        return sheet
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)
