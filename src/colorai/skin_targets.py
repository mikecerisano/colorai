"""Skin-appearance references, targets, and scope-safe candidate derivation.

The workflow is evidence-first and human-gated:

* a filmmaker (or an accepted agent suggestion) supplies a reference labelled
  **accurate skin reference** or **creative look direction**;
* an agent inspects a reviewed mask contact sheet before drafting a target;
* a human approves one target, then each angle derives its *own* bounded,
  disabled ``skin_appearance`` correction from its own masked profile.

MCP may draft and revise; approval, enabling, and rendering stay human-only.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from colorai.color import bt709_to_linear
from colorai.editorial import GROUP_KIND_SETUP, GROUP_KIND_VARIANT
from colorai.face import detect_faces
from colorai.face_masks import (
    MAX_GAP_RATIO,
    MIN_COVERAGE,
    REVIEW_STATES,
    validate_face_mask_track,
)
from colorai.project.models import (
    FaceCorrection,
    FaceMaskTrack,
    FaceTrack,
    MediaAsset,
    Note,
    Shot,
    ShotGroup,
    SkinAppearanceReference,
    SkinAppearanceTarget,
    SkinMetric,
    Subject,
)
from colorai.project.store import ProjectStore
from colorai.skin import skin_mask
from colorai.skin_appearance import (
    apply_profile_transform,
    derive_skin_appearance_parameters,
    masked_skin_profile,
    rgb_linear_to_oklab,
    validate_skin_appearance_parameters,
)

ROLE_ACCURATE = "accurate_skin_reference"
ROLE_CREATIVE = "creative_look_direction"
ROLES = (ROLE_ACCURATE, ROLE_CREATIVE)

SOURCE_PROJECT_FRAME = "project_frame"
SOURCE_EXTERNAL_IMAGE = "external_image"
SOURCE_KINDS = (SOURCE_PROJECT_FRAME, SOURCE_EXTERNAL_IMAGE)

REF_STATE_ACTIVE = "active"
REF_STATE_REJECTED = "rejected"

TARGET_SUGGESTED = "suggested"
TARGET_APPROVED = "approved"
TARGET_REJECTED = "rejected"

CREATIVE_STRENGTH_CAP = 0.50


def _project_root(store: ProjectStore) -> Path:
    db = getattr(store.engine.url, "database", None)
    if not db or db == ":memory:":
        raise ValueError("skin references require a file-backed project database")
    return Path(db).parent


def _load_scope(session, subject_id: int, group_id: int) -> tuple[Subject, ShotGroup, MediaAsset]:
    subject = session.get(Subject, subject_id)
    if subject is None:
        raise ValueError(f"subject {subject_id} not found")
    group = session.get(ShotGroup, group_id)
    if group is None:
        raise ValueError(f"group {group_id} not found")
    if group.kind not in (GROUP_KIND_SETUP, GROUP_KIND_VARIANT):
        raise ValueError("skin targets require a setup or lighting-variant group")
    asset = session.get(MediaAsset, group.asset_id)
    if asset is None or subject.asset_id != group.asset_id:
        raise ValueError("subject and group belong to different assets")
    return subject, group, asset


def _bgr_mean_profile(mean_b: float, mean_g: float, mean_r: float) -> dict[str, list[float]]:
    """Convert a BGR skin mean to a robust OKLab opponent-chroma profile."""
    rgb = np.array([mean_r, mean_g, mean_b], dtype=np.float64)
    lab = rgb_linear_to_oklab(bt709_to_linear(rgb))
    return {
        "mean_ab": [float(lab[1]), float(lab[2])],
        "spread_ab": [0.0, 0.0],
    }


def _measure_region_profile(image_bgr: np.ndarray, box: tuple[int, int, int, int]) -> dict | None:
    """Measure a masked skin profile for one box, intersecting the conservative
    colour skin classifier. Returns ``None`` when the box has no skin pixels."""
    x, y, w, h = (int(v) for v in box)
    if w <= 0 or h <= 0:
        return None
    region = image_bgr[y : y + h, x : x + w]
    if region.size == 0:
        return None
    mask = skin_mask(region)
    if not mask.any():
        return None
    rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    return masked_skin_profile(bt709_to_linear(rgb), mask.astype(np.float64))


def _measure_external_profile(
    image_bgr: np.ndarray, crop_geometry: dict | None, detect=detect_faces
) -> dict | None:
    """Measure a face/skin profile from an external image.

    Uses ``crop_geometry`` (normalized ``x/y/w/h``) when supplied; otherwise
    detects the largest face. Returns ``None`` when no face/skin can be
    established (callers must then reject rather than default to a neutral
    placeholder).
    """
    ah, aw = image_bgr.shape[:2]
    if crop_geometry:
        cx, cy, cw, ch = (
            float(crop_geometry["x"]) * aw,
            float(crop_geometry["y"]) * ah,
            float(crop_geometry["w"]) * aw,
            float(crop_geometry["h"]) * ah,
        )
        return _measure_region_profile(image_bgr, (int(cx), int(cy), int(cw), int(ch)))
    boxes = detect(image_bgr)
    if not boxes:
        return None
    largest = max(boxes, key=lambda b: b[2] * b[3])
    return _measure_region_profile(image_bgr, largest)


def _copy_external(source: Path, store: ProjectStore, content_hash: str) -> Path:
    suffix = source.suffix or ".img"
    managed_dir = _project_root(store) / "references"
    managed_dir.mkdir(parents=True, exist_ok=True)
    managed = managed_dir / f"{content_hash}{suffix}"
    if not managed.exists():
        managed.write_bytes(source.read_bytes())
    return managed


def create_skin_reference(
    store: ProjectStore,
    *,
    subject_id: int,
    group_id: int,
    role: str,
    source_shot_id: int | None = None,
    frame_index: int | None = None,
    external_path: Path | None = None,
    crop_geometry: dict[str, float] | None = None,
) -> SkinAppearanceReference:
    """Create a provenance-backed skin reference (one source mode only).

    External images are copied into project-managed storage by content hash
    (the original is never modified). A project-frame reference must live in
    the target group and contain the target subject's face.
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    if (source_shot_id is None) == (external_path is None):
        raise ValueError("provide exactly one of source_shot_id or external_path")

    with store.session() as session:
        subject, group, asset = _load_scope(session, subject_id, group_id)

        if external_path is not None:
            external_path = Path(external_path)
            if not external_path.exists():
                raise ValueError(f"external reference file does not exist: {external_path}")
            source_kind = SOURCE_EXTERNAL_IMAGE
            data = external_path.read_bytes()
            content_hash = hashlib.sha256(data).hexdigest()
            managed = _copy_external(external_path, store, content_hash)
            source_path = str(external_path)
            managed_path = str(managed)
            source_shot = None
            image = cv2.imread(str(managed), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("external reference image could not be read")
            profile = _measure_external_profile(image, crop_geometry)
            if profile is None:
                raise ValueError(
                    "external reference has no detectable face/skin; supply a reviewed "
                    "crop_geometry or a different image"
                )
        else:
            if frame_index is None:
                raise ValueError("project-frame reference requires frame_index")
            source_kind = SOURCE_PROJECT_FRAME
            source_shot = session.get(Shot, source_shot_id)
            if source_shot is None or source_shot.asset_id != asset.id:
                raise ValueError("project-frame reference shot must belong to the group's asset")
            if source_shot.group_id != group_id:
                raise ValueError("project-frame reference shot must be inside the target group")
            metric = (
                session.query(SkinMetric)
                .filter_by(shot_id=source_shot.id, subject_id=subject_id)
                .first()
            )
            if metric is None:
                raise ValueError("project-frame reference must contain the target subject's face")
            content_hash = hashlib.sha256(
                f"{asset.source_path}:{frame_index}:{source_shot_id}".encode()
            ).hexdigest()
            source_path = asset.source_path
            managed_path = None
            profile = _bgr_mean_profile(metric.mean_b, metric.mean_g, metric.mean_r)

        ref = SkinAppearanceReference(
            subject_id=subject_id,
            asset_id=asset.id,
            group_id=group_id,
            source_kind=source_kind,
            role=role,
            source_path=source_path,
            managed_path=managed_path,
            content_hash=content_hash,
            source_shot_id=source_shot.id if source_shot else None,
            frame_index=frame_index,
            crop_geometry=crop_geometry or {},
            profile=profile,
            state=REF_STATE_ACTIVE,
        )
        session.add(ref)
        session.flush()
        session.refresh(ref)
        return ref


def request_skin_reference(
    store: ProjectStore, *, asset_id: int, subject_id: int, group_id: int, rationale: str
) -> Note:
    """Record an agent's request for a better skin reference (a reviewable note)."""
    with store.session() as session:
        if session.get(Subject, subject_id) is None:
            raise ValueError(f"subject {subject_id} not found")
        note = Note(
            asset_id=asset_id,
            subject_id=subject_id,
            shot_id=None,
            author="agent",
            text=f"skin reference request: {rationale}",
        )
        session.add(note)
        session.flush()
        session.refresh(note)
        return note


def review_face_mask(
    store: ProjectStore,
    mask_track_id: int,
    *,
    review_state: str,
    reason: str,
    author: str = "agent",
) -> FaceMaskTrack:
    """Record a review state on a mask track (evidence gate only).

    ``author="human"`` records an explicit human approval (required for the
    lower-confidence fallback mask); ``author="agent"`` is evidence inspection
    only.
    """
    if review_state not in REVIEW_STATES:
        raise ValueError(f"review_state must be one of {REVIEW_STATES}")
    with store.session() as session:
        mask = session.get(FaceMaskTrack, mask_track_id)
        if mask is None:
            raise ValueError(f"face mask track {mask_track_id} not found")
        if review_state == "approved_for_proposal" and mask.state != "valid":
            raise ValueError("a non-valid mask track cannot be approved for proposals")
        mask.review_state = review_state
        mask.review_reason = reason
        if author == "human":
            mask.human_approved = True
        session.flush()
        session.refresh(mask)
        return mask


def _resolve_reviewed_mask_for_track(session, face_track_id: int, subject_id: int) -> FaceMaskTrack:
    """Resolve the reviewed FaceMaskTrack that belongs to a FaceTrack.

    Never assumes ``FaceMaskTrack.id == FaceTrack.id``; the two tables have
    independent primary-key sequences. Validates the linkage (same subject and
    shot), quality bounds, and the proposal gate (reviewed + human-approved
    fallback).
    """
    track = session.get(FaceTrack, face_track_id)
    if track is None:
        raise ValueError(f"face track {face_track_id} not found")
    if track.state != "valid":
        raise ValueError(f"face track {face_track_id} is not valid")
    mask = (
        session.query(FaceMaskTrack)
        .filter_by(face_track_id=face_track_id)
        .order_by(FaceMaskTrack.id.desc())
        .first()
    )
    if mask is None:
        raise ValueError(f"face track {face_track_id} has no face mask track")
    if mask.shot_id != track.shot_id:
        raise ValueError("face mask track belongs to a different shot")
    if mask.subject_id != subject_id:
        raise ValueError("face mask track belongs to a different subject")
    if mask.state != "valid":
        raise ValueError(f"face mask track {mask.id} is not valid ({mask.state})")
    if mask.review_state != "approved_for_proposal":
        raise ValueError(f"face mask track {mask.id} must be reviewed 'approved_for_proposal'")
    if mask.backend == "fallback" and not mask.human_approved:
        raise ValueError(f"face mask track {mask.id} is a fallback mask and requires human approval")
    if mask.coverage < MIN_COVERAGE:
        raise ValueError(f"face mask track {mask.id} coverage below threshold")
    if mask.max_gap > MAX_GAP_RATIO:
        raise ValueError(f"face mask track {mask.id} gap exceeds threshold")
    return mask


def suggest_skin_target(
    store: ProjectStore,
    *,
    subject_id: int,
    group_id: int,
    reference_id: int | None,
    face_track_id: int,
    parameters: dict,
    rationale: str,
    confidence: float,
    profile: dict | None = None,
) -> SkinAppearanceTarget:
    """Draft a ``suggested`` skin target after the mask evidence is reviewed.

    Resolves the reviewed mask track that belongs to ``face_track_id`` (never
    assumes aligned primary keys), measures a real source profile, applies the
    approved appearance parameters exactly once to produce the canonical target
    profile, and persists both. A zero/neutral placeholder is never accepted.
    """
    params = validate_skin_appearance_parameters(parameters)
    with store.session() as session:
        _load_scope(session, subject_id, group_id)
        mask = _resolve_reviewed_mask_for_track(session, face_track_id, subject_id)

        ref = session.get(SkinAppearanceReference, reference_id) if reference_id else None
        if reference_id is not None and ref is None:
            raise ValueError(f"skin reference {reference_id} not found")
        if ref is not None:
            if ref.state != REF_STATE_ACTIVE:
                raise ValueError("skin reference is not active")
            if ref.subject_id != subject_id or ref.group_id != group_id:
                raise ValueError("skin reference is outside the subject/group scope")
            source_profile = ref.profile
        else:
            source_profile = profile

        if source_profile is None:
            track = session.get(FaceTrack, face_track_id)
            metric = session.get(SkinMetric, track.skin_metric_id) if track else None
            if metric is not None:
                source_profile = _bgr_mean_profile(metric.mean_b, metric.mean_g, metric.mean_r)

        if not source_profile or not source_profile.get("mean_ab"):
            raise ValueError(
                "a skin target requires a real measured source profile; no accurate "
                "reference, external measurement, or supplied profile is available"
            )

        canonical = apply_profile_transform(source_profile, params)

        target = SkinAppearanceTarget(
            subject_id=subject_id,
            group_id=group_id,
            reference_id=reference_id,
            mask_track_id=mask.id,
            profile=source_profile,
            canonical_profile=canonical,
            approved_preview_parameters=dict(parameters),
            state=TARGET_SUGGESTED,
            rationale=rationale,
            confidence=float(confidence),
        )
        session.add(target)
        session.flush()
        session.refresh(target)
        return target


def _set_target_state(store: ProjectStore, target_id: int, state: str) -> SkinAppearanceTarget | None:
    with store.session() as session:
        target = session.get(SkinAppearanceTarget, target_id)
        if target is None:
            return None
        if state == TARGET_APPROVED and target.state != TARGET_SUGGESTED:
            raise ValueError("only a suggested target can be approved")
        target.state = state
        session.flush()
        session.refresh(target)
        return target


def approve_skin_target(store: ProjectStore, target_id: int) -> SkinAppearanceTarget:
    """Human action: approve a suggested skin target (does not enable grades)."""
    target = _set_target_state(store, target_id, TARGET_APPROVED)
    if target is None:
        raise ValueError(f"skin target {target_id} not found")
    return target


def reject_skin_target(store: ProjectStore, target_id: int) -> SkinAppearanceTarget:
    target = _set_target_state(store, target_id, TARGET_REJECTED)
    if target is None:
        raise ValueError(f"skin target {target_id} not found")
    return target


def _latest_reviewed_mask(session, face_track_id: int) -> FaceMaskTrack | None:
    return (
        session.query(FaceMaskTrack)
        .filter_by(face_track_id=face_track_id)
        .order_by(FaceMaskTrack.id.desc())
        .first()
    )


def _latest_valid_track(session, skin_metric_id: int) -> FaceTrack | None:
    return (
        session.query(FaceTrack)
        .filter_by(skin_metric_id=skin_metric_id, state="valid")
        .order_by(FaceTrack.id.desc())
        .first()
    )


def match_skin_target_to_group(
    store: ProjectStore, *, target_id: int, group_id: int
) -> dict:
    """Derive per-shot ``skin_appearance`` evidence for an approved target.

    Enumerates only the exact setup/variant scope, requires an approved target
    and reviewed valid masks, and derives each candidate's parameters from its
    own masked profile. Failed gates become ``uncertain`` evidence, never a
    grade. This function never persists or enables a correction.
    """
    with store.session() as session:
        target = session.get(SkinAppearanceTarget, target_id)
        if target is None:
            raise ValueError(f"skin target {target_id} not found")
        if target.state != TARGET_APPROVED:
            raise ValueError("skin target must be approved before matching")
        if group_id != target.group_id:
            raise ValueError("candidate group must be the exact target setup or variant")

        group = session.get(ShotGroup, group_id)
        if group is None or group.kind not in (GROUP_KIND_SETUP, GROUP_KIND_VARIANT):
            raise ValueError("candidate group must be a setup or lighting variant")

        ref = session.get(SkinAppearanceReference, target.reference_id) if target.reference_id else None
        role = ref.role if ref else ROLE_ACCURATE
        # Candidates move fully toward the canonical target for an accurate
        # reference; a creative look direction is a softer, lower-confidence
        # shift (capped at 0.50).
        strength = CREATIVE_STRENGTH_CAP if role == ROLE_CREATIVE else 1.0

        member_ids = [
            s.id for s in session.query(Shot).filter_by(group_id=group_id).order_by(Shot.index).all()
        ]
        metrics = (
            session.query(SkinMetric)
            .filter(SkinMetric.subject_id == target.subject_id, SkinMetric.shot_id.in_(member_ids))
            .order_by(SkinMetric.shot_id, SkinMetric.face_index)
            .all()
        )

        # The canonical profile is the single authoritative appearance target
        # (the source profile with the approved parameters applied once).
        target_profile = target.canonical_profile or target.profile or {"mean_ab": [0.0, 0.0], "spread_ab": [0.0, 0.0]}

        def _mask_ready(mask) -> bool:
            if mask is None or mask.state != "valid" or mask.review_state != "approved_for_proposal":
                return False
            if mask.backend == "fallback" and not mask.human_approved:
                return False
            return True

        def candidate_for(m: SkinMetric) -> dict:
            track = _latest_valid_track(session, m.id)
            mask = _latest_reviewed_mask(session, track.id) if track else None
            candidate_profile = _bgr_mean_profile(m.mean_b, m.mean_g, m.mean_r)
            if track is None:
                return {"shot_id": m.shot_id, "skin_metric_id": m.id, "state": "uncertain", "reason": "no valid face track"}
            if not _mask_ready(mask):
                return {"shot_id": m.shot_id, "skin_metric_id": m.id, "face_track_id": track.id, "state": "uncertain", "reason": "mask not reviewed for proposal"}
            params = derive_skin_appearance_parameters(candidate_profile, target_profile, strength=strength)
            parameters = {
                "version": 1, "space": "oklab", "luma_mode": "preserve",
                "ab_offset": list(params.ab_offset),
                "ab_scale": list(params.ab_scale),
                "strength": params.strength,
            }
            return {
                "shot_id": m.shot_id,
                "skin_metric_id": m.id,
                "face_track_id": track.id,
                "mask_track_id": mask.id,
                "state": "ready",
                "parameters": parameters,
                "reference_role": role,
            }

        candidates = [candidate_for(m) for m in metrics]
        status = "ok" if len(metrics) >= 2 else "qc_only"
        return {
            "status": status,
            "target_id": target_id,
            "subject_id": target.subject_id,
            "group_id": group_id,
            "reference_role": role,
            "candidates": candidates,
        }


def propose_skin_appearance_correction(
    store: ProjectStore,
    *,
    shot_id: int,
    subject_id: int,
    skin_metric_id: int | None,
    face_track_id: int,
    skin_target_id: int,
    parameters: dict,
    reason: str,
    confidence: float,
    evidence: dict | None = None,
) -> FaceCorrection:
    """Draft a *suggested, disabled* ``skin_appearance`` correction (agent only).

    Enforces the skin-first scope invariants at proposal time: the track
    subject/metric must match, the candidate shot must be inside the target's
    exact group, and the target must be approved with the same subject/scope.
    """
    validate_skin_appearance_parameters(parameters)
    with store.session() as session:
        track = session.get(FaceTrack, face_track_id)
        if track is None:
            raise ValueError(f"face track {face_track_id} not found")
        if track.state != "valid":
            raise ValueError(f"face track {face_track_id} is not valid")
        if track.shot_id != shot_id:
            raise ValueError("face track belongs to a different shot")
        if track.subject_id != subject_id:
            raise ValueError("face track belongs to a different subject")
        if skin_metric_id is not None and track.skin_metric_id != skin_metric_id:
            raise ValueError("face track belongs to a different skin metric")

        shot = session.get(Shot, shot_id)
        if shot is None:
            raise ValueError(f"shot {shot_id} not found")
        if skin_metric_id is not None:
            metric = session.get(SkinMetric, skin_metric_id)
            if metric is None or metric.shot_id != shot_id or metric.subject_id != subject_id:
                raise ValueError("skin metric does not belong to the requested subject/shot")

        target = session.get(SkinAppearanceTarget, skin_target_id)
        if target is None or target.state != TARGET_APPROVED:
            raise ValueError("skin target must be approved")
        if target.subject_id != subject_id:
            raise ValueError("skin target subject does not match the requested subject")
        if shot.group_id != target.group_id:
            raise ValueError("candidate shot is outside the target's exact group")

        mask = _resolve_reviewed_mask_for_track(session, face_track_id, subject_id)

        correction = FaceCorrection(
            shot_id=shot_id,
            subject_id=subject_id,
            skin_metric_id=skin_metric_id,
            face_track_id=face_track_id,
            skin_target_id=skin_target_id,
            reference_group_id=target.group_id,
            reference_shot_id=None,
            kind="skin_appearance",
            parameters=dict(parameters),
            evidence=evidence,
            reason=reason,
            confidence=float(confidence),
            classification="skin_mismatch",
            state="suggested",
            enabled=False,
        )
        session.add(correction)
        session.flush()
        session.refresh(correction)
        return correction
