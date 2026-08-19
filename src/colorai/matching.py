"""Group-aware matching: compare shots only within an approved scope.

The default comparison mode must preserve intent, so ColorAI does **not**
compare every shot to a global median. Matching happens only inside an
explicit, human/agent-defined scope:

    subject × setup family (× optional camera angle, via the group's label)

A scope is only matchable once it has an **approved reference** (a human
decision — see :mod:`colorai.references`). Whole-frame proposals are
deterministic, include their reference and group context, and are persisted
disabled unless asked for. Face-derived skin proposals are **report-only** —
applying them whole-frame would regrade the background, so they are never
persisted until correction rows can carry a tracked, feathered temporal face
mask. Skin is compared **within the subject only** — never across subjects.
Global median matching remains available as an explicit diagnostic
(`colorai.analysis.find_outliers`), not a default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from colorai.analysis import (
    ProposedCorrection,
    load_shot_feature,
    propose_corrections,
)
from colorai.project.models import Correction, Shot, ShotGroup
from colorai.project.store import ProjectStore
from colorai.references import approved_reference_for_scope, effective_reference_shot_id
from colorai.skin_analysis import FaceSkin, face_features, propose_skin_match, skin_balance_correction

GROUP_KIND_SETUP = "setup"
GROUP_KIND_VARIANT = "variant"


@dataclass(frozen=True)
class MatchProposal:
    """One shot's deviation from an approved reference, with full context."""

    shot_id: int
    reference_shot_id: int
    subject_id: int
    group_id: int | None
    reasons: tuple[str, ...]
    #: Whole-frame proposals (exposure/balance/saturation vs the reference).
    corrections: tuple[ProposedCorrection, ...]
    #: Face-region skin proposals (``rgb_balance`` derived from skin samples).
    #: These are **report-only**: applying them whole-frame would regrade the
    #: background, so they are never persisted until correction rows can carry
    #: a tracked, feathered temporal face mask used by preview and render alike.
    skin_corrections: tuple[ProposedCorrection, ...]


@dataclass(frozen=True)
class VariantSkinDeviation:
    """A lighting variant whose subject face-skin differs from the subject
    baseline.

    Cross-variant *whole-frame* differences (window light, background) are
    expected and never corrected; only face/skin consistency is checked. The
    correction is a face-region proposal (report-only, never a whole-frame
    grade).
    """

    variant_id: int
    subject_id: int
    distance: float
    is_issue: bool
    correction: ProposedCorrection | None = None


def _reference_skin(subject_faces: list[FaceSkin], ref_shot_id: int) -> np.ndarray | None:
    """Median skin of the subject's faces in the reference shot (BGR, 0..1)."""
    refs = [f for f in subject_faces if f.shot_id == ref_shot_id]
    if not refs:
        return None
    return np.median([np.array([f.b, f.g, f.r], dtype=np.float64) for f in refs], axis=0)


def match_subject_in_group(
    store: ProjectStore,
    asset_id: int,
    *,
    subject_id: int,
    group_id: int | None = None,
    persist: bool = False,
) -> tuple[list[MatchProposal], str | None]:
    """Propose corrections for a subject's shots against an approved reference.

    Scope: the subject's shots inside ``group_id`` (or all the subject's shots
    when ``group_id`` is ``None``). Requires an approved reference for the
    scope; returns an explanatory ``error`` (with no proposals) otherwise.

    ``persist=True`` stores the proposals as **disabled** ``Correction`` rows
    (still deterministic, non-destructive, and unapplied until a human enables
    them). ``persist=False`` only returns them.
    """
    faces = face_features(store, asset_id)
    subject_faces = [f for f in faces if f.subject_id == subject_id]
    if not subject_faces:
        return [], "subject has no face samples in this asset"

    is_variant = False
    member_shot_ids = {f.shot_id for f in subject_faces}
    if group_id is not None:
        with store.session() as session:
            group = session.get(ShotGroup, group_id)
            if group is None or group.asset_id != asset_id:
                return [], "group not found for this asset"
            if group.kind not in (GROUP_KIND_SETUP, GROUP_KIND_VARIANT):
                return [], "matching is only defined for interview setups or lighting variants"
            is_variant = group.kind == GROUP_KIND_VARIANT
            member_shot_ids &= {
                s.id for s in session.query(Shot).filter_by(group_id=group_id).all()
            }
        if not member_shot_ids:
            return [], "no member shots for this subject in the group"

    if is_variant:
        # A lighting variant has its own approved reference; never inherit the
        # subject hero shot from a different lighting condition.
        ref_shot_id = approved_reference_for_scope(
            store, asset_id=asset_id, subject_id=subject_id, group_id=group_id
        )
        if ref_shot_id is None:
            return [], (
                "no approved reference for this lighting variant — propose and "
                "approve a reference within the variant before matching"
            )
    else:
        ref_shot_id = effective_reference_shot_id(
            store, asset_id=asset_id, subject_id=subject_id, group_id=group_id
        )
        if ref_shot_id is None:
            return [], (
                "no approved reference for this subject/setup scope — propose a "
                "reference and have the human approve it before matching"
            )
    if ref_shot_id not in member_shot_ids:
        return [], "the approved reference shot is not a member of this scope"

    ref_feature = load_shot_feature(store, ref_shot_id)
    ref_skin = _reference_skin(subject_faces, ref_shot_id)

    proposals: list[MatchProposal] = []
    for shot_id in sorted(member_shot_ids):
        if shot_id == ref_shot_id:
            continue
        reasons: list[str] = []
        corrections: list[ProposedCorrection] = []
        skin_corrections: list[ProposedCorrection] = []

        feature = load_shot_feature(store, shot_id)
        if ref_feature is not None and feature is not None:
            deviation = propose_corrections(ref_feature, feature)
            reasons.extend(deviation.reasons)
            corrections.extend(deviation.corrections)

        if ref_skin is not None:
            for face in subject_faces:
                if face.shot_id != shot_id:
                    continue
                skin_fix = propose_skin_match(ref_skin, face)
                if skin_fix is not None:
                    reasons.append("skin")
                    skin_corrections.append(skin_fix)

        if corrections or skin_corrections:
            proposals.append(
                MatchProposal(
                    shot_id=shot_id,
                    reference_shot_id=ref_shot_id,
                    subject_id=subject_id,
                    group_id=group_id,
                    reasons=tuple(dict.fromkeys(reasons)),
                    corrections=tuple(corrections),
                    skin_corrections=tuple(skin_corrections),
                )
            )

    # Persist only whole-frame proposals, and only as disabled rows. Face-
    # region skin proposals are report-only (see MatchProposal) and are never
    # persisted here.
    if persist and proposals:
        with store.session() as session:
            for proposal in proposals:
                for correction in proposal.corrections:
                    session.add(
                        Correction(
                            shot_id=proposal.shot_id,
                            kind=correction.kind,
                            parameters=correction.parameters,
                            enabled=False,  # never auto-applied
                        )
                    )

    return proposals, None


def cross_variant_skin_consistency(
    store: ProjectStore,
    asset_id: int,
    *,
    subject_id: int,
    family_group_id: int,
    tolerance: float = 0.06,
) -> tuple[list[VariantSkinDeviation], str | None]:
    """Check the subject's *face skin* across a setup family's lighting variants.

    Whole-frame differences between variants (window light, background) are
    expected and intentionally **not** corrected here. Only face/skin
    consistency is compared: each variant's median face skin is measured
    against the subject's baseline (its hero shot, else the median of its own
    faces), and a deviation beyond ``tolerance`` is reported as a real issue
    with a **face-region** ``rgb_balance`` proposal. That proposal is
    report-only: it is never persisted as a whole-frame grade (see
    :class:`MatchProposal`). Returns ``(deviations, error)``.
    """
    import numpy as np

    faces = face_features(store, asset_id)
    subject_faces = [f for f in faces if f.subject_id == subject_id]
    if not subject_faces:
        return [], "subject has no face samples in this asset"

    with store.session() as session:
        family = session.get(ShotGroup, family_group_id)
        if family is None or family.asset_id != asset_id or family.kind != GROUP_KIND_SETUP:
            return [], "family group not found or is not a setup family"
        variants = (
            session.query(ShotGroup)
            .filter_by(parent_id=family_group_id, kind=GROUP_KIND_VARIANT)
            .order_by(ShotGroup.id)
            .all()
        )
        variant_shot_ids: dict[int, set[int]] = {}
        for variant in variants:
            variant_shot_ids[variant.id] = {
                s.id for s in session.query(Shot).filter_by(group_id=variant.id).all()
            }

    if not variants:
        return [], "no lighting variants defined for this family"

    def _rgb(f: FaceSkin) -> np.ndarray:
        return np.array([f.b, f.g, f.r], dtype=np.float64)

    baseline = None
    ref_shot_id = effective_reference_shot_id(store, asset_id=asset_id, subject_id=subject_id)
    if ref_shot_id is not None:
        refs = [f for f in subject_faces if f.shot_id == ref_shot_id]
        if refs:
            baseline = np.median([_rgb(f) for f in refs], axis=0)
    if baseline is None:
        baseline = np.median([_rgb(f) for f in subject_faces], axis=0)

    deviations: list[VariantSkinDeviation] = []
    for variant in variants:
        variant_faces = [
            f for f in subject_faces if f.shot_id in variant_shot_ids[variant.id]
        ]
        if not variant_faces:
            continue
        variant_skin = np.median([_rgb(f) for f in variant_faces], axis=0)
        distance = float(np.linalg.norm(variant_skin - baseline))
        correction = skin_balance_correction(
            baseline, (variant_skin[0], variant_skin[1], variant_skin[2]), tolerance=tolerance
        )
        deviations.append(
            VariantSkinDeviation(
                variant_id=variant.id,
                subject_id=subject_id,
                distance=distance,
                is_issue=correction is not None,
                correction=correction,
            )
        )
    return deviations, None
