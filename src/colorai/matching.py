"""Group-aware matching: compare shots only within an approved scope.

The default comparison mode must preserve intent, so ColorAI does **not**
compare every shot to a global median. Matching happens only inside an
explicit, human/agent-defined scope:

    subject × setup family (× optional camera angle, via the group's label)

A scope is only matchable once it has an **approved reference** (a human
decision — see :mod:`colorai.references`). Proposals are deterministic,
include their reference and group context, and are persisted disabled unless
asked for. Skin is compared **within the subject only** — never across
subjects. Global median matching remains available as an explicit diagnostic
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
from colorai.references import effective_reference_shot_id
from colorai.skin_analysis import FaceSkin, face_features, propose_skin_match


@dataclass(frozen=True)
class MatchProposal:
    """One shot's deviation from an approved reference, with full context."""

    shot_id: int
    reference_shot_id: int
    subject_id: int
    group_id: int | None
    reasons: tuple[str, ...]
    corrections: tuple[ProposedCorrection, ...]


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

    member_shot_ids = {f.shot_id for f in subject_faces}
    if group_id is not None:
        with store.session() as session:
            group = session.get(ShotGroup, group_id)
            if group is None or group.asset_id != asset_id:
                return [], "group not found for this asset"
            member_shot_ids &= {
                s.id for s in session.query(Shot).filter_by(group_id=group_id).all()
            }
        if not member_shot_ids:
            return [], "no member shots for this subject in the group"

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
                    corrections.append(skin_fix)

        if corrections:
            proposals.append(
                MatchProposal(
                    shot_id=shot_id,
                    reference_shot_id=ref_shot_id,
                    subject_id=subject_id,
                    group_id=group_id,
                    reasons=tuple(dict.fromkeys(reasons)),
                    corrections=tuple(corrections),
                )
            )

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
