"""Human-approved reference-shot proposals for matching scopes.

A vision agent proposes a reference (hero) shot for a matching scope — a
subject, a setup group, or a subject within a setup group — with a reason and
a confidence. Proposals start ``suggested``; only an explicit human
approve/reject moves them. An **approved** proposal is the effective reference
for group-aware matching in its scope, and approving a subject-scoped proposal
also sets the subject's hero shot (the existing ``Subject.reference_shot_id``
semantics).

Nothing is applied automatically from a proposal: it is a decision record, not
a grade.
"""

from __future__ import annotations

from colorai.project.models import (
    MediaAsset,
    ReferenceProposal,
    Shot,
    ShotGroup,
    Subject,
)
from colorai.project.store import ProjectStore

STATE_SUGGESTED = "suggested"
STATE_APPROVED = "approved"
STATE_REJECTED = "rejected"
STATES = (STATE_SUGGESTED, STATE_APPROVED, STATE_REJECTED)


def _validate_scope(
    store: ProjectStore, asset_id: int, subject_id: int | None, group_id: int | None
) -> None:
    with store.session() as session:
        if session.get(MediaAsset, asset_id) is None:
            raise ValueError(f"asset {asset_id} not found")
        if subject_id is not None:
            subject = session.get(Subject, subject_id)
            if subject is None or subject.asset_id != asset_id:
                raise ValueError(f"subject {subject_id} does not belong to asset {asset_id}")
        if group_id is not None:
            group = session.get(ShotGroup, group_id)
            if group is None or group.asset_id != asset_id:
                raise ValueError(f"group {group_id} does not belong to asset {asset_id}")


def propose_reference(
    store: ProjectStore,
    *,
    asset_id: int,
    shot_id: int,
    reason: str,
    confidence: float,
    subject_id: int | None = None,
    group_id: int | None = None,
    author: str = "agent",
) -> ReferenceProposal:
    """Record a suggested reference shot for a subject and/or setup group."""
    if not reason.strip():
        raise ValueError("reason must not be empty")
    if not (0.0 <= float(confidence) <= 1.0):
        raise ValueError("confidence must be within [0, 1]")
    _validate_scope(store, asset_id, subject_id, group_id)

    with store.session() as session:
        if session.get(Shot, shot_id) is None:
            raise ValueError(f"shot {shot_id} not found")
        proposal = ReferenceProposal(
            asset_id=asset_id,
            subject_id=subject_id,
            group_id=group_id,
            shot_id=shot_id,
            author=author,
            reason=reason,
            confidence=float(confidence),
            state=STATE_SUGGESTED,
        )
        session.add(proposal)
        session.flush()
        session.refresh(proposal)
        return proposal


def list_reference_proposals(store: ProjectStore, asset_id: int) -> list[ReferenceProposal]:
    with store.session() as session:
        return (
            session.query(ReferenceProposal)
            .filter_by(asset_id=asset_id)
            .order_by(ReferenceProposal.id)
            .all()
        )


def get_proposal(store: ProjectStore, proposal_id: int) -> ReferenceProposal | None:
    with store.session() as session:
        return session.get(ReferenceProposal, proposal_id)


def _set_state(store: ProjectStore, proposal_id: int, state: str) -> ReferenceProposal | None:
    if state not in STATES:
        raise ValueError(f"invalid reference state {state!r}")
    with store.session() as session:
        proposal = session.get(ReferenceProposal, proposal_id)
        if proposal is None:
            return None
        proposal.state = state
        if state == STATE_APPROVED and proposal.subject_id is not None:
            subject = session.get(Subject, proposal.subject_id)
            if subject is not None:
                subject.reference_shot_id = proposal.shot_id
        session.flush()
        session.refresh(proposal)
        return proposal


def approve_reference(store: ProjectStore, proposal_id: int) -> ReferenceProposal | None:
    """Approve a proposal (the human's explicit reference decision)."""
    return _set_state(store, proposal_id, STATE_APPROVED)


def reject_reference(store: ProjectStore, proposal_id: int) -> ReferenceProposal | None:
    return _set_state(store, proposal_id, STATE_REJECTED)


def effective_reference_shot_id(
    store: ProjectStore,
    *,
    asset_id: int | None = None,
    subject_id: int | None = None,
    group_id: int | None = None,
) -> int | None:
    """The approved reference shot for a scope (or ``None`` when undecided).

    An approved proposal for the exact scope wins; otherwise a subject's
    explicitly-set hero shot (``Subject.reference_shot_id``) is the fallback.
    Both are human decisions — never an automatic default.
    """
    with store.session() as session:
        query = session.query(ReferenceProposal).filter(
            ReferenceProposal.state == STATE_APPROVED
        )
        if asset_id is not None:
            query = query.filter(ReferenceProposal.asset_id == asset_id)
        if subject_id is not None:
            query = query.filter(ReferenceProposal.subject_id == subject_id)
        else:
            query = query.filter(ReferenceProposal.subject_id.is_(None))
        if group_id is not None:
            query = query.filter(ReferenceProposal.group_id == group_id)
        else:
            query = query.filter(ReferenceProposal.group_id.is_(None))
        latest = query.order_by(ReferenceProposal.id.desc()).first()
        if latest is not None:
            return latest.shot_id

        if subject_id is not None:
            subject = session.get(Subject, subject_id)
            if subject is not None and subject.reference_shot_id is not None:
                return subject.reference_shot_id
    return None
