"""Deterministic organization suggestions for the review UI.

Classifies unassigned shots into three visual buckets — interview/setup
candidates, b-roll / non-interview material, and items needing judgment —
from existing persisted evidence (face rows, subject assignments, temporal
adjacency). Suggestions only: nothing is grouped, excused, or graded here.

Rules are deliberately simple and explainable:

- no face rows                     -> b-roll candidate
- >= 2 distinct assigned subjects  -> needs judgment (multi-person shot)
- faces but no assigned subject    -> needs judgment (weak identity evidence)
- exactly one assigned subject     -> interview candidate; adjacent shots
  (index gap <= ``adjacency_gap``) with the same subject form a cluster.

Every cluster carries a descriptive *suggested* label, an agent-readable
reason string, the relevant subject, and a representative shot — evidence for
a human decision, never identity truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ShotEvidence:
    """Per-shot evidence for classification."""

    shot_id: int
    index: int
    face_subject_ids: tuple[int | None, ...]  # one entry per detected face


@dataclass(frozen=True)
class InterviewCluster:
    """A suggested interview/setup cluster (never applied automatically)."""

    id: str
    label: str
    reason: str
    subject_id: int
    subject_name: str
    representative_shot_id: int
    member_shot_ids: tuple[int, ...]


@dataclass(frozen=True)
class JudgmentItem:
    """An ambiguous shot that needs a human decision."""

    shot_id: int
    reasons: tuple[str, ...]
    subject_ids: tuple[int, ...]


@dataclass(frozen=True)
class OrganizationResult:
    interview_clusters: tuple[InterviewCluster, ...]
    broll_shot_ids: tuple[int, ...]
    judgment: tuple[JudgmentItem, ...]


def suggest_organization(
    shots: Sequence[ShotEvidence],
    subject_names: Mapping[int, str],
    adjacency_gap: int = 3,
) -> OrganizationResult:
    """Classify unassigned shots into interview / b-roll / judgment buckets."""
    clusters: list[InterviewCluster] = []
    broll: list[int] = []
    judgment: list[JudgmentItem] = []

    current: dict | None = None  # in-progress cluster

    def finalize(cluster: dict) -> None:
        member_ids = cluster["member_shot_ids"]
        face_counts: dict[int, int] = cluster["face_counts"]
        representative = member_ids[0]
        for sid in member_ids:  # iteration order is index order; ties go earlier
            if face_counts[sid] > face_counts[representative]:
                representative = sid
        start, end = cluster["start_index"], cluster["last_index"]
        subject_id: int = cluster["subject_id"]
        name = subject_names.get(subject_id, f"subject {subject_id}")
        label = (
            f"{name} interview · shot {start}"
            if start == end
            else f"{name} interview · shots {start}–{end}"
        )
        count = len(member_ids)
        reason = (
            f"{count} adjacent shots share subject {name}"
            if count != 1
            else f"shot {start} shows subject {name}"
        )
        clusters.append(
            InterviewCluster(
                id=f"c{len(clusters) + 1}",
                label=label,
                reason=reason,
                subject_id=subject_id,
                subject_name=name,
                representative_shot_id=representative,
                member_shot_ids=tuple(member_ids),
            )
        )

    for shot in shots:
        assigned = sorted({sid for sid in shot.face_subject_ids if sid is not None})
        if not shot.face_subject_ids:
            broll.append(shot.shot_id)
            continue
        if len(assigned) >= 2:
            names = ", ".join(
                subject_names.get(sid, f"subject {sid}") for sid in assigned
            )
            judgment.append(
                JudgmentItem(
                    shot.shot_id,
                    (f"multi-person shot: {names}",),
                    tuple(assigned),
                )
            )
            continue
        if not assigned:
            judgment.append(
                JudgmentItem(
                    shot.shot_id,
                    ("face detected but no subject assigned",),
                    (),
                )
            )
            continue
        subject_id = assigned[0]
        if (
            current is None
            or current["subject_id"] != subject_id
            or shot.index - current["last_index"] > adjacency_gap
        ):
            if current is not None:
                finalize(current)
            current = {
                "subject_id": subject_id,
                "member_shot_ids": [shot.shot_id],
                "face_counts": {shot.shot_id: len(shot.face_subject_ids)},
                "start_index": shot.index,
                "last_index": shot.index,
            }
        else:
            current["member_shot_ids"].append(shot.shot_id)
            current["face_counts"][shot.shot_id] = len(shot.face_subject_ids)
            current["last_index"] = shot.index

    if current is not None:
        finalize(current)

    return OrganizationResult(
        interview_clusters=tuple(clusters),
        broll_shot_ids=tuple(broll),
        judgment=tuple(judgment),
    )
