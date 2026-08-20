"""Durable organization-plan domain: draft, validate, approve, apply.

An MCP/vision agent drafts a plan that gives every shot of an asset a proposed
editorial destination; a human reviews it; applying the approved plan is one
atomic transaction. The module is deliberately separate from ``editorial``
(which mutates live structure) — a plan is a *proposal* that never changes the
asset until it is approved and applied.

The rules implement `docs/superpowers/specs/2026-08-19-organization-plan-design.md`.
"""

from __future__ import annotations

from typing import Any, Sequence

from colorai.editorial import (
    GROUP_KIND_BROLL,
    GROUP_KIND_SETUP,
    GROUP_KIND_VARIANT,
)
from colorai.project.models import (
    MediaAsset,
    OrganizationPlan,
    OrganizationPlanGroup,
    OrganizationPlanItem,
    ReferenceProposal,
    Shot,
    ShotGroup,
    SkinMetric,
    Subject,
    utcnow,
)
from colorai.project.store import ProjectStore

# Plan states.
STATE_DRAFT = "draft"
STATE_APPROVED = "approved"
STATE_APPLIED = "applied"
STATE_SUPERSEDED = "superseded"
STATES = (STATE_DRAFT, STATE_APPROVED, STATE_APPLIED, STATE_SUPERSEDED)

# Item decisions.
DECISION_PROPOSED = "proposed"
DECISION_ACCEPTED = "accepted"
DECISION_REJECTED = "rejected"
DECISIONS = (DECISION_PROPOSED, DECISION_ACCEPTED, DECISION_REJECTED)

# Item destinations.
DEST_EXISTING_GROUP = "existing_group"
DEST_PLANNED_GROUP = "planned_group"
DEST_BROLL = "broll"
DEST_INTENTIONAL_EXCEPTION = "intentional_exception"
DEST_UNRESOLVED = "unresolved"
DESTINATIONS = (
    DEST_EXISTING_GROUP,
    DEST_PLANNED_GROUP,
    DEST_BROLL,
    DEST_INTENTIONAL_EXCEPTION,
    DEST_UNRESOLVED,
)

# Only these group kinds may appear in a plan or be an assignment target.
_TARGETABLE_KINDS = (GROUP_KIND_SETUP, GROUP_KIND_VARIANT, GROUP_KIND_BROLL)

_LEGACY_BROLL_NAMES = {"broll", "b-roll", "b roll"}


# ---------------------------------------------------------------------------
# B-roll group helpers
# ---------------------------------------------------------------------------

def find_broll_group(session, asset_id: int) -> ShotGroup | None:
    """Return the asset's canonical B-roll group, if any.

    Prefers the explicit ``kind="broll"`` group; falls back to a legacy
    ``generic`` group named ``broll``/``b-roll`` so older projects stay
    readable until normal use migrates them.
    """
    groups = (
        session.query(ShotGroup)
        .filter_by(asset_id=asset_id)
        .order_by(ShotGroup.id)
        .all()
    )
    for g in groups:
        if g.kind == GROUP_KIND_BROLL:
            return g
    for g in groups:
        if g.kind == "generic" and (g.name or "").strip().lower() in _LEGACY_BROLL_NAMES:
            return g
    return None


def _broll_group_id(session, asset_id: int) -> int:
    """Return the canonical B-roll group id, creating it if necessary."""
    existing = find_broll_group(session, asset_id)
    if existing is not None:
        return existing.id
    group = ShotGroup(asset_id=asset_id, name="B-roll", kind=GROUP_KIND_BROLL)
    session.add(group)
    session.flush()
    return group.id


# ---------------------------------------------------------------------------
# Structural validation (used at create/update time)
# ---------------------------------------------------------------------------

def _resolve_group_payloads(groups: Sequence[dict]) -> dict[str, dict]:
    by_key: dict[str, dict] = {}
    for g in groups:
        key = g.get("draft_key")
        if key is None:
            raise ValueError("every planned group needs a draft_key")
        if key in by_key:
            raise ValueError(f"duplicate draft_key {key!r}")
        by_key[key] = g
    return by_key


def structural_errors(
    session,
    asset_id: int,
    groups: Sequence[dict],
    items: Sequence[dict],
) -> list[str]:
    """Blocking structural errors for a plan *payload* (before persistence)."""
    errors: list[str] = []

    shot_ids = {s.id for s in session.query(Shot).filter_by(asset_id=asset_id).all()}
    group_rows = {
        g.id: g
        for g in session.query(ShotGroup).filter_by(asset_id=asset_id).all()
    }
    subject_ids = {s.id for s in session.query(Subject).filter_by(asset_id=asset_id).all()}

    by_key: dict[str, dict] = {}
    for g in groups:
        key = g.get("draft_key")
        if key is None:
            errors.append("every planned group needs a draft_key")
            continue
        if key in by_key:
            errors.append(f"duplicate draft key {key!r}")
            continue
        by_key[key] = g

        kind = g.get("kind", GROUP_KIND_SETUP)
        if kind not in (GROUP_KIND_SETUP, GROUP_KIND_VARIANT):
            errors.append(f"planned group {key!r} has invalid kind {kind!r}")
        parent_key = g.get("parent_draft_key")
        if kind == GROUP_KIND_VARIANT and not parent_key:
            errors.append(f"variant {key!r} has no setup parent")
        if kind == GROUP_KIND_SETUP and parent_key:
            errors.append(f"setup {key!r} must not have a parent")

        existing_id = g.get("existing_group_id")
        if existing_id is not None:
            row = group_rows.get(existing_id)
            if row is None:
                errors.append(f"planned group {key!r} references unknown group {existing_id}")
            elif row.kind not in (GROUP_KIND_SETUP, GROUP_KIND_VARIANT):
                errors.append(f"planned group {key!r} maps onto non-setup group {existing_id}")

        for pid in g.get("participant_ids") or []:
            if pid not in subject_ids:
                errors.append(f"planned group {key!r} references unknown subject {pid}")

    # A variant's parent must resolve to a planned or existing *setup* key.
    for key, g in by_key.items():
        parent_key = g.get("parent_draft_key")
        if parent_key and parent_key not in by_key:
            errors.append(f"variant {key!r} parent {parent_key!r} is not a planned group")

    seen_shots: set[int] = set()
    for i, item in enumerate(items):
        shot_id = item.get("shot_id")
        if shot_id not in shot_ids:
            errors.append(f"item {i} references unknown shot {shot_id}")
            continue
        if shot_id in seen_shots:
            errors.append(f"shot {shot_id} has more than one destination")
        seen_shots.add(shot_id)

        decision = item.get("decision", DECISION_PROPOSED)
        if decision not in DECISIONS:
            errors.append(f"item {i} has invalid decision {decision!r}")

        dst = item.get("destination_type")
        if dst not in DESTINATIONS:
            errors.append(f"item {i} has invalid destination {dst!r}")
            continue
        if dst == DEST_EXISTING_GROUP:
            gid = item.get("target_group_id")
            row = group_rows.get(gid)
            if row is None:
                errors.append(f"item {i} targets unknown group {gid}")
            elif row.kind not in _TARGETABLE_KINDS:
                errors.append(
                    f"item {i} targets group {gid} that is not a setup, variant, or B-roll group"
                )
        elif dst == DEST_PLANNED_GROUP:
            key = item.get("target_draft_key")
            if key not in by_key:
                errors.append(f"item {i} targets unknown planned group {key!r}")

    return errors


# ---------------------------------------------------------------------------
# Plan read/validation
# ---------------------------------------------------------------------------

def _resulting_state(
    session,
    plan: OrganizationPlan,
    plan_groups: dict[str, OrganizationPlanGroup],
    items: Sequence[OrganizationPlanItem],
    broll_id: int | None,
):
    """Compute the post-apply group/excused state for every shot.

    Returns ``(resulting_group, resulting_excused)`` keyed by shot id, starting
    from the current structure and applying only *accepted* items.
    """
    shots = session.query(Shot).filter_by(asset_id=plan.asset_id).all()
    resulting_group = {s.id: s.group_id for s in shots}
    resulting_excused = {s.id: s.excused for s in shots}

    # Resolve planned draft keys to group ids (reuse existing, else synthetic
    # "new" markers; we only need equality for reference checks, so reuse the
    # existing id when present and a unique negative placeholder otherwise).
    created: dict[str, int] = {}
    placeholder = [-1]

    def resolve(key: str) -> int:
        if key in created:
            return created[key]
        pg = plan_groups.get(key)
        if pg is None:
            return placeholder[0]
        if pg.existing_group_id is not None:
            created[key] = pg.existing_group_id
        else:
            created[key] = placeholder[0]
            placeholder[0] -= 1
        return created[key]

    for item in items:
        if item.decision != DECISION_ACCEPTED:
            continue
        shot = item.shot_id
        dst = item.destination_type
        if dst == DEST_EXISTING_GROUP:
            resulting_group[shot] = item.target_group_id
            resulting_excused[shot] = False
        elif dst == DEST_PLANNED_GROUP:
            resulting_group[shot] = resolve(item.target_draft_key) if item.target_draft_key else None
            resulting_excused[shot] = False
        elif dst == DEST_BROLL:
            resulting_group[shot] = broll_id
            resulting_excused[shot] = False
        elif dst == DEST_INTENTIONAL_EXCEPTION:
            resulting_group[shot] = None
            resulting_excused[shot] = True
        # unresolved: leave current state untouched.

    return resulting_group, resulting_excused


def _validate(session, plan: OrganizationPlan) -> tuple[list[str], list[str]]:
    """Full validation: blocking errors and non-blocking warnings."""
    errors: list[str] = []
    warnings: list[str] = []

    shots = session.query(Shot).filter_by(asset_id=plan.asset_id).all()
    shot_by_id = {s.id: s for s in shots}
    groups = (
        session.query(OrganizationPlanGroup).filter_by(plan_id=plan.id).all()
    )
    items = (
        session.query(OrganizationPlanItem).filter_by(plan_id=plan.id).all()
    )
    group_by_key = {g.draft_key: g for g in groups}
    item_by_shot = {i.shot_id: i for i in items}

    # Structural (payload-independent) checks.
    for g in groups:
        if g.kind not in (GROUP_KIND_SETUP, GROUP_KIND_VARIANT):
            errors.append(f"planned group {g.draft_key!r} has invalid kind {g.kind!r}")
        if g.kind == GROUP_KIND_VARIANT and not g.parent_draft_key:
            errors.append(f"variant {g.draft_key!r} has no setup parent")
        if g.kind == GROUP_KIND_SETUP and g.parent_draft_key:
            errors.append(f"setup {g.draft_key!r} must not have a parent")
        if g.parent_draft_key and g.parent_draft_key not in group_by_key:
            errors.append(f"variant {g.draft_key!r} parent {g.parent_draft_key!r} is missing")

    for item in items:
        if item.shot_id not in shot_by_id:
            errors.append(f"item references unknown shot {item.shot_id}")
            continue
        if item.destination_type not in DESTINATIONS:
            errors.append(f"shot {item.shot_id} has invalid destination {item.destination_type!r}")
        if item.destination_type == DEST_EXISTING_GROUP:
            target = session.get(ShotGroup, item.target_group_id)
            if target is None or target.asset_id != plan.asset_id:
                errors.append(f"shot {item.shot_id} targets unknown group {item.target_group_id}")
            elif target.kind not in _TARGETABLE_KINDS:
                errors.append(
                    f"shot {item.shot_id} targets non-setup/variant/B-roll group {item.target_group_id}"
                )
        elif item.destination_type == DEST_PLANNED_GROUP and item.target_draft_key not in group_by_key:
            errors.append(f"shot {item.shot_id} targets unknown planned group {item.target_draft_key!r}")

    # Complete coverage: a draft is expected to give every shot a destination.
    for shot in shots:
        if shot.id not in item_by_shot:
            errors.append(f"shot {shot.id} is missing from the plan (incomplete coverage)")

    # Compute post-apply state for reference-scope and empty-group checks.
    existing_broll = find_broll_group(session, plan.asset_id)
    broll_id = existing_broll.id if existing_broll is not None else None
    resulting_group, resulting_excused = _resulting_state(
        session, plan, group_by_key, items, broll_id
    )

    # Human-approved references must stay inside their exact scope after apply.
    face_subjects: dict[int, set[int]] = {}
    for m in session.query(SkinMetric).filter(SkinMetric.shot_id.in_(shot_by_id.keys())).all():
        if m.subject_id is not None:
            face_subjects.setdefault(m.shot_id, set()).add(m.subject_id)

    for ref in session.query(ReferenceProposal).filter_by(asset_id=plan.asset_id, state="approved").all():
        if ref.group_id is not None and resulting_group.get(ref.shot_id) != ref.group_id:
            errors.append(
                f"approved reference {ref.id} points outside its group scope after apply"
            )
        if ref.subject_id is not None and ref.subject_id not in face_subjects.get(ref.shot_id, set()):
            errors.append(
                f"approved reference {ref.id} points at a shot without its participant"
            )

    # Warnings (non-blocking).
    unresolved = [i for i in items if i.decision != DECISION_ACCEPTED]
    if unresolved:
        warnings.append(f"{len(unresolved)} shot(s) remain unresolved after the plan")

    # Group-size / participant warnings.
    accepted_by_key: dict[str, int] = {}
    for i in items:
        if i.decision == DECISION_ACCEPTED and i.destination_type == DEST_PLANNED_GROUP:
            accepted_by_key[i.target_draft_key] = accepted_by_key.get(i.target_draft_key, 0) + 1

    for g in groups:
        if g.kind != GROUP_KIND_SETUP:
            continue
        member_count = accepted_by_key.get(g.draft_key, 0)
        participant_count = len(g.participant_ids or [])
        if member_count <= 1:
            warnings.append(f"setup {g.name!r} has one shot — prefer QC/manual finishing")
        if participant_count == 0:
            warnings.append(f"setup {g.name!r} has no participant faces")
        elif participant_count == 1:
            warnings.append(f"setup {g.name!r} has a single participant")
        if member_count == 0:
            warnings.append(f"setup {g.name!r} would be empty")

    # Existing groups that become empty.
    member_counts: dict[int, int] = {}
    for shot_id, gid in resulting_group.items():
        if gid is not None:
            member_counts[gid] = member_counts.get(gid, 0) + 1
    for g in session.query(ShotGroup).filter_by(asset_id=plan.asset_id).all():
        if g.id not in member_counts and g.id != broll_id:
            warnings.append(f"group {g.name!r} becomes empty (not deleted automatically)")

    # Accepted B-roll that still contains assigned faces.
    for i in items:
        if (
            i.decision == DECISION_ACCEPTED
            and i.destination_type == DEST_BROLL
            and face_subjects.get(i.shot_id)
        ):
            warnings.append(f"accepted B-roll shot {i.shot_id} contains assigned faces")

    return errors, warnings


# ---------------------------------------------------------------------------
# Public plan operations
# ---------------------------------------------------------------------------

def create_organization_plan(
    store: ProjectStore,
    asset_id: int,
    groups: Sequence[dict],
    items: Sequence[dict],
    summary: str = "",
    author: str = "agent",
) -> OrganizationPlan:
    """Store a draft after structural validation. Never changes the asset."""
    with store.session() as session:
        if session.get(MediaAsset, asset_id) is None:
            raise ValueError(f"asset {asset_id} not found")
        errors = structural_errors(session, asset_id, groups, items)
        if errors:
            raise ValueError("; ".join(errors))

        # A new draft supersedes earlier un-applied drafts.
        session.query(OrganizationPlan).filter(
            OrganizationPlan.asset_id == asset_id,
            OrganizationPlan.state.in_((STATE_DRAFT, STATE_APPROVED)),
        ).update({OrganizationPlan.state: STATE_SUPERSEDED}, synchronize_session=False)

        plan = OrganizationPlan(
            asset_id=asset_id, state=STATE_DRAFT, author=author, summary=summary
        )
        session.add(plan)
        session.flush()

        for g in groups:
            session.add(
                OrganizationPlanGroup(
                    plan_id=plan.id,
                    draft_key=g["draft_key"],
                    name=g.get("name", g["draft_key"]),
                    kind=g.get("kind", GROUP_KIND_SETUP),
                    camera=g.get("camera"),
                    parent_draft_key=g.get("parent_draft_key"),
                    existing_group_id=g.get("existing_group_id"),
                    participant_ids=g.get("participant_ids"),
                    reason=g.get("reason"),
                    confidence=g.get("confidence"),
                )
            )
        for item in items:
            session.add(
                OrganizationPlanItem(
                    plan_id=plan.id,
                    shot_id=item["shot_id"],
                    decision=item.get("decision", DECISION_PROPOSED),
                    destination_type=item["destination_type"],
                    target_group_id=item.get("target_group_id"),
                    target_draft_key=item.get("target_draft_key"),
                    reason=item.get("reason"),
                    confidence=item.get("confidence"),
                    evidence=item.get("evidence"),
                    human_override_reason=item.get("human_override_reason"),
                )
            )
        session.flush()
        session.refresh(plan)
        return plan


def get_organization_plan(store: ProjectStore, plan_id: int) -> dict | None:
    with store.session() as session:
        plan = session.get(OrganizationPlan, plan_id)
        if plan is None:
            return None
        return _plan_dict(session, plan)


def list_organization_plans(store: ProjectStore, asset_id: int) -> list[dict]:
    with store.session() as session:
        plans = (
            session.query(OrganizationPlan)
            .filter_by(asset_id=asset_id)
            .order_by(OrganizationPlan.id.desc())
            .all()
        )
        return [_plan_dict(session, p) for p in plans]


def _plan_dict(session, plan: OrganizationPlan) -> dict:
    groups = (
        session.query(OrganizationPlanGroup).filter_by(plan_id=plan.id).all()
    )
    items = session.query(OrganizationPlanItem).filter_by(plan_id=plan.id).all()
    return {
        "id": plan.id,
        "asset_id": plan.asset_id,
        "state": plan.state,
        "author": plan.author,
        "summary": plan.summary,
        "created_at": plan.created_at.isoformat() if plan.created_at else None,
        "approved_by": plan.approved_by,
        "approved_at": plan.approved_at.isoformat() if plan.approved_at else None,
        "applied_at": plan.applied_at.isoformat() if plan.applied_at else None,
        "groups": [
            {
                "id": g.id,
                "draft_key": g.draft_key,
                "name": g.name,
                "kind": g.kind,
                "camera": g.camera,
                "parent_draft_key": g.parent_draft_key,
                "existing_group_id": g.existing_group_id,
                "participant_ids": g.participant_ids,
                "reason": g.reason,
                "confidence": g.confidence,
            }
            for g in groups
        ],
        "items": [
            {
                "id": i.id,
                "shot_id": i.shot_id,
                "decision": i.decision,
                "destination_type": i.destination_type,
                "target_group_id": i.target_group_id,
                "target_draft_key": i.target_draft_key,
                "reason": i.reason,
                "confidence": i.confidence,
                "evidence": i.evidence,
                "human_override_reason": i.human_override_reason,
            }
            for i in items
        ],
    }


def validate_organization_plan(store: ProjectStore, plan_id: int) -> dict:
    """Return blocking errors and warnings without changing anything."""
    with store.session() as session:
        plan = session.get(OrganizationPlan, plan_id)
        if plan is None:
            return {"error": "plan not found"}
        errors, warnings = _validate(session, plan)
        return {"plan_id": plan_id, "errors": errors, "warnings": warnings}


def update_organization_plan_item(
    store: ProjectStore,
    plan_id: int,
    shot_id: int,
    *,
    decision: str | None = None,
    destination_type: str | None = None,
    target_group_id: int | None = None,
    target_draft_key: str | None = None,
    human_override_reason: str | None = None,
) -> dict | None:
    with store.session() as session:
        plan = session.get(OrganizationPlan, plan_id)
        if plan is None or plan.state != STATE_DRAFT:
            return None
        item = (
            session.query(OrganizationPlanItem)
            .filter_by(plan_id=plan_id, shot_id=shot_id)
            .first()
        )
        if item is None:
            return None
        if decision is not None:
            if decision not in DECISIONS:
                raise ValueError(f"invalid decision {decision!r}")
            item.decision = decision
        if destination_type is not None:
            if destination_type not in DESTINATIONS:
                raise ValueError(f"invalid destination {destination_type!r}")
            item.destination_type = destination_type
        if target_group_id is not None:
            item.target_group_id = target_group_id
        if target_draft_key is not None:
            item.target_draft_key = target_draft_key
        if human_override_reason is not None:
            item.human_override_reason = human_override_reason
        session.flush()
        session.refresh(item)
        return {
            "id": item.id,
            "shot_id": item.shot_id,
            "decision": item.decision,
            "destination_type": item.destination_type,
            "target_group_id": item.target_group_id,
            "target_draft_key": item.target_draft_key,
            "human_override_reason": item.human_override_reason,
        }


def update_organization_plan_group(
    store: ProjectStore,
    plan_id: int,
    draft_key: str,
    *,
    name: str | None = None,
    camera: str | None = None,
    kind: str | None = None,
    parent_draft_key: str | None = None,
) -> dict | None:
    with store.session() as session:
        plan = session.get(OrganizationPlan, plan_id)
        if plan is None or plan.state != STATE_DRAFT:
            return None
        group = (
            session.query(OrganizationPlanGroup)
            .filter_by(plan_id=plan_id, draft_key=draft_key)
            .first()
        )
        if group is None:
            return None
        if name is not None:
            group.name = name
        if camera is not None:
            group.camera = camera or None
        if kind is not None:
            if kind not in (GROUP_KIND_SETUP, GROUP_KIND_VARIANT):
                raise ValueError(f"invalid group kind {kind!r}")
            group.kind = kind
        if parent_draft_key is not None:
            group.parent_draft_key = parent_draft_key
        session.flush()
        session.refresh(group)
        return {
            "id": group.id,
            "draft_key": group.draft_key,
            "name": group.name,
            "kind": group.kind,
            "camera": group.camera,
            "parent_draft_key": group.parent_draft_key,
        }


def approve_organization_plan(
    store: ProjectStore, plan_id: int, approved_by: str = "human"
) -> dict:
    with store.session() as session:
        plan = session.get(OrganizationPlan, plan_id)
        if plan is None:
            return {"error": "plan not found"}
        if plan.state != STATE_DRAFT:
            return {"error": f"plan must be a draft (state={plan.state})"}
        plan.state = STATE_APPROVED
        plan.approved_by = approved_by
        plan.approved_at = utcnow()
        session.flush()
        return {"id": plan.id, "state": plan.state, "approved_by": plan.approved_by}


def apply_organization_plan(store: ProjectStore, plan_id: int) -> dict:
    """Atomically apply an approved plan; re-validate inside the transaction."""
    with store.session() as session:
        plan = session.get(OrganizationPlan, plan_id)
        if plan is None:
            return {"error": "plan not found"}
        if plan.state != STATE_APPROVED:
            return {"error": f"plan must be approved (state={plan.state})"}

        errors, warnings = _validate(session, plan)
        if errors:
            return {"error": "validation failed", "errors": errors}

        plan_groups = {
            g.draft_key: g
            for g in session.query(OrganizationPlanGroup).filter_by(plan_id=plan.id).all()
        }
        items = session.query(OrganizationPlanItem).filter_by(plan_id=plan.id).all()
        shots = {
            s.id: s for s in session.query(Shot).filter_by(asset_id=plan.asset_id).all()
        }

        broll_id = _broll_group_id(session, plan.asset_id)
        created_by_key: dict[str, int] = {}
        created_groups: list[dict] = []
        reused_groups: list[dict] = []
        changed_shots: list[dict] = []

        def ensure_group(key: str) -> int:
            if key in created_by_key:
                return created_by_key[key]
            pg = plan_groups.get(key)
            if pg is None:
                raise ValueError(f"unknown planned group {key!r}")
            if pg.existing_group_id is not None:
                created_by_key[key] = pg.existing_group_id
                reused_groups.append({"draft_key": key, "group_id": pg.existing_group_id})
                return pg.existing_group_id
            parent_id = ensure_group(pg.parent_draft_key) if pg.parent_draft_key else None
            g = ShotGroup(
                asset_id=plan.asset_id,
                name=pg.name,
                kind=pg.kind,
                camera=pg.camera,
                parent_id=parent_id,
            )
            session.add(g)
            session.flush()
            created_by_key[key] = g.id
            created_groups.append(
                {"draft_key": key, "group_id": g.id, "name": g.name, "kind": g.kind}
            )
            return g.id

        for item in items:
            if item.decision != DECISION_ACCEPTED:
                continue
            shot = shots.get(item.shot_id)
            if shot is None:
                continue
            dst = item.destination_type
            if dst == DEST_EXISTING_GROUP:
                shot.group_id = item.target_group_id
                shot.excused = False
            elif dst == DEST_PLANNED_GROUP:
                shot.group_id = ensure_group(item.target_draft_key) if item.target_draft_key else None
                shot.excused = False
            elif dst == DEST_BROLL:
                shot.group_id = broll_id
                shot.excused = False
            elif dst == DEST_INTENTIONAL_EXCEPTION:
                shot.group_id = None
                shot.excused = True
            else:  # unresolved / rejected: leave untouched
                continue
            changed_shots.append(
                {"shot_id": shot.id, "group_id": shot.group_id, "excused": shot.excused}
            )

        plan.state = STATE_APPLIED
        plan.applied_at = utcnow()
        session.flush()

        return {
            "plan_id": plan.id,
            "state": plan.state,
            "warnings": warnings,
            "created_groups": created_groups,
            "reused_groups": reused_groups,
            "changed_shots": changed_shots,
        }
