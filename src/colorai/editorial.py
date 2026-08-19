"""Editorial intelligence: human review state, exceptions, grouping, split/merge.

These are the "editorial" operations that sit above the deterministic engine:
the filmmaker can mark shots approved/rejected, flag a shot's deviation as an
*intentional exception* (so the outlier detector leaves it alone), cluster
shots into scene/camera-family groups, and manually split or merge shots when
automatic detection got a boundary wrong.

Structural edits (split/merge) reset the asset to ``registered`` so a later
``analyze`` re-run re-derives the representative stills/metrics/skin for the
edited shots — without re-running shot detection, which would overwrite the
manual edit.
"""

from __future__ import annotations

from colorai.core.timecode import frames_to_timecode
from colorai.project.models import (
    Correction,
    MediaAsset,
    Shot,
    ShotGroup,
)
from colorai.project.store import ProjectStore

REVIEW_PENDING = "pending"
REVIEW_APPROVED = "approved"
REVIEW_REJECTED = "rejected"
REVIEW_STATUSES = (REVIEW_PENDING, REVIEW_APPROVED, REVIEW_REJECTED)

# Large temporary offset used to renumber shot indices without hitting the
# (asset_id, index) uniqueness constraint mid-statement.
_INDEX_OFFSET = 10_000_000


def _shift_indices(session, asset_id: int, above_index: int, by: int) -> None:
    """Shift every shot with ``index > above_index`` by ``by`` (+1 or -1).

    Uses a temporary offset so SQLite's per-row unique check never collides.
    """
    q = session.query(Shot).filter(Shot.asset_id == asset_id, Shot.index > above_index)
    q.update({Shot.index: Shot.index + _INDEX_OFFSET}, synchronize_session=False)
    session.query(Shot).filter(
        Shot.asset_id == asset_id, Shot.index > above_index + _INDEX_OFFSET
    ).update(
        {Shot.index: Shot.index - _INDEX_OFFSET + by}, synchronize_session=False
    )


# ---------------------------------------------------------------------------
# Review state + exceptions
# ---------------------------------------------------------------------------

def set_review_status(store: ProjectStore, shot_id: int, status: str) -> Shot | None:
    """Set a shot's review status (pending/approved/rejected)."""
    if status not in REVIEW_STATUSES:
        raise ValueError(f"invalid review status {status!r}")
    with store.session() as session:
        shot = session.get(Shot, shot_id)
        if shot is None:
            return None
        shot.review_status = status
        session.flush()
        session.refresh(shot)
        return shot


def set_excused(store: ProjectStore, shot_id: int, excused: bool) -> Shot | None:
    """Mark/unmark a shot as an intentional exception (skip outlier proposals)."""
    with store.session() as session:
        shot = session.get(Shot, shot_id)
        if shot is None:
            return None
        shot.excused = excused
        session.flush()
        session.refresh(shot)
        return shot


# ---------------------------------------------------------------------------
# Scene / camera-family grouping
# ---------------------------------------------------------------------------

GROUP_KIND_SETUP = "setup"
GROUP_KIND_GENERIC = "generic"
GROUP_KIND_VARIANT = "variant"
GROUP_KINDS = (GROUP_KIND_GENERIC, GROUP_KIND_SETUP, GROUP_KIND_VARIANT)


def create_group(
    store: ProjectStore,
    asset_id: int,
    name: str,
    *,
    kind: str = GROUP_KIND_GENERIC,
    camera: str | None = None,
    parent_id: int | None = None,
) -> ShotGroup:
    """Create a shot group (scene / camera family / interview setup / variant).

    ``kind="setup"`` marks an interview/setup family (the matching unit);
    ``kind="variant"`` marks a **lighting variant** within a setup family (its
    ``parent_id`` must point to that family), each carrying its own approved
    reference. ``camera`` is an optional human/agent-assigned angle label.
    """
    if kind not in GROUP_KINDS:
        raise ValueError(f"invalid group kind {kind!r}")
    with store.session() as session:
        if session.get(MediaAsset, asset_id) is None:
            raise ValueError(f"asset {asset_id} not found")
        if kind == GROUP_KIND_VARIANT:
            if parent_id is None:
                raise ValueError("a variant group requires a parent setup family")
            parent = session.get(ShotGroup, parent_id)
            if parent is None or parent.asset_id != asset_id or parent.kind != GROUP_KIND_SETUP:
                raise ValueError("variant parent must be a setup family in the same asset")
        elif parent_id is not None:
            raise ValueError("only 'variant' groups may have a parent")

        group = ShotGroup(
            asset_id=asset_id, name=name, kind=kind, camera=camera, parent_id=parent_id
        )
        session.add(group)
        session.flush()
        session.refresh(group)
        return group


def update_group(
    store: ProjectStore,
    group_id: int,
    *,
    name: str | None = None,
    camera: str | None = None,
    kind: str | None = None,
) -> ShotGroup | None:
    """Update a group's name, camera label, and/or kind.

    Pass ``camera=""`` to clear the label. Kind/camera are human/agent
    decisions — never inferred from pixels.
    """
    if kind is not None and kind not in GROUP_KINDS:
        raise ValueError(f"invalid group kind {kind!r}")
    with store.session() as session:
        group = session.get(ShotGroup, group_id)
        if group is None:
            return None
        if name is not None:
            group.name = name
        if camera is not None:
            group.camera = camera or None
        if kind is not None:
            group.kind = kind
        session.flush()
        session.refresh(group)
        return group


def list_groups(store: ProjectStore, asset_id: int) -> list[ShotGroup]:
    with store.session() as session:
        return (
            session.query(ShotGroup)
            .filter_by(asset_id=asset_id)
            .order_by(ShotGroup.id)
            .all()
        )


def rename_group(store: ProjectStore, group_id: int, name: str) -> ShotGroup | None:
    with store.session() as session:
        group = session.get(ShotGroup, group_id)
        if group is None:
            return None
        group.name = name
        session.flush()
        session.refresh(group)
        return group


def delete_group(store: ProjectStore, group_id: int) -> None:
    """Delete a group; its shots revert to ungrouped (``group_id`` NULL)."""
    with store.session() as session:
        session.query(Shot).filter(Shot.group_id == group_id).update(
            {Shot.group_id: None}, synchronize_session=False
        )
        group = session.get(ShotGroup, group_id)
        if group is not None:
            session.delete(group)


def assign_shot_group(store: ProjectStore, shot_id: int, group_id: int) -> Shot | None:
    with store.session() as session:
        shot = session.get(Shot, shot_id)
        group = session.get(ShotGroup, group_id)
        if shot is None or group is None:
            return None
        if group.asset_id != shot.asset_id:
            raise ValueError("shot and group belong to different assets")
        shot.group_id = group_id
        session.flush()
        session.refresh(shot)
        return shot


def unassign_shot_group(store: ProjectStore, shot_id: int) -> Shot | None:
    with store.session() as session:
        shot = session.get(Shot, shot_id)
        if shot is None:
            return None
        shot.group_id = None
        session.flush()
        session.refresh(shot)
        return shot


# ---------------------------------------------------------------------------
# Manual split / merge
# ---------------------------------------------------------------------------

def _new_shot(
    asset_id: int,
    index: int,
    start: int,
    end: int,
    fps: float,
    *,
    review_status: str,
    excused: bool,
    group_id: int | None,
) -> Shot:
    return Shot(
        asset_id=asset_id,
        index=index,
        start_frame=start,
        end_frame=end,
        start_timecode=frames_to_timecode(start, fps),
        end_timecode=frames_to_timecode(end, fps),
        review_status=review_status,
        excused=excused,
        group_id=group_id,
    )


def split_shot(store: ProjectStore, shot_id: int, at_frame: int) -> tuple[Shot, Shot]:
    """Split one shot into two at ``at_frame`` (which must be strictly inside).

    The two halves inherit the original shot's corrections, review state, and
    group; analysis artifacts (still/metrics/skin) are dropped and the asset is
    reset to ``registered`` so a re-analyze re-derives them for the new shots.
    """
    # Read the shot and its corrections in a short session so the mutation
    # session's identity map stays clean (SQLite can reuse rowids after delete).
    with store.session() as session:
        shot = session.get(Shot, shot_id)
        if shot is None:
            raise ValueError(f"shot {shot_id} not found")
        if not (shot.start_frame < at_frame <= shot.end_frame):
            raise ValueError(
                f"split frame {at_frame} must be strictly inside "
                f"[{shot.start_frame}, {shot.end_frame}]"
            )
        asset_id = shot.asset_id
        fps = shot.asset.frame_rate
        start, end = shot.start_frame, shot.end_frame
        k, group_id = shot.index, shot.group_id
        review_status, excused = shot.review_status, shot.excused
        corrections = [
            (c.kind, c.parameters, c.enabled)
            for c in session.query(Correction)
            .filter_by(shot_id=shot_id)
            .order_by(Correction.id)
            .all()
        ]

    with store.session() as session:
        shot = session.get(Shot, shot_id)
        session.delete(shot)  # cascades its still/metrics/skin/corrections
        session.flush()

        _shift_indices(session, asset_id, k, by=1)

        first = _new_shot(
            asset_id, k, start, at_frame - 1, fps,
            review_status=review_status, excused=excused, group_id=group_id,
        )
        second = _new_shot(
            asset_id, k + 1, at_frame, end, fps,
            review_status=review_status, excused=excused, group_id=group_id,
        )
        session.add_all([first, second])
        session.flush()

        for kind, params, enabled in corrections:
            for target in (first, second):
                session.add(
                    Correction(
                        shot_id=target.id,
                        kind=kind,
                        parameters=params,
                        enabled=enabled,
                    )
                )

        session.query(MediaAsset).filter(MediaAsset.id == asset_id).update(
            {"status": "registered"}
        )
        session.flush()
        session.refresh(first)
        session.refresh(second)
        return first, second


def merge_shots(store: ProjectStore, shot_id_a: int, shot_id_b: int) -> Shot:
    """Merge two adjacent shots into one (lower-index shot survives).

    The survivor keeps its own corrections and appends the dropped shot's;
    analysis artifacts of the dropped shot are removed and the asset is reset
    to ``registered`` for a re-analyze.
    """
    with store.session() as session:
        a = session.get(Shot, shot_id_a)
        b = session.get(Shot, shot_id_b)
        if a is None or b is None:
            raise ValueError("both shots must exist")
        if a.asset_id != b.asset_id:
            raise ValueError("shots belong to different assets")

        first, second = (a, b) if a.index < b.index else (b, a)
        if first.index + 1 != second.index:
            raise ValueError("shots must be adjacent in index to merge")
        if first.end_frame + 1 != second.start_frame:
            raise ValueError("shots must be contiguous in frame bounds to merge")

        asset_id = first.asset_id
        fps = first.asset.frame_rate
        dropped_index = second.index
        new_start, new_end = first.start_frame, second.end_frame
        sec_corrections = [
            (c.kind, c.parameters, c.enabled)
            for c in session.query(Correction)
            .filter_by(shot_id=second.id)
            .order_by(Correction.id)
            .all()
        ]

    with store.session() as session:
        first = session.get(Shot, first.id)
        second = session.get(Shot, second.id)
        session.delete(second)  # cascades its still/metrics/skin/corrections
        session.flush()

        first.start_frame = new_start
        first.end_frame = new_end
        first.start_timecode = frames_to_timecode(new_start, fps)
        first.end_timecode = frames_to_timecode(new_end, fps)

        for kind, params, enabled in sec_corrections:
            session.add(
                Correction(
                    shot_id=first.id,
                    kind=kind,
                    parameters=params,
                    enabled=enabled,
                )
            )

        _shift_indices(session, asset_id, dropped_index, by=-1)

        session.query(MediaAsset).filter(MediaAsset.id == asset_id).update(
            {"status": "registered"}
        )
        session.flush()
        session.refresh(first)
        return first
