"""Per-subject skin-tone consistency analysis.

Skin tone is compared **within a subject**, not globally: two people must not
be pulled toward each other's skin, and the film's dominant look must be
preserved. Faces are grouped into subjects by skin color (a deterministic
greedy clustering — a heuristic, not an identity claim), then each subject's
median skin tone is its reference and every face that deviates from it gets a
proposed ``rgb_balance`` correction.

This is measurement and proposal, not decision: nothing is applied without
approval, and a human can always override the automatic clustering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from colorai.analysis import ProposedCorrection
from colorai.project.models import Shot, SkinMetric
from colorai.project.store import ProjectStore

MIN_GAIN = 0.25
MAX_GAIN = 4.0
DEFAULT_CLUSTER_DISTANCE = 0.12
DEFAULT_MATCH_TOLERANCE = 0.06


@dataclass(frozen=True)
class FaceSkin:
    """One face's skin signature (normalized BGR, 0..1)."""

    shot_id: int
    face_index: int
    b: float
    g: float
    r: float


@dataclass(frozen=True)
class SkinDeviation:
    """One face's deviation from its subject's reference skin tone."""

    shot_id: int
    face_index: int
    subject_id: int
    distance: float
    is_outlier: bool
    corrections: tuple[ProposedCorrection, ...] = ()


def _rgb(f: FaceSkin) -> np.ndarray:
    return np.array([f.b, f.g, f.r], dtype=np.float64)


def _clamp(value: float) -> float:
    return max(MIN_GAIN, min(MAX_GAIN, value))


def skin_features(store: ProjectStore, asset_id: int) -> list[FaceSkin]:
    """Collect every face's skin signature for an asset, in shot order."""
    with store.session() as session:
        rows = (
            session.query(SkinMetric)
            .join(Shot, SkinMetric.shot_id == Shot.id)
            .filter(Shot.asset_id == asset_id)
            .order_by(Shot.index, SkinMetric.face_index)
            .all()
        )
    return [
        FaceSkin(
            row.shot_id,
            row.face_index,
            float(row.mean_b),
            float(row.mean_g),
            float(row.mean_r),
        )
        for row in rows
    ]


def cluster_by_skin(
    features: list[FaceSkin], *, max_distance: float = DEFAULT_CLUSTER_DISTANCE
) -> list[list[FaceSkin]]:
    """Greedy single-pass clustering of faces by skin color.

    Deterministic given a sorted input (callers pass features in shot order).
    ``max_distance`` is the Euclidean distance in normalized BGR space within
    which a face joins an existing subject.
    """
    clusters: list[list[FaceSkin]] = []
    for face in sorted(features, key=lambda f: (f.shot_id, f.face_index)):
        placed = False
        for cluster in clusters:
            centroid = np.mean([_rgb(m) for m in cluster], axis=0)
            if float(np.linalg.norm(_rgb(face) - centroid)) <= max_distance:
                cluster.append(face)
                placed = True
                break
        if not placed:
            clusters.append([face])
    return clusters


def propose_skin_match(
    reference: np.ndarray, target: FaceSkin, *, tolerance: float = DEFAULT_MATCH_TOLERANCE
) -> ProposedCorrection | None:
    """Propose an ``rgb_balance`` correction to match a face to a reference skin.

    ``reference`` is a normalized BGR ``(b, g, r)`` vector; gains are clamped
    so an extreme deviation cannot produce an extreme correction.
    """
    gains = [
        float(_clamp(reference[i] / t)) if t > 1e-3 else 1.0
        for i, t in enumerate((target.b, target.g, target.r))
    ]
    if max(abs(g - 1.0) for g in gains) > tolerance:
        return ProposedCorrection("rgb_balance", {"gain": [round(g, 4) for g in gains]})
    return None


def skin_consistency(
    store: ProjectStore,
    asset_id: int,
    *,
    cluster_distance: float = DEFAULT_CLUSTER_DISTANCE,
    match_tolerance: float = DEFAULT_MATCH_TOLERANCE,
) -> list[SkinDeviation]:
    """Group faces into subjects and flag skin-tone deviations within each.

    Returns one :class:`SkinDeviation` per face (including non-outliers, so the
    caller can see every subject's spread).
    """
    features = skin_features(store, asset_id)
    if not features:
        return []

    deviations: list[SkinDeviation] = []
    for subject_id, cluster in enumerate(cluster_by_skin(features, max_distance=cluster_distance)):
        reference = np.median([_rgb(f) for f in cluster], axis=0)
        for face in cluster:
            distance = float(np.linalg.norm(_rgb(face) - reference))
            correction = propose_skin_match(reference, face, tolerance=match_tolerance)
            deviations.append(
                SkinDeviation(
                    shot_id=face.shot_id,
                    face_index=face.face_index,
                    subject_id=subject_id,
                    distance=distance,
                    is_outlier=correction is not None,
                    corrections=(correction,) if correction else (),
                )
            )
    return deviations
