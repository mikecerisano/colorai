"""Per-subject skin-tone consistency, keyed by *identity*, not skin color.

The grouping key is a face **identity embedding** (OpenCV SFace), because skin
color is exactly the thing that is broken and must never be the grouping
signal. Auto-assignment produces a first guess; the human owns the result via
the editable :class:`~colorai.project.models.Subject` entity (rename, reassign,
merge, split, set a reference shot).

Skin tone is then compared **within a subject**: each subject's reference is
its hero shot (if the human set one) or the median of its own faces, and every
face that deviates gets a proposed ``rgb_balance`` correction. Nothing is
applied without approval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from colorai.analysis import ProposedCorrection
from colorai.project.models import (
    RepresentativeFrame,
    Shot,
    SkinMetric,
    Subject,
)
from colorai.project.store import ProjectStore

MIN_GAIN = 0.25
MAX_GAIN = 4.0
DEFAULT_SIMILARITY = 0.4  # cosine threshold for "same person"
DEFAULT_MATCH_TOLERANCE = 0.06


@dataclass(frozen=True)
class FaceSkin:
    """One face's skin signature and subject membership."""

    skin_metric_id: int
    shot_id: int
    face_index: int
    subject_id: int | None
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


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


def _clamp(value: float) -> float:
    return max(MIN_GAIN, min(MAX_GAIN, value))


def face_features(store: ProjectStore, asset_id: int) -> list[FaceSkin]:
    """Collect every face's skin signature + subject membership for an asset."""
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
            row.id,
            row.shot_id,
            row.face_index,
            row.subject_id,
            float(row.mean_b),
            float(row.mean_g),
            float(row.mean_r),
        )
        for row in rows
    ]


def list_subjects(store: ProjectStore, asset_id: int) -> list[Subject]:
    with store.session() as session:
        return (
            session.query(Subject)
            .filter_by(asset_id=asset_id)
            .order_by(Subject.id)
            .all()
        )


# ---------------------------------------------------------------------------
# Auto-assignment (identity embeddings)
# ---------------------------------------------------------------------------

def cluster_embeddings(
    embeddings: list[np.ndarray], *, similarity: float = DEFAULT_SIMILARITY
) -> list[list[int]]:
    """Agglomerative (average-linkage) clustering of L2-normalized embeddings.

    Returns clusters of indices into ``embeddings``. Clustering over the whole
    set at once (rather than greedily against a running centroid) is more
    robust to noisy per-frame embeddings.
    """
    n = len(embeddings)
    if n == 0:
        return []
    clusters: list[list[int]] = [[i] for i in range(n)]
    centroids: list[np.ndarray] = [np.asarray(e, dtype=np.float64) for e in embeddings]

    while True:
        best_i, best_j, best_sim = -1, -1, -1.0
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                sim = float(np.dot(centroids[i], centroids[j]))
                if sim > best_sim:
                    best_i, best_j, best_sim = i, j, sim
        if best_i == -1 or best_sim < similarity:
            break
        clusters[best_i].extend(clusters[best_j])
        merged = np.mean([np.asarray(embeddings[k], dtype=np.float64) for k in clusters[best_i]], axis=0)
        centroids[best_i] = _unit(merged)
        del clusters[best_j]
        del centroids[best_j]

    return clusters


def auto_assign_subjects(
    store: ProjectStore, asset_id: int, *, similarity: float = DEFAULT_SIMILARITY
) -> list[Subject]:
    """Assign every face to a subject by identity embedding.

    Existing subjects for the asset are reset first. Embeddings are clustered
    with average linkage; each cluster becomes a subject. Aligns by
    ``face_index`` so skin and identity stay consistent.
    """
    import cv2

    from colorai.face import analyze_faces

    with store.session() as session:
        for subject in session.query(Subject).filter_by(asset_id=asset_id).all():
            session.delete(subject)
        session.flush()

        shots = session.query(Shot).filter_by(asset_id=asset_id).order_by(Shot.index).all()
        embeddings: list[np.ndarray] = []
        metric_ids: list[int] = []

        for shot in shots:
            rf = session.query(RepresentativeFrame).filter_by(shot_id=shot.id).first()
            if rf is None or not rf.image_path:
                continue
            image = cv2.imread(rf.image_path, cv2.IMREAD_COLOR)
            if image is None:
                continue

            faces = analyze_faces(image)
            skin_by_index = {
                m.face_index: m
                for m in session.query(SkinMetric).filter_by(shot_id=shot.id).all()
            }
            for index, face in enumerate(faces):
                embedding = face["embedding"]
                metric = skin_by_index.get(index)
                if embedding is None or metric is None:
                    continue
                embeddings.append(embedding)
                metric_ids.append(metric.id)

        clusters = cluster_embeddings(embeddings, similarity=similarity)
        subjects: list[Subject] = []
        for cluster in clusters:
            subject = Subject(asset_id=asset_id, name=f"Person {len(subjects) + 1}")
            session.add(subject)
            session.flush()
            subjects.append(subject)
            for k in cluster:
                session.query(SkinMetric).filter_by(id=metric_ids[k]).update(
                    {"subject_id": subject.id}
                )

        session.flush()
        for subject in subjects:
            session.refresh(subject)
        return subjects


# ---------------------------------------------------------------------------
# Editable operations
# ---------------------------------------------------------------------------

def create_subject(store: ProjectStore, asset_id: int, name: str) -> Subject:
    subject = Subject(asset_id=asset_id, name=name)
    with store.session() as session:
        session.add(subject)
        session.flush()
        session.refresh(subject)
    return subject


def rename_subject(store: ProjectStore, subject_id: int, name: str) -> Subject | None:
    with store.session() as session:
        subject = session.get(Subject, subject_id)
        if subject is None:
            return None
        subject.name = name
        session.flush()
        session.refresh(subject)
    return subject


def set_reference(
    store: ProjectStore, subject_id: int, shot_id: int | None
) -> Subject | None:
    """Set (or clear) the hero shot whose skin is this subject's target."""
    with store.session() as session:
        subject = session.get(Subject, subject_id)
        if subject is None:
            return None
        subject.reference_shot_id = shot_id
        session.flush()
        session.refresh(subject)
    return subject


def assign_face(store: ProjectStore, skin_metric_id: int, subject_id: int) -> None:
    """Move one face into a subject (rearrange groups)."""
    with store.session() as session:
        session.query(SkinMetric).filter_by(id=skin_metric_id).update(
            {"subject_id": subject_id}
        )


def unassign_face(store: ProjectStore, skin_metric_id: int) -> None:
    with store.session() as session:
        session.query(SkinMetric).filter_by(id=skin_metric_id).update(
            {"subject_id": None}
        )


def merge_subjects(store: ProjectStore, keep_id: int, drop_id: int) -> None:
    """Move all of ``drop_id``'s faces into ``keep_id`` and delete it."""
    with store.session() as session:
        session.query(SkinMetric).filter_by(subject_id=drop_id).update(
            {"subject_id": keep_id}
        )
        subject = session.get(Subject, drop_id)
        if subject is not None:
            session.delete(subject)


def delete_subject(store: ProjectStore, subject_id: int) -> None:
    """Delete a subject, leaving its faces unassigned (``subject_id=NULL``)."""
    with store.session() as session:
        session.query(SkinMetric).filter_by(subject_id=subject_id).update(
            {"subject_id": None}
        )
        subject = session.get(Subject, subject_id)
        if subject is not None:
            session.delete(subject)


# ---------------------------------------------------------------------------
# Per-subject skin matching
# ---------------------------------------------------------------------------

def propose_skin_match(
    reference: np.ndarray, target: FaceSkin, *, tolerance: float = DEFAULT_MATCH_TOLERANCE
) -> ProposedCorrection | None:
    """Propose an ``rgb_balance`` correction to match a face to a reference skin."""
    gains = [
        float(_clamp(reference[i] / t)) if t > 1e-3 else 1.0
        for i, t in enumerate((target.b, target.g, target.r))
    ]
    if max(abs(g - 1.0) for g in gains) > tolerance:
        return ProposedCorrection("rgb_balance", {"gain": [round(g, 4) for g in gains]})
    return None


def _subject_reference(store: ProjectStore, subject: Subject, faces: list[FaceSkin]) -> np.ndarray:
    """The subject's reference skin: hero shot if set, else the median of its faces."""
    if subject.reference_shot_id is not None:
        hero = [f for f in faces if f.shot_id == subject.reference_shot_id]
        if hero:
            return np.median([_rgb(f) for f in hero], axis=0)
    return np.median([_rgb(f) for f in faces], axis=0)


def skin_consistency(
    store: ProjectStore,
    asset_id: int,
    *,
    match_tolerance: float = DEFAULT_MATCH_TOLERANCE,
) -> list[SkinDeviation]:
    """Flag skin-tone deviations within each subject (not across subjects)."""
    features = face_features(store, asset_id)
    if not features:
        return []

    by_subject: dict[int, list[FaceSkin]] = {}
    for f in features:
        if f.subject_id is not None:
            by_subject.setdefault(f.subject_id, []).append(f)

    deviations: list[SkinDeviation] = []
    for subject in list_subjects(store, asset_id):
        faces = by_subject.get(subject.id, [])
        if not faces:
            continue
        reference = _subject_reference(store, subject, faces)
        for face in faces:
            distance = float(np.linalg.norm(_rgb(face) - reference))
            correction = propose_skin_match(reference, face, tolerance=match_tolerance)
            deviations.append(
                SkinDeviation(
                    shot_id=face.shot_id,
                    face_index=face.face_index,
                    subject_id=subject.id,
                    distance=distance,
                    is_outlier=correction is not None,
                    corrections=(correction,) if correction else (),
                )
            )
    return deviations
