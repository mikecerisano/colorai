"""Shot-to-shot consistency analysis.

Detects shots whose representative-frame metrics deviate from a reference and
proposes *deterministic* corrections (the same kinds implemented in
:mod:`colorai.correction`) to bring them into line. This is measurement and
proposal, not decision: nothing is applied without approval, and nothing is
normalized toward a single "ideal" — the reference is either an explicit shot
or the median shot, which preserves the film's dominant look.

Math is intentionally simple and auditable:

* luminance mismatch -> an ``exposure`` gain (``ref_luma / shot_luma``)
* channel imbalance (after removing the exposure component) -> ``rgb_balance``
* saturation mismatch -> a ``saturation`` amount

Gains are clamped to ``[MIN_GAIN, MAX_GAIN]`` so an extreme deviation cannot
produce an extreme, unstable correction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from colorai.project.models import Correction, FrameMetrics, Shot
from colorai.project.store import ProjectStore

DEFAULT_LUMA_TOL_STOPS = 0.3
DEFAULT_BALANCE_TOL = 0.05
DEFAULT_SATURATION_TOL = 0.15
MIN_GAIN = 0.25
MAX_GAIN = 4.0


@dataclass(frozen=True)
class ShotFeature:
    """The metrics of one shot's representative frame, as analyzed."""

    shot_id: int
    luma_mean: float
    r_mean: float
    g_mean: float
    b_mean: float
    saturation_mean: float


@dataclass(frozen=True)
class ProposedCorrection:
    kind: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ShotDeviation:
    """One shot's deviation from the reference and the proposed fix."""

    shot_id: int
    luma_delta_stops: float
    is_outlier: bool
    reasons: tuple[str, ...] = ()
    corrections: tuple[ProposedCorrection, ...] = ()


def _clamp_gain(value: float) -> float:
    return max(MIN_GAIN, min(MAX_GAIN, value))


def load_shot_features(store: ProjectStore, asset_id: int) -> list[ShotFeature]:
    """Collect one feature vector per shot from its stored metrics."""
    features: list[ShotFeature] = []
    with store.session() as session:
        shots = (
            session.query(Shot).filter_by(asset_id=asset_id).order_by(Shot.index).all()
        )
        for shot in shots:
            metrics = (
                session.query(FrameMetrics)
                .filter_by(shot_id=shot.id)
                .order_by(FrameMetrics.id)
                .first()
            )
            if metrics is None or metrics.luma_mean is None:
                continue
            features.append(
                ShotFeature(
                    shot_id=shot.id,
                    luma_mean=float(metrics.luma_mean),
                    r_mean=float(metrics.r_mean or 0.0),
                    g_mean=float(metrics.g_mean or 0.0),
                    b_mean=float(metrics.b_mean or 0.0),
                    saturation_mean=float(metrics.saturation_mean or 0.0),
                )
            )
    return features


def median_reference(features: list[ShotFeature]) -> ShotFeature:
    """The shot whose luma is closest to the median — a robust default reference."""
    if not features:
        raise ValueError("no shot features to choose a reference from")
    target = float(np.median([f.luma_mean for f in features]))
    return min(features, key=lambda f: abs(f.luma_mean - target))


def feature_from_image(path: str) -> ShotFeature:
    """Compute a reference feature vector from an arbitrary still image.

    Enables matching a shot against an external reference still (another
    shot's grade, a hero still, a client reference), not just the median shot.
    """
    from colorai.metrics import metrics_from_path

    m = metrics_from_path(path)
    return ShotFeature(
        shot_id=0,  # not a persisted shot; identity is irrelevant for matching
        luma_mean=float(m["luma_mean"]),
        r_mean=float(m["r_mean"]),
        g_mean=float(m["g_mean"]),
        b_mean=float(m["b_mean"]),
        saturation_mean=float(m["saturation_mean"]),
    )


def load_shot_feature(store: ProjectStore, shot_id: int) -> ShotFeature | None:
    """Load a single shot's feature vector from its stored metrics."""
    with store.session() as session:
        metrics = (
            session.query(FrameMetrics)
            .filter_by(shot_id=shot_id)
            .order_by(FrameMetrics.id)
            .first()
        )
    if metrics is None or metrics.luma_mean is None:
        return None
    return ShotFeature(
        shot_id=shot_id,
        luma_mean=float(metrics.luma_mean),
        r_mean=float(metrics.r_mean or 0.0),
        g_mean=float(metrics.g_mean or 0.0),
        b_mean=float(metrics.b_mean or 0.0),
        saturation_mean=float(metrics.saturation_mean or 0.0),
    )


def match_shot_to_reference(
    store: ProjectStore,
    shot_id: int,
    reference_image_path: str,
    **tolerances: float,
) -> ShotDeviation:
    """Propose corrections to match a shot to an arbitrary reference still."""
    shot = load_shot_feature(store, shot_id)
    if shot is None:
        raise ValueError(f"shot {shot_id} has no metrics")
    reference = feature_from_image(reference_image_path)
    return propose_corrections(reference, shot, **tolerances)


def propose_corrections(
    reference: ShotFeature,
    shot: ShotFeature,
    *,
    luma_tol_stops: float = DEFAULT_LUMA_TOL_STOPS,
    balance_tol: float = DEFAULT_BALANCE_TOL,
    saturation_tol: float = DEFAULT_SATURATION_TOL,
) -> ShotDeviation:
    """Compare ``shot`` to ``reference`` and propose corrections if it deviates."""
    reasons: list[str] = []
    corrections: list[ProposedCorrection] = []

    # Luminance -> exposure gain.
    if shot.luma_mean > 1e-6 and reference.luma_mean > 1e-6:
        luma_ratio = reference.luma_mean / shot.luma_mean
        delta_stops = math.log2(luma_ratio)
    else:
        luma_ratio = 1.0
        delta_stops = 0.0 if shot.luma_mean > 1e-6 else float("inf")

    if abs(delta_stops) > luma_tol_stops:
        reasons.append("luma")
        if math.isfinite(delta_stops):
            corrections.append(
                ProposedCorrection("exposure", {"gain": round(_clamp_gain(luma_ratio), 4)})
            )

    # Channel balance, with the exposure component removed so we correct
    # color rather than double-counting exposure.
    gains = [
        _clamp_gain((ref_c / shot_c) / luma_ratio)
        if shot_c > 1e-6 and luma_ratio > 0
        else 1.0
        for ref_c, shot_c in (
            (reference.r_mean, shot.r_mean),
            (reference.g_mean, shot.g_mean),
            (reference.b_mean, shot.b_mean),
        )
    ]
    if any(abs(g - 1.0) > balance_tol for g in gains):
        reasons.append("channel_balance")
        corrections.append(
            ProposedCorrection(
                "rgb_balance", {"gain": [round(g, 4) for g in gains]}
            )
        )

    # Saturation.
    if shot.saturation_mean > 1e-6 and reference.saturation_mean > 1e-6:
        sat_ratio = reference.saturation_mean / shot.saturation_mean
        if abs(sat_ratio - 1.0) > saturation_tol:
            reasons.append("saturation")
            corrections.append(
                ProposedCorrection("saturation", {"amount": round(_clamp_gain(sat_ratio), 4)})
            )

    return ShotDeviation(
        shot_id=shot.shot_id,
        luma_delta_stops=delta_stops,
        is_outlier=bool(reasons),
        reasons=tuple(reasons),
        corrections=tuple(corrections),
    )


def find_outliers(
    store: ProjectStore,
    asset_id: int,
    *,
    reference_shot_id: int | None = None,
    luma_tol_stops: float = DEFAULT_LUMA_TOL_STOPS,
    balance_tol: float = DEFAULT_BALANCE_TOL,
    saturation_tol: float = DEFAULT_SATURATION_TOL,
) -> list[ShotDeviation]:
    """Analyze all shots of an asset against a reference and return deviations."""
    features = load_shot_features(store, asset_id)
    if not features:
        return []

    if reference_shot_id is not None:
        reference = next(
            (f for f in features if f.shot_id == reference_shot_id), None
        )
        if reference is None:
            raise ValueError(f"reference shot {reference_shot_id} not found")
    else:
        reference = median_reference(features)

    return [
        propose_corrections(
            reference,
            f,
            luma_tol_stops=luma_tol_stops,
            balance_tol=balance_tol,
            saturation_tol=saturation_tol,
        )
        for f in features
        if f.shot_id != reference.shot_id
    ]


def persist_proposals(
    store: ProjectStore,
    deviations: Iterable[ShotDeviation],
    *,
    enabled: bool = True,
) -> list[Correction]:
    """Persist proposed corrections as ``Correction`` rows (for approval)."""
    created: list[Correction] = []
    with store.session() as session:
        for deviation in deviations:
            for proposal in deviation.corrections:
                correction = Correction(
                    shot_id=deviation.shot_id,
                    kind=proposal.kind,
                    parameters=proposal.parameters,
                    enabled=enabled,
                )
                session.add(correction)
                created.append(correction)
        session.flush()
        for correction in created:
            session.refresh(correction)
    return created
