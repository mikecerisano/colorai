"""Tests for shot-to-shot consistency analysis."""

from __future__ import annotations

import pytest

from colorai.analysis import (
    ShotFeature,
    find_outliers,
    persist_proposals,
    propose_corrections,
)
from colorai.project import Correction, FrameMetrics, ProjectStore, make_shots


def _feature(shot_id, luma, r=None, g=None, b=None, sat=0.0):
    return ShotFeature(
        shot_id=shot_id,
        luma_mean=luma,
        r_mean=luma if r is None else r,
        g_mean=luma if g is None else g,
        b_mean=luma if b is None else b,
        saturation_mean=sat,
    )


REF = _feature(1, 0.5)


def test_exposure_proposed_for_darker_shot():
    dev = propose_corrections(REF, _feature(2, 0.25))
    assert dev.is_outlier
    assert "luma" in dev.reasons
    kinds = [c.kind for c in dev.corrections]
    assert "exposure" in kinds
    exposure = next(c for c in dev.corrections if c.kind == "exposure")
    assert exposure.parameters["gain"] == pytest.approx(2.0)


def test_channel_balance_proposed():
    dev = propose_corrections(REF, _feature(2, 0.5, r=0.5, g=0.4, b=0.4))
    assert dev.is_outlier
    balance = next(c for c in dev.corrections if c.kind == "rgb_balance")
    assert balance.parameters["gain"] == pytest.approx([1.0, 1.25, 1.25])


def test_saturation_proposed():
    ref = _feature(1, 0.5, sat=0.4)
    dev = propose_corrections(ref, _feature(2, 0.5, sat=0.1))
    assert dev.is_outlier
    assert any(c.kind == "saturation" for c in dev.corrections)


def test_within_tolerance_not_outlier():
    dev = propose_corrections(REF, _feature(2, 0.51))
    assert not dev.is_outlier
    assert dev.corrections == ()


def test_zero_luma_flagged_without_exposure():
    # A black shot cannot be exposure-corrected to match; flag but no gain.
    dev = propose_corrections(REF, _feature(2, 0.0))
    assert dev.is_outlier
    assert not any(c.kind == "exposure" for c in dev.corrections)


def test_gain_is_clamped():
    dev = propose_corrections(REF, _feature(2, 0.01))
    exposure = next(c for c in dev.corrections if c.kind == "exposure")
    assert exposure.parameters["gain"] == 4.0  # clamped to MAX_GAIN


def _build_three_shot_asset(store):
    project = store.create_project("analysis test")
    asset = store.add_asset(project.id, source_path="/media/m.mov", frame_rate=25.0)
    shots = make_shots(asset, [(0, 24), (25, 49), (50, 74)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)

    rows = [
        (shots[0].id, 0.5, 0.5, 0.5, 0.5, 0.0),
        (shots[1].id, 0.25, 0.25, 0.25, 0.25, 0.0),
        (shots[2].id, 0.5, 0.5, 0.4, 0.4, 0.0),
    ]
    with store.session() as session:
        for shot_id, luma, r, g, b, sat in rows:
            session.add(
                FrameMetrics(
                    shot_id=shot_id,
                    frame_index=0,
                    luma_mean=luma,
                    r_mean=r,
                    g_mean=g,
                    b_mean=b,
                    saturation_mean=sat,
                )
            )
        session.commit()
    return asset, shots


def test_find_outliers_against_median():
    store = ProjectStore.create(":memory:")
    asset, shots = _build_three_shot_asset(store)

    outliers = find_outliers(store, asset.id)
    by_shot = {d.shot_id: d for d in outliers}

    assert shots[0].id not in by_shot  # the median reference is excluded
    assert by_shot[shots[1].id].is_outlier  # darker -> exposure
    assert by_shot[shots[2].id].is_outlier  # channel imbalance -> rgb_balance
    assert any(c.kind == "exposure" for c in by_shot[shots[1].id].corrections)
    assert any(c.kind == "rgb_balance" for c in by_shot[shots[2].id].corrections)


def test_find_outliers_explicit_reference():
    store = ProjectStore.create(":memory:")
    asset, shots = _build_three_shot_asset(store)

    # Reference shot 2 (luma 0.25) makes the brighter shots outliers.
    outliers = find_outliers(store, asset.id, reference_shot_id=shots[1].id)
    assert shots[1].id not in {d.shot_id for d in outliers}
    assert all(d.luma_delta_stops < 0 for d in outliers)  # others are brighter


def test_persist_proposals():
    store = ProjectStore.create(":memory:")
    asset, shots = _build_three_shot_asset(store)
    outliers = find_outliers(store, asset.id)

    created = persist_proposals(store, outliers)
    assert len(created) == 2

    with store.session() as session:
        persisted = session.query(Correction).order_by(Correction.shot_id).all()
    assert {c.kind for c in persisted} == {"exposure", "rgb_balance"}
    assert all(c.enabled for c in persisted)
