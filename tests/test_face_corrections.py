"""Tests for persisted face tracks and local face corrections."""

from __future__ import annotations

from colorai.project import (
    FaceCorrection,
    FaceTrack,
    ProjectStore,
    Shot,
    SkinMetric,
    Subject,
    make_shots,
)
from colorai.skin_analysis import create_subject


def _fixture():
    store = ProjectStore.create(":memory:")
    project = store.create_project("face corrections")
    asset = store.add_asset(
        project.id, source_path="/media/m.mov", frame_rate=25.0, width=1920, height=1080
    )
    shots = make_shots(asset, [(0, 49)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)
    alice = create_subject(store, asset.id, "Alice")
    with store.session() as session:
        session.add(
            SkinMetric(
                shot_id=shots[0].id, face_index=0,
                mean_b=0.30, mean_g=0.30, mean_r=0.50,
                sample_pixels=100, subject_id=alice.id,
                bbox_x=100, bbox_y=120, bbox_w=200, bbox_h=240,
            )
        )
        session.commit()
    with store.session() as session:
        metric = session.query(SkinMetric).filter_by(shot_id=shots[0].id).first()
    return store, asset, shots[0], alice, metric


def test_face_track_and_correction_persist_with_defaults():
    store, asset, shot, alice, metric = _fixture()

    with store.session() as session:
        track = FaceTrack(
            shot_id=shot.id, skin_metric_id=metric.id, subject_id=alice.id,
            source_width=1920, source_height=1080, analysis_scale=480,
            keyframes=[[0, 0.1, 0.1, 0.2, 0.2], [25, 0.1, 0.1, 0.2, 0.2]],
            sample_count=2, tracked_count=2, coverage=1.0, max_gap=0.0,
            skin_stability=0.02, median_bgr=[0.30, 0.30, 0.50], state="valid",
        )
        session.add(track)
        session.flush()
        session.refresh(track)

        correction = FaceCorrection(
            shot_id=shot.id, subject_id=alice.id, skin_metric_id=metric.id,
            face_track_id=track.id, reference_shot_id=shot.id,
            kind="rgb_balance", parameters={"gain": [1.0, 0.95, 0.90]},
            reason="slightly warm", classification="skin_mismatch", confidence=0.8,
        )
        session.add(correction)
        session.flush()
        session.refresh(correction)

        assert correction.state == "suggested"
        assert correction.enabled is False
        assert correction.kind == "rgb_balance"
        assert correction.face_track_id == track.id

    with store.session() as session:
        assert session.query(FaceTrack).count() == 1
        assert session.query(FaceCorrection).count() == 1


def test_face_correction_cannot_enable_without_approval_by_default():
    store, asset, shot, alice, metric = _fixture()
    with store.session() as session:
        correction = FaceCorrection(
            shot_id=shot.id, subject_id=alice.id, skin_metric_id=metric.id,
            kind="rgb_balance", parameters={"gain": [1.0, 1.0, 1.0]},
            reason="", classification="skin_mismatch",
        )
        session.add(correction)
        session.commit()
        session.refresh(correction)
        assert correction.enabled is False
        assert correction.state == "suggested"


# -- pure mask compositor ----------------------------------------------------

import cv2
import numpy as np
import pytest

from colorai.face_corrections import (
    FaceCorrectionSpec,
    _interpolate_box,
    _max_gap_ratio,
    apply_face_corrections,
    validate_gain,
)


def _skin_bgr():
    # A YCrCb skin-like value (BGR) so colorai.skin.skin_mask returns True.
    return (89, 97, 148)


def _spec(shot_region, gain=(1.10, 1.0, 1.0), keyframes=((0, 0.25, 0.25, 0.5, 0.5),)):
    return FaceCorrectionSpec(id=1, gain=tuple(gain), keyframes=tuple(keyframes), source_width=64, source_height=64)


def _image_with_skin_region():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :] = (0, 128, 0)  # green background (RGB)
    bgr = _skin_bgr()
    img[16:48, 16:48] = (bgr[2], bgr[1], bgr[0])  # skin-like RGB
    return img


def test_validate_gain_caps():
    assert validate_gain([1.0, 1.0, 1.0]) == (1.0, 1.0, 1.0)
    assert validate_gain([0.90, 1.10, 0.95]) == (0.90, 1.10, 0.95)
    with pytest.raises(ValueError):
        validate_gain([1.0, 1.2, 1.0])
    with pytest.raises(ValueError):
        validate_gain([0.5, 1.0, 1.0])
    with pytest.raises(ValueError):
        validate_gain([1.0, 1.0])


def test_interpolate_box_clamps_and_interpolates():
    kf = ((0, 0.1, 0.1, 0.2, 0.2), (10, 0.3, 0.3, 0.4, 0.4))
    assert _interpolate_box(kf, -5) == (0.1, 0.1, 0.2, 0.2)
    assert _interpolate_box(kf, 5) == pytest.approx((0.2, 0.2, 0.3, 0.3))
    assert _interpolate_box(kf, 99) == (0.3, 0.3, 0.4, 0.4)


def test_max_gap_ratio():
    kf = ((0, 0.1, 0.1, 0.2, 0.2), (10, 0.1, 0.1, 0.2, 0.2))
    # duration = 11 frames; gaps: start->0 = 0, 0->10 = 9, 10->end=0 => 9/11
    assert _max_gap_ratio(kf, 0, 10) == pytest.approx(9 / 11)
    assert _max_gap_ratio([], 0, 10) == 1.0


def test_apply_face_corrections_changes_skin_not_background():
    img = _image_with_skin_region()
    out = apply_face_corrections(img, [_spec(16, gain=(1.10, 1.0, 1.0))], frame_index=0)
    # Background (non-skin, outside box) is bit-identical.
    assert (out[0:8, 0:8] == img[0:8, 0:8]).all()
    # The skin region inside the box changed (red gain applied).
    assert not (out[20:40, 20:40] == img[20:40, 20:40]).all()


def test_apply_face_corrections_leaves_second_face_unchanged():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :] = (0, 128, 0)  # green background (RGB)
    bgr = _skin_bgr()
    rgb = (bgr[2], bgr[1], bgr[0])
    img[8:24, 8:24] = rgb    # participant A
    img[40:56, 40:56] = rgb  # participant B (fully outside A's box)
    spec = FaceCorrectionSpec(
        id=1, gain=(1.10, 1.0, 1.0),
        keyframes=((0, 0.125, 0.125, 0.25, 0.25),),
        source_width=64, source_height=64,
    )
    out = apply_face_corrections(img, [spec], frame_index=0)
    # A's box changed, but the second participant and background did not.
    assert not (out[8:24, 8:24] == img[8:24, 8:24]).all()
    assert (out[40:56, 40:56] == img[40:56, 40:56]).all()
    assert (out[0:8, 0:8] == img[0:8, 0:8]).all()


def test_apply_face_corrections_is_deterministic():
    img = _image_with_skin_region()
    spec = _spec(16, gain=(0.95, 1.05, 1.10))
    a = apply_face_corrections(img, [spec], frame_index=0)
    b = apply_face_corrections(img, [spec], frame_index=0)
    assert (a == b).all()


def test_apply_face_corrections_empty_is_identity():
    img = _image_with_skin_region()
    assert (apply_face_corrections(img, [], frame_index=0) == img).all()


# -- track builder -----------------------------------------------------------

from colorai.face_corrections import build_face_track


def _fake_extract(frame_boxes):
    def extract(video_path, frame_index, out_path, fps=None, scale=None):
        import cv2 as _cv2
        import numpy as _np

        _cv2.imwrite(
            str(out_path), _np.full((270, 480, 3), _skin_bgr(), dtype=_np.uint8)
        )
        return out_path

    return extract


def test_build_face_track_success():
    store, asset, shot, alice, metric = _fixture()

    def detect(image):
        return [(25, 30, 50, 60)]

    track = build_face_track(store, metric.id, samples=16, scale=480, extract=_fake_extract(None), detect=detect)
    assert track.state == "valid"
    assert track.coverage == 1.0
    assert len(track.keyframes) == 16


def test_build_face_track_fails_on_low_coverage():
    store, asset, shot, alice, metric = _fixture()

    def detect(image):
        return []

    track = build_face_track(store, metric.id, samples=4, scale=480, extract=_fake_extract(None), detect=detect)
    assert track.state == "failed"
    assert track.coverage == 0.0
    assert "coverage" in track.failure_reason


def test_build_face_track_fails_on_excessive_gap():
    store, asset, shot, alice, metric = _fixture()

    def detect(image):
        return [(25, 30, 50, 60)]

    # Succeed on 12 of 16 samples (75% coverage) but miss four consecutive
    # middle samples, leaving an untracked span above the 20% cap.
    seen = {"n": 0}
    def sparse_detect(image):
        seen["n"] += 1
        if seen["n"] in (7, 8, 9, 10):
            return []
        return [(25, 30, 50, 60)]

    track = build_face_track(store, metric.id, samples=16, scale=480, extract=_fake_extract(None), detect=sparse_detect)
    assert track.state == "failed"
    assert "gap" in track.failure_reason


def test_preview_applies_face_correction_via_shared_compositor(tmp_path):
    from colorai.correction import load_corrected_still
    from colorai.face_corrections import (
        approve_face_correction,
        enable_face_correction,
        propose_face_correction,
    )
    from colorai.project import (
        FaceTrack,
        ProjectStore,
        make_representative_frame,
        make_shots,
    )

    store = ProjectStore.create(":memory:")
    project = store.create_project("preview face")
    asset = store.add_asset(project.id, source_path="/media/m.mov", frame_rate=25.0, width=64, height=64)
    shots = make_shots(asset, [(0, 9)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)
    shot = shots[0]
    alice = create_subject(store, asset.id, "Alice")

    with store.session() as session:
        metric = SkinMetric(
            shot_id=shot.id, face_index=0,
            mean_b=0.30, mean_g=0.30, mean_r=0.50,
            sample_pixels=100, subject_id=alice.id,
            bbox_x=16, bbox_y=16, bbox_w=32, bbox_h=32,
        )
        session.add(metric)
        session.flush()
        track = FaceTrack(
            shot_id=shot.id, skin_metric_id=metric.id, subject_id=alice.id,
            source_width=64, source_height=64, analysis_scale=64,
            keyframes=[[0, 0.25, 0.25, 0.5, 0.5], [9, 0.25, 0.25, 0.5, 0.5]],
            sample_count=2, tracked_count=2, coverage=1.0, max_gap=0.0,
            skin_stability=0.01, median_bgr=[0.30, 0.30, 0.50], state="valid",
        )
        session.add(track)
        session.flush()
        metric_id, track_id = metric.id, track.id

        still = tmp_path / "still.png"
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:, :] = (0, 128, 0)  # green background (RGB)
        img[16:48, 16:48] = (_skin_bgr()[2], _skin_bgr()[1], _skin_bgr()[0])  # skin (RGB)
        cv2.imwrite(str(still), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        session.add(make_representative_frame(shot, 0, image_path=str(still), frame_rate=25.0))
        session.commit()

    c = propose_face_correction(
        store, shot_id=shot.id, subject_id=alice.id, skin_metric_id=metric_id,
        face_track_id=track_id, reference_shot_id=shot.id, reference_group_id=None,
        reason="warm", confidence=0.8, classification="skin_mismatch",
        gain=(1.10, 1.0, 1.0),
    )
    approve_face_correction(store, c.id)
    enable_face_correction(store, c.id)

    out_bgr = load_corrected_still(store, shot)
    out = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    assert not (out[20:40, 20:40] == img[20:40, 20:40]).all()  # skin changed
    assert (out[0:8, 0:8] == img[0:8, 0:8]).all()  # background unchanged
