"""Tests for image metrics."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from colorai.metrics import compute_frame_metrics, metrics_from_path, store_frame_metrics
from colorai.project import ProjectStore, Shot, make_shots


def _bgr(b, g, r, size=(8, 8)):
    return np.full((size[0], size[1], 3), [b, g, r], dtype=np.uint8)


def test_black_frame():
    m = compute_frame_metrics(np.zeros((16, 16, 3), dtype=np.uint8))
    assert m["luma_mean"] == 0.0
    assert m["luma_std"] == 0.0
    assert m["luma_max"] == 0.0
    assert m["saturation_mean"] == 0.0
    assert m["r_mean"] == m["g_mean"] == m["b_mean"] == 0.0


def test_white_frame():
    m = compute_frame_metrics(np.full((16, 16, 3), 255, dtype=np.uint8))
    assert m["luma_mean"] == pytest.approx(1.0)
    assert m["luma_min"] == pytest.approx(1.0)
    assert m["r_mean"] == pytest.approx(1.0)
    assert m["saturation_mean"] == pytest.approx(0.0, abs=1e-6)


def test_mid_gray_frame():
    m = compute_frame_metrics(_bgr(128, 128, 128))
    assert m["luma_mean"] == pytest.approx(128 / 255, abs=1e-3)
    assert m["r_mean"] == m["g_mean"] == m["b_mean"] == pytest.approx(128 / 255, abs=1e-3)
    assert m["luma_std"] == pytest.approx(0.0, abs=1e-9)


def test_pure_red_frame():
    # BGR order: red is (0, 0, 255).
    m = compute_frame_metrics(_bgr(0, 0, 255))
    assert m["r_mean"] == pytest.approx(1.0)
    assert m["g_mean"] == pytest.approx(0.0)
    assert m["b_mean"] == pytest.approx(0.0)
    assert m["luma_mean"] == pytest.approx(0.2126, abs=1e-3)
    assert m["saturation_mean"] > 0.0


def test_gradient_spreads_percentiles():
    # Horizontal luma ramp 0..255 in all channels.
    ramp = np.tile(np.arange(256, dtype=np.uint8), (8, 1))
    img = np.dstack([ramp, ramp, ramp])
    m = compute_frame_metrics(img)
    assert m["luma_min"] == pytest.approx(0.0, abs=1e-9)
    assert m["luma_max"] == pytest.approx(1.0, abs=1e-9)
    assert m["luma_median"] == pytest.approx(0.5, abs=0.01)
    assert m["luma_p5"] < m["luma_median"] < m["luma_p95"]


def test_grayscale_input():
    m = compute_frame_metrics(np.full((8, 8), 128, dtype=np.uint8))
    assert m["luma_mean"] == pytest.approx(128 / 255, abs=1e-3)


def test_metrics_from_path(tmp_path):
    out = tmp_path / "gray.png"
    cv2.imwrite(str(out), np.full((16, 16, 3), [128, 128, 128], dtype=np.uint8))
    m = metrics_from_path(str(out))
    assert m["luma_mean"] == pytest.approx(128 / 255, abs=1e-3)


def test_store_frame_metrics(tmp_path):
    store = ProjectStore.create(":memory:")
    project = store.create_project("metrics test")
    asset = store.add_asset(
        project.id, source_path="/media/m.mov", frame_rate=25.0
    )
    with store.session() as session:
        session.add_all(make_shots(asset, [(0, 24)]))
        session.flush()
        shot = session.query(Shot).one()

    metrics = compute_frame_metrics(_bgr(128, 128, 128))
    row = store_frame_metrics(store, shot, 0, metrics)

    assert row.luma_mean == pytest.approx(128 / 255, abs=1e-3)
    assert row.frame_index == 0
    assert row.shot_id == shot.id
