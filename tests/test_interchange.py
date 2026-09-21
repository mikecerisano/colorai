"""Tests for Resolve interchange (CDL / baked LUT / EDL / XML / package)."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from colorai.correction import apply_corrections
from colorai.interchange import (
    bake_cube_text,
    export_edl,
    export_fcp_xml,
    export_package,
    fold_sop,
    format_cube_3d,
    sop_to_cdl_xml,
)
from colorai.lutcube import apply_cube, parse_cube
from colorai.project import ProjectStore, make_shots
from colorai.project.models import Correction


def _stack(*items):
    return [(kind, dict(params)) for kind, params in items]


def test_fold_single_cdl_is_exact():
    slope, offset, power, exact, _ = fold_sop(
        _stack(("cdl", {"slope": [1.1, 1.0, 0.9], "offset": [0.01, 0.0, 0.0], "power": [1.0, 1.0, 1.0]}))
    )
    assert exact is True
    assert slope == pytest.approx((1.1, 1.0, 0.9))
    assert offset == pytest.approx((0.01, 0.0, 0.0))


def test_fold_gain_chain_composes_in_order():
    slope, offset, power, exact, _ = fold_sop(
        _stack(
            ("exposure", {"gain": 2.0}),
            ("offset", {"value": 0.1}),
            ("rgb_balance", {"gain": [1.0, 0.5, 1.0]}),
        )
    )
    assert exact is True
    assert slope == pytest.approx((2.0, 1.0, 2.0))
    assert offset == pytest.approx((0.1, 0.05, 0.1))
    assert power == pytest.approx((1.0, 1.0, 1.0))


def test_fold_identity_ops_skip_silently():
    _, _, _, exact, _ = fold_sop(
        _stack(("exposure", {"gain": 1.5}), ("saturation", {"amount": 1.0}), ("contrast", {"amount": 1.0}))
    )
    assert exact is True


def test_fold_power_then_gain_is_not_exact():
    _, _, _, exact, reason = fold_sop(
        _stack(
            ("cdl", {"power": [1.2, 1.2, 1.2]}),
            ("exposure", {"gain": 1.1}),
        )
    )
    assert exact is False
    assert "baked .cube" in reason


def test_fold_two_powers_is_not_exact():
    _, _, _, exact, _ = fold_sop(
        _stack(
            ("cdl", {"power": [1.2, 1.2, 1.2]}),
            ("cdl", {"power": [1.1, 1.1, 1.1]}),
        )
    )
    assert exact is False


def test_fold_contrast_is_not_exact():
    _, _, _, exact, reason = fold_sop(_stack(("contrast", {"amount": 1.2})))
    assert exact is False
    assert "contrast" in reason


def test_cdl_xml_maps_stored_power_to_asc():
    # Stored power P applies as (i*s+o)^(1/P); ASC writes the exponent directly.
    text = sop_to_cdl_xml([
        ("shot_000", "shot_000 test", (2.0, 1.0, 1.0), (0.0, 0.0, 0.0), (2.0, 1.0, 1.0))
    ])
    root = ET.fromstring(text)
    assert root.tag.endswith("ColorDecisionList")
    power = root.find(".//{*}Power").text.split()
    assert [float(v) for v in power] == pytest.approx([0.5, 1.0, 1.0])
    slope = root.find(".//{*}Slope").text.split()
    assert [float(v) for v in slope] == pytest.approx([2.0, 1.0, 1.0])


def test_bake_round_trips_through_parse_and_matches_direct():
    stack = _stack(("exposure", {"gain": 1.1}), ("offset", {"value": 0.01}))
    text = bake_cube_text(stack, size=17, title="test")
    lut = parse_cube(text)
    assert lut.is_3d and lut.size == 17
    rng = np.random.default_rng(7)
    pixels = rng.random((40, 40, 3))
    direct = np.asarray(apply_corrections(pixels, stack), dtype=np.float64)
    via_lut = np.asarray(apply_cube(lut, pixels), dtype=np.float64)
    interior = ((direct > 0.02) & (direct < 0.98)).all(axis=-1)
    assert via_lut[interior] == pytest.approx(direct[interior], abs=0.005)
    assert np.abs(via_lut - direct).max() < 0.02


def test_bake_clip_kink_is_bounded():
    # Clipping is a kink: trilinear interpolation overshoots where the grade
    # slams into black/white. The bake stays honest by bounding it — Resolve
    # sees the same clipped grade, off by at most one node span of smoothing.
    stack = _stack(("exposure", {"gain": 1.5}), ("saturation", {"amount": 1.2}))
    lut = parse_cube(bake_cube_text(stack, size=33, title="test"))
    rng = np.random.default_rng(7)
    pixels = rng.random((20, 30, 3))
    direct = np.asarray(apply_corrections(pixels, stack), dtype=np.float64)
    via_lut = np.asarray(apply_cube(lut, pixels), dtype=np.float64)
    assert np.abs(via_lut - direct).max() < 0.08


def test_format_cube_writer_shape():
    table = np.zeros((5, 5, 5, 3))
    text = format_cube_3d(table, 5, title="t")
    assert parse_cube(text).table.shape == (5, 5, 5, 3)


def _store_with_grades(tmp_path):
    db = tmp_path / "project.sqlite3"
    store = ProjectStore.create(db)
    project = store.create_project("film")
    asset = store.add_asset(project.id, source_path="/media/m.mov", frame_rate=25.0)
    shots = make_shots(asset, [(0, 24), (25, 49), (50, 74)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)
        session.add(Correction(shot_id=shots[0].id, kind="exposure", parameters={"gain": 1.5}, enabled=True))
        session.add(Correction(shot_id=shots[1].id, kind="contrast", parameters={"amount": 1.2}, enabled=True))
        session.add(Correction(shot_id=shots[1].id, kind="exposure", parameters={"gain": 1.1}, enabled=False))
        session.commit()
    return store, asset, shots


def test_export_package_writes_cdl_cube_edl_xml_manifest(tmp_path):
    store, asset, shots = _store_with_grades(tmp_path)
    out = tmp_path / "resolve"
    manifest = export_package(store, asset.id, out, lut_size=9)

    assert (out / "shot_000.cdl").exists()
    assert (out / "shot_001.cube").exists()
    assert not (out / "shot_002.cdl").exists()
    assert (out / "timeline.edl").exists()
    assert (out / "timeline.xml").exists()
    assert json.loads((out / "manifest.json").read_text())["asset_id"] == asset.id

    by_index = {s["index"]: s for s in manifest["shots"]}
    assert by_index[0]["format"] == "cdl"
    assert by_index[1]["format"] == "cube"
    assert "contrast" in by_index[1]["note"]
    assert by_index[2]["format"] == "none"


def test_export_edl_uses_exclusive_outs(tmp_path):
    _store, _asset, shots = _store_with_grades(tmp_path)
    text = export_edl(shots, fps=25.0)
    assert "FCM: NON-DROP FRAME" in text
    # Shot 0 covers frames 0..24; the EDL out is the exclusive frame 25.
    assert "00:00:00:00 00:00:01:00 00:00:00:00 00:00:01:00" in text


def test_export_fcp_xml_parses_with_two_clips(tmp_path):
    _store, _asset, shots = _store_with_grades(tmp_path)
    text = export_fcp_xml(shots, fps=25.0, master_name="m.mov", master_path="/media/m.mov")
    root = ET.fromstring(text)
    clips = root.findall(".//clipitem")
    assert len(clips) == 3
    assert clips[0].find("start").text == "0"
    assert clips[0].find("end").text == "25"


def test_export_package_rejects_missing_asset_and_log(tmp_path):
    store, asset, _shots = _store_with_grades(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        export_package(store, 9999, tmp_path / "x")
    with store.session() as session:
        session.query(Correction).delete()
        db_asset = session.get(type(asset), asset.id)
        db_asset.transfer = "slog3"
        session.commit()
    with pytest.raises(ValueError, match="never guessed"):
        export_package(store, asset.id, tmp_path / "y")


def test_export_package_grades_pq_transfer_natively(tmp_path):
    store, asset, _shots = _store_with_grades(tmp_path)
    with store.session() as session:
        db_asset = session.get(type(asset), asset.id)
        db_asset.transfer = "smpte2084"
        session.commit()
    out = tmp_path / "resolve-pq"
    manifest = export_package(store, asset.id, out, lut_size=9)
    assert manifest["transfer"] == "smpte2084"
    assert (out / "shot_000.cdl").exists()  # exact SOP path is transfer-agnostic
    cube_text = (out / "shot_001.cube").read_text()
    assert "smpte2084" in cube_text  # baked LUT is labeled transfer-native


def test_bake_rejects_lattice_beyond_cap():
    from colorai.interchange import MAX_LUT_SIZE, bake_cube_text

    with pytest.raises(ValueError, match=r"in \[2, 64\]"):
        bake_cube_text([("exposure", {"gain": 1.1})], size=MAX_LUT_SIZE + 1)
    with pytest.raises(ValueError, match=r"in \[2, 64\]"):
        bake_cube_text([("exposure", {"gain": 1.1})], size=1)
