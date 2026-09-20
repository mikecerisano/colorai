"""Tests for the CLI."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from colorai import __version__
from colorai.cli import build_parser, main


def test_help_exits_zero(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "colorai" in out
    assert "analyze" in out
    assert "ui" in out


def test_version():
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--version"])
    assert exc.value.code == 0


def test_ui_subcommand_declared():
    parser = build_parser()
    args = parser.parse_args(["ui", "--project", "/tmp/x.sqlite3", "--port", "9000"])
    assert args.command == "ui"
    assert args.port == 9000


def test_render_subcommand_declared():
    parser = build_parser()
    args = parser.parse_args(
        ["render", "--project", "/tmp/x.sqlite3", "--out", "/tmp/y.mp4", "--crf", "20"]
    )
    assert args.command == "render"
    assert args.out == "/tmp/y.mp4"
    assert args.crf == 20


def test_version_matches_package():
    from importlib.metadata import version

    assert version("colorai") == __version__


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
def test_analyze_end_to_end(tmp_path, capsys):
    # Two shots: 25 frames black, 25 frames white at 25 fps.
    clip = tmp_path / "master.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error",
            "-f", "lavfi", "-t", "1", "-i", "color=c=black:size=32x32:rate=25",
            "-f", "lavfi", "-t", "1", "-i", "color=c=white:size=32x32:rate=25",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0,format=yuv420p",
            "-c:v", "mpeg4", "-y", str(clip),
        ],
        check=True,
    )

    db = tmp_path / "project.sqlite3"
    assert main(["analyze", str(clip), "--project", str(db)]) == 0

    out = capsys.readouterr().out
    assert "shots : 2" in out
    assert "stills: 2" in out
    assert "metrics: 2" in out

    # Results must be persisted and stills written to disk.
    from sqlalchemy import create_engine, text

    engine = create_engine(f"sqlite+pysqlite:///{db}")
    with engine.connect() as conn:
        shots = conn.execute(text("SELECT COUNT(*) FROM shots")).scalar()
        frames = conn.execute(text("SELECT COUNT(*) FROM representative_frames")).scalar()
        metrics = conn.execute(text("SELECT COUNT(*) FROM frame_metrics")).scalar()
        assert (shots, frames, metrics) == (2, 2, 2)
    assert len(list((tmp_path / "stills").rglob("*.png"))) == 2


def test_export_subcommand_declared():
    parser = build_parser()
    args = parser.parse_args(
        ["export", "--project", "/tmp/x.sqlite3", "--out-dir", "/tmp/resolve"]
    )
    assert args.command == "export"
    assert args.out_dir == "/tmp/resolve"
    assert args.lut_size == 33


def test_export_end_to_end(tmp_path):
    from colorai.project import ProjectStore, make_shots
    from colorai.project.models import Correction

    db = tmp_path / "project.sqlite3"
    store = ProjectStore.create(db)
    project = store.create_project("film")
    asset = store.add_asset(project.id, source_path="/media/m.mov", frame_rate=25.0)
    shots = make_shots(asset, [(0, 24), (25, 49)])
    with store.session() as session:
        session.add_all(shots)
        session.flush()
        for s in shots:
            session.refresh(s)
        session.add(
            Correction(shot_id=shots[0].id, kind="exposure",
                       parameters={"gain": 1.5}, enabled=True)
        )
        session.commit()

    out_dir = tmp_path / "resolve"
    assert main(["export", "--project", str(db), "--out-dir", str(out_dir)]) == 0
    assert (out_dir / "shot_000.cdl").exists()
    assert (out_dir / "timeline.edl").exists()
    assert (out_dir / "timeline.xml").exists()
    assert (out_dir / "manifest.json").exists()


def test_open_subcommand_declared():
    parser = build_parser()
    args = parser.parse_args(["open", "/media/film.mov"])
    assert args.command == "open"
    assert args.master == "/media/film.mov"
    assert args.projects_dir == "data"
    assert args.port == 8000


def test_project_path_for_master_sanitizes(tmp_path):
    from colorai.cli import project_path_for_master

    p = project_path_for_master(tmp_path, "/media/My Film (final).mov")
    assert p.parent.parent == tmp_path
    assert p.name == "project.sqlite3"
    assert ".." not in p.parts
    assert p.parent.name.startswith("My_Film")


def test_analyze_and_open_accept_transfer():
    parser = build_parser()
    assert parser.parse_args(["analyze", "/m.mov"]).transfer is None
    assert parser.parse_args(["analyze", "/m.mov", "--transfer", "pq"]).transfer == "pq"
    assert parser.parse_args(["open", "/m.mov", "--transfer", "hlg"]).transfer == "hlg"
