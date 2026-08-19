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
