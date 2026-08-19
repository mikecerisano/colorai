"""Tests for the CLI scaffold."""

from __future__ import annotations

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


def test_unimplemented_command_exits_nonzero(capsys):
    assert main(["analyze", "/tmp/master.mov"]) == 1
    err = capsys.readouterr().out
    assert "not implemented" in err


def test_version_matches_package():
    from importlib.metadata import version

    assert version("colorai") == __version__
