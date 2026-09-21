"""Tests for media probing and ingest."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from colorai.core.timecode import is_drop_frame
from colorai.ingest import ingest_media
from colorai.media.probe import _parse_rate, probe_media
from colorai.project import ProjectStore


ffmpeg = shutil.which("ffmpeg")
ffprobe = shutil.which("ffprobe")
requires_ffmpeg = pytest.mark.skipif(
    ffmpeg is None or ffprobe is None, reason="ffmpeg/ffprobe not available"
)


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory):
    out = tmp_path_factory.mktemp("media") / "sample.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=25",
            "-t",
            "2",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "mpeg4",
            "-y",
            str(out),
        ],
        check=True,
    )
    return out


@pytest.fixture(scope="module")
def audio_only(tmp_path_factory):
    out = tmp_path_factory.mktemp("media") / "tone.wav"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.1",
            "-y",
            str(out),
        ],
        check=True,
    )
    return out


def test_parse_rate():
    assert _parse_rate("25/1") == 25.0
    assert _parse_rate("30000/1001") == pytest.approx(29.97, abs=1e-4)
    assert _parse_rate("24.5") == 24.5


@requires_ffmpeg
def test_probe_media_generated_clip(sample_video):
    probe = probe_media(sample_video)
    assert probe.frame_rate == 25.0
    assert probe.width == 160
    assert probe.height == 120
    assert probe.pixel_format == "yuv420p"
    assert probe.codec_name == "mpeg4"
    assert probe.duration_seconds == pytest.approx(2.0, abs=0.1)
    assert probe.frame_count == pytest.approx(50, abs=2)
    assert probe.file_size_bytes is not None and probe.file_size_bytes > 0


@requires_ffmpeg
def test_probe_asset_fields_non_none(sample_video):
    fields = probe_media(sample_video).asset_fields()
    essential = {
        "file_size_bytes",
        "width",
        "height",
        "frame_count",
        "duration_seconds",
        "pixel_format",
        "codec_name",
    }
    assert essential <= set(fields)
    assert all(v is not None for v in fields.values())


@requires_ffmpeg
def test_probe_defaults_untagged_to_bt709(sample_video):
    # The lavfi testsrc encode carries no explicit color tags, so both the
    # transfer and primaries should fall back to the BT.709 working space.
    probe = probe_media(sample_video)
    assert probe.transfer == "bt709"
    assert probe.color_space == "bt709"


@requires_ffmpeg
def test_probe_audio_only_raises(audio_only):
    with pytest.raises(ValueError):
        probe_media(audio_only)


@requires_ffmpeg
def test_ingest_media_registers_asset(sample_video):
    store = ProjectStore.create(":memory:")
    project = store.create_project("ingest test")
    asset = ingest_media(store, project.id, sample_video)

    assert asset.project_id == project.id
    assert asset.source_path == str(sample_video)
    assert asset.frame_rate == 25.0
    assert asset.timecode_format == "NDF"
    assert asset.width == 160
    assert asset.height == 120
    assert asset.status == "registered"


def test_drop_frame_detection_for_ntsc_rates():
    # Frame-rate parsing is what keeps ingest's timecode_format correct.
    assert is_drop_frame(_parse_rate("30000/1001")) is True
    assert is_drop_frame(_parse_rate("60000/1001")) is True
    assert is_drop_frame(_parse_rate("25/1")) is False


@requires_ffmpeg
def test_ingest_declared_transfer_override(sample_video):
    store = ProjectStore.create(":memory:")
    project = store.create_project("declared")
    asset = ingest_media(store, project.id, sample_video, transfer="pq")
    assert asset.transfer == "pq"


@requires_ffmpeg
def test_ingest_declared_transfer_rejects_log(sample_video):
    store = ProjectStore.create(":memory:")
    project = store.create_project("declared bad")
    with pytest.raises(ValueError, match="never guessed"):
        ingest_media(store, project.id, sample_video, transfer="slog3")


def test_parse_rate_rejects_sentinels():
    from colorai.media.probe import _parse_rate

    assert _parse_rate("30000/1001") == pytest.approx(30000 / 1001, abs=1e-9)
    assert _parse_rate("25") == 25.0
    for bad in ("0/0", "N/A", "", "  ", "abc", "25/0"):
        with pytest.raises(ValueError, match="unparseable frame rate"):
            _parse_rate(bad)


def test_probe_media_tolerates_na_fields(monkeypatch):
    import json

    import colorai.media.probe as probe_mod

    payload = {
        "streams": [{
            "codec_type": "video",
            "width": 1920,
            "height": 1080,
            "avg_frame_rate": "0/0",
            "r_frame_rate": "25/1",
            "nb_frames": "N/A",
            "pix_fmt": "yuv420p",
        }],
        "format": {"duration": "N/A", "size": "N/A"},
    }

    class _Result:
        stdout = json.dumps(payload)

    monkeypatch.setattr(
        probe_mod, "subprocess",
        type("Sub", (), {"run": staticmethod(lambda *a, **k: _Result())})(),
    )
    probe = probe_mod.probe_media("/media/odd.mov")
    assert probe.frame_rate == 25.0  # fell back from 0/0 to r_frame_rate
    assert probe.frame_count is None
    assert probe.duration_seconds is None
    assert probe.file_size_bytes is None


def test_probe_media_rejects_unusable_rate_and_garbage(monkeypatch):
    import colorai.media.probe as probe_mod

    class _Result:
        def __init__(self, stdout):
            self.stdout = stdout

    bad_rate = {"streams": [{"codec_type": "video", "avg_frame_rate": "N/A"}], "format": {}}
    monkeypatch.setattr(
        probe_mod, "subprocess",
        type("Sub", (), {"run": staticmethod(lambda *a, **k: _Result(""))})(),
    )
    with pytest.raises(ValueError, match="unparseable output"):
        probe_mod.probe_media("/media/empty.mov")

    monkeypatch.setattr(
        probe_mod, "subprocess",
        type("Sub", (), {"run": staticmethod(
            lambda *a, **k: _Result(__import__("json").dumps(bad_rate)))})(),
    )
    with pytest.raises(ValueError, match="no usable frame rate"):
        probe_mod.probe_media("/media/norate.mov")
