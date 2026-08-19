"""Media ingest: probe a source master and register it in the project."""

from __future__ import annotations

import hashlib
from pathlib import Path

from colorai.media.probe import MediaProbe, probe_media
from colorai.project.models import MediaAsset
from colorai.project.store import ProjectStore

__all__ = [
    "MediaProbe",
    "compute_source_hash",
    "ingest_media",
    "probe_media",
]

# Bytes sampled from the head, middle, and tail for the content fingerprint.
# Reading three small windows is fast even on multi-GB masters while still
# catching virtually any real content change.
_HASH_SAMPLE_BYTES = 64 * 1024


def compute_source_hash(path: str | Path, *, sample_bytes: int = _HASH_SAMPLE_BYTES) -> str:
    """Fast, robust source identity for a media file.

    Returns ``"<size>:<sha256>"`` where the hash covers the file size plus
    three sampled windows (head, middle, tail). This is not a cryptographic
    guarantee of uniqueness, but it is deterministic and cheap, which is what
    resumability needs: recognize the same master on a re-run and detect that
    it changed.
    """
    src = Path(path)
    size = src.stat().st_size
    digest = hashlib.sha256()
    with src.open("rb") as f:
        digest.update(f.read(sample_bytes))  # head
        if size > 2 * sample_bytes:
            f.seek(size // 2)
            digest.update(f.read(sample_bytes))  # middle
        f.seek(max(0, size - sample_bytes))
        digest.update(f.read(sample_bytes))  # tail
    return f"{size}:{digest.hexdigest()}"


def ingest_media(
    store: ProjectStore, project_id: int, path: str | Path
) -> MediaAsset:
    """Probe ``path`` and register it as a source master on ``project_id``.

    Non-destructive: only the probe metadata, a fast content fingerprint, and
    the path are recorded; the source file is never touched.
    """
    probe = probe_media(path)
    return store.add_asset(
        project_id,
        source_path=probe.source_path,
        frame_rate=probe.frame_rate,
        source_hash=compute_source_hash(path),
        **probe.asset_fields(),
    )
