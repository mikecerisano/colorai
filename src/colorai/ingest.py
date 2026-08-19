"""Media ingest: probe a source master and register it in the project."""

from __future__ import annotations

from pathlib import Path

from colorai.media.probe import MediaProbe, probe_media
from colorai.project.models import MediaAsset
from colorai.project.store import ProjectStore

__all__ = ["MediaProbe", "ingest_media", "probe_media"]


def ingest_media(
    store: ProjectStore, project_id: int, path: str | Path
) -> MediaAsset:
    """Probe ``path`` and register it as a source master on ``project_id``.

    Non-destructive: only the probe metadata and path are recorded; the source
    file is never touched.
    """
    probe = probe_media(path)
    return store.add_asset(
        project_id,
        source_path=probe.source_path,
        frame_rate=probe.frame_rate,
        **probe.asset_fields(),
    )
