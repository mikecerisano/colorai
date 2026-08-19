# Status

Current progress as of the initial foundation pass. 140 tests passing.

## Done

- **Package scaffold** — `pyproject.toml`, editable install, pinned lockfile,
  `.gitignore`, CLI entry point.
- **Timecode core** — SMPTE NDF/DF conversion with correct drop-frame
  accounting for 29.97/59.94, exhaustively round-trip tested.
- **Project model + store** — SQLAlchemy/SQLite with `Project`, `MediaAsset`,
  `Shot`, `RepresentativeFrame`, `FrameMetrics`, `Correction`; foreign-key
  enforcement and transactional sessions.
- **Ingest** — ffprobe probing with exact rational frame-rate parsing.
- **Shot detection** — PySceneDetect adapter, inclusive 0-based bounds.
- **Representative frames** — frame-accurate ffmpeg extraction (middle frame).
- **Image metrics** — luma percentiles/dispersion, RGB means, saturation proxy.
- **Analysis pipeline** — `colorai analyze` runs ingest → shots → frames →
  metrics and persists everything.
- **Review UI** — `colorai ui` serves a shot-by-shot review page.
- **Correction schema + deterministic transform** — `cdl`, `exposure`,
  `offset`, `rgb_balance`, `contrast`, `saturation`, `hue_rotate`, with
  validation and a non-destructive preview render.
- **Correction review API** — JSON endpoints to add/toggle/delete corrections
  and a live corrected preview endpoint, surfaced in the review UI.
- **Skin segmentation experiment** — color-only YCrCb heuristic as a
  placeholder for coverage measurement.
- **Docs** — `architecture.md`, `dependency-audit.md`, `research-notes.md`,
  `AGENTS.md`, this file.

## Not yet done (next)

- Alembic migrations (schema currently via `create_all`).
- Full interactive approve/reject UX in the review UI (API + preview exist).
- Real face/skin detection (replace the color heuristic with a local model).
- Content-aware representative-frame selection (currently middle frame).
- Shot-to-shot color-outlier detection and candidate correction generation.
- Reference-shot matching and a full approval workflow.
- Generative restoration for damaged intervals (deterministic recovery first).
- Seek-optimized still extraction for long-form media.

## Verified

- `colorai analyze` runs end-to-end on a real encoded master (shots, stills,
  metrics, DB rows all confirmed).
- 140 tests across timecode, project model, ingest, shot detection, frames,
  metrics, pipeline, correction, skin, UI, and CLI.
