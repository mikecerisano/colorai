# ColorAI

**Local-first AI finishing and color-QC assistant for professionally finished video.**

ColorAI analyzes a baked Rec.709 master shot-by-shot, measures what's actually
there, and proposes **deterministic, temporally stable** corrections. It's
built around one idea: the deterministic engine is the *body* (measure +
execute), and an LLM/agent is the *brain* (judge + explain) — with the
filmmaker always holding final approval.

- **Non-destructive** — the source master is never touched; every analysis and
  correction is an explicit row in a local SQLite project database.
- **Deterministic grading** — corrections are exposure / offset / RGB balance /
  ASC CDL / contrast / saturation / hue-rotate / tone curves / `.cube` LUTs,
  never generative repainting of normal footage. Baked masters are decoded
  with the **display-referred sRGB/BT.1886 EOTF**, graded in linear light, and
  re-encoded once (stacked corrections compose in a single float pass); the
  **BT.709 camera OETF** pair is provided separately for scene-linear
  interchange.
- **Identity-aware skin QC** — faces are grouped by a face-recognition
  embedding (not skin color), then matched *within each subject* so two people
  are never pulled toward each other.
- **Temporal** — a face can be tracked across a shot for a stable skin
  signature and a propagated mask, instead of a single-frame snapshot.
- **Full-master export** — approved shot corrections render across the real
  frames to a new master (the same deterministic transforms the preview
  shows), with the source's audio, subtitles, chapters, metadata, and color
  tags preserved; non-Rec.709 transfers, decoder failures, and incomplete
  output are rejected rather than silently emitted.
- **Resumable + editable** — re-analysis is cached by source identity; manual
  split/merge, review/approval state, intentional-exception flags, and
  scene/camera-family grouping all survive a re-run. Pre-Alembic project
  databases open safely and migrate in place.
- **Interview/setup-aware matching** — the matching unit is *subject × setup
  family × camera angle* (labels assigned by a human or vision agent, never
  inferred from pixels). A setup family can hold **lighting variants** (each
  with its own approved reference) so natural window-light changes are matched
  within a variant, never flattened across one. An agent proposes reference
  shots with reasons and confidence; a human must approve before group-scoped
  match proposals appear. Proposals are deterministic, disabled, and carry
  their reference + group context. Global median matching stays an explicit
  diagnostic, not a default.
- **Lower-third name suggestions** — persistent lower-third text is detected
  and OCR'd (local Tesseract CLI, optional), split into a candidate name vs
  role/affiliation, and associated with the visible subject only for
  single-person shots (multi-person shots stay unassigned). Suggestions show
  source timecode + crop with Accept / Edit / Ignore, are **evidence, not
  identity truth**, and never overwrite a human-confirmed name.
- **Temporal QC** — flicker, highlight/shadow measurements, and blank /
  duplicate damaged-frame signatures, alongside blur-pulse detection — all
  reported as *evidence, not defects* for comparing similar shots.
- **Agent-ready (MCP)** — `colorai mcp` exposes the whole engine (including
  real image frames) to Claude Code / Codex / ChatGPT, with agent reasoning
  persisted as reviewable notes.

## Quick start

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[web,dev]"        # core + review UI + tests
# optional extras: ".[face]"       -> MediaPipe landmarks for precise skin sampling
#                  ".[agent]"      -> MCP server for LLM/agent integration
#                  ".[generative]" -> onnxruntime for the RIFE/LaMa restoration tier

# Analyze a master end-to-end (ingest -> shots -> frames -> metrics -> skin)
.venv/bin/colorai analyze /path/to/master.mov --project data/project.sqlite3
# Re-analyzing the same unchanged master resumes from cache; --force re-detects

# Render the master with approved shot corrections applied to a new file
.venv/bin/colorai render --project data/project.sqlite3 --out /path/to/graded.mp4

# Start the local review UI (shots, subjects, notes, corrections, tracking)
.venv/bin/colorai ui --project data/project.sqlite3 --port 8000

# Expose the engine to an agent over MCP (stdio)
.venv/bin/colorai mcp

# Apply schema migrations explicitly (optional; analyze auto-creates fresh DBs)
.venv/bin/colorai db migrate --project data/project.sqlite3
```

`ffmpeg`/`ffprobe` must be on `PATH`.

## The pipeline

`colorai analyze` runs, non-destructively, and is **resumable**: re-running on
an unchanged master returns the cached analysis, and re-running after a manual
edit only re-derives what changed.

1. **Ingest** — ffprobe the master (exact rational frame rate, so 29.97 drop-frame
   is handled correctly) and record a fast content fingerprint.
2. **Shot detection** — PySceneDetect, stored as inclusive 0-based frame bounds
   + SMPTE timecode (skipped when shots already exist, so manual edits survive).
3. **Representative frames** — seek-optimized extraction (middle frame, or
   content-aware "sharpest").
4. **Metrics** — luma percentiles, RGB means, saturation, sharpness.
5. **Skin + identity** — YuNet face detection, SFace identity embeddings,
   per-face skin sampling (MediaPipe landmarks optional), grouped into editable
   subjects.

Then the review UI / MCP surface turns those measurements into decisions:
detect deviations, propose corrections, render a live corrected preview, and
record reasoning.

## Where the intelligence lives

- **Deterministic layer** (`src/colorai/`) — timecode, project model, ingest,
  shot detection, metrics, skin analysis, tracking, correction transforms,
  full-master render, editorial state (split/merge, approval, grouping),
  temporal QC, restoration (deterministic first, generative gated), review UI.
- **Agent layer** — an external LLM drives the same surface via MCP. It can
  read measurements, *see* frames (`get_shot_still`/`get_shot_frame` return
  real images), regroup faces, adjust skin samples, tune corrections, and
  annotate why — all reviewable and reversible.

## Docs

- `docs/architecture.md` — design and decisions
- `docs/status.md` — what's done and what's next
- `docs/dependency-audit.md` — reuse vs. build, licensing
- `docs/research-notes.md` — Resolve integration, color matching, restoration,
  Apple Silicon inference
- `AGENTS.md` — contributor / agent handoff

## Tests

```bash
.venv/bin/python -m pytest
```

429 tests, including exhaustive SMPTE drop-frame round-trips, real
ffmpeg-encoded fixtures, and end-to-end pipeline + MCP checks. Tests that need
`ffmpeg` skip automatically when it's absent.

## License

MIT
