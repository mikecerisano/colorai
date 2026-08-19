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
  ASC CDL / contrast / saturation / hue-rotate, never generative repainting of
  normal footage. Grades are applied in **linear BT.709 light** (scene-referred),
  so exposure gain 2 is one stop and CDL slope/offset/power are physically
  meaningful; `hue_rotate` is a display-referred perceptual op.
- **Identity-aware skin QC** — faces are grouped by a face-recognition
  embedding (not skin color), then matched *within each subject* so two people
  are never pulled toward each other.
- **Temporal** — a face can be tracked across a shot for a stable skin
  signature and a propagated mask, instead of a single-frame snapshot.
- **Agent-ready (MCP)** — `colorai mcp` exposes the whole engine (including
  real image frames) to Claude Code / Codex / ChatGPT, with agent reasoning
  persisted as reviewable notes.

## Quick start

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[web,dev]"        # core + review UI + tests
# optional extras: ".[face]"  -> MediaPipe landmarks for precise skin sampling
#                  ".[agent]" -> MCP server for LLM/agent integration

# Analyze a master end-to-end (ingest -> shots -> frames -> metrics -> skin)
.venv/bin/colorai analyze /path/to/master.mov --project data/project.sqlite3

# Start the local review UI (shots, subjects, notes, corrections, tracking)
.venv/bin/colorai ui --project data/project.sqlite3 --port 8000

# Expose the engine to an agent over MCP (stdio)
.venv/bin/colorai mcp

# Apply schema migrations explicitly (optional; analyze auto-creates fresh DBs)
.venv/bin/colorai db migrate --project data/project.sqlite3
```

`ffmpeg`/`ffprobe` must be on `PATH`.

## The pipeline

`colorai analyze` runs, non-destructively:

1. **Ingest** — ffprobe the master (exact rational frame rate, so 29.97 drop-frame
   is handled correctly).
2. **Shot detection** — PySceneDetect, stored as inclusive 0-based frame bounds
   + SMPTE timecode.
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
  restoration (deterministic first, generative gated), review UI.
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

250 tests, including exhaustive SMPTE drop-frame round-trips, real
ffmpeg-encoded fixtures, and end-to-end pipeline + MCP checks. Tests that need
`ffmpeg` skip automatically when it's absent.

## License

MIT
