# AGENTS.md — Contributor handoff

Guidance for the next engineer or coding agent working in this repository.
Read `docs/architecture.md` for design and rationale; this file is the
"how do I work here" reference.

## What this is

ColorAI is a local-first AI finishing / color-QC assistant for professionally
finished video. The core pipeline (`analyze`) and a review UI already work end
to end. See `docs/status.md` for exactly what is and isn't done.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[web,dev]"
```

`ffmpeg` and `ffprobe` must be on `PATH` (Homebrew: `brew install ffmpeg`).
Tests that need them skip automatically when they are absent.

## Run tests

```bash
.venv/bin/python -m pytest            # everything
.venv/bin/python -m pytest tests/test_timecode.py   # one module
```

There are 100+ tests including exhaustive drop-frame round-trips and real
ffmpeg-encoded fixtures. Keep them green.

## Layout

```
src/colorai/
  __init__.py        version
  cli.py             argparse entry point (colorai analyze / ui / db)
  ingest.py          probe + register a master
  media/probe.py     ffprobe -> MediaAsset metadata
  shotdetect.py      PySceneDetect -> inclusive shot bounds
  frames.py          representative still selection + extraction
  metrics.py         image statistics + sharpness
  analysis.py        shot-to-shot consistency + reference matching
  skin_analysis.py   per-subject skin-tone matching
  correction.py      deterministic correction transforms + preview
  face.py            YuNet detection + SFace identity + skin sampling
  skin.py            color-only skin heuristic (experiment)
  restoration.py     deterministic recovery + generative boundary
  pipeline.py        analyze_master orchestration
  ui.py              FastAPI review app + correction/analysis API
  core/timecode.py   SMPTE timecode <-> frame conversion
  project/models.py  SQLAlchemy model (Project/Asset/Shot/...)
  project/store.py   ProjectStore + construction helpers
  templates/         Jinja2 UI templates
  models/            bundled ONNX models (YuNet)
  migrations/        Alembic env + versions
tests/               pytest suite, one file per module
docs/                architecture, status, audit, research notes
```

## Non-negotiable conventions

- **Frame numbers are zero-based and inclusive.** Shot bounds are
  `[start_frame, end_frame]`. The one exception is PySceneDetect, which is
  half-open `[start, end)`; convert at the boundary of `shotdetect.py` and
  nowhere else.
- **Store both frame and timecode.** Derive timecode with the helpers in
  `core/timecode.py` and `project/store.py`; never hand-write a timecode
  string in a new module.
- **Drop-frame only for 29.97/59.94.** `core/timecode.py` enforces this and
  uses `;` as the DF separator. PySceneDetect must stay `>= 0.7` (0-based
  `frame_num`).
- **Non-destructive.** Never open a source master for writing. Results live in
  the project SQLite database and the stills directory.
- **Deterministic, temporally stable corrections.** Per-shot corrections apply
  the same transform to every frame; no per-frame generative grading.
- **Measurements ≠ decisions.** A metric/statistical difference is data, not
  automatically an error. Preserve filmic intent.

## How to add a pipeline stage

1. Add a pure function (probe/detect/compute) with a narrow input and a typed
   return, in its own module.
2. Persist via `ProjectStore` + the existing `make_shots` /
   `make_representative_frame` helpers so timecode derivation is centralized.
3. Add a test in `tests/test_<module>.py`. Use `tmp_path` and, where media is
   needed, encode a tiny ffmpeg fixture (see `tests/test_shotdetect.py`) and
   skip when `ffmpeg` is absent.
4. Wire it into `pipeline.analyze_master` if it belongs in the default run.

## Commit hygiene

- Commit at meaningful milestones with a one-line summary of intent.
- Never commit secrets, API keys, generated media, large models, caches, or the
  `data/` directory (all gitignored). `_scratch/` is ignored for ad-hoc checks.

## Current gaps (do not assume they exist)

- Install/bundle the generative models (RIFE + LaMa ONNX) and wire the loader
  (`docs/research-notes.md`); deterministic recovery is complete.
- Any future schema change needs a new Alembic revision (machinery exists).

## Schema migrations

Alembic is configured (`src/colorai/alembic.ini`, `src/colorai/migrations/`).
Apply with `colorai db migrate --project <path>`, and add a new revision for
any model change:

```bash
COLORAI_DB_URL=sqlite+pysqlite:////tmp/seed.sqlite3 \
  .venv/bin/alembic -c src/colorai/alembic.ini revision --autogenerate -m "..."
```
