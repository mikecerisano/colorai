# Dependency and reuse audit

What ColorAI reuses versus builds, and why. The guiding rule: **do not build a
proprietary replacement for a mature tool without a strong reason** (product
brief, section 36). We build the domain logic that *is* the product; we reuse
boring, battle-tested infrastructure.

## Runtime dependencies (`pyproject.toml`)

| Package | Constraint | Role | Reused vs built |
| --- | --- | --- | --- |
| numpy | `>=1.26,<3` | array math for metrics and (future) transforms | reused |
| opencv-contrib-python | `>=4.9,<6` | image I/O, color conversion, YuNet detector, cv2 backend for scene detect | reused |
| pydantic | `>=2.6,<3` | (reserved) typed config/validation surface | reused |
| pillow | `>=10,<12` | still image decode/encode in tests and tooling | reused |
| scenedetect | `>=0.7,<1` | content-based shot detection | reused |
| scipy | `>=1.11,<2` | (reserved) statistics/optimization for later analysis | reused |
| tqdm | `>=4.66` | progress reporting for long-form processing | reused |
| rich | `>=13` | console output | reused |
| sqlalchemy | `>=2.0,<3` | ORM / persistence | reused |
| alembic | `>=1.13,<2` | schema migrations | reused |

Web extra: `fastapi`, `uvicorn[standard]`, `jinja2`, `python-multipart` — the
review UI.

Dev extra: `pytest`, `pytest-cov`, `httpx` (FastAPI test client).

Face extra: `mediapipe` — FaceMesh landmark sampling for precise skin tone.
It is optional; without it, `face.py` falls back to YuNet bounding boxes.

## External tools (not pip)

- **ffmpeg / ffprobe** — decoding, frame-accurate still extraction, media
  probing. We shell out to it; we do not reimplement demuxing or encoding.
- **GitHub CLI (`gh`)** — repository creation/push only, not a runtime dep.
- **OpenCV YuNet face detector** — a bundled ONNX model
  (`src/colorai/models/face_detection_yunet_2023mar.onnx`, ~230 KB) powers
  local face detection with no runtime download.

## What we build ourselves (and why that's the product)

- `core/timecode.py` — SMPTE timecode <-> frame conversion including correct
  drop-frame accounting. Small enough to get exactly right and exhaustively
  test; it is a core correctness surface, not a reinvention of a media tool.
- `project/` — the SQLAlchemy model and `ProjectStore`. This *is* the
  non-destructive project/sidecar format the product promises.
- `media/probe.py` — a thin, opinionated mapping from ffprobe JSON to the
  `MediaAsset` model (rational frame-rate parsing for drop-frame correctness).
- `shotdetect.py` — a thin adapter over PySceneDetect that owns the one
  off-by-one conversion (half-open -> inclusive) and persists results.
- `frames.py`, `metrics.py`, `pipeline.py`, `ui.py` — selection, statistics,
  orchestration, review UX. The QC value-add.
- `cli.py` — the local deterministic CLI.

## Deferred / deliberately not yet introduced

- **scikit-image / scikit-learn** — aspirational (listed in the original
  dependency probe) but not yet needed; add when a concrete algorithm requires
  them rather than pre-emptively.
- **Generative-restoration models** — decided (RIFE temporal + LaMa spatial,
  ONNX) but not bundled; the loader reads `COLORAI_GENERATIVE_MODEL_DIR` and
  the deterministic tier needs no model (`research-notes.md`).
- **OpenColorIO / Resolve APIs** — future integration targets, documented in
  `research-notes.md`, not yet dependencies.

## Licensing

All direct dependencies are permissively licensed (BSD/MIT/Apache-2.0 family);
none impose copyleft obligations on this project. ffmpeg is invoked as a
separate process (LGPL/GPL build-dependent); ColorAI does not link against it.
PySceneDetect is BSD-3-Clause. The project itself is MIT.
