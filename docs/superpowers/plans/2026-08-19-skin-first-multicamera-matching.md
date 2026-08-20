# Skin-First Multicamera Matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add conservative, tracked, face-region skin matching with preview/render parity.

**Architecture:** Persist face tracks and local face corrections separately from whole-frame corrections. Use a shared pure compositor in preview and render. MCP drafts only; the UI approves and enables work.

**Tech Stack:** Python 3.12, SQLAlchemy/Alembic, NumPy/OpenCV, FastAPI/Jinja, ffmpeg, pytest.

**Spec:** `docs/superpowers/specs/2026-08-19-skin-first-multicamera-matching-design.md`

## Global Constraints

- Source media remains read-only; frame indices stay zero-based and inclusive.
- Version one allows only local `rgb_balance` gains within `[0.90, 1.10]`.
- Preview and render call the same pure mask compositor.
- MCP cannot approve, enable, or delete face corrections.
- Invalid enabled face corrections abort render before output starts.

### Task 1: Persistence and migration

**Files:** `src/colorai/project/models.py`, a new Alembic revision, `tests/test_face_corrections.py`, and legacy migration tests.

- [ ] Write failing tests for `FaceTrack` and `FaceCorrection`: track keyframes/quality/state persistence; a correction defaults to `suggested` and disabled; `(shot, subject, metric, track)` foreign-key integrity.
- [ ] Run `.venv/bin/python -m pytest tests/test_face_corrections.py -q` and confirm RED.
- [ ] Add models and migration. `FaceTrack` records normalized sampled boxes, dimensions, coverage, max gap, stability, median BGR, state, and failure reason. `FaceCorrection` records scope, parameters, evidence, classification, rationale, state, and enabled state.
- [ ] Run model/migration tests through GREEN and commit `Add persisted face tracks and corrections`.

### Task 2: Track builder and pure mask compositor

**Files:** `src/colorai/tracking.py`, new `src/colorai/face_corrections.py`, `tests/test_tracking.py`, `tests/test_face_corrections.py`.

- [ ] Write failing tests for normalized keyframe interpolation, low coverage, excessive gaps, source-resolution mask bounds, gain caps, alpha feathering, deterministic result, and a two-person fixture that leaves the other face/background bit-identical.
- [ ] Run focused tests and confirm RED.
- [ ] Implement `build_face_track(store, skin_metric_id, samples=16)` with >=75% coverage and <=20% maximum gap. Store failed tracks rather than guessing.
- [ ] Implement `apply_face_corrections(image_rgb, frame_index, corrections)`: interpolate stored boxes, use current deterministic skin mask, feather plus face-oval falloff, apply linear RGB balance to a copy, then alpha composite in stable correction-ID order.
- [ ] Run focused tests through GREEN and commit `Add tracked face mask compositor`.

### Task 3: Skin-first evidence and draft-only MCP surface

**Files:** `src/colorai/matching.py`, `src/colorai/face_corrections.py`, `src/colorai/mcp_server.py`, `tests/test_matching.py`, new `tests/test_mcp_skin_matching.py`.

- [ ] Write failing tests: exact approved reference is required; a one-shot participant returns `QC only`; different camera backgrounds never return whole-frame corrections in skin-first matching; only valid tracks and `skin_mismatch` classification create a suggested disabled proposal; MCP registry excludes approve/enable/delete tools.
- [ ] Run focused tests and confirm RED.
- [ ] Implement `skin_first_match_subject_setup`, `skin_matching_workspace`, `build_face_track`, labelled track contact sheet, and `propose_face_correction`. Require in-scope subject/shot/track and gain cap. Keep existing whole-frame matching as labelled composition-sensitive diagnostic.
- [ ] Run tests through GREEN and commit `Add draft-only skin matching MCP tools`.

### Task 4: Preview/render integration and fail-safe preflight

**Files:** `src/colorai/correction.py`, `src/colorai/render.py`, `tests/test_correction.py`, `tests/test_render.py`.

- [ ] Write failing tests for preview/render pixel parity on the same fixture frame, outside-mask pixels unchanged, and an invalid enabled correction aborting before an output file exists.
- [ ] Run focused tests and confirm RED.
- [ ] Load only approved+enabled local corrections. Apply whole-frame corrections first and the shared face compositor second. Preflight linkage, valid track, dimensions, keyframes, gaps, and capped parameters before starting ffmpeg.
- [ ] Run tests through GREEN and commit `Render approved face corrections with preview parity`.

### Task 5: Human review UI and API

**Files:** `src/colorai/ui.py`, `src/colorai/templates/index.html`, new `tests/test_face_corrections_api.py`, `tests/test_review_ui.py`.

- [ ] Write failing API/HTML tests: `QC only` for single-shot people; full-frame context and face mask overlay; suggestions cannot enable before approval; a human can approve, reject, mark intentional, then enable an approved valid correction without a page reload.
- [ ] Run focused tests and confirm RED.
- [ ] Add Skin matching inside setup/variant workspace with reference/candidate crops, temporal evidence, rationale, confidence, explicit actions, and masked before/after preview. Keep approval/enable endpoints out of MCP.
- [ ] Run tests through GREEN and commit `Add human review for skin matching corrections`.

### Task 6: Documentation and verification

**Files:** `README.md`, `AGENTS.md`, `docs/architecture.md`, `docs/status.md`.

- [ ] Document local-only BT.709 skin matching, agent draft/human approval, and render failure safety; label whole-frame matching composition-sensitive diagnostic.
- [ ] Run `.venv/bin/python -m pytest -q`; inspect migration head and a tiny ffmpeg preview/render fixture.
- [ ] Commit `Document skin-first multicamera matching`.
