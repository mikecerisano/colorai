# Agent-guided skin target Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a human approve an internal or external skin-appearance target, then let an MCP agent draft deterministic, tracked, face-local corrections for matching interview angles.

**Architecture:** Add provenance-backed skin targets and reviewable mask tracks beside the existing `FaceTrack` and `FaceCorrection` layer. Analyse masked faces in OKLab, but apply one bounded, deterministic `skin_appearance` transform under the same temporal mask in preview and render. MCP can inspect and draft; approval, enabling, and render remain human-only UI actions.

**Tech Stack:** Python 3.12, SQLAlchemy/Alembic/SQLite, NumPy, OpenCV, optional MediaPipe Face Landmarker, FastAPI/Jinja2, MCP, ffmpeg fixtures, pytest.

**Spec:** `docs/superpowers/specs/2026-08-19-agent-guided-skin-target-design.md`

## Global Constraints

- Keep source masters and external reference originals read-only; project-managed reference derivatives are content-addressed copies.
- Use zero-based, inclusive absolute frame numbers and derive all timecodes through existing helpers.
- Support only already-supported Rec.709 masters; reject log, HDR, and unsupported transfers.
- Produce no generated pixels, beauty smoothing, face replacement, or whole-frame grade in this feature.
- Apply only human-approved and enabled face corrections; MCP may never approve, enable, reject, delete, or render.
- Use one pure compositor for still preview and `render_master`; any invalid enabled correction aborts before output exists.
- Do not copy or vendor code from the research projects named in the spec. Audit any new dependency and model license before adding it.
- Add a new Alembic migration for each schema change and preserve legacy-open migration behaviour.

---

## File structure

| File | Responsibility |
| --- | --- |
| `src/colorai/project/models.py` | Persist references, mask tracks, targets, and the target foreign key on `FaceCorrection`. |
| `src/colorai/migrations/versions/f5e6a7b8c9d0_add_skin_targets_and_mask_tracks.py` | Upgrade existing projects without changing prior rows. |
| `src/colorai/skin_appearance.py` | Pure OKLab conversion, robust masked profile measurement, parameter validation, target-to-candidate derivation. |
| `src/colorai/face_masks.py` | Build, validate, interpolate, render, and contact-sheet temporal face masks. |
| `src/colorai/face_corrections.py` | Extend validated correction specs and the shared compositor with `skin_appearance`. |
| `src/colorai/skin_targets.py` | Reference intake, target lifecycle, scope validation, and disabled candidate proposal lifecycle. |
| `src/colorai/mcp_server.py` | Draft-only skin-target and mask-evidence tools. |
| `src/colorai/ui.py` | Typed API payloads, workspace data, reference/mask/target endpoints, human-only actions. |
| `src/colorai/templates/index.html` | Skin target review UI with visible mask and in-place updates. |
| `tests/test_skin_appearance.py` | Pure colour/profile/parameter tests. |
| `tests/test_face_masks.py` | Mask-track construction, interpolation, containment, and review-state tests. |
| `tests/test_skin_targets.py` | Persistence, provenance, scope, and target/proposal lifecycle tests. |
| `tests/test_mcp_skin_targets.py` | MCP draft-only surface and evidence-gate tests. |
| `tests/test_skin_targets_api.py` | API/UI interaction tests. |
| `tests/test_face_render_parity.py` | Extend the existing real ffmpeg parity fixture to `skin_appearance`. |

## Task 1: Persist references, mask tracks, and approved targets

**Files:**
- Modify: `src/colorai/project/models.py`
- Modify: `src/colorai/project/__init__.py`
- Create: `src/colorai/migrations/versions/f5e6a7b8c9d0_add_skin_targets_and_mask_tracks.py`
- Modify: `tests/test_migrations.py`
- Create: `tests/test_skin_targets.py`

**Interfaces:**
- Produces `SkinAppearanceReference`, `FaceMaskTrack`, and `SkinAppearanceTarget` ORM models.
- Produces `FaceCorrection.skin_target_id: int | None` and permits `kind="skin_appearance"` at the schema layer only.
- Later tasks consume the model names and their exact enum strings.

- [ ] **Step 1: Write failing persistence tests**

```python
def test_skin_target_records_provenance_and_safe_defaults(tmp_path):
    store, asset, subject, group, shot = _fixture(tmp_path)
    with store.session() as session:
        ref = SkinAppearanceReference(
            subject_id=subject.id, asset_id=asset.id, group_id=group.id,
            source_kind="project_frame", role="accurate_skin_reference",
            content_hash="a" * 64, source_shot_id=shot.id,
            frame_index=shot.start_frame, crop_geometry={"x": .1, "y": .1, "w": .2, "h": .2},
        )
        session.add(ref); session.flush()
        target = SkinAppearanceTarget(
            subject_id=subject.id, group_id=group.id, reference_id=ref.id,
            profile={"mean_ab": [0.01, 0.02], "spread_ab": [.01, .01]},
            approved_preview_parameters={"version": 1},
        )
        session.add(target); session.flush()
        assert target.state == "suggested"
        assert ref.state == "active"
```

Add migration assertions that a database at the previous head upgrades to the
new head and that existing `FaceTrack`/`FaceCorrection` rows remain readable.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `.venv/bin/python -m pytest tests/test_skin_targets.py tests/test_migrations.py -q`

Expected: import/model failures for the three new records.

- [ ] **Step 3: Add models and migration**

Add the following exact fields and constraints:

```python
class SkinAppearanceReference(Base):
    __tablename__ = "skin_appearance_references"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    subject_id: Mapped[int] = mapped_column(ForeignKey("subjects.id", ondelete="CASCADE"), nullable=False)
    asset_id: Mapped[int | None] = mapped_column(ForeignKey("media_assets.id", ondelete="CASCADE"))
    group_id: Mapped[int | None] = mapped_column(ForeignKey("shot_groups.id", ondelete="SET NULL"))
    source_kind: Mapped[str] = mapped_column(String(24), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    source_path: Mapped[str | None] = mapped_column(Text)
    managed_path: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_shot_id: Mapped[int | None] = mapped_column(ForeignKey("shots.id", ondelete="SET NULL"))
    frame_index: Mapped[int | None] = mapped_column(Integer)
    crop_geometry: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="active")

class FaceMaskTrack(Base):
    __tablename__ = "face_mask_tracks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    face_track_id: Mapped[int] = mapped_column(ForeignKey("face_tracks.id", ondelete="CASCADE"), nullable=False, unique=True)
    shot_id: Mapped[int] = mapped_column(ForeignKey("shots.id", ondelete="CASCADE"), nullable=False)
    subject_id: Mapped[int] = mapped_column(ForeignKey("subjects.id", ondelete="CASCADE"), nullable=False)
    backend: Mapped[str] = mapped_column(String(32), nullable=False)
    backend_version: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy: Mapped[str] = mapped_column(String(32), nullable=False)
    landmark_keyframes: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    coverage: Mapped[float] = mapped_column(Float, nullable=False)
    max_gap: Mapped[float] = mapped_column(Float, nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="valid")
    review_state: Mapped[str] = mapped_column(String(32), nullable=False, default="unreviewed")
    review_reason: Mapped[str] = mapped_column(Text, nullable=False, default="")

class SkinAppearanceTarget(Base):
    __tablename__ = "skin_appearance_targets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    subject_id: Mapped[int] = mapped_column(ForeignKey("subjects.id", ondelete="CASCADE"), nullable=False)
    group_id: Mapped[int] = mapped_column(ForeignKey("shot_groups.id", ondelete="CASCADE"), nullable=False)
    reference_id: Mapped[int | None] = mapped_column(ForeignKey("skin_appearance_references.id", ondelete="SET NULL"))
    profile: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    approved_preview_parameters: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="suggested")
    rationale: Mapped[str] = mapped_column(Text, nullable=False, default="")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
```

The migration adds the tables, `face_corrections.skin_target_id`, and indexes
for `subject_id`, `group_id`, `shot_id`, and `face_track_id`. It does not alter
or backfill older correction semantics.

- [ ] **Step 4: Run focused tests and full migration chain**

Run: `.venv/bin/python -m pytest tests/test_skin_targets.py tests/test_migrations.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/colorai/project src/colorai/migrations/versions tests/test_skin_targets.py tests/test_migrations.py
git commit -m "Add persisted skin targets and mask tracks"
```

## Task 2: Add pure OKLab profile and correction math

**Files:**
- Create: `src/colorai/skin_appearance.py`
- Create: `tests/test_skin_appearance.py`

**Interfaces:**
- Produces `SkinAppearanceParameters`, `validate_skin_appearance_parameters`, `rgb_linear_to_oklab`, `oklab_to_rgb_linear`, `masked_skin_profile`, `derive_skin_appearance_parameters`, and `apply_skin_appearance`.
- Consumes linear RGB float arrays and alpha masks of matching height/width.
- Later tasks use validated parameter dictionaries with keys `version`, `space`, `luma_mode`, `ab_offset`, `ab_scale`, and `strength`.

- [ ] **Step 1: Write failing pure-math tests**

```python
def test_oklab_round_trip_is_close_for_linear_rgb():
    rgb = np.array([[[.12, .38, .71], [.84, .31, .09]]], dtype=np.float64)
    assert np.allclose(oklab_to_rgb_linear(rgb_linear_to_oklab(rgb)), rgb, atol=1e-6)

def test_chroma_only_transform_preserves_oklab_lightness():
    rgb = np.full((8, 8, 3), [.55, .30, .21])
    out = apply_skin_appearance(rgb, _params(offset=(-.012, .004)), np.ones((8, 8)))
    assert np.allclose(rgb_linear_to_oklab(out)[..., 0], rgb_linear_to_oklab(rgb)[..., 0], atol=1e-6)

def test_invalid_skin_appearance_parameters_reject_before_render():
    with pytest.raises(ValueError, match="ab_offset"):
        validate_skin_appearance_parameters({"version": 1, "ab_offset": [float("nan"), 0]})
```

Add tests proving an all-zero alpha is identity, profile sampling excludes low
alpha pixels, and a source profile with excess positive `a` derives a negative
bounded `ab_offset` toward a target profile.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `.venv/bin/python -m pytest tests/test_skin_appearance.py -q`

Expected: import failures.

- [ ] **Step 3: Implement one bounded perceptual transform**

Use the published OKLab matrix conversion directly in this new, independently
written module. Implement only this parameter contract:

```python
@dataclass(frozen=True)
class SkinAppearanceParameters:
    version: int
    luma_mode: str
    ab_offset: tuple[float, float]
    ab_scale: tuple[float, float]
    strength: float

def derive_skin_appearance_parameters(
    candidate: dict[str, list[float]], target: dict[str, list[float]], *, strength: float
) -> SkinAppearanceParameters: ...

def apply_skin_appearance(
    linear_rgb: np.ndarray, params: SkinAppearanceParameters, alpha: np.ndarray
) -> np.ndarray: ...
```

Require `version == 1`, `space == "oklab"`, `luma_mode == "preserve"`,
`0.0 <= strength <= 1.0`, `0.85 <= ab_scale[i] <= 1.15`, and
`-0.04 <= ab_offset[i] <= 0.04`. `apply_skin_appearance` preserves the
original OKLab L component, transforms only a/b, then blends by alpha.

- [ ] **Step 4: Run focused tests**

Run: `.venv/bin/python -m pytest tests/test_skin_appearance.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/colorai/skin_appearance.py tests/test_skin_appearance.py
git commit -m "Add bounded perceptual skin appearance math"
```

## Task 3: Build reviewable temporal face-mask tracks

**Files:**
- Create: `src/colorai/face_masks.py`
- Modify: `src/colorai/project/models.py`
- Modify: `tests/test_face_corrections.py`
- Create: `tests/test_face_masks.py`

**Interfaces:**
- Produces `build_face_mask_track`, `validate_face_mask_track`, `interpolate_mask_geometry`, `render_face_mask`, and `make_face_mask_contact_sheet`.
- Consumes a valid `FaceTrack`; produces a `FaceMaskTrack` in `valid`, `failed`, or `unsafe` state with `review_state="unreviewed"`.
- Later tasks require `review_state == "approved_for_proposal"` before drafting a skin target or correction.

- [ ] **Step 1: Write failing mask geometry and review tests**

```python
def test_landmark_mask_excludes_protected_eye_mouth_and_hair_pixels():
    geometry = _synthetic_landmark_geometry()
    mask = render_face_mask((96, 96), geometry, strategy="landmark_skin")
    assert mask[30, 32] == 0.0       # left eye
    assert mask[65, 48] == 0.0       # mouth
    assert mask[5, 48] == 0.0        # hair/outside oval
    assert mask[52, 48] > 0.5        # cheek

def test_unreviewed_mask_cannot_be_used_for_a_proposal(store):
    mask = _valid_mask_track(store, review_state="unreviewed")
    with pytest.raises(ValueError, match="approved_for_proposal"):
        validate_face_mask_track(store, mask.id, require_review=True)
```

Add an interpolation test across two landmark keyframes and a failure test for
coverage below `MIN_COVERAGE` or a gap above `MAX_GAP_RATIO`.

- [ ] **Step 2: Run focused tests and verify they fail**

Run: `.venv/bin/python -m pytest tests/test_face_masks.py -q`

Expected: module/import failures.

- [ ] **Step 3: Implement a backend protocol and conservative fallback**

```python
class LandmarkDetector(Protocol):
    def __call__(self, image_bgr: np.ndarray, box_xywh: tuple[int, int, int, int]) -> dict[str, np.ndarray] | None: ...

def build_face_mask_track(
    store: ProjectStore, face_track_id: int, *, detector: LandmarkDetector | None = None,
    strategy: str = "landmark_skin", samples: int = 16,
) -> FaceMaskTrack: ...

def render_face_mask(
    shape_hw: tuple[int, int], geometry: dict, *, strategy: str
) -> np.ndarray: ...
```

When `detector` returns landmarks, store normalized polygons for face oval,
eyes, brows, mouth/lips, and hairline boundary. Construct a feathered face
skin alpha by filling the oval, subtracting protected polygons, intersecting
with the existing conservative colour skin mask, and applying a Gaussian
feather. When no landmark backend is installed, store
`backend="fallback"`, `strategy="face_oval_skin"`, and clearly label that
fallback in evidence. Do not pretend that it protects facial features it did
not detect.

Add a `make_face_mask_contact_sheet` helper that renders the actual alpha as a
cyan overlay on sampled source frames with a yellow face box and frame index.

- [ ] **Step 4: Add an optional MediaPipe adapter behind an extra**

Add a `mediapipe` optional dependency only after recording its package/model
license and version in `pyproject.toml` and `docs/dependency-audit.md`. The
adapter implements `LandmarkDetector`; tests inject synthetic landmarks and do
not download a model. If the extra or model is unavailable,
`build_face_mask_track` must return the labelled fallback or a clear error,
never silently substitute a different model.

- [ ] **Step 5: Run focused tests**

Run: `.venv/bin/python -m pytest tests/test_face_masks.py tests/test_face_corrections.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/colorai/face_masks.py src/colorai/project/models.py pyproject.toml docs/dependency-audit.md tests/test_face_masks.py tests/test_face_corrections.py
git commit -m "Add reviewable temporal face mask tracks"
```

## Task 4: Extend the shared compositor and render preflight

**Files:**
- Modify: `src/colorai/face_corrections.py`
- Modify: `tests/test_face_corrections.py`
- Modify: `tests/test_render.py`
- Modify: `tests/test_face_render_parity.py`

**Interfaces:**
- Extends `FaceCorrectionSpec` with `kind`, `parameters`, and `mask_track_id`/mask geometry.
- `apply_face_corrections(image_rgb, corrections, frame_index)` remains the only face compositor entry point.
- `load_face_correction_specs` rejects an enabled `skin_appearance` correction lacking a reviewed valid mask track or valid parameters.

- [ ] **Step 1: Write failing compositor and preflight tests**

```python
def test_skin_appearance_changes_cheek_but_not_eye_lip_or_second_face():
    image, geometry = _two_face_fixture()
    spec = _appearance_spec(geometry, ab_offset=(-.02, .0))
    out = apply_face_corrections(image, [spec], frame_index=0)
    assert not np.array_equal(out[48, 40], image[48, 40])  # cheek
    assert np.array_equal(out[30, 32], image[30, 32])      # eye
    assert np.array_equal(out[65, 48], image[65, 48])      # lip
    assert np.array_equal(out[48, 80], image[48, 80])      # other participant

def test_enabled_skin_appearance_requires_reviewed_mask_before_output(tmp_path):
    store, correction = _enabled_appearance_with_mask(review_state="unreviewed")
    with pytest.raises(ValidationError, match="approved_for_proposal"):
        render_master(store, correction.shot.asset_id, tmp_path / "out.mp4")
    assert not (tmp_path / "out.mp4").exists()
```

Extend `tests/test_face_render_parity.py` with a moving mask geometry fixture,
then compare decoded first and middle output frames to `load_corrected_still`
within codec tolerance.

- [ ] **Step 2: Run focused tests and verify they fail**

Run: `.venv/bin/python -m pytest tests/test_face_corrections.py tests/test_render.py tests/test_face_render_parity.py -q`

Expected: assertions fail because only RGB gains and box masks exist.

- [ ] **Step 3: Implement the correction union**

Refactor the spec to carry the row's kind and validated parameter object:

```python
@dataclass(frozen=True)
class FaceCorrectionSpec:
    id: int
    kind: str
    parameters: dict[str, object]
    keyframes: tuple[tuple[int, float, float, float, float], ...]
    mask_geometry_keyframes: tuple[dict[str, object], ...]
    source_width: int
    source_height: int
```

Keep legacy `rgb_balance` output bit-for-bit compatible. For
`skin_appearance`, resolve the interpolated mask geometry, render its alpha at
the current input size, convert only the cropped masked region to linear RGB,
call `apply_skin_appearance`, encode once, and composite in stable correction
ID order. The previous `covered` exclusion policy still prevents overlapping
face corrections from double-grading a pixel.

Update `_validate_face_correction_row` to branch by kind, validate the exact
version-one parameter contract, validate the linked `SkinAppearanceTarget`, and
require a valid `FaceMaskTrack` whose `review_state` is
`approved_for_proposal`. Do not weaken any existing `FaceTrack`, subject,
metric, or scope checks.

- [ ] **Step 4: Run focused parity and preflight tests**

Run: `.venv/bin/python -m pytest tests/test_face_corrections.py tests/test_render.py tests/test_face_render_parity.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/colorai/face_corrections.py tests/test_face_corrections.py tests/test_render.py tests/test_face_render_parity.py
git commit -m "Render reviewed perceptual face corrections"
```

## Task 5: Implement references, targets, and scope-safe candidate derivation

**Files:**
- Create: `src/colorai/skin_targets.py`
- Modify: `src/colorai/face_corrections.py`
- Modify: `tests/test_skin_targets.py`
- Modify: `tests/test_matching.py`

**Interfaces:**
- Produces `create_skin_reference`, `request_skin_reference`, `review_face_mask`, `suggest_skin_target`, `approve_skin_target`, and `match_skin_target_to_group`.
- `match_skin_target_to_group` returns evidence/candidates but creates corrections only through explicit `propose_skin_appearance_correction`.
- All candidate corrections default to `state="suggested"`, `enabled=False`.

- [ ] **Step 1: Write failing lifecycle and scope tests**

```python
def test_external_reference_is_copied_by_hash_and_retains_provenance(tmp_path):
    reference = tmp_path / "accurate.jpg"; reference.write_bytes(b"pixels")
    created = create_skin_reference(store, subject_id=alice.id, group_id=group.id,
                                    external_path=reference, role="accurate_skin_reference")
    assert Path(created.managed_path).read_bytes() == b"pixels"
    assert created.source_path == str(reference)
    assert created.content_hash == hashlib.sha256(b"pixels").hexdigest()

def test_target_match_rejects_cross_variant_and_unreviewed_mask(store):
    with pytest.raises(ValueError, match="same setup or variant"):
        match_skin_target_to_group(store, target_id=target.id, group_id=other_variant.id)
```

Add tests for project-frame reference scope, creative-reference confidence
reduction, missing original external file after import, target approval
sequence, one-shot `qc_only`, and distinct parameters for two candidates with
different masked profiles.

- [ ] **Step 2: Run focused tests and verify they fail**

Run: `.venv/bin/python -m pytest tests/test_skin_targets.py tests/test_matching.py -q`

Expected: import/function failures.

- [ ] **Step 3: Implement the lifecycle in one focused service module**

Use these signatures:

```python
def create_skin_reference(store: ProjectStore, *, subject_id: int, group_id: int,
                          role: str, source_shot_id: int | None = None,
                          frame_index: int | None = None, external_path: Path | None = None,
                          crop_geometry: dict[str, float] | None = None) -> SkinAppearanceReference: ...
def review_face_mask(store: ProjectStore, mask_track_id: int, *, review_state: str, reason: str) -> FaceMaskTrack: ...
def suggest_skin_target(store: ProjectStore, *, subject_id: int, group_id: int,
                         reference_id: int | None, face_track_id: int,
                         parameters: dict, rationale: str, confidence: float) -> SkinAppearanceTarget: ...
def match_skin_target_to_group(store: ProjectStore, *, target_id: int, group_id: int) -> dict: ...
```

`create_skin_reference` accepts exactly one source mode. It copies external
bytes to a project `references/<sha256>.<suffix>` path without modifying the
original. A project-frame reference must be in the target group and contain the
target subject's face. A target can only use an active reference with the same
asset/subject/group scope. `review_face_mask` accepts only the three states in
the spec and never upgrades a failed track.

`match_skin_target_to_group` enumerates only the exact setup or variant,
requires an approved target and reviewed valid masks, derives each candidate's
parameters from its own robust masked profile, and returns `uncertain` evidence
instead of a grade on any failed gate. `creative_look_direction` caps strength
at `0.50`; accurate references may use the normal bounded range. It never
alters or enables an existing correction.

- [ ] **Step 4: Run focused tests**

Run: `.venv/bin/python -m pytest tests/test_skin_targets.py tests/test_matching.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/colorai/skin_targets.py src/colorai/face_corrections.py tests/test_skin_targets.py tests/test_matching.py
git commit -m "Add scope-safe skin target matching"
```

## Task 6: Expose draft-only MCP evidence and proposal tools

**Files:**
- Modify: `src/colorai/mcp_server.py`
- Create: `tests/test_mcp_skin_targets.py`
- Modify: `tests/test_mcp_skin_matching.py`
- Modify: `AGENTS.md`

**Interfaces:**
- Produces the exact MCP tools `skin_target_workspace`, `request_skin_reference`, `create_skin_appearance_reference`, `list_skin_appearance_references`, `build_face_mask_track`, `get_face_mask_contact_sheet`, `review_face_mask_evidence`, `suggest_skin_appearance_target`, `list_skin_appearance_targets`, and `match_skin_target_to_setup`.
- Consumes the Task 3–5 service functions.
- Does not expose any human-only approval/enable/reject/delete/render action.

- [ ] **Step 1: Write failing MCP-surface tests**

```python
def test_skin_target_mcp_surface_is_draft_only():
    names = {tool.name for tool in asyncio.run(mcp_server.mcp.list_tools())}
    assert {"skin_target_workspace", "get_face_mask_contact_sheet", "match_skin_target_to_setup"} <= names
    assert not {"approve_skin_target", "enable_face_correction", "render_master"} & names

def test_mcp_cannot_draft_before_mask_review(tmp_path):
    out = mcp_server.suggest_skin_appearance_target(str(db), subject.id, group.id, reference.id, track.id,
                                                     {"version": 1, "space": "oklab", "luma_mode": "preserve", "ab_offset": [0, 0], "ab_scale": [1, 1], "strength": .5},
                                                     "test", .8)
    assert "approved_for_proposal" in out["error"]
```

- [ ] **Step 2: Run focused tests and verify they fail**

Run: `.venv/bin/python -m pytest tests/test_mcp_skin_targets.py tests/test_mcp_skin_matching.py -q`

Expected: unknown tool/function failures.

- [ ] **Step 3: Add tools and agent instructions**

Make each mutating MCP tool return structured error dictionaries instead of
raising raw exceptions. `get_face_mask_contact_sheet` returns an MCP image.
`review_face_mask_evidence` records only an agent review state/reason and may
set `needs_rebuild` or `unsafe`; it cannot approve a target or grade. Add this
instruction near the MCP matching guidance:

```text
Before drafting a skin target or skin_appearance correction, inspect source
frames and get_face_mask_contact_sheet. Never claim a true skin tone without
an accurate human-designated reference. Treat a creative reference as a softer
look direction and leave uncertain candidates ungraded.
```

- [ ] **Step 4: Run MCP tests**

Run: `.venv/bin/python -m pytest tests/test_mcp_skin_targets.py tests/test_mcp_skin_matching.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/colorai/mcp_server.py AGENTS.md tests/test_mcp_skin_targets.py tests/test_mcp_skin_matching.py
git commit -m "Add draft-only MCP skin target tools"
```

## Task 7: Build the human skin-target review flow

**Files:**
- Modify: `src/colorai/ui.py`
- Modify: `src/colorai/templates/index.html`
- Create: `tests/test_skin_targets_api.py`
- Modify: `tests/test_face_corrections_api.py`
- Modify: `tests/test_review_ui.py`

**Interfaces:**
- Produces UI endpoints for reference intake, mask contact-sheet retrieval, target proposal/approval/rejection, and human-only correction approval/enabling.
- Extends `_workspace` with `skin_targets` per group and participant.
- Uses the existing view-preserving fetch helper; no action navigates back to the root view.

- [ ] **Step 1: Write failing API/UI tests**

```python
def test_skin_target_workspace_shows_reference_mask_and_disabled_proposal(tmp_path):
    client, ids = _target_fixture(tmp_path)
    html = client.get("/").text
    assert "Skin target" in html
    assert "Accurate skin reference" in html
    assert "Review mask" in html
    assert "Agent suggestion" in html
    assert "Approve target" in html

def test_target_actions_preserve_selected_setup_view(tmp_path):
    client, ids = _target_fixture(tmp_path)
    response = client.post(f"/api/skin-targets/{ids.target}/approve")
    assert response.json() == {"id": ids.target, "state": "approved"}
    assert "selectedGroup" in client.get("/").text
```

Add tests for rejected references/targets, external-image upload hash mismatch,
mask-overlay PNG response, no enabled correction before target approval, and
the existing skin-matching UI remaining readable for legacy `rgb_balance` rows.

- [ ] **Step 2: Run focused tests and verify they fail**

Run: `.venv/bin/python -m pytest tests/test_skin_targets_api.py tests/test_face_corrections_api.py tests/test_review_ui.py -q`

Expected: missing endpoints and markup assertions.

- [ ] **Step 3: Add typed API endpoints and workspace data**

Define Pydantic payloads for reference import, mask review, target draft, and
target update. Add endpoints with these human-only boundaries:

```text
POST /api/assets/{asset_id}/skin-references
GET  /api/face-mask-tracks/{mask_track_id}/contact-sheet.png
POST /api/face-mask-tracks/{mask_track_id}/review
POST /api/skin-targets/{target_id}/approve
POST /api/skin-targets/{target_id}/reject
POST /api/face-corrections/{correction_id}/approve
POST /api/face-corrections/{correction_id}/enable
```

Validate form/file paths server-side; never accept a client-supplied managed
path. Return JSON errors with HTTP 400 for scope/validation faults and 404 for
missing rows. Ensure `_workspace` returns the exact evidence needed by the
template: role, provenance, target state, reference crop URL, mask state and
reason, contact-sheet URL, candidate state, confidence, parameters, and
before/after URLs.

- [ ] **Step 4: Implement the focused UI**

Add a `Skin target` panel inside the existing selected-group workspace, not a
new global screen. Keep this order: participant → reference/status → visible
mask evidence → agent target previews → cross-angle candidates. Use labelled
badges (`accurate reference`, `creative direction`, `unreviewed mask`,
`needs rebuild`, `unsafe`, `agent suggestion`, `approved target`, `uncertain`).
Make all action handlers call the existing fetch/view-preservation helper and
refresh only workspace data. The target preview uses an in-place before/after
wipe over the same source crop plus a full-frame context thumbnail; it must
not rely on a generated face image.

- [ ] **Step 5: Run API/UI tests**

Run: `.venv/bin/python -m pytest tests/test_skin_targets_api.py tests/test_face_corrections_api.py tests/test_review_ui.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/colorai/ui.py src/colorai/templates/index.html tests/test_skin_targets_api.py tests/test_face_corrections_api.py tests/test_review_ui.py
git commit -m "Add human review for skin appearance targets"
```

## Task 8: Verify end-to-end safety, document it, and run a real-clip review

**Files:**
- Modify: `docs/architecture.md`
- Modify: `docs/status.md`
- Modify: `README.md`
- Modify: `tests/test_face_render_parity.py`
- Modify: `tests/test_render.py`

**Interfaces:**
- Documents the exact agent/human boundary and `skin_appearance` colour model.
- Produces evidence from an approved target only; does not commit media or database scratch outputs.

- [ ] **Step 1: Write the final failing integration tests**

```python
def test_render_aborts_before_output_for_enabled_target_with_cross_scope_reference(tmp_path):
    store, asset, correction = _cross_scope_enabled_target_fixture()
    output = tmp_path / "out.mp4"
    with pytest.raises(ValidationError, match="scope"):
        render_master(store, asset.id, output)
    assert not output.exists()
```

Add a real ffmpeg fixture with two tracked faces and moving geometry; assert the
selected face changes, protected eye/lip/background pixels remain unchanged
within codec tolerance, and preview/render frames match.

- [ ] **Step 2: Run the failing integration tests**

Run: `.venv/bin/python -m pytest tests/test_face_render_parity.py tests/test_render.py -q`

Expected: FAIL until all validation paths are present.

- [ ] **Step 3: Close any validation gaps and document actual behaviour**

Update docs to say that version one accepts project/external references,
requires agent-inspected mask evidence, applies deterministic OKLab chroma-only
corrections under a temporal face mask, and leaves uncertain candidates
ungraded. State the optional MediaPipe dependency and fallback limits plainly.
Do not claim the system determines a person's true skin tone.

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest -q`

Expected: PASS, including migration, MCP, API/UI, and ffmpeg tests (ffmpeg
tests may skip only when ffmpeg is unavailable).

- [ ] **Step 5: Perform the real-clip review without enabling or rendering**

Use stdio MCP against `_scratch/chunk_proj.sqlite3`:

```text
1. Locate the black-shirt participant/setup corresponding to feature timecode
   00:34:10:19–00:34:14:07.
2. Inspect source frames and the face-mask contact sheet.
3. Create one reference only after the filmmaker labels it accurate or creative.
4. Draft, but do not approve or enable, at most three target previews.
5. Inspect the previews and record why any candidate is uncertain.
```

The pass succeeds if it leaves reviewable evidence and disabled proposals, or
if it correctly leaves no proposal because the evidence is insufficient.

- [ ] **Step 6: Commit**

```bash
git add docs/architecture.md docs/status.md README.md tests/test_face_render_parity.py tests/test_render.py
git commit -m "Document and verify skin target workflow"
```

## Plan self-review

### Spec coverage

- Internal/external references and accurate-vs-creative roles: Tasks 1, 5, and 7.
- Agent-inspected, reviewable masks with rebuild/unsafe outcomes: Tasks 3, 6, and 7.
- Perceptual chroma-only target model and deterministic compositor: Tasks 2 and 4.
- Individual candidate grades constrained to subject/setup/variant: Task 5.
- Draft-only MCP and human-only approval/enable/render: Tasks 4, 6, and 7.
- Preview/render parity and abort-before-output safety: Tasks 4 and 8.
- Documentation, license boundaries, and real-clip review: Tasks 3 and 8.

### Consistency checks

- All persisted model names and states match the approved design.
- `FaceMaskTrack.review_state="approved_for_proposal"` is the sole proposal gate in Tasks 3–7.
- All version-one `skin_appearance` parameters use `space="oklab"` and `luma_mode="preserve"`.
- Existing `rgb_balance` corrections remain supported and are regression-tested.
