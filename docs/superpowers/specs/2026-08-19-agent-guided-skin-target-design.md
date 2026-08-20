# Agent-guided skin targets

## Purpose

ColorAI currently matches a person to an approved shot with a tracked,
face-local RGB balance. That is useful for small camera drift. It cannot repair
a face that is visibly too red, magenta, or warm when every available shot is a
poor reference.

This design adds a reviewable skin-target workflow for finished Rec.709 video.
It lets a filmmaker supply an accurate reference from the project or from an
external image, or accept a cautious agent-proposed target. The engine then
derives a deterministic, face-local correction for each angle. It never
generates or repaints pixels.

The primary use case is the same person across a multicamera interview. It is
not automatic beauty retouching, whole-frame look transfer, log conversion, or
a replacement for a colorist's global grade.

## Principles

1. A skin target is evidence, not an assertion about a person's "correct"
   complexion. The filmmaker labels a reference as either **accurate skin
   reference** or **creative look direction**.
2. The agent may inspect, draft, and revise. Only the human approves a target,
   approves or enables a correction, and renders.
3. A local skin correction preserves the source face's texture, identity,
   lighting shape, and luminance by default. It never changes the room, shirt,
   or another participant.
4. Every preview and export uses the same pure compositor and the same stored
   parameters. Source media and external-reference originals remain untouched.
5. A mask is evidence. The agent must inspect a contact sheet with mask
   overlays before it proposes a correction. A low-quality or unreviewed mask
   blocks a proposal.

## Research inputs and licensing

The implementation is original. No code, model weights, or ComfyUI glue will
be copied or vendored from these projects without a separate dependency and
license review.

| Input | Adopted idea | Deliberate boundary |
| --- | --- | --- |
| [LumiVideo](https://arxiv.org/html/2604.02409v1) | perception, reasoning, execution, and reflection should lead to editable parameters, not generated video pixels | It targets autonomous log base grades. ColorAI remains human-approved and face-local for this feature. |
| [skin-color-correction](https://github.com/rodrigorcz/skin-color-correction) | use an explicitly accurate reference, face segmentation, and colour-distribution transfer | ColorAI adds temporal tracks, scope checks, mask review, and preview/render parity. |
| [Eric Color Correction](https://github.com/EricRollei/Eric_Color_Correction_ComfyUI) | skin-line/luminance diagnostics, multiple mask strategies, and reusable corrections | Its license is not a dependency option. Treat it as product research only. |
| [Skin Tone Checker](https://github.com/kangabru/skin-tone-checker) | diagnostic guidance should not dictate a single ideal skin tone | Do not copy its GPL code or encode a universal skin-tone line as truth. |
| [ComfyUI-VideoColorGrading](https://github.com/kijai/ComfyUI-VideoColorGrading) | reference-driven LUT work is a useful future whole-frame workflow | A global LUT is out of scope for face-local correction. |
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) | dense landmarks can define eyes, lips, brows, and facial boundaries | Add it only as an optional, separately audited model dependency. |

## User workflow

### 1. Establish a target

Inside a selected setup or lighting variant, the user selects a participant and
opens **Skin target**.

The page shows three choices:

- **Use project frame** — select an in-scope frame containing that participant.
- **Import external reference** — import a still or photo. ColorAI copies it
  into project-managed reference storage by content hash, preserving the
  original path and hash for provenance.
- **Ask the agent for a starting point** — the agent records a request for an
  accurate reference if one would materially improve confidence. If none is
  available, it may draft a conservative source-derived preview labelled
  *agent suggestion*, never *true skin tone*.

Every imported or selected reference has one required role:

- `accurate_skin_reference`: the agent may use masked face chroma statistics
  as target evidence.
- `creative_look_direction`: the agent may use it as a soft preference and
  must state that it is not an accuracy claim.

The agent never copies reference exposure, local contrast, skin texture,
lighting direction, or background. It transfers only a bounded face-appearance
profile.

### 2. Build and inspect a mask track

For the selected subject and shot, ColorAI builds a mask track. It uses dense
face landmarks when the optional backend is installed; otherwise it retains the
existing tracked box, face-oval falloff, and conservative skin mask as a
labelled fallback.

A landmark mask includes separate protected regions for eyes, brows, lips,
teeth, hairline, and the exterior face boundary. It has a feathered skin region
that excludes those regions. The system stores normalized landmark keyframes,
their coverage, gaps, confidence, backend/version, and mask strategy. It does
not persist a large raster mask for every frame.

The UI and MCP expose a contact sheet of source frames with the exact rendered
mask overlay. The agent must inspect it and store one of:

- `approved_for_proposal`
- `needs_rebuild`, with a reason and a requested strategy
- `unsafe`, with a reason

The agent may request a rebuilt mask using a different conservative strategy;
it may not silently expand a mask over eyes, teeth, hair, or another face. A
human can see the same evidence and may reject a mask. Only
`approved_for_proposal` tracks can receive a skin-target proposal.

### 3. Draft target appearances

Given a valid mask and reference evidence, the agent drafts at most three
disabled target previews. Each proposal names its intent, such as:

- `neutralize excess red/magenta`
- `neutralize while retaining warmth`
- `match approved accurate reference`

The review screen shows the reference, source face crop, mask overlay,
corrected crop, full-frame context, and a split/wipe comparison. It also shows
an explanation, confidence, reference role, and the effective bounded
parameters. The user can approve one appearance target, reject it, or ask for
a different direction in natural language.

### 4. Match other angles

After a target is approved, the agent examines other valid tracks in the same
`subject × setup/variant` scope. It derives an individual, disabled correction
per shot. It does not copy the first shot's numbers: each correction uses the
candidate face's own masked measurements and preserves that shot's luminance.

The agent must label a candidate `uncertain` rather than grade it when a mask
is unsafe, the face is too small, lighting materially changes, a track is
unstable, or the reference is merely creative direction. One-shot participants
remain QC-only unless the user creates a target for a later matching scope.

## Deterministic colour model

### Analysis space

Colour analysis runs on decoded display-linear BT.709 RGB, converted into a
perceptual opponent space such as OKLab. The measured profile excludes masked
eyes, teeth, lips, brows, hair, extreme highlights, and deep shadows. It
records robust centre and spread for lightness and opponent chroma, not a
single average pixel.

For an `accurate_skin_reference`, the agent may compare the candidate and
reference masked chroma profiles. For a `creative_look_direction`, it may bias
a suggestion toward the reference but must reduce confidence and state that the
target is aesthetic. For no reference, it may suggest a bounded reduction of a
clear red/magenta cast, but it must not infer ethnicity or claim accuracy.

### Correction kind

`FaceCorrection.kind="skin_appearance"` is added alongside the existing
`rgb_balance`. Its versioned parameters describe a bounded transformation in
the perceptual space:

```json
{
  "version": 1,
  "space": "oklab",
  "luma_mode": "preserve",
  "ab_offset": [-0.012, 0.006],
  "ab_scale": [0.94, 0.98],
  "strength": 0.72
}
```

`ab_offset` and `ab_scale` affect chroma only. `luma_mode="preserve"` is the
default and applies no lightness change. A later, explicit bounded lightness
mode may be added only after validation. Values are finite, conservative, and
validated by a single parameter validator shared by the API, preview, and
render preflight.

The compositor converts only pixels under the temporal face mask to the
perceptual space, applies the transform, converts back to linear BT.709 RGB,
and alpha-composites the result over the original. It has no smoothing,
beautification, inpainting, or generated-pixel stage.

### Target derivation

An approved `SkinAppearanceTarget` stores its subject, group scope, provenance,
masked profile, approved correction preview, and state. A candidate correction
is derived from the difference between the candidate's masked profile and the
approved target profile. The algorithm clamps its output and records the raw
measurement, clamp, target ID, and rationale in `FaceCorrection.evidence`.

## Persistence

Add an Alembic revision and the following records.

```
SkinAppearanceReference
  subject_id, asset_id?, group_id?
  source_kind: project_frame | external_image
  role: accurate_skin_reference | creative_look_direction
  source_path, content_hash, source_shot_id?, frame_index?, crop_geometry
  state: active | rejected

FaceMaskTrack
  face_track_id, subject_id, shot_id
  backend, backend_version, strategy
  landmark_keyframes, coverage, max_gap, state, review_state, review_reason

SkinAppearanceTarget
  subject_id, group_id, reference_id?
  profile, approved_preview_parameters, state: suggested | approved | rejected
  rationale, confidence
```

`FaceCorrection` gains nullable `skin_target_id` and accepts
`kind="skin_appearance"`. Existing `rgb_balance` corrections keep their
current behaviour and validation. All scoped relationships must resolve to the
same project asset, subject, shot, and setup/variant scope.

## MCP contract

MCP remains draft-only. New tools are read-only or draft/revise operations:

- `skin_target_workspace`
- `request_skin_reference`
- `create_skin_appearance_reference`
- `list_skin_appearance_references`
- `build_face_mask_track`
- `get_face_mask_contact_sheet`
- `review_face_mask_evidence`
- `suggest_skin_appearance_target`
- `list_skin_appearance_targets`
- `match_skin_target_to_setup`

Tool instructions require the agent to inspect mask contact sheets and source
frames before drafting. MCP cannot approve a reference or target, approve,
enable, reject, delete a face correction, or render media.

## Review UI

Add a **Skin target** workspace to each setup/variant:

1. Participant selector and target status.
2. Reference intake, role badge, provenance, crop, and a visible request when
   the agent needs a better reference.
3. Mask review card with overlay, contact sheet, backend/coverage/gap status,
   and agent review rationale.
4. Target-preview comparison with source/reference/corrected face crops,
   full-frame before/after wipe, parameter summary, and approve/reject.
5. Cross-angle proposal strip that distinguishes ready-to-review corrections
   from uncertain/QC-only candidates.

Human actions use in-place API updates. They must not reset selections or
return the user to the home view.

## Failure handling

- Missing external files leave the stored derivative and provenance visible;
  they do not invalidate already-approved deterministic parameters.
- A missing, unsafe, or unreviewed mask blocks target and candidate proposals.
- A target/reference outside the subject/group/asset scope is rejected at
  creation time.
- An unsupported source transfer (log, HDR, or untagged non-Rec.709) retains
  ColorAI's current colour-management refusal.
- An invalid enabled `skin_appearance` correction aborts render before output
  begins, exactly as existing invalid face corrections do.

## Tests and acceptance criteria

1. Unit tests convert known OKLab values and confirm neutral, identity, bounded
   chroma-only, and invalid parameter cases.
2. A synthetic two-face frame proves that a skin-appearance transform changes
   only the selected tracked/masked face; eye, lip, hair, and background
   protection pixels remain within tolerance.
3. A moving-face ffmpeg fixture proves preview/render parity and temporal mask
   interpolation over representative motion.
4. Reference tests cover project-frame and external-image provenance, hash
   stability, role semantics, missing source files, and cross-asset rejection.
5. Mask-review tests prove that no target or correction can be proposed before
   the contact-sheet review state is approved.
6. Scope tests prove per-shot matching derives individual parameters and never
   crosses subject, setup, lighting variant, or participant boundaries.
7. API/UI tests cover reference intake, visible mask overlays, target approval,
   disabled candidate proposals, and view-preserving in-place actions.
8. A final real-clip review compares the black-shirt interview at feature
   timecode `00:34:10:19–00:34:14:07`: one human-approved source-derived or
   accurately referenced target, then one separately derived candidate angle.
   This is a review exercise, not an automatic acceptance test.

## Explicitly out of scope

- Diffusion, face replacement, beauty smoothing, texture alteration, or any
  generated pixel output.
- Whole-frame colour matching and global LUT generation.
- Automatic claims about a person's true or ideal skin tone.
- Log/HDR/OCIO support, which remains a separate managed-colour project.
- Automatic approval, enabling, or full-film export.
