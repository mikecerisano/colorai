# Skin-First Multicamera Matching Design

## Goal

Make ColorAI useful for its primary finishing task: within one approved
interview setup and lighting variant, help a human make the *same person's*
skin tone consistent across camera angles without treating normal differences
in framing, windows, wardrobe, or background as a whole-frame grade problem.

The system must render every approved face-region correction identically in
preview and in a full-master export. Nothing is enabled or rendered from an
agent's judgment until a human approves it.

## Scope and non-goals

This is for an already professionally finished, BT.709/Rec.709 delivery
master. It adds conservative local skin corrections, not a dailies color
pipeline.

In scope:

- same-subject, same-setup/variant temporal skin measurement;
- persisted face tracks and soft masks;
- conservative masked RGB balance corrections;
- agent evidence and classification;
- human review, approval, preview, and delivery render parity.

Out of scope:

- automatic whole-frame matching across camera angles;
- beauty retouching, face reshaping, texture synthesis, or generative work;
- log/RAW/OCIO workflows;
- automatic approval;
- skin matching between different people;
- applying a correction to an entire frame because a face sample differs.

## Editorial rules

- The unit of comparison is **subject × exact setup/lighting variant**.
  A parent setup with variants is never a cross-variant whole-frame match
  scope.
- The same person can have a different intended look in another location,
  lighting variant, or day. No fallback reference crosses those boundaries.
- A visible bright window, wide shot, different background, or a different
  camera angle is not evidence of a color error.
- A one-shot participant is QC-only. It has no skin-match target and receives
  no automatic correction proposal.
- The agent may classify evidence as `skin_mismatch`, `intentional_lighting`,
  or `uncertain`. Only `skin_mismatch` can carry a correction proposal.
- The correction operates only under a tracked, feathered face-skin mask. It
  must not alter the background or a second participant in the same shot.

## User workflow

1. A human approves one reference frame per participant and exact matching
   scope. The reference frame must visibly contain that participant.
2. An agent calls the skin-first workspace for a scope. It receives face crops,
   temporal track summaries, representative frames, reference context, and
   measured skin differences.
3. The agent reviews the images, not merely the numbers. It writes one of the
   three classifications per candidate shot and a concise reason. It may add a
   conservative RGB-balance proposal only for `skin_mismatch`.
4. The review UI displays the reference, candidate frame, face box/mask
   overlay, temporal confidence, agent reason, and a masked before/after
   comparison. The human accepts, edits, rejects, or marks the shot
   intentional.
5. An accepted correction becomes enabled only through a human UI action.
6. Preview and full-master render use the same stored track/mask specification
   and compositing code. A render refuses to proceed if an enabled correction
   lacks a valid track instead of silently applying it to the wrong pixels.

## Data model

Use a separate face-local layer rather than overloading whole-frame
`Correction` rows.

### `FaceTrack`

Persist one reusable track for a detected face in a shot:

- `id`, `shot_id`, `skin_metric_id`, `subject_id`;
- source frame dimensions and analysis scale;
- sampled inclusive frame indices and normalized `(x, y, width, height)` face
  boxes as JSON keyframes;
- `sample_count`, `tracked_count`, coverage, maximum gap, skin stability,
  temporal median BGR, and creation time;
- state: `valid` or `failed`, with a failure reason.

The source asset remains read-only. The track records derived metadata only.
It is built from the same subject/face identity selected on the representative
frame. Track keyframes are interpolated during preview/render; no per-frame
face detector is allowed in export.

### `FaceCorrection`

Persist the human-reviewable local correction:

- `id`, `shot_id`, `subject_id`, `skin_metric_id`, `face_track_id`;
- reference shot/group IDs, kind, parameters, evidence, agent reason,
  confidence, classification, and creation time;
- `state`: `suggested`, `approved`, or `rejected`;
- `enabled`: false by default. It can become true only after approval in the
  review UI.

Version one supports only `kind="rgb_balance"`, with three linear-light gains
clamped to `[0.90, 1.10]`. That cap makes the feature a finishing adjustment,
not a substitute for primary grading. The agent may suggest no correction.

`evidence` stores the reference and candidate temporal skin summaries,
stability/coverage measurements, and the agent's classification rationale so
later agents and the filmmaker can understand the decision.

## Track and mask pipeline

1. Start from the persisted `SkinMetric` bounding box on the representative
   frame. Validate that it belongs to the requested subject and shot.
2. Sample at least 16 inclusive frames across the shot at a stable analysis
   scale. Re-detect and IoU-associate the selected face, as the current tracker
   does, but preserve every successful normalized box as a keyframe.
3. Accept a track only when coverage is at least 75%, the largest untracked
   gap is no more than 20% of the shot duration, and temporal skin stability is
   below the documented threshold. Otherwise persist a failed track and offer
   QC evidence only.
4. At preview/render time, interpolate the stored boxes to the current source
   frame. Within the box, calculate the deterministic existing color-only skin
   mask; feather it with a Gaussian edge at source resolution; multiply it by
   a conservative face-oval falloff so isolated hand/background skin pixels do
   not receive the correction.
5. Apply the RGB balance to a copy of the already whole-frame-corrected frame
   in linear light, then alpha composite only the masked pixels back over that
   frame. For multiple enabled face corrections, apply rows in stable ID order
   and prevent later masks from replacing already-covered pixels beyond their
   remaining alpha.

The same pure `apply_face_corrections(frame, face_corrections, frame_index)`
function serves corrected-still previews and `render.py`. It accepts an image,
stored track specs, and correction parameters; it performs no database or
filesystem work itself.

## Proposal and matching logic

Add `skin_first_match_subject_setup` beside the existing whole-frame matcher.
It requires:

- an approved exact-scope reference;
- at least two subject-visible shots in that scope;
- valid temporal tracks for reference and candidate;
- reference/candidate skin metrics that exceed a small deadband.

It returns candidates with temporal evidence but does not persist a correction.
The deterministic layer may calculate a conservative candidate balance. The
vision agent decides whether the evidence is an actual skin mismatch,
intentional lighting, or uncertain. It must compare face crops at the same
approximate scale and review representative frames before proposing a
correction.

The existing `match_subject_setup` remains a diagnostic for whole-frame
measurements. Its results must be labelled **composition-sensitive diagnostic**
in MCP/UI and must never be presented as the primary skin-matching workflow.

## MCP contract

MCP remains the agent's read/draft surface, not a source of approval.

- `skin_matching_workspace(project, asset_id, subject_id, group_id)` returns
  the exact reference, member shots, face IDs/boxes, crops, stored tracks,
  temporal skin summaries, and existing local proposals.
- `build_face_track(project, skin_metric_id, samples=16)` derives and persists
  a track, returning coverage/stability/error. It does not enable a grade.
- `get_face_track_contact_sheet(project, face_track_id)` returns labelled
  samples with face boxes for visual inspection.
- `skin_first_match_subject_setup(...)` returns deterministic candidate
  evidence only.
- `propose_face_correction(...)` accepts a valid track, explicit
  classification, reason, confidence, and a capped RGB-balance parameter set;
  it stores a `suggested`, disabled `FaceCorrection` only when classification
  is `skin_mismatch`.
- list/get/update draft tools allow an agent to revise its evidence while a
  correction remains suggested.

MCP must not expose approval, enable, or deletion of face corrections. Those
are human review-UI actions.

## Review UI

Add a clear **Skin matching** section inside a selected setup or variant:

- one row per participant, with `Reference approved`, `Needs reference`,
  `QC only`, or `No valid track` status;
- reference and candidate face crops side by side, with full-frame context;
- a toggle to reveal the soft mask/face box, especially for multi-person
  frames;
- agent label, confidence, temporal coverage/stability, and exact rationale;
- three prominent actions: **Approve correction**, **Reject**, and **Mark
  intentional**; no ambiguous generic toggle;
- a masked before/after preview that makes the untouched background visible.

Show a deliberate empty state for a single-shot participant: “No matching
target. Inspect skin QC or add a manual finishing correction.” This is useful
information, not an error.

## Render and safety rules

`render.py` loads enabled whole-frame corrections and enabled approved face
corrections per shot. It performs whole-frame corrections first, then the
masked face layer, using the pure shared compositor.

Before rendering, validate every enabled face correction:

- its correction is approved, belongs to the shot/asset, and references a
  valid track for the same subject/face;
- its source dimensions and track keyframes are usable;
- the RGB-balance parameters remain within the conservative cap;
- no required mask/track interval has an invalid gap.

If any validation fails, abort before producing delivery output. Do not skip a
face correction, fall back to a whole-frame correction, or silently create a
partial master.

## Tests and acceptance criteria

- Migration tests open existing databases without changing shots, corrections,
  or reference proposals.
- Unit tests cover keyframe interpolation, soft-mask feathering, mask bounds,
  alpha compositing, gain caps, multiple-face isolation, and deterministic
  output.
- Tracking tests cover successful tracks, insufficient coverage, excessive
  gaps, and stable normalized boxes at different source resolutions.
- Matching tests prove that two camera angles with different backgrounds do
  not create a whole-frame correction in the skin-first path; only the chosen
  participant's measured face region can produce a candidate.
- API/MCP tests prove draft-only agent permissions and reject attempts to
  approve/enable via MCP.
- Preview/render parity tests use a tiny ffmpeg fixture with a moving face-like
  region and assert matching pixels for the same frame, plus unchanged pixels
  outside the feathered mask.
- A two-person fixture proves a correction for one participant does not alter
  the other participant or the background.
- A render preflight test proves an invalid enabled face correction aborts with
  no output master.

## Deliberate limitations

The current color-only skin mask and box tracker are conservative tools, not
semantic face segmentation. Low-confidence, occluded, profile, fast-motion,
or poorly tracked faces must stay report-only. More capable segmentation or
model-based retouching can be added later behind the same persisted track and
review boundary.
