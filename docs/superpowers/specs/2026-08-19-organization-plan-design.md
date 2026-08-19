# Organization Plan Design

## Goal

Let a vision-capable MCP agent propose a complete, reviewable editorial
organization for an analyzed asset before it changes any shot assignment. A
human reviews the proposal and explicitly applies the accepted decisions as
one validated transaction.

## Scope

This feature serves ColorAI's current use case: a professionally finished
Rec.709 edit that needs interview setups, lighting variants, participants,
B-roll, and intentional exceptions organized before reference selection and
matching. It does not infer grades, apply corrections, or make color-managed
media decisions.

## Principles

- A measurement, detected face, or model judgment is evidence, not an
  editorial decision.
- Planning never changes the asset.
- The human approves the organization; an agent may only draft or revise it.
- Applying a plan is atomic. Invalid accepted decisions apply nothing.
- The applied result records the plan, its evidence, its author, and its
  approval so later agents can understand the editorial structure.
- B-roll, intentional exceptions, and unresolved shots remain distinguishable.
  `excused` continues to mean an intentional exception, never a generic inbox
  state.

## User flow

1. The agent calls `organization_workspace` and retrieves contact sheets or
   individual stills for the proposed clusters.
2. It writes a **draft**. The draft gives every shot one proposed destination:
   an existing setup or variant, a proposed setup or variant, B-roll,
   intentional exception, or unresolved.
3. The review UI opens the draft as a visual storyboard. Each decision shows
   representative frames, time range, named participants, confidence, and a
   plain-language reason. It does not show a flat timecode-only inbox.
4. The human accepts, changes, or rejects individual decisions. A rejected
   decision becomes unresolved unless the human selects a different
   destination.
5. The UI runs validation and shows errors before enabling **Apply accepted
   plan**.
6. Applying creates requested groups and variants, assigns accepted shots,
   puts accepted B-roll into the visible B-roll bin, marks accepted intentional
   exceptions as `excused`, and leaves unresolved/rejected shots untouched.
7. The UI shows a post-apply report. The next workflow is per-participant
   reference selection and matching, not more organization.

## Editorial model

### Destinations

Every plan item has exactly one destination:

| Destination | Meaning | Apply behavior |
| --- | --- | --- |
| existing group | A confirmed current setup or lighting variant | Assign the shot to that group and clear `excused`. |
| planned setup or variant | A group the plan will create | Create the group, then assign the shot and clear `excused`. |
| B-roll | Editorial material excluded from setup matching | Assign to the asset's canonical B-roll group and clear `excused`. |
| intentional exception | Material intentionally excluded from matching/QC outlier triage | Unassign it and set `excused=True`. |
| unresolved | Needs a later decision | Do not change its current assignment or exception state. |

`broll` becomes an explicit `ShotGroup.kind`, rather than relying on a magic
generic-group name. Matching rejects it just as it rejects other non-setup
groups. Existing legacy B-roll bins remain readable and are recognized by the
current compatibility rule until migrated by normal use.

### Groups and variants

A planned group has a stable draft key, a name, kind, optional camera label,
optional parent key, participant subject IDs, reason, and confidence.

- A `setup` is a real interview/location/editorial family.
- A `variant` is a lighting condition inside one setup. It must name a planned
  or existing `setup` parent.
- A camera label describes an angle within the same lighting condition. It is
  not inferred or required for grouping.
- One setup can have several people. People are participants, not duplicate
  groups. References and skin evaluation are selected per participant.

The agent may use visual evidence such as wardrobe, background, framing,
lighting direction, visible window state, and temporal proximity. It must not
infer a setup from identity or time adjacency alone. In particular, B-roll can
contain a known face, and the same person can appear in several unrelated
interviews.

## Persistence

Add a durable plan layer, separate from `ShotGroup` and `Shot`:

- `OrganizationPlan`: `id`, `asset_id`, `state` (`draft`, `approved`,
  `applied`, `superseded`), `author`, `summary`, `created_at`,
  `approved_by`, `approved_at`, and `applied_at`.
- `OrganizationPlanGroup`: `id`, `plan_id`, stable `draft_key`, `name`,
  `kind`, `camera`, `parent_draft_key`, optional `existing_group_id`,
  participant IDs, `reason`, and `confidence`.
- `OrganizationPlanItem`: `id`, `plan_id`, `shot_id`, `decision`
  (`proposed`, `accepted`, `rejected`), destination type, optional target
  group/draft key, `reason`, `confidence`, representative evidence, and an
  optional human override reason.

The unique key `(plan_id, shot_id)` makes it impossible for one draft to give
a shot two destinations. A later draft supersedes an earlier un-applied draft;
applied plans remain immutable audit records.

## MCP contract

The MCP server adds a planning surface. Its instructions explicitly say that
agents may draft and revise plans but must not approve or apply one.

- `organization_workspace(project, asset_id)` returns the current groups,
  ungrouped/intentional/B-roll states, subjects, per-shot face assignments,
  existing references, and an agent-readable validation summary.
- `get_shot_contact_sheet(project, shot_ids, columns=5)` returns a labelled
  image for visual comparison. It complements, rather than replaces,
  `get_shot_still` and `get_shot_frame`.
- `create_organization_plan(project, asset_id, groups, items, summary,
  author="agent")` stores a draft after structural validation.
- `get_organization_plan` and `list_organization_plans` expose the complete
  draft, evidence, human overrides, and validation report.
- `update_organization_plan_item` and `update_organization_plan_group` let an
  agent incorporate a human's requested edits while the plan remains a draft.
- `validate_organization_plan` returns blocking errors and warnings without
  changing anything.
- `approve_organization_plan` and `apply_organization_plan` are human-only
  operations in UI copy and MCP instructions. Apply requires `state=approved`
  and re-runs validation inside the transaction.

The contract validates a reference separately from organization: future
reference proposals must identify an exact participant and must use a shot
inside the exact setup/variant scope that contains that participant's face.
Invalid proposals fail immediately instead of surviving until matching.

## Validation

Blocking errors:

- a shot is missing from a plan that claims complete coverage;
- an item names an unknown shot, group, or subject, or crosses assets;
- duplicate or contradictory destinations exist for one shot;
- a planned variant has no setup parent;
- a target existing group is not a setup, variant, or canonical B-roll group;
- a planned group has an invalid kind or a duplicate draft key;
- a human-approved reference points outside its exact group/participant
  scope.

Warnings do not prevent an apply:

- a setup has one shot or one participant, so it should receive QC/manual
  finishing rather than matching;
- a setup has no participant faces;
- a group becomes empty;
- a setup mixes clearly different lighting/wardrobe/background evidence;
- accepted B-roll contains assigned faces;
- unresolved shots remain after the plan.

The post-apply report repeats warnings and lists every changed shot and created
group. It never silently deletes an empty setup; the human chooses whether to
remove it.

## Review UI

Replace the first-run “Needs organization” experience with an **Organization
draft** view when an active draft exists. It has four visible sections:

1. **Proposed setup families**: setup cards with nested variant lanes,
   representative thumbnails, participants, shot count, confidence, and
   reasons.
2. **B-roll**: a visible, reversible bin with thumbnails and reasons.
3. **Needs a decision**: unresolved or rejected cards with visual evidence and
   explicit destination controls.
4. **Validation and apply**: clear blocking errors, warnings, plan coverage,
   and one disabled-until-valid apply button.

Edits stay local in the draft and batch-save; selecting a destination must not
reload the page or reset unrelated selections. The normal setup workspace
continues to display only applied editorial structure.

## Current-project protection

The existing documentary clip is already organized. No migration or first-run
agent should reinterpret it. Any new plan for that asset starts from its
current setup/variant/B-roll/intentional structure and defaults to a
non-destructive audit draft. The user must explicitly request a reorganization
before an agent proposes changes to locked assignments.

## Testing and acceptance

- Unit tests cover plan schema validation, variant-parent rules, one-decision
  per shot, B-roll semantics, reference-scope validation, and idempotent
  post-apply reporting.
- MCP tests cover a vision-agent sequence: workspace → contact sheet → draft
  → validation; and confirm MCP draft calls cannot change shots.
- API/UI tests cover visual draft sections, durable human overrides, no reload
  on staged edits, and apply failure rollback.
- Migration tests prove existing project databases open with no changed shot
  assignments.
- An end-to-end fixture verifies that one subject may appear in two visual
  interview families, a shared two-person setup remains one setup with two
  participants, and a lighting change becomes sibling variants rather than a
  whole-frame match target.

## Out of scope

- Automatic approval or application of an agent plan.
- Color corrections, LUTs, OCIO, or log-media support.
- Face-masked skin correction rendering.
- Automatic camera-angle inference.
- Replacing human editorial judgment with a classifier.
