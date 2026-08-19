# AI Finishing / Color QC System — Full Build Brief

You are the primary implementation engineer for a new local-first professional video finishing application.

This is **not merely an MVP exercise**. Build the architecture for the full envisioned system, while implementing it in sensible vertical slices so that useful functionality appears early and later features do not require a rewrite.

Other agents — primarily OpenAI Codex and Claude Code — will join the repository later today to review architecture, test behavior, critique design decisions, fix issues, and implement additional components. You should therefore optimize not only for shipping code, but for **clarity, modularity, testability, documentation, and easy multi-agent handoff**.

Do not optimize for conserving tokens. Investigate deeply, implement aggressively, test thoroughly, and document decisions.

---

# 1. PRODUCT VISION

Build a local-first **AI finishing and quality-control assistant for professionally finished video**.

The target workflow is:

A filmmaker/colorist exports a nearly finished Rec.709 master — potentially a 60–120 minute ProRes feature film weighing hundreds of gigabytes.

The application analyzes the film shot-by-shot and identifies visual inconsistencies or defects such as:

* inconsistent skin tone between cameras
* incorrect skin hue, saturation, or luminance
* white-balance discrepancies
* shot-to-shot exposure mismatches
* inconsistent camera matching
* crushed blacks
* clipped highlights
* excessive saturation
* unwanted tint
* mixed-lighting problems
* anomalous softness
* motion blur anomalies
* stabilization artifacts
* Gyroflow-style post-stabilization blur pulses
* rolling-shutter / fast-camera-motion artifacts
* flicker
* individual damaged frames
* other visual anomalies detectable from temporal or perceptual analysis

The system must then propose **deterministic, temporally stable corrections**.

The system must NOT generatively repaint normal footage merely because a model thinks it can make it prettier.

The philosophy is:

**Analyze intelligently. Correct deterministically whenever possible. Use generative reconstruction only for genuinely damaged temporal intervals where deterministic correction cannot recover the missing image.**

Human approval is central.

The tool should behave like an obsessive AI finishing assistant, not an autonomous “make movie good” button.

---

# 2. CORE PRODUCT PRINCIPLES

The application must follow these principles throughout its architecture.

## Local-first

Large media stays local.

A 500 GB ProRes master should not require uploading to a cloud service.

Video decoding, shot detection, frame extraction, image statistics, segmentation, rendering, and ideally restoration inference should run locally.

LLM/agent reasoning may be driven externally through tools such as Codex CLI or Claude Code, but the application itself must expose local deterministic APIs/CLI/MCP interfaces.

## Non-destructive

Never overwrite source media.

Every analysis and correction must be represented in an explicit project/database/sidecar format.

Users must be able to reproduce, modify, disable, reject, or remove any correction.

## Deterministic grading

Ordinary grading corrections should be represented through deterministic operations such as:

* exposure
* offset
* RGB balance
* temperature/tint style chromatic adaptation
* contrast/pivot
* saturation
* hue rotation
* hue-vs-hue
* hue-vs-saturation
* luma-vs-saturation
* shadow/midtone/highlight balance
* ASC CDL
* 1D curves
* 3D LUTs
* OpenColorIO transforms
* qualified skin corrections
* masks
* temporally propagated masks

Do not use generative image synthesis for routine grading.

## Temporal stability

A correction approved for a shot must produce the same mathematical transformation on every relevant frame unless intentionally keyframed.

Avoid temporal flicker caused by per-frame model decisions.

## Human approval

The system should detect, measure, propose, and preview.

The human decides whether to apply.

Support actions conceptually equivalent to:

Approve
Reject
Adjust
Apply to shot
Apply to source/camera family
Apply to matching shots
Mark as intentional exception

## Explainability

Every recommendation should say:

what the system detected
why it thinks it is inconsistent
what reference it compared against
what correction it proposes
how confident it is
whether the operation is global or selective

## Source/family awareness

The system should understand that a feature is not merely 130,000 unrelated frames.

It should discover and/or allow the user to identify relationships such as:

same interview
same subject
same camera
same source roll
same lighting setup
same scene
same visual treatment
intentional exceptions

This is critical.

---

# 3. EXISTING OPEN-SOURCE PROJECTS TO AUDIT BEFORE REIMPLEMENTING ANYTHING

Before writing replacements for solved problems, inspect and evaluate existing projects.

At minimum, investigate:

`isaacrowntree/color-grade-ai`

This is particularly relevant. It reportedly analyzes Rec.709 material, proposes corrections, supports reference matching, processes clips, generates `.cube` LUTs, and has before/after preview functionality.

Determine whether to:

use directly as a dependency
fork/extend
borrow architecture/code
or replace selected components

Document the decision.

Also investigate:

`kijai/ComfyUI-VideoColorGrading`

This wraps research around reference-driven video color grading / LUT generation. Determine whether its models or algorithms are appropriate for generating candidate reference-match transforms.

Investigate:

`breakthrough/PySceneDetect`

Likely use this rather than reinventing shot-boundary detection.

Investigate:

`facebookresearch/sam2`

Potentially useful for temporal subject/skin/region segmentation and mask propagation.

Investigate:

`hahnec/color-matcher`

Potential baseline for deterministic statistical color transfer.

Investigate:

Academy Software Foundation `OpenColorIO`

Use professional color transform infrastructure where appropriate instead of inventing foundational color-management math.

Investigate existing Resolve MCP projects, especially the current Resolve MCP implementation by Samuel Gursky and other actively maintained Resolve MCP repositories.

Do not assume any of these projects are production-ready merely because they exist.

Audit:

license
maintenance status
test coverage
API design
performance
Apple Silicon compatibility
Python/package health
GPU dependencies
whether components can run headlessly
whether integrating them would create architectural lock-in

Write the results into:

`docs/research/open-source-audit.md`

---

# 4. PRIMARY INPUT

The core application must accept a baked video master such as:

ProRes 422 LT
ProRes 422
ProRes 422 HQ
ProRes 4444
DNxHR
other professional mezzanine codecs

Assume Rec.709 initially.

Do not hard-wire the entire architecture to Rec.709, however.

The project model should be future-compatible with:

Rec.709 Gamma 2.4
Rec.709-A handling
Display P3
HDR / PQ
HLG
ACES / scene-referred workflows
camera originals

The first production target is display-referred Rec.709 finishing.

---

# 5. INGEST PIPELINE

Create a robust media ingest layer.

It should inspect the master using ffprobe or equivalent and persist:

file path
codec
container
resolution
frame rate
duration
timecode where available
audio configuration
color primaries
transfer function
matrix coefficients
bit depth
pixel format
metadata
file hash or robust identity

Never repeatedly scan a massive master unnecessarily.

Create a project database/cache so analysis is resumable.

If a two-hour master has already been analyzed, restarting the app should not repeat hours of extraction.

---

# 6. SHOT DETECTION

Implement automatic shot-boundary detection.

Start with PySceneDetect or another proven detector.

Support:

hard cuts
fades
dissolves where reasonably detectable
manual boundary correction
merged/split shots

Store each shot as a first-class entity.

Each shot should have:

shot ID
timeline start timecode
timeline end timecode
start/end frame
duration
representative frame references
analysis state
family assignments
correction state
review state

The application must allow re-running detection without destroying manually approved shot metadata.

---

# 7. REPRESENTATIVE FRAME EXTRACTION

Do not send every frame through expensive AI analysis.

For every shot, extract representative samples.

Initial sampling strategy can include:

early frame
25%
50%
75%
late frame

Avoid frames too close to transitions.

For long shots, support adaptive sampling.

If significant temporal change is detected, increase sample density.

Examples:

daylight changes during interview
person enters/leaves frame
lighting shifts
camera changes exposure
flash/strobe events

Use FFmpeg efficiently.

Preserve original frame numbers/timecodes.

---

# 8. IMAGE METRICS ENGINE

Create a deterministic metrics pipeline independent of any LLM.

For each sampled frame and relevant ROI compute useful measurements such as:

luma statistics
RGB channel means/medians
histograms
percentiles
black point
white point
highlight clipping
shadow clipping
chroma
saturation
color cast
estimated white balance
local contrast
global contrast
sharpness
blur score
directional blur score
noise estimate
flicker-related temporal metrics
optical flow statistics
frame-to-frame residuals

Use appropriate perceptual/color spaces where useful:

Lab
OKLab / OKLCH
ICtCp where applicable
linear RGB
scene/display-referred representations

Document all math.

Do not rely solely on RGB averages.

---

# 9. FACE / PERSON / SKIN ANALYSIS

This is one of the most important product subsystems.

Detect people/faces in representative frames.

Track identity across shots where possible.

Allow human correction of identity associations.

For faces, create robust skin sampling regions.

Avoid contaminating measurements with:

eyes
eyebrows
hair
lips where appropriate
teeth
background
clothing

Use segmentation rather than crude rectangular face-box averaging.

Potentially use SAM2 or another segmentation model.

Calculate skin statistics including:

median hue
median chroma
median luminance
distribution
highlight region
shadow region
estimated color cast
distance from surrounding shots/reference

Do NOT enforce a universal “correct skin tone.”

Skin tone varies by person, ethnicity, lighting, creative intent, and scene.

The preferred reference should usually be:

the same person
in the same interview/setup
on an approved camera/reference shot

The application must model **consistency relative to an approved reference**, not normalize all humans to some global target.

---

# 10. SHOT FAMILY / INTERVIEW CLUSTERING

Build a system that can group shots into families.

Possible features:

visual embeddings
face identity
background similarity
camera metadata if available
temporal proximity
framing similarity
color statistics
manual tags
filename/source metadata when using a Resolve project

Conceptual families may include:

Dad Hero Interview
Camera A
Camera B
Camera C
Morning Roll
Afternoon Roll
Open Mic
Studio Performance
Archive
Phone inserts

Allow hierarchical grouping.

For example:

Dad Hero Interview
→ Camera A
→ Roll 1

Dad Hero Interview
→ Camera A
→ Roll 2

Dad Hero Interview
→ Camera B
→ Roll 1

The system should allow one approved reference for:

an entire family
one camera
one source roll
one individual shot

---

# 11. COLOR CONSISTENCY QC

Create a QC engine that identifies probable outliers.

The system should answer questions such as:

Which shots of this interview have skin hue inconsistent with the reference?

Which camera is systematically 0.3 stops darker?

Which source roll becomes increasingly warm?

Which shot has significantly higher saturation?

Which shots have unusually clipped skin highlights?

Which camera has greener neutrals?

Which B-roll insert does not visually belong between adjacent shots?

The system should rank issues by severity and confidence.

Do not flag tiny differences as problems merely because numbers differ.

Provide configurable tolerances.

Support “intentional look” exemptions.

---

# 12. AI / VISION REASONING LAYER

Create an abstraction so different reasoning agents/models can be used.

Potential clients may include:

Codex CLI
Claude Code
DeepSeek API
OpenAI APIs
Anthropic APIs
local vision-language models
future models

Do NOT tightly couple the core application to one API provider.

The application should expose structured analysis data and representative visual material to an agent.

The reasoning layer may answer questions such as:

Does this mismatch look intentional?

Which of these four cameras is the best reference?

Does this face appear too green, or is the warm environment intentionally affecting skin?

Does the numeric outlier correspond to a visually meaningful problem?

What correction strategy is appropriate?

The model should output structured recommendations rather than prose only.

Define JSON schemas.

---

# 13. CORRECTION REPRESENTATION

Design a formal correction graph/schema.

A correction should be capable of representing at least:

global shot correction
family-level correction
skin-only correction
masked correction
time-varying correction
approved exception

Operations should be composable.

Example conceptual graph:

Source
→ global exposure/WB correction
→ camera match
→ skin correction
→ highlight recovery
→ creative transform
→ output

Represent corrections in a portable deterministic form.

Candidate representations include:

ASC CDL
OpenColorIO transforms
1D curves
3D LUTs
custom matrix operations
HSL curves
mask + transform pairs

Do not reduce every correction to a LUT if a LUT is the wrong representation.

---

# 14. REFERENCE MATCHING

Implement multiple candidate matching algorithms.

Do not trust one method.

Potential candidate generators:

statistical color matching
ASC-CDL optimization
matrix/chromatic adaptation
curve optimization
3D LUT generation
ML reference-conditioned LUT generation
skin-constrained optimization

For each mismatch, allow the system to generate one or more candidate corrections.

The review UI should show:

Original
Candidate
Reference

Ideally as:

still frames
wipe
side-by-side
short synchronized loops

---

# 15. SKIN-SELECTIVE CORRECTION

This should eventually be a first-class correction mode.

Pipeline concept:

detect person
segment face/skin
propagate stable mask temporally
soften/regularize mask
apply limited deterministic correction inside mask

Corrections may adjust:

hue
saturation
luma
tint
highlight balance

Avoid uncanny plastic skin.

Preserve natural variation.

Masks must not shimmer frame-to-frame.

Mask temporal coherence matters more than single-frame perfection.

---

# 16. TEMPORAL STABILIZATION ARTIFACT / MOTION-BLUR QC

Create a separate visual defect detection subsystem.

Goal:

Detect short intervals where stabilizers such as Gyroflow have successfully stabilized geometry but the original violent camera movement created baked motion blur / rolling-shutter deformation that appears as a pulse, smear, wobble, or “flicker.”

These artifacts may last only:

1 frame
3 frames
8 frames
10–15 frames

Possible detection signals:

sudden sharpness drop
directional blur increase
optical-flow residual anomaly
deformation inconsistency
temporal feature mismatch
localized warping
difference from neighboring stable frames

The QC engine should produce intervals such as:

Shot 184
TC 00:37:14:08–00:37:14:16
9 suspicious frames
Probable stabilization/motion-blur artifact
Confidence 0.92

Human can:

Ignore
Mark intentional
Repair
Preview proposed repair

---

# 17. FRAME / SHORT-INTERVAL RESTORATION

This is where generative methods are allowed.

Do NOT generatively alter an entire normal shot.

Use generative/video restoration only for explicitly detected and approved damaged intervals.

Preferred hierarchy:

First attempt deterministic or restoration-based deblur.

If recoverable information exists, use video deblurring / neighboring-frame reconstruction.

For severely corrupted frames, use temporally conditioned generative reconstruction.

Use good frames before and after the damaged interval as constraints.

Example:

frames 176–183 good
frames 184–191 damaged
frames 192–199 good

Only replace 184–191.

The repaired interval must match:

geometry
subject identity
texture
grain/noise character
exposure
color
motion trajectory
preceding frames
following frames

No identity drift.

No invented objects.

No temporal shimmer.

No altered facial features.

Research open-source video restoration, deblurring, interpolation, diffusion, and temporally conditioned reconstruction models that can run locally or through a local inference server.

Prefer Apple-Silicon-capable options where realistic.

Also support remote GPU inference later.

Abstract the restoration engine.

---

# 18. GYROFLOW AWARENESS

Investigate whether Gyroflow project/gyro metadata can improve restoration.

Gyroflow knows camera motion.

For footage with available gyro data, determine whether the restoration system can use:

camera rotational trajectory
exposure duration
rolling-shutter timing
lens model
stabilization transform

as priors for blur estimation/deconvolution.

This may eventually become a meaningful differentiator.

Do not make the first implementation dependent on Gyroflow internals, but research it and create an integration design.

Write findings to:

`docs/research/gyroflow-restoration.md`

---

# 19. FLICKER / DEFLICKER QC

Analyze temporal luminance and chroma variations.

Detect probable:

fluorescent flicker
LED refresh flicker
exposure pumping
grade/cache anomalies
single-frame flashes

Differentiate intentional lighting changes from periodic flicker where possible.

Support proposed repair via deterministic temporal filtering or existing deflicker algorithms.

---

# 20. REVIEW APPLICATION

Build a useful local review UI.

Do not wait until the end to build UI.

A local web application is acceptable initially.

The review UI should ultimately support:

timeline-like issue navigation
filter by interview
filter by camera
filter by issue type
filter by severity
filter by confidence
filter by approval state

For each issue display:

timecode
shot ID
family
camera/source if known
issue description
metric differences
reference frame
original
proposed correction
difference/wipe if useful

Actions:

Approve
Reject
Adjust
Apply to family
Apply only here
Mark intentional
Defer

For temporal defects, provide looping playback around the issue.

---

# 21. PROJECT DATABASE

Use a robust local project representation.

SQLite is a reasonable default unless a better architecture emerges.

Persist:

media
shots
frames
metrics
faces
identities
families
references
issues
proposed corrections
approved corrections
masks
restoration intervals
render outputs
model/version metadata
analysis version
user decisions

Migrations must be supported.

Do not store giant extracted frames directly in SQLite.

Use cache directories with stable IDs.

---

# 22. RENDERING

Implement a deterministic render pipeline.

Initially, FFmpeg + OpenColorIO / GPU transforms may be appropriate.

Requirements:

preserve original timeline frame count
preserve frame rate
preserve audio sync
preserve timecode where possible
support ProRes output on macOS
support high-quality chroma handling
avoid accidental range shifts
avoid Gamma 2.4 / Rec.709-A surprises
preserve audio bit depth/layout when possible

Allow rendering:

single corrected shot
issue preview
selected range
entire master

For unchanged frames, investigate whether stream-copy / segment reuse or other strategies can reduce unnecessary re-encoding, but correctness is more important than premature optimization.

---

# 23. OUTPUTS

The system should be able to produce:

corrected master video
QC report
JSON sidecar
CSV issue report
correction manifest
LUT/CDL exports where applicable
preview clips
contact sheets
before/after stills

Eventually:

Resolve markers
Resolve grade artifacts
DRX/CDL/LUT/DCTL outputs
Resolve MCP application of approved corrections

---

# 24. DAVINCI RESOLVE INTEGRATION

Build this as a standalone system first, but design an integration layer.

Use existing Resolve MCP/scripting projects rather than recreating hundreds of Resolve API wrappers.

Potential Resolve mode:

analyze current Resolve timeline
extract display-referred frames
understand groups/remotes/local grades
identify QC problems
drop timeline markers
propose changes
apply only safe/exposed operations
export corrective LUT/CDL/DRX assets
allow human to perform unsupported fine-grained changes

The system must work even when Resolve cannot expose every Color-page parameter.

Standalone baked-master mode remains essential.

---

# 25. AGENT / MCP INTERFACE

Expose the application so Codex and Claude can control it.

Provide:

CLI
structured JSON output
local HTTP API if useful
MCP server if appropriate

Example conceptual commands:

analyze-project
detect-shots
analyze-shot
analyze-interview
find-color-outliers
find-skin-outliers
find-stabilization-artifacts
generate-correction
render-preview
approve-correction
reject-correction
render-master
export-report

Agent operations must support read-only mode.

Default agent mode should NEVER modify/render-overwrite anything without explicit action.

---

# 26. PERFORMANCE / APPLE SILICON

Primary development system:

Apple Silicon Mac
M4 Pro
48 GB unified memory

Target machines may include:

M1 Max
M2
M3
M4
future Apple Silicon

Investigate:

Metal acceleration
MLX
Core ML
PyTorch MPS
FFmpeg VideoToolbox
OpenColorIO GPU processing
memory-mapped workflows
bounded queues

A 500 GB master must not require being loaded into RAM.

Process incrementally.

Avoid memory leaks.

Memory/resource discipline is a product requirement.

---

# 27. CACHE ARCHITECTURE

Everything expensive should be cached by content/version identity.

Examples:

shot boundaries
extracted frames
embeddings
segmentation masks
metrics
face identity
candidate corrections
preview renders

Cache keys should incorporate:

source identity
frame range
algorithm version
model version
parameters

Changing one grading algorithm should not invalidate scene detection.

---

# 28. TESTING

Write real tests.

Include:

unit tests for color math
shot-boundary tests
frame/timecode conversion tests
render round-trip tests
audio-sync tests
mask stability tests where possible
database migration tests
correction serialization tests
CLI tests
sample-project integration tests

Create synthetic test video when practical.

Examples:

two identical shots with +0.5 stop exposure difference
known WB shift
known saturation change
generated skin-tone patch changes
single-frame blur anomaly
multi-frame flicker
known cut locations

The system should detect these reliably.

---

# 29. QUALITY SAFEGUARDS

Never silently modify source files.

Never generatively repair a frame without recording that fact.

Every generated/reconstructed frame must be traceable.

Store:

model
model version
seed where relevant
inputs
reference frames
parameters
output path
approved/rejected state

Provide a visual indicator for synthetic/restored intervals.

Professional finishing requires provenance.

---

# 30. REPOSITORY STRUCTURE

Design a clean monorepo.

A reasonable conceptual structure might be:

`apps/review-ui`

`packages/core`

`packages/media`

`packages/scene-detection`

`packages/color-analysis`

`packages/skin-analysis`

`packages/corrections`

`packages/restoration`

`packages/render`

`packages/resolve`

`packages/agent`

`packages/mcp`

`docs`

`tests`

Do not blindly use this exact structure if investigation suggests something better.

Document architecture in:

`docs/architecture.md`

---

# 31. MULTI-AGENT HANDOFF

Codex and Claude will join development later.

Create:

`AGENTS.md`

It must describe:

product mission
architecture
repository map
how to run
how to test
current implementation status
known problems
important design constraints
areas safe for parallel work
areas requiring coordination
open questions
decisions already made

Also maintain:

`docs/status.md`

with:

completed
in progress
next
blocked
needs review

Do not put critical design knowledge only in chat/session memory.

Commit it to documentation.

---

# 32. DEVELOPMENT STRATEGY

Do NOT attempt to write the entire application in one giant untested pass.

Build vertical functionality continuously.

However, do NOT intentionally architect a throwaway MVP.

The architecture should anticipate the complete system from day one.

Suggested progression:

Media ingest and project database.

Shot detection and representative frame extraction.

Deterministic metrics.

Basic local review UI.

Face/skin analysis.

Shot-family/reference system.

Color-outlier detection.

Candidate deterministic corrections.

Before/after approval workflow.

Preview rendering.

Full-master rendering.

Resolve bridge.

Temporal artifact detection.

Restoration system.

Agent/MCP controls.

Polish/performance.

You may reorder based on technical findings.

---

# 33. FIRST REAL TEST CASE

The first real-world target is a finished documentary.

Approximately feature length.

4K UHD.

23.976 fps.

Rec.709 master.

Contains:

multiple Sony cameras
FX6
FX3
a7S-series
possibly Burano
S-Log3 material transformed to Rec.709
S-Cinetone material
multiple interviews
mixed lighting
camera mismatch
some recovered/corrupted footage that was manually repaired
skin-tone inconsistencies
Gyroflow-stabilized shots
some stabilization artifact intervals
noise reduction
deflicker
archive/B-roll
stylized finale

The application should be useful for finding the last 5–10% of inconsistencies in a film that is already mostly professionally graded.

Do not assume the entire movie should have one look.

Consistency is contextual.

---

# 34. EXAMPLE DESIRED COLOR-QC BEHAVIOR

Suppose four cameras shot the same person.

Three match.

One is warmer/more magenta.

The system should discover that the fourth is the outlier.

It should not average all four cameras and move the correct three toward the wrong one.

It should reason that three agreeing cameras create evidence of a reference cluster.

It should propose something like:

Camera D / Roll 2

Skin hue: +4.2° toward magenta relative to reference cluster
Skin chroma: +7%
Face luminance: +0.12 stop
Overall WB: slightly warmer

Suggested correction:

skin hue −3.5°
skin saturation −4%
global warmth −small amount

Confidence: 0.91

Reference:

Camera A / Roll 2
Camera B / Roll 2
Camera C / Roll 2

Then preview.

The human approves or rejects.

---

# 35. EXAMPLE DESIRED STABILIZATION-QC BEHAVIOR

Suppose a 12-second shot has Gyroflow stabilization.

The shot is generally stable.

At 00:38:12:14 the original camera jerked violently.

Frames 14–21 contain strong directional motion blur and deformation.

Gyroflow stabilized the geometry but the blur creates an 8-frame visual pulse.

The system should flag:

Probable stabilization artifact
8 frames
high directional blur anomaly
neighboring frames sharp
geometry otherwise stable

Then offer:

Ignore
Inspect
Attempt restoration

If restoration is requested:

use good frames before and after
attempt temporal deblur/reconstruction
render only damaged interval
show loop:

Original / Repaired

User approves before insertion.

---

# 36. DO NOT DO THESE THINGS

Do not build an “AI filter.”

Do not generatively repaint the whole movie.

Do not normalize every human to one skin color.

Do not assume statistical difference means visual error.

Do not discard filmic intent.

Do not overwrite originals.

Do not depend entirely on one cloud API.

Do not build proprietary replacements for FFmpeg, PySceneDetect, OpenColorIO, etc. without strong reason.

Do not hardcode the project around this one documentary.

Do not use per-frame independent generative grading that creates flicker.

Do not make hidden corrections.

Do not require Resolve for standalone operation.

Do not implement the UI before the underlying project/correction model is coherent, but do create a usable review UI early enough to test actual workflows.

---

# 37. DELIVERABLES FOR YOUR FIRST MAJOR PASS

Work continuously until you have produced a substantive foundation.

At minimum I want:

A functioning repository.

An open-source dependency/reuse audit.

Architecture documentation.

A persistent project model/database.

Professional media ingest.

Shot detection.

Representative frame extraction.

Initial image metrics.

A basic local review UI showing detected shots and representative frames.

Initial face/skin detection experiments.

A documented correction schema.

A working deterministic preview-transform pipeline.

A CLI.

Tests.

AGENTS.md.

Status documentation.

Research notes on:

Resolve integration
reference color matching
skin segmentation
video restoration
Gyroflow metadata potential
Apple Silicon inference options

If feasible in the available run, proceed further into:

color-outlier detection
reference-shot matching
candidate correction generation
approval workflow
preview rendering

Do not stop merely because the minimum list is complete if you have time and tokens available.

---

# 38. HOW TO WORK

Start by inspecting the environment and creating a development plan.

Then investigate existing repositories.

Do not ask me basic implementation questions you can resolve through research or reasonable engineering judgment.

If a decision is reversible, choose a reasonable default and document it.

If a decision has major architectural consequences, document alternatives and reasoning.

Use commits at meaningful milestones.

Do not commit secrets, API keys, generated media, giant models, or caches.

Use `.gitignore` appropriately.

Keep dependencies pinned/reproducible.

Prefer boring, maintainable solutions.

Optimize for a codebase another strong engineer or coding agent can understand immediately.

---

# 39. FINAL PRODUCT MENTAL MODEL

This should eventually feel like:

**Resolve's professional finishing mentality**

plus

**an AI assistant that can watch every shot without fatigue**

plus

**automated visual QC**

plus

**deterministic professional color correction**

plus

**selective modern generative restoration for genuinely damaged frames**

with the filmmaker always retaining control.

The product is not trying to replace the colorist.

It is trying to give the colorist a tireless second set of eyes, excellent measurement tools, automated propagation, and repair capabilities that do not currently exist in one coherent workflow.

Build toward that full vision.
