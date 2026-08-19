# Research notes

Working notes on integration and analysis topics. Items marked
**to-verify** need hands-on validation before being treated as fact; the rest
reflect the current plan and known ecosystem state. Update these as
experiments land.

## 1. Resolve integration

- DaVinci Resolve reads/writes **ASC CDL** (slope/offset/power) via the
  "ColorTrace"/CDL round-trip and per-clip CDL nodes. Our `Correction` schema
  already models `cdl` first-class, so a Resolve interchange can map 1:1.
- Resolve also exports/imports **1D and 3D LUTs** (`.cube`, `.dctl`) and
  **EDL/XML/ALE** for shot lists. Shot boundaries + timecode are the natural
  interchange points; our shot model already stores both frame numbers and
  SMPTE timecode.
- `.cube` LUTs are now first-class (`lut` correction kind, `lutcube.py`). They
  are applied in the **linear BT.709 working space** and clamped to the file's
  `DOMAIN_MIN`/`DOMAIN_MAX`; a Resolve LUT authored in log or display space
  must be converted to the working space first (to-verify: whether to add a
  `space` tag for automatic display↔linear remapping).
- Scripting options: the **DaVinci Resolve scripting API** (Python/Lua, local
  Studio license), or file-based interchange (LUT/CDL/EDL) that requires no
  Resolve license at all. **Plan:** prefer file-based interchange for the
  standalone path, use the scripting API opportunistically where available.
- **to-verify**: exact CDL node semantics (ascending vs descending order,
  power clamp behavior) and ALE timecode formatting for DF vs NDF.

## 2. Reference color matching

Goal: match shot-to-shot appearance against a reference (another shot or a
reference still), not just equalize statistics.

- A robust, deterministic baseline is **histogram/percentile matching in a
  perceptual space** (log or Rec.709), then expressing the result as an
  invertible, temporally stable grade (CDL / gain / offset) rather than a
  per-pixel LUT that flickers.
- **Color transfer** (Reinhard et al.) in a decorrelated color space is a
  common starting point but must be converted to a *single global transform*
  per shot to preserve temporal stability.
- **OpenColorIO** is the right substrate for managed color transforms and
  color-space-consistent matching; add it when the transform pipeline needs
  managed color, not before.
- **to-verify**: whether shot-to-shot exposure matching should use luma
  percentiles (robust to outliers) vs. mean, and how to keep skin hue stable
  while matching neutral balance.

## 3. Skin segmentation

- Detection is real and local: OpenCV **YuNet** (bundled ONNX) finds face
  boxes; optional **MediaPipe FaceMesh** (``colorai[face]``) adds 468
  landmarks so skin is sampled precisely from the forehead/cheeks while
  avoiding eyes/lips/hair. Both are wired in `face.py` behind one interface.
- The committed `skin.py` remains a **color-only YCrCb heuristic** used to
  select skin *pixels* within a located face region — it is not itself a
  face detector.
- A stronger detector (InsightFace) can be swapped in later behind the same
  narrow interface if the workflow demands it.
- Critical product guardrail: never "normalize every human to one skin color";
  skin metrics describe deviation, they do not prescribe a target tone.

## 4. Video restoration

- Reserved for **genuinely damaged temporal intervals** only, never routine
  grading. Two tiers:
  - **Deterministic** (implemented in `restoration.py`): nearest-good-frame,
    cross-dissolve blend, temporal median for flicker/dead pixels. No model,
    fully predictable.
  - **Generative** second, only where deterministic recovery cannot restore
    the missing image. **Model selection (decided):** **RIFE** for temporal
    frame interpolation and **LaMa** for spatial inpainting, both ONNX and run
    locally via ONNX Runtime on Apple Silicon (section 6). Loader reads
    `COLORAI_GENERATIVE_MODEL_DIR`; the interface stays explicit and
    approval-gated (Original/Repaired loop) until models are installed.
- Flicker is often removable deterministically (temporal low-pass on a
  per-frame exposure estimate); the brief's stabilization-artifact case
  (directional blur pulse) is the harder, generative-tier case.

## 5. Gyroflow metadata potential

- Gyroflow writes stabilization data and can export trajectories; its project
  files describe frame-by-frame transforms. If a master was stabilized by
  Gyroflow, that metadata could flag the *intervals* where the original
  camera motion was violent — exactly where post-stabilization blur pulses
  appear (see the brief's section 35 example).
- **Plan:** treat Gyroflow metadata as an optional hint channel for
  *where to look* for stabilization artifacts, then confirm with actual blur
  measurement on those intervals rather than trusting the metadata alone.
- **to-verify**: Gyroflow project schema/version stability and whether the
  relevant motion/rotation fields are machine-readable per frame.

## 6. Apple Silicon inference options

- On M-series, local inference targets CPU + GPU (Metal) + ANE. Practically:
  - **Core ML** (ANE) is lowest power but the most friction to convert models
    and keep reproducible.
  - **ONNX Runtime / PyTorch with MPS** is the most portable local path and is
    the recommended default for the modest models we'd need (face detection,
    small segmentation, inpainting).
  - **llama.cpp / mlx** if any LLM reasoning runs locally; otherwise LLM/agent
    reasoning is expected to be *external* (Codex CLI / Claude Code) per the
    product brief, with ColorAI exposing local deterministic APIs.
- **Decision:** standardize on ONNX Runtime for bundled inference models; keep
  everything behind an interface so Core ML can be added later without
  rewiring callers.
