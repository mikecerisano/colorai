"""Temporal quality-control measurements.

Extends the deterministic engine beyond a single representative still into
temporal signals, all measurements (not decisions):

* :func:`flicker_intervals` — frame-to-frame luma oscillation (a flicker
  signature), from a luma sequence.
* :func:`clip_flags` / :func:`shot_clip_report` — clipped highlights and
  crushed blacks from stored per-shot luma percentiles.
* :func:`detect_blank_frames` / :func:`detect_duplicate_frames` — damaged-frame
  signatures (decode-failed black/white frames and frozen duplicates).
* :func:`rolling_shutter_metric` / :func:`skew_intervals` — row-dependent
  inter-frame shear (a rolling-shutter / wobble signature), from consecutive
  frame pairs. Uniform global motion gives near-zero skew; band-dependent
  shear gives high skew. A measurement, not a defect: fast pans vibrate, and
  only a human judges whether the shear reads as an artifact.

These functions sample a contiguous window of frames (the first ``samples``
from ``start``) so flicker/duplicate/shear detection sees consecutive frames;
long-form scanning is a performance nicety.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from colorai.frames import extract_frame
from colorai.project.models import FrameMetrics, Shot
from colorai.project.store import ProjectStore

_LUMA = (0.2126, 0.7152, 0.0722)


def _luma_mean(image_bgr: np.ndarray) -> float:
    """Mean BT.709 luma of an HxWx3 BGR image, normalized to [0, 1]."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    luma = _LUMA[0] * rgb[..., 0] + _LUMA[1] * rgb[..., 1] + _LUMA[2] * rgb[..., 2]
    return float(luma.mean())


def flicker_intervals(
    values: list[float], *, threshold: float = 0.05, min_oscillations: int = 2
) -> list[tuple[int, int]]:
    """Find runs of frame-to-frame luma oscillation (flicker).

    ``values`` is a per-frame mean luma sequence. A run is a stretch of
    consecutive frames whose adjacent deltas all exceed ``threshold`` and
    *alternate sign*. Returns inclusive ``(start_index, end_index)`` runs.
    """
    n = len(values)
    if n < 3:
        return []
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    count = 0
    prev_sign = 0
    for i in range(1, n):
        delta = values[i] - values[i - 1]
        if abs(delta) >= threshold:
            sign = 1 if delta > 0 else -1
            if prev_sign != 0 and sign != prev_sign and start is not None:
                count += 1
            else:
                start, count = i - 1, 1
            prev_sign = sign
        else:
            if start is not None and count >= min_oscillations:
                intervals.append((start, i - 1))
            start, count, prev_sign = None, 0, 0
    if start is not None and count >= min_oscillations:
        intervals.append((start, n - 1))
    return intervals


def clip_flags(
    luma_p5: float, luma_p95: float, *, clip: float = 0.98, crush: float = 0.02
) -> dict[str, bool]:
    """Flag clipped highlights / crushed blacks from luma percentiles (0..1)."""
    return {"clipped": luma_p95 >= clip, "crushed": luma_p5 <= crush}


def shot_clip_report(
    store: ProjectStore, asset_id: int, *, clip: float = 0.98, crush: float = 0.02
) -> list[dict]:
    """Per-shot highlight/shadow report from stored ``FrameMetrics``.

    These are **measurements, not defects**: bright windows, practicals,
    speculars, dark furniture, wardrobe, and stylized contrast are normal in
    a nearly finished Rec.709 master. Each row carries an interpretation note
    so agents use the signal as evidence when comparing otherwise similar
    shots, not as an automatic "fix" trigger.
    """
    out: list[dict] = []
    with store.session() as session:
        shots = (
            session.query(Shot).filter_by(asset_id=asset_id).order_by(Shot.index).all()
        )
        for shot in shots:
            m = session.query(FrameMetrics).filter_by(shot_id=shot.id).first()
            if m is None or m.luma_p5 is None or m.luma_p95 is None:
                continue
            clipped = m.luma_p95 >= clip
            crushed = m.luma_p5 <= crush
            notes = []
            if clipped:
                notes.append(
                    "bright content near the ceiling (windows/practicals/speculars) — "
                    "a measurement, not a defect"
                )
            if crushed:
                notes.append(
                    "deep shadows near black — a measurement, not a defect"
                )
            out.append(
                {
                    "shot_id": shot.id,
                    "luma_p5": round(m.luma_p5, 4),
                    "luma_p95": round(m.luma_p95, 4),
                    "clipped": clipped,
                    "crushed": crushed,
                    "note": "; ".join(notes),
                }
            )
    return out


@dataclass(frozen=True)
class BlankFrame:
    frame_index: int
    kind: str  # "black" or "white"


def blank_frames_from_lumas(
    lumas: dict[int, float], *, black: float = 0.02, white: float = 0.98
) -> list[BlankFrame]:
    """Flag near-uniform black/white frames from ``frame -> mean luma``."""
    return [
        BlankFrame(i, "black" if v <= black else "white")
        for i, v in sorted(lumas.items())
        if v <= black or v >= white
    ]


def duplicate_intervals_from_hashes(
    frame_indices: list[int], hashes: list[str], *, min_run: int = 2
) -> list[tuple[int, int]]:
    """Find runs of identical consecutive frames (frozen/duplicate frames).

    ``hashes[i]`` is a per-frame content digest; ``frame_indices`` is the
    corresponding absolute frame number. Returns inclusive intervals.
    """
    if len(frame_indices) != len(hashes):
        raise ValueError("frame_indices and hashes must align")
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for i in range(1, len(frame_indices)):
        if hashes[i] == hashes[i - 1]:
            if start is None:
                start = frame_indices[i - 1]
        else:
            if start is not None and frame_indices[i - 1] - start + 1 >= min_run:
                intervals.append((start, frame_indices[i - 1]))
            start = None
    if start is not None and frame_indices[-1] - start + 1 >= min_run:
        intervals.append((start, frame_indices[-1]))
    return intervals


def _gray(image_bgr: np.ndarray) -> np.ndarray:
    """BT.709-ish grayscale of an HxWx3 BGR image, normalized to [0, 1]."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0


def band_shifts(
    prev_bgr: np.ndarray, curr_bgr: np.ndarray, *, bands: int = 8, max_lag: int = 8
) -> list[int]:
    """Estimate per-band horizontal motion between two frames, in pixels.

    Each of ``bands`` horizontal strips gets one integer shift (positive =
    motion right) from the normalized cross-correlation of its column-luma
    profile over ``[-max_lag, max_lag]``. Flat (textureless) bands carry no
    motion information and report 0. Both frames must be same-shaped HxWx3.
    """
    if (
        prev_bgr.ndim != 3
        or curr_bgr.ndim != 3
        or prev_bgr.shape != curr_bgr.shape
        or prev_bgr.shape[2] != 3
    ):
        raise ValueError("prev_bgr and curr_bgr must be same-shaped HxWx3 images")
    if bands < 2:
        raise ValueError("bands must be >= 2")
    if max_lag < 0:
        raise ValueError("max_lag must be >= 0")
    h, w = prev_bgr.shape[:2]
    if h < bands:
        raise ValueError("frame height must be >= bands")

    prev_g, curr_g = _gray(prev_bgr), _gray(curr_bgr)
    edges = sorted({round(h * i / bands) for i in range(bands + 1)})
    shifts: list[int] = []
    for top, bottom in zip(edges[:-1], edges[1:]):
        p = prev_g[top:bottom].mean(axis=0) - prev_g[top:bottom].mean()
        c = curr_g[top:bottom].mean(axis=0) - curr_g[top:bottom].mean()
        best_lag, best_corr = 0, -2.0
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                po, co = p[:lag], c[-lag:]
            elif lag > 0:
                po, co = p[lag:], c[:-lag]
            else:
                po, co = p, c
            denom = float(np.linalg.norm(po) * np.linalg.norm(co))
            corr = float(po @ co / denom) if denom > 1e-12 else 0.0
            if corr > best_corr + 1e-12 or (
                abs(corr - best_corr) <= 1e-12 and abs(lag) < abs(best_lag)
            ):
                best_lag, best_corr = lag, corr
        shifts.append(best_lag if best_corr > 0.0 else 0)
    return shifts


def rolling_shutter_metric(
    prev_bgr: np.ndarray, curr_bgr: np.ndarray, *, bands: int = 8, max_lag: int = 8
) -> dict:
    """Row-dependent shear between two consecutive frames.

    Returns ``band_shifts`` (per-band horizontal motion), ``mean_shift``
    (uniform/global component), ``skew`` (max − min band shift — the shear
    signature), and a ``note`` framing the numbers as measurement, not defect.
    """
    shifts = band_shifts(prev_bgr, curr_bgr, bands=bands, max_lag=max_lag)
    return {
        "band_shifts": shifts,
        "mean_shift": round(sum(shifts) / len(shifts), 3),
        "skew": max(shifts) - min(shifts),
        "note": (
            "row-dependent inter-frame shear — a measurement, not a defect; "
            "uniform pans give near-zero skew, band-dependent shear suggests "
            "rolling-shutter / wobble worth a human look"
        ),
    }


def skew_intervals(
    skews: list[float], *, skew_threshold: float = 1.5, min_run: int = 2
) -> list[tuple[int, int]]:
    """Find runs of consecutive frame pairs whose shear meets ``skew_threshold``.

    ``skews[i]`` is the skew between pair ``i`` and ``i + 1``; returns
    inclusive ``(start_pair, end_pair)`` runs of length >= ``min_run``.
    """
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for i, skew in enumerate(skews):
        if skew >= skew_threshold:
            if start is None:
                start = i
        else:
            if start is not None and i - start >= min_run:
                intervals.append((start, i - 1))
            start = None
    if start is not None and len(skews) - start >= min_run:
        intervals.append((start, len(skews) - 1))
    return intervals


def _sample_lumas(
    video_path: str | Path,
    start: int,
    end: int,
    fps: float,
    samples: int,
    scale: int | None,
) -> dict[int, float]:
    """Decode a contiguous window of up to ``samples`` frames and return luma."""
    probe_dir = Path(tempfile.mkdtemp(prefix="colorai_qc_"))
    try:
        lumas: dict[int, float] = {}
        last = min(end, start + samples - 1)
        for frame_index in range(start, last + 1):
            still = extract_frame(
                video_path, frame_index, probe_dir / f"{frame_index}.png",
                fps=fps, scale=scale,
            )
            image = cv2.imread(str(still), cv2.IMREAD_COLOR)
            if image is not None:
                lumas[frame_index] = _luma_mean(image)
        return lumas
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)


def detect_flicker(
    video_path: str | Path,
    start: int,
    end: int,
    fps: float,
    *,
    samples: int = 24,
    scale: int | None = 480,
    threshold: float = 0.05,
    min_oscillations: int = 2,
) -> list[tuple[int, int]]:
    """Detect flicker in a contiguous window of ``[start, end]`` (inclusive)."""
    lumas = _sample_lumas(video_path, start, end, fps, samples, scale)
    ordered = sorted(lumas)
    values = [lumas[i] for i in ordered]
    runs = flicker_intervals(values, threshold=threshold, min_oscillations=min_oscillations)
    return [(ordered[a], ordered[b]) for a, b in runs]


def detect_blank_frames(
    video_path: str | Path,
    start: int,
    end: int,
    fps: float,
    *,
    samples: int = 24,
    scale: int | None = 480,
) -> list[BlankFrame]:
    """Flag near-black / near-white frames in a contiguous window."""
    lumas = _sample_lumas(video_path, start, end, fps, samples, scale)
    return blank_frames_from_lumas(lumas)


def detect_rolling_shutter(
    video_path: str | Path,
    start: int,
    end: int,
    fps: float,
    *,
    samples: int = 12,
    scale: int | None = 480,
    bands: int = 8,
    max_lag: int = 8,
    skew_threshold: float = 1.5,
    min_run: int = 2,
) -> list[tuple[int, int]]:
    """Flag runs of consecutive pairs with high inter-frame shear.

    Decodes a contiguous window of up to ``samples`` frames, measures
    :func:`rolling_shutter_metric` on each consecutive pair, and returns
    inclusive ``(start_frame, end_frame)`` runs whose pair skews stay above
    ``skew_threshold`` (via :func:`skew_intervals`). A measurement, not a
    defect — fast pans legitimately shear.
    """
    probe_dir = Path(tempfile.mkdtemp(prefix="colorai_qc_rs_"))
    try:
        frames: dict[int, np.ndarray] = {}
        last = min(end, start + samples - 1)
        for frame_index in range(start, last + 1):
            still = extract_frame(
                video_path, frame_index, probe_dir / f"{frame_index}.png",
                fps=fps, scale=scale,
            )
            image = cv2.imread(str(still), cv2.IMREAD_COLOR)
            if image is not None:
                frames[frame_index] = image
        ordered = sorted(frames)
        skews = [
            rolling_shutter_metric(frames[a], frames[b], bands=bands, max_lag=max_lag)["skew"]
            for a, b in zip(ordered[:-1], ordered[1:])
            if b == a + 1
        ]
        runs = skew_intervals(
            skews, skew_threshold=skew_threshold, min_run=min_run
        )
        return [(ordered[a], ordered[b] + 1) for a, b in runs]
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)
