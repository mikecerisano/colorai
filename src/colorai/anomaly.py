"""Deterministic temporal anomaly detection.

Finds **blur pulses**: short runs of consecutive low-sharpness frames
sandwiched between sharp frames — the signature of Gyroflow-style
post-stabilization, where geometry is stable but a few frames carried motion
blur. Sharpness uses Laplacian variance (:func:`colorai.metrics.frame_sharpness`),
a first-order proxy; a directional blur metric is a documented refinement.

All frame numbers are inclusive and zero-based.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from colorai.frames import extract_frame
from colorai.metrics import frame_sharpness
from colorai.tracking import sample_frames

DEFAULT_RATIO_THRESHOLD = 0.5
DEFAULT_MIN_RUN = 2
DEFAULT_SAMPLES = 16


@dataclass(frozen=True)
class BlurPulse:
    """A detected run of low-sharpness frames."""

    start_frame: int
    end_frame: int  # inclusive
    num_frames: int
    min_ratio: float  # lowest sharpness / baseline sharpness


def blur_pulses_from_scores(
    scores: dict[int, float],
    *,
    ratio_threshold: float = DEFAULT_RATIO_THRESHOLD,
    min_run: int = DEFAULT_MIN_RUN,
) -> list[BlurPulse]:
    """Find runs of consecutive low-sharpness frames in a ``frame -> sharpness`` map.

    A frame is "blurred" when its sharpness is below ``ratio_threshold`` times
    the shot's median sharpness. Runs shorter than ``min_run`` are ignored.
    """
    if not scores:
        return []
    baseline = float(np.median(list(scores.values())))
    if baseline <= 0:
        return []

    frames = sorted(scores)
    pulses: list[BlurPulse] = []
    run_start: int | None = None
    run_min = 1.0
    previous: int | None = None

    for frame in frames:
        ratio = scores[frame] / baseline
        blurred = ratio < ratio_threshold
        if blurred and run_start is None:
            run_start, run_min = frame, ratio
        elif blurred:
            run_min = min(run_min, ratio)
        elif run_start is not None and previous is not None:
            pulses.append(
                BlurPulse(run_start, previous, previous - run_start + 1, run_min)
            )
            run_start = None
        previous = frame

    if run_start is not None:
        pulses.append(
            BlurPulse(run_start, frames[-1], frames[-1] - run_start + 1, run_min)
        )
    return [p for p in pulses if p.num_frames >= min_run]


def detect_blur_pulses(
    video_path: str | Path,
    start: int,
    end: int,
    fps: float,
    *,
    samples: int = DEFAULT_SAMPLES,
    scale: int | None = 480,
    ratio_threshold: float = DEFAULT_RATIO_THRESHOLD,
    min_run: int = DEFAULT_MIN_RUN,
) -> list[BlurPulse]:
    """Sample sharpness across ``[start, end]`` and flag blur pulses."""
    probe_dir = Path(tempfile.mkdtemp(prefix="colorai_anomaly_"))
    try:
        scores: dict[int, float] = {}
        for frame_index in sample_frames(start, end, samples):
            still = extract_frame(
                video_path, frame_index, probe_dir / f"{frame_index}.png",
                fps=fps, scale=scale,
            )
            image = cv2.imread(str(still), cv2.IMREAD_COLOR)
            if image is not None:
                scores[frame_index] = frame_sharpness(image)
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)

    return blur_pulses_from_scores(
        scores, ratio_threshold=ratio_threshold, min_run=min_run
    )
