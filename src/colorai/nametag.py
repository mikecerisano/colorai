"""Lower-third name suggestions for subjects.

A lower-third is **evidence, not identity truth**. This module detects the
lower-third region of a shot's representative still, runs OCR (Tesseract via
its CLI — no Python binding needed), splits the first line as a *candidate
name* and the rest as *role/affiliation*, and associates the candidate with a
subject only when the shot has exactly one distinct subject (multi-person shots
stay unassigned for human review).

Suggestions are persisted with their raw text, confidence, source
timecode, and a crop path for verification. A suggestion is ``suggested`` until
a human accepts/ignores it; accepting never overwrites a human-confirmed
subject name.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from colorai.project.models import (
    MediaAsset,
    NameSuggestion,
    RepresentativeFrame,
    Shot,
    SkinMetric,
    Subject,
)
from colorai.project.store import ProjectStore

_LOWER_THIRD_FRACTION = 0.22

STATE_SUGGESTED = "suggested"
STATE_ACCEPTED = "accepted"
STATE_IGNORED = "ignored"
STATES = (STATE_SUGGESTED, STATE_ACCEPTED, STATE_IGNORED)


def tesseract_path() -> str | None:
    return shutil.which("tesseract")


def ocr_available() -> bool:
    return tesseract_path() is not None


def ocr_status() -> dict:
    return {"tesseract": tesseract_path(), "available": ocr_available()}


def _parse_tsv(tsv: str) -> list[dict]:
    """Parse Tesseract TSV output into per-line ``{text, confidence, box}``."""
    words: dict[tuple[str, str, str], list[tuple[str, str, int, int, int, int]]] = {}
    for row in tsv.splitlines():
        cols = row.split("\t")
        if len(cols) < 12 or cols[0] != "5":  # word-level rows only
            continue
        block, par, line = cols[2], cols[3], cols[4]
        left, top, width, height = (int(cols[6]), int(cols[7]), int(cols[8]), int(cols[9]))
        conf, text = cols[10], cols[11].strip()
        if not text:
            continue
        words.setdefault((block, par, line), []).append((text, conf, left, top, width, height))

    lines: list[dict] = []
    for key in sorted(words, key=lambda k: (int(k[0]), int(k[1]), int(k[2]))):
        group = words[key]
        text = " ".join(w[0] for w in group)
        confs = [float(w[1]) for w in group if w[1] not in ("-1", "")]
        confidence = (sum(confs) / len(confs)) / 100.0 if confs else 0.0
        xs = [w[2] for w in group]
        ys = [w[3] for w in group]
        x2 = [w[2] + w[4] for w in group]
        y2 = [w[3] + w[5] for w in group]
        box = (min(xs), min(ys), max(x2) - min(xs), max(y2) - min(ys))
        lines.append({"text": text, "confidence": round(confidence, 4), "box": list(box)})
    return lines


def _run_tesseract_tsv(image_path: str | Path, lang: str = "eng") -> str:
    binary = tesseract_path()
    if binary is None:
        raise RuntimeError(
            "tesseract is not installed; install it (`brew install tesseract`) "
            "to enable lower-third name suggestions"
        )
    proc = subprocess.run(
        [binary, str(image_path), "stdout", "-l", lang, "tsv"],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


def ocr_lines(image_bgr: np.ndarray, *, lang: str = "eng") -> list[dict]:
    """OCR an HxWx3 BGR image into per-line ``{text, confidence, box}``."""
    fd, tmp = tempfile.mkstemp(suffix=".png")
    try:
        cv2.imwrite(tmp, image_bgr)
        return _parse_tsv(_run_tesseract_tsv(tmp, lang))
    finally:
        Path(tmp).unlink(missing_ok=True)


def lower_third_crop(image_bgr: np.ndarray, fraction: float = _LOWER_THIRD_FRACTION) -> np.ndarray:
    """Crop the bottom ``fraction`` of a frame (the lower-third region)."""
    h = image_bgr.shape[0]
    return image_bgr[int(h * (1.0 - fraction)):, :]


def split_name_role(lines: list[dict]) -> tuple[str, str | None]:
    """Split OCR lines into ``(candidate_name, role_text)``.

    The first non-empty line is treated as the person's name (title-cased);
    remaining lines are role/affiliation. Heuristic, not identity truth.
    """
    texts = [line["text"].strip() for line in lines if line["text"].strip()]
    if not texts:
        return "", None
    name = texts[0].strip()
    role = " | ".join(texts[1:]).strip() or None
    return name, role


def associate_subject(store: ProjectStore, shot_id: int) -> int | None:
    """The shot's subject, only when it has *exactly one* distinct subject.

    Returns ``None`` for zero or multiple subjects (conservative multi-person
    handling).
    """
    with store.session() as session:
        rows = (
            session.query(SkinMetric.subject_id)
            .filter(SkinMetric.shot_id == shot_id, SkinMetric.subject_id.isnot(None))
            .distinct()
            .all()
        )
    ids = [r[0] for r in rows]
    return ids[0] if len(ids) == 1 else None


def extract_and_store_suggestions(
    store: ProjectStore,
    asset: MediaAsset,
    shots: list[Shot],
    frames: list[RepresentativeFrame],
    crops_dir: str | Path,
    *,
    ocr: Callable[[np.ndarray], list[dict]] | None = None,
    min_confidence: float = 0.4,
    fraction: float = _LOWER_THIRD_FRACTION,
) -> list[NameSuggestion]:
    """OCR lower-thirds for each shot and persist name suggestions.

    ``ocr`` is injectable for tests; defaults to :func:`ocr_lines`. The lower-
    third crop is saved under ``crops_dir`` for human verification.
    """
    ocr_fn = ocr if ocr is not None else ocr_lines
    crops = Path(crops_dir)
    crops.mkdir(parents=True, exist_ok=True)

    with store.session() as session:
        already = {
            s.shot_id for s in session.query(NameSuggestion).filter_by(asset_id=asset.id).all()
        }

    created: list[NameSuggestion] = []
    for shot, frame in zip(shots, frames):
        if shot.id in already:
            continue  # already suggested for this shot (idempotent re-analysis)
        if frame.image_path is None:
            continue
        image = cv2.imread(frame.image_path, cv2.IMREAD_COLOR)
        if image is None:
            continue

        region = lower_third_crop(image, fraction=fraction)
        lines = ocr_fn(region)
        name, role = split_name_role(lines)
        if not name:
            continue
        confidence = max((line["confidence"] for line in lines), default=0.0)
        if confidence < min_confidence:
            continue

        crop_path = crops / f"shot_{shot.index:04d}_lowerthird.png"
        cv2.imwrite(str(crop_path), region)
        subject_id = associate_subject(store, shot.id)

        with store.session() as session:
            suggestion = NameSuggestion(
                asset_id=asset.id,
                subject_id=subject_id,
                shot_id=shot.id,
                candidate_name=name.title(),
                raw_text="\n".join(line["text"].strip() for line in lines if line["text"].strip()),
                role_text=role,
                confidence=confidence,
                timecode=shot.start_timecode,
                crop_path=str(crop_path),
                state=STATE_SUGGESTED,
            )
            session.add(suggestion)
            session.flush()
            session.refresh(suggestion)
            created.append(suggestion)
    return created


def list_suggestions(store: ProjectStore, asset_id: int) -> list[NameSuggestion]:
    with store.session() as session:
        return (
            session.query(NameSuggestion)
            .filter_by(asset_id=asset_id)
            .order_by(NameSuggestion.id)
            .all()
        )


def accept_suggestion(
    store: ProjectStore, suggestion_id: int, *, name: str | None = None
) -> NameSuggestion | None:
    """Accept a suggestion; optionally with a human-edited ``name``.

    Renames the subject only when its name has not been human-confirmed. A
    confirmed name is never overwritten — the suggestion is still marked
    accepted.
    """
    with store.session() as session:
        suggestion = session.get(NameSuggestion, suggestion_id)
        if suggestion is None:
            return None
        suggestion.state = STATE_ACCEPTED
        if suggestion.subject_id is not None:
            subject = session.get(Subject, suggestion.subject_id)
            if subject is not None and not subject.name_confirmed:
                subject.name = (name or suggestion.candidate_name).strip()
                subject.name_confirmed = True
        session.flush()
        session.refresh(suggestion)
        return suggestion


def ignore_suggestion(store: ProjectStore, suggestion_id: int) -> NameSuggestion | None:
    with store.session() as session:
        suggestion = session.get(NameSuggestion, suggestion_id)
        if suggestion is None:
            return None
        suggestion.state = STATE_IGNORED
        session.flush()
        session.refresh(suggestion)
        return suggestion


def assign_suggestion(
    store: ProjectStore, suggestion_id: int, subject_id: int
) -> NameSuggestion | None:
    """Attach an unassigned (multi-person) suggestion to a subject for review."""
    with store.session() as session:
        suggestion = session.get(NameSuggestion, suggestion_id)
        subject = session.get(Subject, subject_id)
        if suggestion is None or subject is None:
            return None
        if subject.asset_id != suggestion.asset_id:
            raise ValueError("subject and suggestion belong to different assets")
        suggestion.subject_id = subject_id
        session.flush()
        session.refresh(suggestion)
        return suggestion
