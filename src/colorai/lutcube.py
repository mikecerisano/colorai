"""Resolve/IRIDAS ``.cube`` LUT parsing and application.

The ``lut`` correction kind references a ``.cube`` file by path and applies it
in the **linear BT.709 working space** (the same scene-referred space the rest
of the grade runs in). This module is the deterministic, side-effect-free body:

* :func:`parse_cube` — parse 1D/3D ``.cube`` text (``LUT_1D_SIZE`` /
  ``LUT_3D_SIZE``, ``DOMAIN_MIN``/``DOMAIN_MAX`` and the ``LUT_*_INPUT_RANGE``
  variants, comments, ``TITLE``).
* :func:`apply_cube` — trilinear (3D) or linear (1D) interpolation with
  clamping outside the LUT's declared domain.
* :func:`cube_content_hash` — a content fingerprint so a persisted ``lut``
  correction can record exactly which file version it used.
* :func:`load_cube` — parse with a small cache keyed by ``(path, mtime, size)``.

**Domain semantics.** A ``.cube`` authored for a display or log space is *not*
remapped here — it is interpreted in the working space's ``[0, 1]`` linear
domain (clamped to the file's ``DOMAIN_MIN``/``DOMAIN_MAX``). Converting a
display/log LUT into the linear working space is a caller/agent responsibility
(see ``docs/research-notes.md``).

LUT files are read-only; nothing here opens them for writing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CubeLUT:
    """A parsed ``.cube`` LUT."""

    size: int
    is_3d: bool
    domain_min: tuple[float, float, float]
    domain_max: tuple[float, float, float]
    #: shape ``(size, 3)`` for 1D, ``(size, size, size, 3)`` for 3D (indexed
    #: ``[r, g, b, channel]`` after normalizing the file's red-fastest order).
    table: np.ndarray


def _split_tokens(line: str) -> list[str]:
    return line.strip().split()


def parse_cube(text: str) -> CubeLUT:
    """Parse ``.cube`` text into a :class:`CubeLUT` (raises ``ValueError``)."""
    size: int | None = None
    is_3d: bool | None = None
    domain_min = [0.0, 0.0, 0.0]
    domain_max = [1.0, 1.0, 1.0]
    data: list[list[float]] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = _split_tokens(line)
        if not parts:
            continue
        head = parts[0].upper()
        if head == "LUT_1D_SIZE":
            size, is_3d = int(parts[1]), False
        elif head == "LUT_3D_SIZE":
            size, is_3d = int(parts[1]), True
        elif head == "DOMAIN_MIN":
            domain_min = [float(x) for x in parts[1:4]]
        elif head == "DOMAIN_MAX":
            domain_max = [float(x) for x in parts[1:4]]
        elif head == "LUT_1D_INPUT_RANGE":
            domain_min = [float(parts[1])] * 3
            domain_max = [float(parts[2])] * 3
        elif head == "LUT_3D_INPUT_RANGE":
            domain_min = [float(x) for x in parts[1:4]]
            domain_max = [float(x) for x in parts[4:7]]
        elif head in ("TITLE",):
            continue
        else:
            data.append([float(x) for x in parts[:3]])

    if size is None or is_3d is None:
        raise ValueError("missing LUT_1D_SIZE or LUT_3D_SIZE")
    expected = size ** 3 if is_3d else size
    if len(data) != expected:
        raise ValueError(f"expected {expected} LUT entries, got {len(data)}")

    table = np.asarray(data, dtype=np.float32)
    if is_3d:
        # .cube files list red-fastest, then green, then blue (flat index
        # r + N*g + N^2*b). Reshape yields [b, g, r, ch]; transpose to the
        # canonical [r, g, b, ch] orientation the applier uses.
        table = table.reshape(size, size, size, 3).transpose(2, 1, 0, 3)
    else:
        table = table.reshape(size, 3)

    return CubeLUT(
        size=size,
        is_3d=is_3d,
        domain_min=(float(domain_min[0]), float(domain_min[1]), float(domain_min[2])),
        domain_max=(float(domain_max[0]), float(domain_max[1]), float(domain_max[2])),
        table=table,
    )


def parse_cube_file(path: str | Path) -> CubeLUT:
    """Parse a ``.cube`` file on disk (read-only)."""
    return parse_cube(Path(path).read_text(encoding="utf-8"))


def cube_content_hash(path: str | Path) -> str:
    """SHA-256 of a ``.cube`` file's bytes (streamed for large files)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


_CACHE: dict[tuple[str, int, int], CubeLUT] = {}


def load_cube(path: str | Path) -> CubeLUT:
    """Parse ``path`` with a cache keyed by ``(path, mtime_ns, size)``.

    The cache invalidates when the file changes, so an edited LUT is picked up
    without restarting the process.
    """
    p = Path(path)
    stat = p.stat()
    key = (str(p.resolve()), stat.st_mtime_ns, stat.st_size)
    lut = _CACHE.get(key)
    if lut is None:
        lut = parse_cube_file(p)
        _CACHE[key] = lut
    return lut


def _domain_scale(lut: CubeLUT) -> tuple[np.ndarray, np.ndarray]:
    dmin = np.asarray(lut.domain_min, dtype=np.float32)
    dmax = np.asarray(lut.domain_max, dtype=np.float32)
    span = dmax - dmin
    span = np.where(span == 0.0, 1.0, span)
    return dmin, span


def apply_cube_1d(lut: CubeLUT, rgb: np.ndarray) -> np.ndarray:
    """Linear interpolation of a 1D ``.cube`` on ``(..., 3)`` linear RGB."""
    n = lut.size
    dmin, span = _domain_scale(lut)
    pos = np.clip((rgb.astype(np.float32) - dmin) / span * (n - 1), 0.0, n - 1)
    i0 = np.floor(pos).astype(np.int32)
    i1 = np.minimum(i0 + 1, n - 1)
    frac = pos - i0
    out = np.empty(rgb.shape, dtype=np.float32)
    for k in range(3):
        t0 = lut.table[i0[..., k], k]
        t1 = lut.table[i1[..., k], k]
        out[..., k] = t0 + (t1 - t0) * frac[..., k]
    return out


def apply_cube_3d(lut: CubeLUT, rgb: np.ndarray) -> np.ndarray:
    """Trilinear interpolation of a 3D ``.cube`` on ``(..., 3)`` linear RGB."""
    n = lut.size
    dmin, span = _domain_scale(lut)
    pos = np.clip((rgb.astype(np.float32) - dmin) / span * (n - 1), 0.0, n - 1)
    i0 = np.floor(pos).astype(np.int32)
    i1 = np.minimum(i0 + 1, n - 1)
    frac = pos - i0

    out = np.zeros(rgb.shape, dtype=np.float32)
    for ci in (0, 1):
        for cj in (0, 1):
            for ck in (0, 1):
                idx = np.stack(
                    [
                        i0[..., 0] if ci == 0 else i1[..., 0],
                        i0[..., 1] if cj == 0 else i1[..., 1],
                        i0[..., 2] if ck == 0 else i1[..., 2],
                    ],
                    axis=-1,
                )
                w = (
                    ((1.0 - frac[..., 0]) if ci == 0 else frac[..., 0])
                    * ((1.0 - frac[..., 1]) if cj == 0 else frac[..., 1])
                    * ((1.0 - frac[..., 2]) if ck == 0 else frac[..., 2])
                )
                out += lut.table[idx[..., 0], idx[..., 1], idx[..., 2]] * w[..., None]
    return out


def apply_cube(lut: CubeLUT, rgb: np.ndarray) -> np.ndarray:
    """Apply a parsed ``.cube`` to ``(..., 3)`` linear RGB (1D or 3D)."""
    return apply_cube_3d(lut, rgb) if lut.is_3d else apply_cube_1d(lut, rgb)
