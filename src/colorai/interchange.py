"""Resolve interchange: export grades + editorial decisions for NLEs.

A ColorAI session ends in one of two places: a rendered master
(:mod:`colorai.render`) or a working colorist's timeline. This module is the
second path — file-based, no Resolve license needed:

* **ASC CDL** (``.cdl``) per shot when the enabled correction stack is
  exactly representable as one slope/offset/power triple (``cdl``,
  ``exposure``, ``offset``, ``rgb_balance`` — composed in order).
* **Baked ``.cube`` LUT** per shot for everything else (contrast, saturation,
  curves, LUTs, mixed stacks): the actual transform sampled on a lattice, so
  Resolve sees the same grade the preview shows.
* **CMX3600 EDL** (``timeline.edl``) and **Final Cut Pro 7 XML**
  (``timeline.xml``, which Resolve imports) carrying shot boundaries with
  source timecode.

Only *enabled* corrections are exported — the same rule the preview and the
render path use. Grades bake transfer-natively (BT.709, PQ, or HLG timelines);
like render, export refuses transfers without a known EOTF pair rather than
baking a mis-graded LUT.

Power-convention note: :mod:`colorai.correction` stores CDL ``power`` such
that the transform is ``(i*s+o)^(1/power)``; ASC CDL defines the exponent
directly as ``(i*s+o)^power``. Export therefore writes ``1/power`` — the
round-trip is exact, and the mapping is stated here so nobody has to guess.
"""

from __future__ import annotations

import json
import xml.sax.saxutils as saxutils
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from colorai.color import is_gradeable_transfer
from colorai.correction import apply_corrections
from colorai.core.timecode import frames_to_timecode, is_drop_frame
from colorai.project.models import Correction, MediaAsset, Shot
from colorai.project.store import ProjectStore

#: Correction kinds that fold exactly into one ASC CDL SOP triple.
_SOP_EXACT_KINDS = ("cdl", "exposure", "offset", "rgb_balance")

#: Upper bound for baked LUT lattices: the lattice holds ``size**3`` pixels,
#: so 64 (262k) is generous and anything larger is a hung export, not a grade.
MAX_LUT_SIZE = 64


def _vec3(value: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if value is None:
        return default
    return (float(value[0]), float(value[1]), float(value[2]))


def fold_sop(
    corrections: Iterable[tuple[str, dict[str, Any]]],
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], bool, str]:
    """Fold a correction stack into one CDL ``(slope, offset, power)`` triple.

    Returns ``(slope, offset, power, exact, reason)``. ``exact`` is True only
    when every op is CDL-expressible and the composition stays a single SOP:
    gains/offsets compose freely, but a second non-unity power — or any gain
    after a power — cannot fold into one triple. Identity ops (saturation 1,
    contrast 1) are skipped silently. ``power`` here uses the pipeline's
    stored convention (applied as ``1/power``); the ASC mapping happens in
    :func:`sop_to_cdl_xml`.
    """
    slope = [1.0, 1.0, 1.0]
    offset = [0.0, 0.0, 0.0]
    power = [1.0, 1.0, 1.0]
    has_power = False
    for kind, params in corrections:
        if kind == "cdl":
            s = _vec3(params.get("slope"), (1.0, 1.0, 1.0))
            o = _vec3(params.get("offset"), (0.0, 0.0, 0.0))
            p = _vec3(params.get("power"), (1.0, 1.0, 1.0))
        elif kind == "exposure":
            g = float(params.get("gain", 1.0))
            s, o, p = (g, g, g), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
        elif kind == "offset":
            v = float(params.get("value", 0.0))
            s, o, p = (1.0, 1.0, 1.0), (v, v, v), (1.0, 1.0, 1.0)
        elif kind == "rgb_balance":
            g = _vec3(params.get("gain"), (1.0, 1.0, 1.0))
            s, o, p = g, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
        elif kind == "saturation" and float(params.get("amount", 1.0)) == 1.0:
            continue
        elif kind == "contrast" and float(params.get("amount", 1.0)) == 1.0:
            continue
        else:
            what = kind if kind in _SOP_EXACT_KINDS else kind
            return (
                (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0),
                False,
                f"{what} is not CDL-expressible; shot will get a baked .cube instead",
            )
        if has_power and (s != (1.0, 1.0, 1.0) or o != (0.0, 0.0, 0.0)):
            return (
                (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0),
                False,
                "gain/offset after a CDL power cannot fold into one SOP triple; "
                "shot will get a baked .cube instead",
            )
        if p != (1.0, 1.0, 1.0):
            if has_power:
                return (
                    (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0),
                    False,
                    "two CDL powers cannot fold into one SOP triple; "
                    "shot will get a baked .cube instead",
                )
            has_power = True
        slope = [a * b for a, b in zip(slope, s)]
        offset = [a * b + c for a, b, c in zip(offset, s, o)]
        power = [a * b for a, b in zip(power, p)]
    return (
        (slope[0], slope[1], slope[2]),
        (offset[0], offset[1], offset[2]),
        (power[0], power[1], power[2]),
        True,
        "exact",
    )


def _fmt_triple(values: tuple[float, float, float]) -> str:
    return f"{values[0]:.6f} {values[1]:.6f} {values[2]:.6f}"


def sop_to_cdl_xml(
    decisions: list[tuple[str, str, tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]],
) -> str:
    """Format ``(cc_id, description, slope, offset, power)`` rows as ASC CDL.

    ``power`` uses the pipeline's stored convention; the written exponent is
    ``1/power`` per the ASC definition (see module docstring).
    """
    lines = ['<ColorDecisionList xmlns="urn:ASC:CDL:v1.2">']
    for cc_id, description, slope, offset, power in decisions:
        asc_power = tuple(1.0 / p for p in power)
        lines.append("<ColorDecision>")
        lines.append(f'<ColorCorrection id="{saxutils.escape(cc_id)}">')
        lines.append("<SOPNode>")
        lines.append(f"<Description>{saxutils.escape(description)}</Description>")
        lines.append(f"<Slope>{_fmt_triple(slope)}</Slope>")
        lines.append(f"<Offset>{_fmt_triple(offset)}</Offset>")
        lines.append(f"<Power>{_fmt_triple(asc_power)}</Power>")
        lines.append("</SOPNode>")
        lines.append("</ColorCorrection>")
        lines.append("</ColorDecision>")
    lines.append("</ColorDecisionList>")
    return "\n".join(lines) + "\n"


def format_cube_3d(
    table: np.ndarray, size: int, *, title: str = "ColorAI baked grade"
) -> str:
    """Serialize a ``(size, size, size, 3)`` ``[r, g, b]`` table as ``.cube`` text."""
    lines = [
        f'TITLE "{title}"',
        f"LUT_3D_SIZE {size}",
        "DOMAIN_MIN 0.0 0.0 0.0",
        "DOMAIN_MAX 1.0 1.0 1.0",
    ]
    for b in range(size):
        for g in range(size):
            for r in range(size):
                v = table[r, g, b]
                lines.append(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    return "\n".join(lines) + "\n"


def bake_cube_text(
    corrections: list[tuple[str, dict[str, Any]]], *, size: int = 33,
    title: str = "ColorAI baked grade", transfer: str | None = "bt709",
) -> str:
    """Bake a correction stack into a transfer-native 3D ``.cube``.

    The identity lattice is run through :func:`apply_corrections` — the exact
    transform the preview shows — so the baked LUT reproduces the grade in any
    NLE that applies it on a timeline of the same transfer. Fidelity is tight
    on smooth grades; where the grade clips to black/white the kink is
    smoothed over about one node span (bounded, never a different grade).
    """
    if size < 2 or size > MAX_LUT_SIZE:
        raise ValueError(f"lut size must be in [2, {MAX_LUT_SIZE}]")
    grid = np.linspace(0.0, 1.0, size, dtype=np.float64)
    lattice = np.empty((size * size, size, 3), dtype=np.float64)
    for b in range(size):
        for g in range(size):
            row = b * size + g
            lattice[row, :, 0] = grid
            lattice[row, :, 1] = grid[g]
            lattice[row, :, 2] = grid[b]
    out = np.asarray(
        apply_corrections(lattice, corrections, transfer=transfer), dtype=np.float64
    )
    table = np.empty((size, size, size, 3), dtype=np.float64)
    for b in range(size):
        for g in range(size):
            for r in range(size):
                table[r, g, b] = out[b * size + g, r]
    return format_cube_3d(table, size, title=title)


def export_edl(
    shots: list[Shot], *, fps: float, title: str = "ColorAI timeline"
) -> str:
    """Format shots as a CMX3600 EDL (assembly, record timecode = source).

    Outs are exclusive (``end_frame + 1``); bounds stay zero-based inclusive
    everywhere else per project convention.
    """
    drop = is_drop_frame(fps)
    lines = [f"TITLE: {title}", f"FCM: {'DROP FRAME' if drop else 'NON-DROP FRAME'}", ""]
    for n, shot in enumerate(shots, start=1):
        src_in = shot.start_timecode
        src_out = frames_to_timecode(shot.end_frame + 1, fps)
        lines.append(
            f"{n:03d}  AX       V     C        {src_in} {src_out} {src_in} {src_out}"
        )
        lines.append(f"* FROM CLIP NAME: shot_{shot.index:03d}")
        lines.append("")
    return "\n".join(lines)


def _fcp_rate(fps: float) -> tuple[int, bool, str]:
    timebase = int(round(fps))
    drop = is_drop_frame(fps)
    return timebase, drop, "DF" if drop else "NDF"


def export_fcp_xml(
    shots: list[Shot], *, fps: float, master_name: str, master_path: str,
    sequence_name: str = "ColorAI timeline",
) -> str:
    """Format shots as a Final Cut Pro 7 XML timeline (Resolve-importable)."""
    timebase, ntsc, displayformat = _fcp_rate(fps)
    ntsc_str = "TRUE" if ntsc else "FALSE"
    items: list[str] = []
    for shot in shots:
        duration = shot.end_frame - shot.start_frame + 1
        items.append(
            "<clipitem>"
            f"<name>shot_{shot.index:03d}</name>"
            f"<start>{shot.start_frame}</start>"
            f"<end>{shot.end_frame + 1}</end>"
            f"<in>{shot.start_frame}</in>"
            f"<out>{shot.end_frame + 1}</out>"
            f"<rate><timebase>{timebase}</timebase><ntsc>{ntsc_str}</ntsc></rate>"
            "<file>"
            f"<name>{saxutils.escape(master_name)}</name>"
            f"<pathurl>file://{saxutils.escape(master_path)}</pathurl>"
            f"<rate><timebase>{timebase}</timebase><ntsc>{ntsc_str}</ntsc></rate>"
            "</file>"
            "</clipitem>"
        )
    track = "<track>" + "".join(items) + "</track>"
    return (
        '<xmeml version="4"><sequence>'
        f"<name>{saxutils.escape(sequence_name)}</name>"
        f"<rate><timebase>{timebase}</timebase><ntsc>{ntsc_str}</ntsc></rate>"
        f"<media><video>{track}</video></media>"
        "<timecode>"
        f"<rate><timebase>{timebase}</timebase><ntsc>{ntsc_str}</ntsc></rate>"
        f"<string>{shots[0].start_timecode if shots else '00:00:00:00'}</string>"
        "<frame>0</frame>"
        f"<displayformat>{displayformat}</displayformat>"
        "</timecode>"
        "</sequence></xmeml>\n"
    )


def _enabled_stack(session_corrections: list[Correction]) -> list[tuple[str, dict]]:
    return [(c.kind, dict(c.parameters)) for c in session_corrections if c.enabled]


def export_package(
    store: ProjectStore, asset_id: int, out_dir: str | Path, *, lut_size: int = 33
) -> dict[str, Any]:
    """Export one asset's grades + editorial decisions to ``out_dir``.

    Writes per-shot ``shot_NNN.cdl`` (exact SOP) or ``shot_NNN.cube`` (baked),
    ``timeline.edl``, ``timeline.xml``, and ``manifest.json``. Never touches
    the source master or the project database. Raises ``ValueError`` for a
    missing asset or a non-Rec.709 transfer.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with store.session() as session:
        asset = session.get(MediaAsset, asset_id)
        if asset is None:
            raise ValueError(f"asset {asset_id} not found")
        if not is_gradeable_transfer(asset.transfer):
            from colorai.color import non_gradeable_reason

            raise ValueError(non_gradeable_reason(asset.transfer))
        transfer = asset.transfer
        shots = (
            session.query(Shot).filter_by(asset_id=asset_id)
            .order_by(Shot.start_frame).all()
        )
        corrections_by_shot: dict[int, list[Correction]] = {}
        for c in (
            session.query(Correction)
            .order_by(Correction.id).all()
        ):
            corrections_by_shot.setdefault(c.shot_id, []).append(c)
        fps = asset.frame_rate
        source_path = asset.source_path

    manifest_shots: list[dict[str, Any]] = []
    for shot in shots:
        stack = _enabled_stack(corrections_by_shot.get(shot.id, ()))
        label = f"shot_{shot.index:03d}"
        entry: dict[str, Any] = {
            "shot_id": shot.id,
            "index": shot.index,
            "start_frame": shot.start_frame,
            "end_frame": shot.end_frame,
            "start_timecode": shot.start_timecode,
            "end_timecode": shot.end_timecode,
        }
        if not stack:
            entry.update({"grade_file": None, "format": "none", "note": "no enabled corrections"})
        else:
            slope, offset, power, exact, reason = fold_sop(stack)
            if exact:
                text = sop_to_cdl_xml([(
                    label,
                    f"{label} {shot.start_timecode} -> {shot.end_timecode} [{transfer}]",
                    slope, offset, power,
                )])
                (out / f"{label}.cdl").write_text(text, encoding="utf-8")
                entry.update({"grade_file": f"{label}.cdl", "format": "cdl", "note": reason})
            else:
                text = bake_cube_text(
                    stack, size=lut_size, title=f"ColorAI {label} [{transfer}]",
                    transfer=transfer,
                )
                (out / f"{label}.cube").write_text(text, encoding="utf-8")
                entry.update({"grade_file": f"{label}.cube", "format": "cube", "note": reason})
        manifest_shots.append(entry)

    (out / "timeline.edl").write_text(
        export_edl(shots, fps=fps, title=f"ColorAI asset {asset_id}"), encoding="utf-8"
    )
    master = Path(source_path)
    (out / "timeline.xml").write_text(
        export_fcp_xml(
            shots, fps=fps, master_name=master.name,
            master_path=master.as_posix(),
            sequence_name=f"ColorAI asset {asset_id}",
        ),
        encoding="utf-8",
    )
    manifest = {
        "asset_id": asset_id,
        "fps": fps,
        "source_path": source_path,
        "transfer": transfer,
        "files": {"edl": "timeline.edl", "xml": "timeline.xml"},
        "shots": manifest_shots,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
