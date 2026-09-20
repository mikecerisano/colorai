"""ColorAI command-line interface.

``analyze`` runs the full pipeline (ingest -> shot detection -> representative
frames -> metrics). ``open`` analyzes a master and serves the review UI in one
command. ``render`` exports a full master with approved corrections applied.
``export`` writes a Resolve interchange package (per-shot CDL/baked LUT, EDL,
FCP7 XML). ``ui`` starts the review server. ``db migrate`` applies Alembic
schema migrations to a project database. ``mcp`` starts the MCP server (stdio)
for agent integration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from colorai import __version__

_DESCRIPTION = (
    "Local-first AI finishing and color-QC assistant for professionally "
    "finished video."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="colorai", description=_DESCRIPTION)
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p_analyze = sub.add_parser(
        "analyze",
        help="Analyze a master end-to-end (ingest -> shots -> frames -> metrics).",
    )
    p_analyze.add_argument("master", help="Path to the baked Rec.709 master.")
    p_analyze.add_argument(
        "--project", default="data/project.sqlite3", help="Project database path."
    )
    p_analyze.add_argument(
        "--force", action="store_true", help="Re-analyze even if results are cached."
    )
    p_analyze.add_argument(
        "--transfer", default=None,
        help="Declare the master's transfer (bt709/pq/hlg) when untagged; never inferred.",
    )

    p_open = sub.add_parser(
        "open",
        help="Analyze a master and open it in the review UI (one command).",
    )
    p_open.add_argument("master", help="Path to the baked Rec.709 master.")
    p_open.add_argument(
        "--projects-dir", default="data", help="Directory holding per-master projects."
    )
    p_open.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    p_open.add_argument(
        "--force", action="store_true", help="Re-analyze even if results are cached."
    )
    p_open.add_argument(
        "--transfer", default=None,
        help="Declare the master's transfer (bt709/pq/hlg) when untagged; never inferred.",
    )

    p_ui = sub.add_parser(
        "ui",
        help="Start the local review UI.",
    )
    p_ui.add_argument(
        "--project", default="data/project.sqlite3", help="Project database path."
    )
    p_ui.add_argument("--port", type=int, default=8000, help="Port to listen on.")

    p_render = sub.add_parser(
        "render",
        help="Render an asset with its approved shot corrections applied.",
    )
    p_render.add_argument(
        "--project", default="data/project.sqlite3", help="Project database path."
    )
    p_render.add_argument(
        "--asset", type=int, default=None, help="Asset id (defaults to the first asset)."
    )
    p_render.add_argument("--out", required=True, help="Output video path.")
    p_render.add_argument("--codec", default="libx264", help="Output codec.")
    p_render.add_argument("--crf", type=int, default=18, help="Quality (lower = better).")
    p_render.add_argument(
        "--preset", default="medium", help="x264 speed/quality preset."
    )
    p_render.add_argument(
        "--jobs", type=int, default=4,
        help="Parallel transform workers (1 = serial).",
    )

    p_export = sub.add_parser(
        "export",
        help="Export a Resolve interchange package (CDL/LUT + EDL + XML).",
    )
    p_export.add_argument(
        "--project", default="data/project.sqlite3", help="Project database path."
    )
    p_export.add_argument(
        "--asset", type=int, default=None, help="Asset id (defaults to the first asset)."
    )
    p_export.add_argument("--out-dir", required=True, help="Output directory.")
    p_export.add_argument(
        "--lut-size", type=int, default=33, help="Baked .cube lattice size."
    )

    p_db = sub.add_parser("db", help="Database management.")
    db_sub = p_db.add_subparsers(dest="db_command", metavar="COMMAND")
    p_db_migrate = db_sub.add_parser("migrate", help="Apply pending schema migrations.")
    p_db_migrate.add_argument(
        "--project", default="data/project.sqlite3", help="Project database path."
    )

    sub.add_parser(
        "mcp",
        help="Start the MCP server (stdio) for Claude Code / Codex / ChatGPT integration.",
    )

    return parser


def _analyze_into(
    project_path: Path, master: str, *, force: bool,
    transfer: str | None = None,
):
    """Analyze ``master`` into the project database at ``project_path``."""
    from colorai.pipeline import analyze_master
    from colorai.project import ProjectStore

    # create_all is idempotent, so this both creates a fresh database and
    # safely opens an existing one without touching its data.
    store = ProjectStore.create(project_path)

    projects = store.list_projects()
    if projects:
        project_id = projects[0].id
    else:
        project_id = store.create_project(Path(master).stem).id

    stills_dir = project_path.parent / "stills"
    return analyze_master(
        store, project_id, master, stills_dir=stills_dir, resume=not force,
        transfer=transfer,
    )


def _serve(project_path: Path, port: int) -> int:
    import uvicorn

    from colorai.project import ProjectStore
    from colorai.ui import create_app

    store = ProjectStore.create(project_path)
    stills_dir = project_path.parent / "stills"
    app = create_app(store, stills_dir)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0


def _run_analyze(args: argparse.Namespace) -> int:
    result = _analyze_into(
        Path(args.project), args.master, force=args.force,
        transfer=args.transfer,
    )

    print(f"asset : {result.asset.source_path}")
    print(f"shots : {len(result.shots)}")
    print(f"stills: {len(result.representative_frames)}")
    print(f"metrics: {len(result.metrics)}")
    return 0


def _run_ui(args: argparse.Namespace) -> int:
    return _serve(Path(args.project), args.port)


def project_path_for_master(projects_dir: str | Path, master: str) -> Path:
    """Per-master project database path: ``<dir>/<stem>/project.sqlite3``.

    The stem is sanitized for directory use so odd filenames cannot escape
    the projects directory.
    """
    import re

    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(master).stem).strip("._") or "master"
    return Path(projects_dir) / stem[:80] / "project.sqlite3"


def _run_open(args: argparse.Namespace) -> int:
    project_path = project_path_for_master(args.projects_dir, args.master)
    result = _analyze_into(
        project_path, args.master, force=args.force, transfer=args.transfer
    )
    print(f"asset : {result.asset.source_path}")
    print(f"shots : {len(result.shots)}")
    print(f"review: http://127.0.0.1:{args.port}/?asset_id={result.asset.id}")
    return _serve(project_path, args.port)


def _run_render(args: argparse.Namespace) -> int:
    from colorai.project import ProjectStore
    from colorai.render import render_master

    store = ProjectStore.open(args.project)
    with store.session() as session:
        from colorai.project.models import MediaAsset

        asset = (
            session.get(MediaAsset, args.asset)
            if args.asset is not None
            else session.query(MediaAsset).order_by(MediaAsset.id).first()
        )
        if asset is None:
            print("error: no asset found in project")
            return 1
        asset_id = asset.id

    print(f"rendering asset {asset_id} -> {args.out}")
    out = render_master(
        store,
        asset_id,
        args.out,
        codec=args.codec,
        crf=args.crf,
        preset=args.preset,
        jobs=args.jobs,
    )
    print(f"rendered {out}")
    return 0


def _run_export(args: argparse.Namespace) -> int:
    from colorai.project import ProjectStore
    from colorai.interchange import export_package
    from colorai.project.models import MediaAsset

    store = ProjectStore.open(args.project)
    with store.session() as session:
        asset = (
            session.get(MediaAsset, args.asset)
            if args.asset is not None
            else session.query(MediaAsset).order_by(MediaAsset.id).first()
        )
        if asset is None:
            print("error: no asset found in project")
            return 1
        asset_id = asset.id

    print(f"exporting asset {asset_id} -> {args.out_dir}")
    try:
        manifest = export_package(store, asset_id, args.out_dir, lut_size=args.lut_size)
    except ValueError as exc:
        print(f"error: {exc}")
        return 1
    graded = sum(1 for s in manifest["shots"] if s["format"] != "none")
    print(f"exported {len(manifest['shots'])} shots ({graded} graded) + timeline.edl/xml")
    return 0


def _run_db_migrate(args: argparse.Namespace) -> int:
    import os

    import colorai
    from alembic import command
    from alembic.config import Config

    pkg_dir = Path(colorai.__file__).resolve().parent
    cfg = Config(str(pkg_dir / "alembic.ini"))
    cfg.set_main_option("script_location", str(pkg_dir / "migrations"))
    os.environ["COLORAI_DB_URL"] = (
        f"sqlite+pysqlite:///{Path(args.project).resolve().as_posix()}"
    )
    command.upgrade(cfg, "head")
    print(f"migrated {args.project}")
    return 0


def _run_mcp(args: argparse.Namespace) -> int:
    from colorai.mcp_server import main as mcp_main

    mcp_main()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "analyze":
        return _run_analyze(args)
    if args.command == "open":
        return _run_open(args)
    if args.command == "ui":
        return _run_ui(args)
    if args.command == "render":
        return _run_render(args)
    if args.command == "export":
        return _run_export(args)
    if args.command == "db" and args.db_command == "migrate":
        return _run_db_migrate(args)
    if args.command == "mcp":
        return _run_mcp(args)
    print(f"error: 'colorai {args.command}' is not implemented yet")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
