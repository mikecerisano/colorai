"""ColorAI command-line interface.

``analyze`` runs the full pipeline (ingest -> shot detection -> representative
frames -> metrics). ``ui`` starts the review server. ``db migrate`` applies
Alembic schema migrations to a project database.
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

    p_ui = sub.add_parser(
        "ui",
        help="Start the local review UI.",
    )
    p_ui.add_argument(
        "--project", default="data/project.sqlite3", help="Project database path."
    )
    p_ui.add_argument("--port", type=int, default=8000, help="Port to listen on.")

    p_db = sub.add_parser("db", help="Database management.")
    db_sub = p_db.add_subparsers(dest="db_command", metavar="COMMAND")
    p_db_migrate = db_sub.add_parser("migrate", help="Apply pending schema migrations.")
    p_db_migrate.add_argument(
        "--project", default="data/project.sqlite3", help="Project database path."
    )

    return parser


def _run_analyze(args: argparse.Namespace) -> int:
    from colorai.pipeline import analyze_master
    from colorai.project import ProjectStore

    project_path = Path(args.project)
    # create_all is idempotent, so this both creates a fresh database and
    # safely opens an existing one without touching its data.
    store = ProjectStore.create(project_path)

    projects = store.list_projects()
    if projects:
        project_id = projects[0].id
    else:
        project_id = store.create_project(Path(args.master).stem).id

    stills_dir = project_path.parent / "stills"
    result = analyze_master(store, project_id, args.master, stills_dir=stills_dir)

    print(f"asset : {result.asset.source_path}")
    print(f"shots : {len(result.shots)}")
    print(f"stills: {len(result.representative_frames)}")
    print(f"metrics: {len(result.metrics)}")
    return 0


def _run_ui(args: argparse.Namespace) -> int:
    import uvicorn

    from colorai.project import ProjectStore
    from colorai.ui import create_app

    project_path = Path(args.project)
    store = ProjectStore.create(project_path)
    stills_dir = project_path.parent / "stills"
    app = create_app(store, stills_dir)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "analyze":
        return _run_analyze(args)
    if args.command == "ui":
        return _run_ui(args)
    if args.command == "db" and args.db_command == "migrate":
        return _run_db_migrate(args)
    print(f"error: 'colorai {args.command}' is not implemented yet")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
