"""ColorAI command-line interface.

Currently a thin scaffold: ``--help`` / ``--version`` work, and the
``analyze`` / ``ui`` commands are declared with their final argument shape
but report that they are not implemented yet. Each command gets a real
implementation as the corresponding pipeline lands.
"""

from __future__ import annotations

import argparse

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
        description="Analyze a master end-to-end (not yet implemented).",
    )
    p_analyze.add_argument("master", help="Path to the baked Rec.709 master.")
    p_analyze.add_argument(
        "--project", default="data/project.sqlite3", help="Project database path."
    )

    p_ui = sub.add_parser(
        "ui",
        help="Start the local review UI.",
        description="Start the local review UI (not yet implemented).",
    )
    p_ui.add_argument(
        "--project", default="data/project.sqlite3", help="Project database path."
    )
    p_ui.add_argument("--port", type=int, default=8000, help="Port to listen on.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    print(f"error: 'colorai {args.command}' is not implemented yet")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
