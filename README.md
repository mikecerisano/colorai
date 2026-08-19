# ColorAI

Local-first AI finishing and color-QC assistant for professionally finished video.

ColorAI analyzes a baked Rec.709 master shot-by-shot, detects visual
inconsistencies and defects, proposes **deterministic, temporally stable**
corrections, and reserves generative restoration only for genuinely damaged
temporal intervals. The filmmaker always retains approval authority.

See `docs/architecture.md` for the design, `AGENTS.md` for contributor
handoff, and `docs/status.md` for current progress.

## Quick start

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[web,dev]"

# Show the CLI
.venv/bin/colorai --help

# Analyze a master end-to-end (ingest -> shots -> frames -> metrics)
.venv/bin/colorai analyze /path/to/master.mov --project data/project.sqlite3

# Start the local review UI
.venv/bin/colorai ui --project data/project.sqlite3 --port 8000
```

## License

MIT
