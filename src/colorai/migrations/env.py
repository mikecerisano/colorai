"""Alembic migration environment for ColorAI's project database."""

from __future__ import annotations

import os

from alembic import context
from sqlalchemy import create_engine, pool

from colorai.project.models import Base

config = context.config
target_metadata = Base.metadata


def _database_url() -> str:
    """The database URL, overridable via ``COLORAI_DB_URL`` for CLI/tests."""
    return os.environ.get("COLORAI_DB_URL", config.get_main_option("sqlalchemy.url"))


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_database_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
