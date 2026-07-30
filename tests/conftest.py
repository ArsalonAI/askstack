"""Test fixtures.

Tests run against `<database>_test`, never the database in `.env`. The schema
tests downgrade to base on teardown, so pointing them at the dev database
would silently destroy an ingested corpus — an hour of GitHub API calls — every
time someone ran pytest.
"""

from pathlib import Path

import psycopg
import pytest
from alembic.config import Config
from psycopg import sql

from app.config import settings

REPO_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = REPO_ROOT / "scripts" / "migrations" / "alembic.ini"
EXTENSIONS = ("vector", "pg_trgm")


def _test_database_url() -> str:
    base, _, name = settings.database_url.rpartition("/")
    return f"{base}/{name}_test"


def _maintenance_url() -> str:
    base, _, _ = settings.database_url.rpartition("/")
    return f"{base}/postgres"


@pytest.fixture(scope="session")
def test_database() -> str:
    """Create `<database>_test` and its extensions if they are missing."""
    url = _test_database_url()
    name = url.rpartition("/")[2]

    with psycopg.connect(_maintenance_url(), autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (name,)
        ).fetchone()
        if not exists:
            try:
                conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
            except psycopg.errors.InsufficientPrivilege as exc:
                # In docker-compose the app role owns the cluster, so this
                # never fires there. On a hand-rolled server it will.
                raise RuntimeError(
                    f"cannot create database {name!r}. Either grant the role "
                    f"createdb:\n  psql -d postgres -c 'ALTER ROLE "
                    f"{settings.database_url.partition('://')[2].partition(':')[0]} "
                    f"CREATEDB;'\nor create it by hand:\n  createdb {name}"
                ) from exc

    with psycopg.connect(url, autocommit=True) as conn:
        for ext in EXTENSIONS:
            try:
                conn.execute(
                    sql.SQL("CREATE EXTENSION IF NOT EXISTS {}").format(
                        sql.Identifier(ext)
                    )
                )
            except psycopg.errors.InsufficientPrivilege as exc:
                raise RuntimeError(
                    f"cannot create extension {ext!r} in {name}. Run as a superuser:\n"
                    f'  psql -d {name} -c "CREATE EXTENSION IF NOT EXISTS {ext};"'
                ) from exc
    return url


@pytest.fixture(scope="session")
def alembic_config(test_database) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(ALEMBIC_INI.parent))
    cfg.set_main_option("sqlalchemy.url", test_database.replace(
        "postgresql://", "postgresql+psycopg://", 1
    ))
    return cfg


@pytest.fixture
def db(test_database):
    """A live connection. Deliberately errors rather than skipping — a schema
    test that quietly no-ops when Postgres is down is worse than no test."""
    with psycopg.connect(test_database, autocommit=True) as conn:
        yield conn
