"""areas.yaml loading and resolution — TRD §5.5."""

from pathlib import Path

import asyncpg
import pytest
from alembic import command

from app.config import settings
from app.facts.areas import UnknownArea, load_areas_file, resolve, sync_areas

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
async def conn(alembic_config, test_database):
    command.downgrade(alembic_config, "base")
    command.upgrade(alembic_config, "head")
    connection = await asyncpg.connect(test_database)
    try:
        yield connection
    finally:
        await connection.close()
        command.downgrade(alembic_config, "base")


def test_the_committed_areas_file_parses():
    areas = load_areas_file(REPO_ROOT / settings.areas_file)
    names = {a.name for a in areas}
    assert {"auth", "routing"} <= names  # the two TRD §5.5 names explicitly
    assert all(a.path_globs for a in areas), "an area with no globs matches nothing"


def test_globs_become_prefix_patterns():
    (auth,) = [
        a for a in load_areas_file(REPO_ROOT / settings.areas_file) if a.name == "auth"
    ]
    assert "fastapi/security/%" in auth.sql_prefixes()
    assert "tests/test_security%" in auth.sql_prefixes()


def test_duplicate_names_are_rejected(tmp_path):
    path = tmp_path / "areas.yaml"
    path.write_text(
        "- name: auth\n  path_globs: [a/**]\n- name: auth\n  path_globs: [b/**]\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_areas_file(path)


async def test_sync_replaces_the_table(conn):
    areas = load_areas_file(REPO_ROOT / settings.areas_file)
    assert await sync_areas(conn, areas) == len(areas)
    await sync_areas(conn, areas)  # curated file wins; no duplicate-key error
    assert await conn.fetchval("SELECT count(*) FROM areas") == len(areas)


async def test_unknown_area_raises_rather_than_returning_empty(conn):
    """§5.5: "no commits in the payments area" and "there is no area called
    payments" must not look the same to the manager."""
    await sync_areas(conn, load_areas_file(REPO_ROOT / settings.areas_file))
    assert (await resolve(conn, "auth")).name == "auth"
    with pytest.raises(UnknownArea) as exc:
        await resolve(conn, "payments")
    assert "payments" in str(exc.value)
    assert "auth" in str(exc.value)  # the error names what does exist
