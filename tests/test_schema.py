"""Migration 0001 round-trips and matches TRD §4.

Round-tripping (upgrade → assert → downgrade → assert empty) is what catches a
`downgrade()` that drops tables in the wrong order, which otherwise only shows
up the first time someone needs to roll back.
"""

import pytest
from alembic import command

TABLES = {
    # corpus
    "chunks",
    # facts layer
    "pull_requests",
    "pr_files",
    "pr_reviews",
    "commits",
    "commit_files",
    "issues",
    "issue_labels",
    "releases",
    "areas",
    # sessions
    "sessions",
    "messages",
    # memory
    "memories",
    "memory_audit",
    # tools
    "tool_defs",
    # eval
    "eval_runs",
    # ingest completion marker (TRD §4.2)
    "ingest_runs",
}

HNSW_INDEXES = {
    "chunks_embedding_idx",
    "memories_embedding_idx",
    "tool_defs_embedding_idx",
}

# index name -> a fragment that must appear in its WHERE clause
PARTIAL_INDEXES = {
    "pr_merged_at_idx": "state = 'merged'",
    "pr_milestone_idx": "milestone IS NOT NULL",
    "issues_milestone_idx": "milestone IS NOT NULL",
    "memories_live_idx": "valid_to IS NULL",
    "ingest_runs_complete_idx": "completed_at IS NOT NULL",
}


@pytest.fixture(scope="module", autouse=True)
def _migrated(alembic_config):
    command.downgrade(alembic_config, "base")
    command.upgrade(alembic_config, "head")
    yield
    command.downgrade(alembic_config, "base")


def _public_tables(db) -> set[str]:
    rows = db.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
    ).fetchall()
    return {r[0] for r in rows} - {"alembic_version"}


def test_every_table_exists(db):
    assert _public_tables(db) == TABLES


def test_vector_columns_are_384_dimensional(db):
    rows = db.execute(
        """
        SELECT c.relname, a.atttypmod
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_type t ON t.oid = a.atttypid
        WHERE t.typname = 'vector'
          AND a.attname = 'embedding'
          AND c.relkind = 'r'    -- the HNSW indexes carry the column too
        """
    ).fetchall()
    # pgvector stores the declared dimension in atttypmod verbatim.
    assert {r[0] for r in rows} == {"chunks", "memories", "tool_defs"}
    assert all(r[1] == 384 for r in rows), rows


def test_hnsw_indexes_use_hnsw(db):
    rows = db.execute(
        """
        SELECT i.relname, am.amname
        FROM pg_class i
        JOIN pg_index x ON x.indexrelid = i.oid
        JOIN pg_am am ON am.oid = i.relam
        WHERE i.relname = ANY(%s)
        """,
        (sorted(HNSW_INDEXES),),
    ).fetchall()
    assert {r[0] for r in rows} == HNSW_INDEXES
    assert all(r[1] == "hnsw" for r in rows), rows


def test_partial_indexes_keep_their_predicates(db):
    rows = db.execute(
        "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = 'public'"
    ).fetchall()
    defs = dict(rows)
    for name, predicate in PARTIAL_INDEXES.items():
        assert name in defs, f"{name} missing"
        assert " WHERE " in defs[name], f"{name} is not partial: {defs[name]}"
        normalized = defs[name].replace('"', "").replace("::text", "")
        assert predicate.lower() in normalized.lower(), defs[name]


def test_tsv_and_trigram_indexes_exist(db):
    rows = db.execute(
        """
        SELECT i.relname, am.amname
        FROM pg_class i
        JOIN pg_index x ON x.indexrelid = i.oid
        JOIN pg_am am ON am.oid = i.relam
        WHERE i.relname IN ('chunks_tsv_idx', 'memories_entities_idx')
        """
    ).fetchall()
    assert {r[0]: r[1] for r in rows} == {
        "chunks_tsv_idx": "gin",
        "memories_entities_idx": "gin",
    }


def test_downgrade_leaves_nothing_behind(alembic_config, db):
    command.downgrade(alembic_config, "base")
    try:
        assert _public_tables(db) == set()
    finally:
        command.upgrade(alembic_config, "head")
