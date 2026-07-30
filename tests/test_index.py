"""Delta detection and the semantic index — TRD §5.3.

The skip path is the point of this module, and it is asserted by **counting
embed calls**, not by comparing row counts. Row counts look identical whether
a chunk was skipped or silently re-embedded, so they cannot tell you the cache
has stopped working — they can only tell you the data is still there.
"""

import asyncpg
import numpy as np
import pytest
from alembic import command

from app.ingest.chunking import RawChunk
from app.ingest.index import sync_chunks


class CountingEmbedder:
    """Deterministic stand-in: no model download, and it counts its calls."""

    model_id = "test-embedder"
    dim = 384

    def __init__(self) -> None:
        self.calls = 0
        self.texts: list[str] = []

    def embed(self, texts):
        self.calls += 1
        self.texts.extend(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            out[row, hash(text) % self.dim] = 1.0
        return out

    def embed_query(self, text):
        return self.embed([text])[0]


def _chunk(chunk_id: str, content: str, source: str = "docs") -> RawChunk:
    return RawChunk(
        id=chunk_id,
        source=source,
        path="a.md",
        anchor="A",
        content=content,
        token_count=len(content.split()),
    )


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


@pytest.fixture
def embedder():
    return CountingEmbedder()


async def test_first_run_inserts_and_embeds(conn, embedder):
    stats = await sync_chunks(conn, embedder, [_chunk("a", "one"), _chunk("b", "two")])
    assert (stats.inserted, stats.updated, stats.skipped) == (2, 0, 0)
    assert stats.embedded == 2
    assert await conn.fetchval("SELECT count(*) FROM chunks") == 2


async def test_unchanged_chunks_cost_zero_embed_calls(conn, embedder):
    """§5.3: "present and content_sha matches -> skip entirely (no embed call)".
    This is what makes a re-run take a minute instead of half an hour."""
    chunks = [_chunk("a", "one"), _chunk("b", "two")]
    await sync_chunks(conn, embedder, chunks)
    before = embedder.calls

    stats = await sync_chunks(conn, embedder, chunks)

    assert stats.skipped == 2
    assert stats.embedded == 0
    assert embedder.calls == before, "re-embedded an unchanged chunk"


async def test_changed_content_is_reembedded(conn, embedder):
    await sync_chunks(conn, embedder, [_chunk("a", "one")])
    stats = await sync_chunks(conn, embedder, [_chunk("a", "one, revised")])
    assert (stats.inserted, stats.updated, stats.skipped) == (0, 1, 0)
    assert stats.embedded == 1
    assert await conn.fetchval("SELECT content FROM chunks WHERE id='a'") == "one, revised"


async def test_orphaned_chunks_are_deleted(conn, embedder):
    """A renamed doc must not leave a chunk that still retrieves and cites a
    path that no longer exists."""
    await sync_chunks(conn, embedder, [_chunk("a", "one"), _chunk("b", "two")])
    stats = await sync_chunks(conn, embedder, [_chunk("a", "one")])
    assert stats.deleted == 1
    assert await conn.fetchval("SELECT count(*) FROM chunks") == 1


async def test_prune_can_be_disabled_for_partial_runs(conn, embedder):
    """`--limit` indexes a subset; pruning would delete everything else."""
    await sync_chunks(conn, embedder, [_chunk("a", "one"), _chunk("b", "two")])
    stats = await sync_chunks(conn, embedder, [_chunk("a", "one")], prune=False)
    assert stats.deleted == 0
    assert await conn.fetchval("SELECT count(*) FROM chunks") == 2


async def test_conflicting_duplicate_ids_are_rejected(conn, embedder):
    """Two different chunks sharing an ID would silently drop one."""
    with pytest.raises(ValueError, match="share the id"):
        await sync_chunks(conn, embedder, [_chunk("a", "one"), _chunk("a", "different")])


async def test_identical_duplicates_are_collapsed(conn, embedder):
    stats = await sync_chunks(conn, embedder, [_chunk("a", "one"), _chunk("a", "one")])
    assert stats.inserted == 1


async def test_tsv_indexes_decomposed_identifiers(conn, embedder):
    """ADR 12: `HTTPException` must match a query saying "http exception"."""
    await sync_chunks(conn, embedder, [_chunk("a", "raise HTTPException(404)", "code")])
    hit = await conn.fetchval(
        "SELECT count(*) FROM chunks WHERE tsv @@ plainto_tsquery('english', $1)",
        "http exception",
    )
    assert hit == 1


async def test_embedding_is_stored_at_the_declared_dimension(conn, embedder):
    await sync_chunks(conn, embedder, [_chunk("a", "one")])
    dim = await conn.fetchval("SELECT vector_dims(embedding) FROM chunks WHERE id='a'")
    assert dim == 384


async def test_empty_input_is_a_no_op(conn, embedder):
    stats = await sync_chunks(conn, embedder, [])
    assert stats.as_dict()["inserted"] == 0
    assert embedder.calls == 0
