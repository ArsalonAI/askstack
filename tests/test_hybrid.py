"""Dense + sparse retrieval and RRF fusion — TRD §6.1–6.3.

The fusion arithmetic is tested as a pure function; the arms are tested against
a real index, because the two failures that matter here are both invisible in
a mock: a `SET LOCAL` that never applied, and a sparse query that skipped
identifier decomposition. Neither raises — they just quietly return worse
results.
"""

import asyncpg
import numpy as np
import pytest
from alembic import command

from app.ingest.chunking import RawChunk
from app.ingest.index import sync_chunks
from app.retrieval.hybrid import RRF_K, HybridRetriever, fuse


class OneHotEmbedder:
    """Deterministic stand-in — no model download, no torch import.

    Each text maps to a single hot dimension, so cosine similarity is 1.0 for
    identical text and 0.0 otherwise. That makes the dense arm's ordering
    predictable enough to assert on.
    """

    model_id = "test-embedder"
    dim = 384

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            out[row, hash(text) % self.dim] = 1.0
        return out

    def embed_query(self, text):
        return self.embed([text])[0]


def _chunk(chunk_id: str, content: str, source: str = "docs", path: str = "a.md") -> RawChunk:
    return RawChunk(
        id=chunk_id,
        source=source,
        path=path,
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


class TestFuse:
    def test_arithmetic(self):
        """RRF(d) = Σ 1/(k + rank). Hand-computed, not golden-file."""
        scores = fuse(["a", "b"], ["b", "c"])
        assert scores["a"] == pytest.approx(1 / (RRF_K + 1))
        assert scores["b"] == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))
        assert scores["c"] == pytest.approx(1 / (RRF_K + 2))

    def test_single_arm_match_still_scores(self):
        """ADR 3: fusion must never punish a term-only or vector-only match."""
        scores = fuse(["only-dense"], [])
        assert scores["only-dense"] > 0

    def test_document_in_both_arms_outranks_either_alone(self):
        scores = fuse(["solo", "both"], ["both"])
        assert scores["both"] > scores["solo"]

    def test_empty_arms(self):
        assert fuse([], []) == {}


class TestArms:
    async def test_sparse_arm_matches_decomposed_identifier(self, conn):
        """ADR 12, the silent-recall-loss regression.

        "http exception" must reach the chunk containing `HTTPException`. The
        indexed side decomposes identifiers; if the *query* side stops doing
        the same, nothing errors and recall just drops.
        """
        await sync_chunks(
            conn,
            OneHotEmbedder(),
            [
                _chunk("docs:a.md#raise", "Raise HTTPException to return an error."),
                _chunk("docs:a.md#other", "Completely unrelated prose about tortoises."),
            ],
        )
        retriever = HybridRetriever(conn, OneHotEmbedder(), hybrid=True)
        results = await retriever.search("http exception", k=5)
        assert "docs:a.md#raise" in [c.id for c in results]

    async def test_dense_only_when_hybrid_disabled(self, conn):
        """Ablation axis A must actually change behaviour.

        With a one-hot embedder the query vector is orthogonal to every stored
        chunk, so the dense arm ranks arbitrarily and the lexical match is not
        privileged. If the sparse arm were still running, the exact-term chunk
        would come first.
        """
        await sync_chunks(
            conn,
            OneHotEmbedder(),
            [_chunk(f"docs:a.md#c{i}", f"chunk {i} about tortoises") for i in range(5)]
            + [_chunk("docs:a.md#hit", "the word gorgonzola appears here")],
        )
        hybrid = HybridRetriever(conn, OneHotEmbedder(), hybrid=True)
        assert [c.id for c in await hybrid.search("gorgonzola", k=3)][0] == "docs:a.md#hit"

        dense_only = HybridRetriever(conn, OneHotEmbedder(), hybrid=False)
        results = await dense_only.search("gorgonzola", k=3)
        assert results, "dense-only must still return something"
        assert [c.id for c in results][0] != "docs:a.md#hit"

    async def test_source_filter(self, conn):
        await sync_chunks(
            conn,
            OneHotEmbedder(),
            [
                _chunk("docs:a.md#x", "shared vocabulary here", source="docs"),
                _chunk("code:b.py:L1-L2", "shared vocabulary here", source="code", path="b.py"),
            ],
        )
        retriever = HybridRetriever(conn, OneHotEmbedder(), hybrid=True)
        results = await retriever.search("shared vocabulary", k=10, sources=["code"])
        assert {c.source for c in results} == {"code"}

    async def test_k_bounds_the_result(self, conn):
        await sync_chunks(
            conn,
            OneHotEmbedder(),
            [_chunk(f"docs:a.md#c{i}", f"tortoises paragraph {i}") for i in range(12)],
        )
        retriever = HybridRetriever(conn, OneHotEmbedder(), hybrid=True)
        assert len(await retriever.search("tortoises", k=4)) == 4

    async def test_citation_is_the_chunk_id(self, conn):
        """§5.1: the id *is* the citation, and `gold_chunks` are these strings."""
        await sync_chunks(
            conn, OneHotEmbedder(), [_chunk("docs:a.md#anchor", "tortoises again")]
        )
        retriever = HybridRetriever(conn, OneHotEmbedder(), hybrid=True)
        (result,) = await retriever.search("tortoises", k=1)
        assert result.citation == result.id == "docs:a.md#anchor"
