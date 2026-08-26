"""Dense + sparse retrieval over `chunks`, fused by RRF — TRD §6.1–6.3.

Two arms answer the same query badly in opposite directions. The dense arm
matches paraphrase and misses exact identifiers; the sparse arm matches
`HTTPException` exactly and misses "how do I return a 404". Fusing their *ranks*
rather than their scores is ADR 3: cosine similarity and `ts_rank_cd` share no
scale, and any normalization between them would need per-corpus calibration.

This module owns no chunking and no embedding-model choice. It reads what
`app/ingest/index.py` wrote, through the same identifier decomposition the
sparse arm was indexed with — see `_sparse`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from contextlib import asynccontextmanager, contextmanager

import asyncpg
import numpy as np

from app.config import settings
from app.ingest.identifiers import tsv_input
from app.interfaces import Chunk, Embedder, Source

# §6.1: the HNSW default of 40 measurably costs recall at this corpus size, and
# 100 stays well inside the latency budget. Interpolated into SQL because `SET`
# takes no bind parameters — it is an int constant, never user input.
EF_SEARCH = 100

# §6.3: each arm contributes its top 50 to the fusion.
ARM_LIMIT = 50

# The k from the original RRF paper, deliberately untuned (ADR 3). Tuning it
# against the golden set would be fitting to the eval.
RRF_K = 60

_SELECT = "id, source, path, anchor, content"

DENSE_SQL = f"""
SELECT {_SELECT}, 1 - (embedding <=> $1::vector) AS score
FROM chunks
WHERE ($2::text[] IS NULL OR source = ANY($2))
ORDER BY embedding <=> $1::vector
LIMIT $3
"""

SPARSE_SQL = f"""
SELECT {_SELECT}, ts_rank_cd(tsv, plainto_tsquery('english', $1)) AS score
FROM chunks
WHERE tsv @@ plainto_tsquery('english', $1)
  AND ($2::text[] IS NULL OR source = ANY($2))
ORDER BY score DESC
LIMIT $3
"""


class _NullSpan:
    def update(self, **kwargs) -> None: ...


class _NullTrace:
    """Stands in when no tracer is passed, so the hot path has no `if trace:`
    branches and the retriever needs no observability stack to run."""

    @contextmanager
    def span(self, name: str, **kwargs):
        yield _NullSpan()


_NO_TRACE = _NullTrace()


def _vector_literal(vector: np.ndarray) -> str:
    """pgvector's text input form. asyncpg has no native codec for it.

    Deliberately identical to the encoder in `app/ingest/index.py`: a query
    encoded to different precision than the passages would compare vectors
    that were never the same numbers.
    """
    return "[" + ",".join(f"{v:.7g}" for v in vector) + "]"


def _chunk(row: asyncpg.Record, score: float) -> Chunk:
    # `chunks.id` *is* the citation (§5.1 grammar) — there is no second column,
    # and the golden set's `gold_chunks` are these strings verbatim.
    return Chunk(
        id=row["id"],
        source=row["source"],
        path=row["path"],
        anchor=row["anchor"],
        content=row["content"],
        citation=row["id"],
        score=score,
    )


def fuse(*arms: Sequence[str], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion over ranked ID lists.

        RRF(d) = Σ_arms 1 / (k + rank_arm(d))

    A document present in only one arm still scores, so fusion never punishes
    a term-only or vector-only match — that property is the reason RRF is here
    rather than a weighted sum.
    """
    scores: dict[str, float] = {}
    for arm in arms:
        for rank, chunk_id in enumerate(arm, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    """`Retriever` over the semantic index.

    Takes a connection or a pool. With a pool the two arms run concurrently per
    §2.1; with a single connection they cannot — asyncpg serialises one
    connection and raises on overlapping queries — so they run in sequence.
    The results are identical either way; only wall time differs.
    """

    def __init__(
        self,
        db: asyncpg.Connection | asyncpg.Pool,
        embedder: Embedder,
        *,
        hybrid: bool | None = None,
    ) -> None:
        self.db = db
        self.embedder = embedder
        # Ablation axis A. Defaults to the flag rather than hardcoding True, so
        # an eval cell can construct its own without mutating global settings.
        self.hybrid = settings.hybrid_enabled if hybrid is None else hybrid

    @asynccontextmanager
    async def _acquire(self):
        if isinstance(self.db, asyncpg.Pool):
            async with self.db.acquire() as conn:
                yield conn
        else:
            yield self.db

    async def _dense(
        self, query_vec: np.ndarray, sources: list[str] | None
    ) -> list[asyncpg.Record]:
        async with self._acquire() as conn:
            # `SET LOCAL` is a no-op outside a transaction, which would silently
            # leave ef_search at the default and cost recall with no error.
            async with conn.transaction():
                await conn.execute(f"SET LOCAL hnsw.ef_search = {EF_SEARCH}")
                return await conn.fetch(
                    DENSE_SQL, _vector_literal(query_vec), sources, ARM_LIMIT
                )

    async def _sparse(self, query: str, sources: list[str] | None) -> list[asyncpg.Record]:
        # The query goes through the same decomposition as the indexed side
        # (ADR 12). Without it "http exception" is stemmed as prose and never
        # reaches the `httpexception` token ingest wrote — a silent recall loss
        # with no error anywhere.
        async with self._acquire() as conn:
            return await conn.fetch(SPARSE_SQL, tsv_input(query), sources, ARM_LIMIT)

    async def search(
        self,
        query: str,
        k: int,
        sources: Sequence[Source] | None = None,
        trace=None,
    ) -> list[Chunk]:
        """`trace` is the §12 span parent. Optional and defaulted so the
        retriever stays usable — and testable — with no observability stack;
        §14.4 needs retrieval metrics reproducible with no network at all."""
        source_filter = list(sources) if sources else None
        trace = trace or _NO_TRACE

        # Encoding is CPU-bound and blocks the loop for tens of milliseconds;
        # off-thread so the sparse arm is not waiting on it.
        query_vec = await asyncio.to_thread(self.embedder.embed_query, query)

        if not self.hybrid:
            # Ablation axis A off: dense only, ranked by cosine similarity.
            with trace.span("retrieve.dense", metadata={"k": k, "ef_search": EF_SEARCH}) as s:
                rows = await self._dense(query_vec, source_filter)
                s.update(metadata={"n_results": len(rows), "hybrid": False})
            return [_chunk(row, float(row["score"])) for row in rows[:k]]

        if isinstance(self.db, asyncpg.Pool):
            with trace.span("retrieve.dense", metadata={"ef_search": EF_SEARCH}), trace.span(
                "retrieve.sparse"
            ):
                dense_rows, sparse_rows = await asyncio.gather(
                    self._dense(query_vec, source_filter),
                    self._sparse(query, source_filter),
                )
        else:
            with trace.span("retrieve.dense", metadata={"ef_search": EF_SEARCH}):
                dense_rows = await self._dense(query_vec, source_filter)
            with trace.span("retrieve.sparse"):
                sparse_rows = await self._sparse(query, source_filter)

        by_id = {row["id"]: row for row in [*dense_rows, *sparse_rows]}
        dense_ids = [row["id"] for row in dense_rows]
        sparse_ids = [row["id"] for row in sparse_rows]
        with trace.span(
            "retrieve.fuse",
            metadata={
                "rrf_k": RRF_K,
                "n_in": len(by_id),
                "n_out": min(k, len(by_id)),
                # How much the two arms agreed. A persistent zero means one arm
                # is contributing nothing and the ablation is measuring itself.
                "overlap": len(set(dense_ids) & set(sparse_ids)),
            },
        ):
            scores = fuse(dense_ids, sparse_ids)
        # Ties broken by id so a run is reproducible — two chunks matched by
        # one arm at adjacent ranks score identically often enough that dict
        # order would otherwise leak into recall@5.
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [_chunk(by_id[chunk_id], score) for chunk_id, score in ranked[:k]]
