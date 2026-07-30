"""Chunks -> the semantic index, with delta detection — TRD §5.3.

Embedding dominates ingest wall time, so the whole point of this module is the
skip path: a chunk whose `content_sha` is unchanged costs zero embed calls.
That is what makes a re-run take a minute instead of half an hour, and it is
worth asserting directly rather than inferring from row counts.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import asyncpg

from app.ingest.chunking import RawChunk
from app.ingest.identifiers import tsv_input
from app.interfaces import Embedder

log = logging.getLogger(__name__)

EMBED_BATCH = 256


@dataclass
class IndexStats:
    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    deleted: int = 0
    embedded: int = 0
    by_source: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "inserted": self.inserted,
            "updated": self.updated,
            "skipped": self.skipped,
            "deleted": self.deleted,
            "embedded": self.embedded,
            "by_source": self.by_source,
        }


def _vector_literal(vector) -> str:
    """pgvector's text input form. asyncpg has no native codec for it."""
    return "[" + ",".join(f"{v:.7g}" for v in vector) + "]"


async def sync_chunks(
    conn: asyncpg.Connection,
    embedder: Embedder,
    chunks: Sequence[RawChunk],
    *,
    prune: bool = True,
) -> IndexStats:
    """Reconcile `chunks` against the table.

    Per §5.3, comparing `content_sha` against the stored row by `id`:
      absent  -> insert, embed
      same    -> skip entirely, no embed call
      differs -> re-embed, update in place
      orphan  -> delete
    """
    stats = IndexStats()
    if not chunks:
        return stats

    # Duplicate IDs would make the reconciliation ambiguous and silently drop
    # content — better to fail here than to index two-thirds of a file.
    seen: dict[str, RawChunk] = {}
    for chunk in chunks:
        if chunk.id in seen and seen[chunk.id].content != chunk.content:
            raise ValueError(f"two different chunks share the id {chunk.id!r}")
        seen[chunk.id] = chunk
    chunks = list(seen.values())

    existing = {
        row["id"]: row["content_sha"]
        for row in await conn.fetch(
            "SELECT id, content_sha FROM chunks WHERE id = ANY($1::text[])",
            [c.id for c in chunks],
        )
    }

    pending: list[RawChunk] = []
    for chunk in chunks:
        stored = existing.get(chunk.id)
        if stored is None:
            pending.append(chunk)
            stats.inserted += 1
        elif stored != chunk.content_sha:
            pending.append(chunk)
            stats.updated += 1
        else:
            stats.skipped += 1
        stats.by_source[chunk.source] = stats.by_source.get(chunk.source, 0) + 1

    for start in range(0, len(pending), EMBED_BATCH):
        batch = pending[start : start + EMBED_BATCH]
        vectors = embedder.embed([c.content for c in batch])
        stats.embedded += len(batch)
        async with conn.transaction():
            await conn.executemany(
                """
                INSERT INTO chunks (
                    id, source, path, anchor, content, content_sha,
                    token_count, embedding, tsv
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8::vector,to_tsvector('english',$9))
                ON CONFLICT (id) DO UPDATE SET
                    source = EXCLUDED.source, path = EXCLUDED.path,
                    anchor = EXCLUDED.anchor, content = EXCLUDED.content,
                    content_sha = EXCLUDED.content_sha,
                    token_count = EXCLUDED.token_count,
                    embedding = EXCLUDED.embedding, tsv = EXCLUDED.tsv,
                    ingested_at = now()
                """,
                [
                    (
                        c.id,
                        c.source,
                        c.path,
                        c.anchor,
                        c.content,
                        c.content_sha,
                        c.token_count,
                        _vector_literal(vector),
                        tsv_input(c.content),
                    )
                    for c, vector in zip(batch, vectors, strict=True)
                ],
            )
        log.info("  indexed %d/%d", min(start + EMBED_BATCH, len(pending)), len(pending))

    if prune:
        # Stored but no longer produced -> delete. Without this a renamed doc
        # leaves a chunk that still retrieves and cites a path that is gone.
        deleted = await conn.fetch(
            "DELETE FROM chunks WHERE id <> ALL($1::text[]) RETURNING id",
            [c.id for c in chunks],
        )
        stats.deleted = len(deleted)

    return stats
