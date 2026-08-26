"""Versioned memory persistence — TRD §3.4 and §4.1.

**Nothing here mutates or deletes a row.** `UPDATE` appears exactly twice, both
times closing a validity window (`valid_to`, `superseded_by`) on a row whose
content is never touched. That is ADR 4, and it is what makes ADR 5 defensible:
the agent writes memory without an approval gate *because* every write is
attributable and reversible. A store that edited in place would concede both.

This module owns persistence and nothing else. What is worth remembering is the
Manager's problem (§2.2), and how memories are extracted is M4's.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from app.interfaces import CreatedBy, Memory, MemType

if TYPE_CHECKING:  # pragma: no cover
    import asyncpg

_COLUMNS = """
    id, revision, user_id, mem_type, content, entities, confidence,
    valid_from, valid_to, created_by, source_session_id, source_ids, trace_id
"""

# Same value the chunk index uses (§6.1): the default of 40 measurably costs
# recall, and 100 stays well inside the latency budget.
EF_SEARCH = 100

# Decay is applied in SQL, not in Python, so `k` selects the top-k *after*
# ranking. Ranking k already-fetched rows cannot promote a slightly less
# similar but far fresher memory into the set, which is the entire purpose of
# the episodic half-life (§8.4).
SEARCH_SQL = """
SELECT {columns}, 1 - (embedding <=> $2::vector) AS similarity
FROM memories
WHERE user_id = $1 AND mem_type = $3 AND valid_to IS NULL
ORDER BY (1 - (embedding <=> $2::vector))
         * CASE WHEN $4::float IS NULL THEN 1.0
                ELSE power(
                    0.5,
                    EXTRACT(EPOCH FROM (now() - valid_from)) / 86400.0 / $4::float
                )
           END
         * CASE WHEN $5 THEN confidence ELSE 1.0 END DESC
LIMIT $6
"""


def vector_literal(vector: np.ndarray) -> str:
    """pgvector's text input form — asyncpg has no native codec for it.

    Same 7-significant-digit format `app.retrieval.hybrid` uses. Two modules
    encoding vectors to different precision would compare memories against
    query vectors that are not quite the ones the index was built from.
    """
    return "[" + ",".join(f"{v:.7g}" for v in vector) + "]"


def _row_to_memory(row: Any) -> Memory:
    return Memory(
        id=row["id"],
        user_id=row["user_id"],
        mem_type=row["mem_type"],
        content=row["content"],
        entities=list(row["entities"] or []),
        confidence=float(row["confidence"]),
        revision=row["revision"],
        valid_from=row["valid_from"],
        valid_to=row["valid_to"],
        created_by=row["created_by"],
        source_session_id=row["source_session_id"],
        source_ids=list(row["source_ids"] or []),
        trace_id=row["trace_id"],
    )


def _audit_payload(memory: Memory) -> str:
    """What the audit row stores. The embedding is deliberately excluded — it is
    derived from `content`, and 384 floats per audit row would dwarf the fact."""
    return json.dumps(
        {
            "id": memory.id,
            "revision": memory.revision,
            "user_id": memory.user_id,
            "mem_type": memory.mem_type,
            "content": memory.content,
            "entities": list(memory.entities),
            "confidence": memory.confidence,
            "created_by": memory.created_by,
            "source_session_id": memory.source_session_id,
            "source_ids": list(memory.source_ids),
        },
        default=str,
    )


class MemoryNotFound(LookupError):
    """No memory with that id. §11.3 maps this to 404."""


class RevisionNotFound(LookupError):
    """`to_revision` exceeds the memory's history. §11.3 maps this to 400."""


class PostgresMemoryStore:
    """§3.4 over asyncpg.

    Takes the pool rather than a connection, for the reason the M2 eval run
    found the hard way: holding one connection across a whole agent turn
    deadlocks, because the turn needs connections *inside* it.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    # ----------------------------------------------------------------- writes

    async def write(
        self,
        user_id: str,
        mem_type: MemType,
        content: str,
        *,
        embedding: np.ndarray,
        entities: Sequence[str] = (),
        confidence: float = 1.0,
        created_by: CreatedBy = "agent",
        source_session_id: str | None = None,
        source_ids: Sequence[str] = (),
        trace_id: str | None = None,
        supersedes: str | None = None,
    ) -> Memory:
        """Append a new memory at revision 1.

        `write` creates a *new* identity. Updating a memory in place is a
        revision of the same `id` and is reached through `revert` (§4.1); a
        `write` that also revised would need two ids in one signature and would
        make "which row does this supersede" ambiguous.

        The window close and the insert share one transaction because §4.1
        requires exactly one live revision per `id` and that invariant is held
        in application code, not by constraint — a partial commit here would
        leave two live rows or none, and neither is recoverable by reading.
        """
        memory_id = f"mem_{uuid.uuid4().hex[:16]}"
        async with self.pool.acquire() as conn, conn.transaction():
            superseded: Memory | None = None
            if supersedes is not None:
                superseded = await self._close_window(
                    conn, supersedes, superseded_by=memory_id
                )

            row = await conn.fetchrow(
                f"""
                INSERT INTO memories (
                    id, revision, user_id, mem_type, content, embedding,
                    entities, confidence, created_by, source_session_id,
                    source_ids, trace_id
                )
                VALUES ($1, 1, $2, $3, $4, $5::vector, $6, $7, $8, $9, $10, $11)
                RETURNING {_COLUMNS}
                """,
                memory_id,
                user_id,
                mem_type,
                content,
                vector_literal(embedding),
                list(entities),
                confidence,
                created_by,
                source_session_id,
                list(source_ids),
                trace_id,
            )
            memory = _row_to_memory(row)
            await self._audit(
                conn, memory, op="create", before=None, actor=created_by, trace_id=trace_id
            )
            if superseded is not None:
                # A second audit row against the *superseded* id. Reading one
                # memory's history should show that it was replaced without
                # having to search the whole table for its successor.
                await self._audit(
                    conn,
                    memory,
                    op="supersede",
                    before=_audit_payload(superseded),
                    actor=created_by,
                    trace_id=trace_id,
                    memory_id=superseded.id,
                    revision=superseded.revision,
                )
        return memory

    async def _close_window(
        self, conn: Any, memory_id: str, *, superseded_by: str
    ) -> Memory:
        row = await conn.fetchrow(
            f"""
            UPDATE memories SET valid_to = now(), superseded_by = $2
            WHERE id = $1 AND valid_to IS NULL
            RETURNING {_COLUMNS}
            """,
            memory_id,
            superseded_by,
        )
        if row is None:
            raise MemoryNotFound(memory_id)
        return _row_to_memory(row)

    async def revert(self, memory_id: str, to_revision: int, *, actor: str) -> Memory:
        """Append revision max+1 carrying `to_revision`'s content.

        A revert is a new revision, never a rollback. The audit trail has to
        show that a revert happened, and a destructive rollback would erase
        precisely the evidence that matters — see §3.4.
        """
        async with self.pool.acquire() as conn, conn.transaction():
            target = await conn.fetchrow(
                "SELECT *, embedding::text AS embedding_text FROM memories"
                " WHERE id = $1 AND revision = $2",
                memory_id,
                to_revision,
            )
            if target is None:
                exists = await conn.fetchval(
                    "SELECT 1 FROM memories WHERE id = $1 LIMIT 1", memory_id
                )
                raise (MemoryNotFound if not exists else RevisionNotFound)(memory_id)

            live = await conn.fetchrow(
                f"SELECT {_COLUMNS} FROM memories WHERE id = $1 AND valid_to IS NULL",
                memory_id,
            )
            next_revision = await conn.fetchval(
                "SELECT max(revision) + 1 FROM memories WHERE id = $1", memory_id
            )
            if live is not None:
                await conn.execute(
                    "UPDATE memories SET valid_to = now()"
                    " WHERE id = $1 AND valid_to IS NULL",
                    memory_id,
                )

            row = await conn.fetchrow(
                f"""
                INSERT INTO memories (
                    id, revision, user_id, mem_type, content, embedding,
                    entities, confidence, created_by, source_session_id,
                    source_ids, trace_id
                )
                VALUES ($1, $2, $3, $4, $5, $6::vector, $7, $8, 'human', $9, $10, $11)
                RETURNING {_COLUMNS}
                """,
                memory_id,
                next_revision,
                target["user_id"],
                target["mem_type"],
                target["content"],
                target["embedding_text"],
                list(target["entities"] or []),
                float(target["confidence"]),
                target["source_session_id"],
                list(target["source_ids"] or []),
                target["trace_id"],
            )
            memory = _row_to_memory(row)
            await self._audit(
                conn,
                memory,
                op="revert",
                before=_audit_payload(_row_to_memory(live)) if live else None,
                actor=actor,
                trace_id=target["trace_id"],
            )
        return memory

    async def _audit(
        self,
        conn: Any,
        memory: Memory,
        *,
        op: str,
        before: str | None,
        actor: str,
        trace_id: str | None,
        memory_id: str | None = None,
        revision: int | None = None,
    ) -> None:
        await conn.execute(
            "INSERT INTO memory_audit"
            " (memory_id, revision, op, before, after, actor, trace_id)"
            " VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6, $7)",
            memory_id or memory.id,
            revision if revision is not None else memory.revision,
            op,
            before,
            _audit_payload(memory),
            actor,
            trace_id,
        )

    # ------------------------------------------------------------------ reads

    async def search(
        self,
        user_id: str,
        query_vec: np.ndarray,
        mem_type: MemType,
        k: int,
        *,
        recency_halflife_days: float | None = None,
        confidence_weighted: bool = False,
    ) -> list[Memory]:
        """Live memories of one type, ranked per §8.4.

        The two flags compose into the three arms that section specifies:
        semantic ranks by `similarity * effective_confidence`, which is a
        180-day half-life *and* the stored confidence; episodic by
        `similarity * 0.5 ** (age/30)`, six times faster because episodic facts
        are situational; procedural by similarity alone, no decay.

        Returns `Memory`, which carries `confidence` as stored. The decayed
        value is the Manager's to compute and display (§8.3); persisting it or
        smuggling it back through this dataclass would make a derived number
        indistinguishable from the stored one.
        """
        async with self.pool.acquire() as conn, conn.transaction():
            # `SET LOCAL` outside a transaction is a silent no-op, which would
            # leave ef_search at the default and cost recall with no error.
            await conn.execute(f"SET LOCAL hnsw.ef_search = {EF_SEARCH}")
            rows = await conn.fetch(
                SEARCH_SQL.format(columns=_COLUMNS),
                user_id,
                vector_literal(query_vec),
                mem_type,
                recency_halflife_days,
                confidence_weighted,
                k,
            )
        return [_row_to_memory(row) for row in rows]

    async def live(
        self, user_id: str, mem_type: MemType | None = None
    ) -> list[Memory]:
        """Every live memory, newest first. Backs `GET /memory` and the
        `user_profile_vector` the semantic arm needs (§8.4)."""
        rows = await self.pool.fetch(
            f"SELECT {_COLUMNS} FROM memories"
            " WHERE user_id = $1 AND valid_to IS NULL"
            " AND ($2::text IS NULL OR mem_type = $2)"
            " ORDER BY valid_from DESC",
            user_id,
            mem_type,
        )
        return [_row_to_memory(row) for row in rows]

    async def embeddings(self, user_id: str, mem_type: MemType) -> np.ndarray:
        """Live embeddings for one type, as an (n, dim) array.

        Separate from `live()` because `Memory` deliberately carries no vector —
        only the profile vector needs them, and putting a 384-float array on
        every memory would push it through the SSE layer and the audit rows.
        """
        rows = await self.pool.fetch(
            "SELECT embedding::text AS embedding FROM memories"
            " WHERE user_id = $1 AND mem_type = $2 AND valid_to IS NULL",
            user_id,
            mem_type,
        )
        if not rows:
            return np.empty((0, 0), dtype=np.float32)
        return np.array(
            [[float(v) for v in r["embedding"].strip("[]").split(",")] for r in rows],
            dtype=np.float32,
        )

    async def history(self, memory_id: str) -> list[Memory]:
        """All revisions, oldest first."""
        rows = await self.pool.fetch(
            f"SELECT {_COLUMNS} FROM memories WHERE id = $1 ORDER BY revision",
            memory_id,
        )
        if not rows:
            raise MemoryNotFound(memory_id)
        return [_row_to_memory(row) for row in rows]

    async def audit(self, memory_id: str) -> list[dict]:
        """The audit trail for one memory, newest first — PRD §5.5."""
        rows = await self.pool.fetch(
            "SELECT memory_id, revision, op, before, after, actor, trace_id, at"
            " FROM memory_audit WHERE memory_id = $1 ORDER BY at DESC, id DESC",
            memory_id,
        )
        return [dict(row) for row in rows]
