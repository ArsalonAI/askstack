"""Versioned memory persistence — TRD §3.4 and §4.1.

Against a real database, because every claim this module makes is about
transactional behaviour: exactly one live revision per id, a window close and
an insert that commit together, and an audit row for every write. None of those
survive being mocked — a mock would happily report success for a store that
lost half of each write.

ADR 5 lets the agent write memory with no approval gate *because* ADR 4 makes
every write attributable and reversible. These tests are what make that trade
real rather than asserted.
"""

import asyncpg
import numpy as np
import pytest
from alembic import command

from app.memory.store import (
    MemoryNotFound,
    PostgresMemoryStore,
    RevisionNotFound,
)

DIM = 384


def vec(seed: int) -> np.ndarray:
    """A deterministic unit vector. Distinct seeds are orthogonal, which is all
    the ranking assertions below need."""
    out = np.zeros(DIM, dtype=np.float32)
    out[seed % DIM] = 1.0
    return out


def partial(seed: int, other: int) -> np.ndarray:
    """A unit vector at cosine ~0.707 to `vec(seed)`.

    Ranking tests that pit two signals against each other need the loser on one
    axis to still score on the other. Two orthogonal one-hot vectors give the
    runner-up a similarity of exactly 0, and no decay factor can promote 0 —
    which tests multiplication by zero rather than the ranking rule.
    """
    out = vec(seed) + vec(other)
    return out / np.linalg.norm(out)


@pytest.fixture
async def pool(alembic_config, test_database):
    command.downgrade(alembic_config, "base")
    command.upgrade(alembic_config, "head")
    created = await asyncpg.create_pool(test_database, min_size=1, max_size=4)
    try:
        yield created
    finally:
        await created.close()
        command.downgrade(alembic_config, "base")


@pytest.fixture
async def store(pool):
    return PostgresMemoryStore(pool)


async def _session(pool, session_id: str = "sess_1", user_id: str = "u1") -> str:
    await pool.execute(
        "INSERT INTO sessions (id, user_id) VALUES ($1, $2)", session_id, user_id
    )
    return session_id


class TestWrite:
    async def test_write_returns_a_live_revision_1(self, store):
        memory = await store.write("u1", "semantic", "prefers async", embedding=vec(1))
        assert memory.revision == 1
        assert memory.valid_to is None
        assert memory.content == "prefers async"
        assert memory.created_by == "agent"

    async def test_provenance_round_trips(self, store, pool):
        await _session(pool)
        memory = await store.write(
            "u1",
            "episodic",
            "asked about Depends() scoping",
            embedding=vec(2),
            entities=["pr:15806"],
            confidence=0.7,
            created_by="extraction",
            source_session_id="sess_1",
            source_ids=["msg_1", "msg_2"],
            trace_id="trace_abc",
        )
        assert memory.entities == ["pr:15806"]
        assert memory.source_session_id == "sess_1"
        assert memory.source_ids == ["msg_1", "msg_2"]
        assert memory.trace_id == "trace_abc"
        assert memory.confidence == pytest.approx(0.7)

    async def test_every_write_leaves_an_audit_row(self, store):
        """PRD §5.5 — provenance on every write, without exception."""
        memory = await store.write("u1", "semantic", "x", embedding=vec(3))
        trail = await store.audit(memory.id)
        assert [row["op"] for row in trail] == ["create"]
        assert trail[0]["actor"] == "agent"

    async def test_audit_excludes_the_embedding(self, store):
        """384 floats per audit row would dwarf the fact being audited, and the
        vector is derived from content anyway."""
        memory = await store.write("u1", "semantic", "x", embedding=vec(4))
        after = (await store.audit(memory.id))[0]["after"]
        assert "embedding" not in after


class TestSupersession:
    async def test_supersede_closes_the_old_window(self, store):
        old = await store.write("u1", "semantic", "prefers sync", embedding=vec(5))
        new = await store.write(
            "u1", "semantic", "prefers async", embedding=vec(6), supersedes=old.id
        )
        live = await store.live("u1", "semantic")
        assert [m.id for m in live] == [new.id]

        history = await store.history(old.id)
        assert history[0].valid_to is not None

    async def test_the_old_row_records_what_replaced_it(self, store, pool):
        old = await store.write("u1", "semantic", "a", embedding=vec(7))
        new = await store.write(
            "u1", "semantic", "b", embedding=vec(8), supersedes=old.id
        )
        superseded_by = await pool.fetchval(
            "SELECT superseded_by FROM memories WHERE id = $1", old.id
        )
        assert superseded_by == new.id

    async def test_supersession_audits_against_the_superseded_id(self, store):
        """Reading one memory's trail must show it was replaced, without
        searching the whole table for its successor."""
        old = await store.write("u1", "semantic", "a", embedding=vec(9))
        await store.write("u1", "semantic", "b", embedding=vec(10), supersedes=old.id)
        assert [row["op"] for row in await store.audit(old.id)] == ["supersede", "create"]

    async def test_superseding_an_unknown_memory_raises(self, store):
        with pytest.raises(MemoryNotFound):
            await store.write("u1", "semantic", "b", embedding=vec(11), supersedes="nope")

    async def test_a_failed_supersede_writes_nothing(self, store):
        """The window close and the insert are one transaction. A partial commit
        would leave a memory whose predecessor is still live."""
        with pytest.raises(MemoryNotFound):
            await store.write("u1", "semantic", "b", embedding=vec(12), supersedes="nope")
        assert await store.live("u1") == []

    async def test_content_is_never_edited(self, store):
        """ADR 4. The superseded row keeps its original text forever."""
        old = await store.write("u1", "semantic", "prefers sync", embedding=vec(13))
        await store.write(
            "u1", "semantic", "prefers async", embedding=vec(14), supersedes=old.id
        )
        assert (await store.history(old.id))[0].content == "prefers sync"


class TestRevert:
    async def test_revert_appends_rather_than_rolling_back(self, store):
        """§3.4 — a destructive rollback would erase the evidence that a revert
        happened, which is the one thing the audit trail must show."""
        original = await store.write("u1", "semantic", "good fact", embedding=vec(15))
        await store.revert(original.id, 1, actor="arsalon")
        history = await store.history(original.id)
        assert [m.revision for m in history] == [1, 2]
        assert history[1].content == "good fact"

    async def test_revert_is_attributed_to_a_human(self, store):
        original = await store.write("u1", "semantic", "x", embedding=vec(16))
        reverted = await store.revert(original.id, 1, actor="arsalon")
        assert reverted.created_by == "human"
        trail = await store.audit(original.id)
        assert trail[0]["op"] == "revert"
        assert trail[0]["actor"] == "arsalon"

    async def test_only_one_revision_stays_live(self, store):
        """§4.1's invariant, held in application code rather than by constraint."""
        original = await store.write("u1", "semantic", "x", embedding=vec(17))
        await store.revert(original.id, 1, actor="a")
        await store.revert(original.id, 1, actor="b")
        live = await store.live("u1", "semantic")
        assert len(live) == 1
        assert live[0].revision == 3

    async def test_revert_preserves_the_embedding(self, store):
        """A reverted memory must still be findable. Re-embedding on revert
        would need the model in the request path and could drift."""
        original = await store.write("u1", "semantic", "findable", embedding=vec(18))
        await store.revert(original.id, 1, actor="a")
        found = await store.search("u1", vec(18), "semantic", 5)
        assert [m.id for m in found] == [original.id]

    async def test_unknown_memory_raises_not_found(self, store):
        with pytest.raises(MemoryNotFound):
            await store.revert("nope", 1, actor="a")

    async def test_unknown_revision_raises_revision_not_found(self, store):
        """§11.3 distinguishes these: 404 versus 400."""
        original = await store.write("u1", "semantic", "x", embedding=vec(19))
        with pytest.raises(RevisionNotFound):
            await store.revert(original.id, 99, actor="a")


class TestSearch:
    async def test_only_live_memories_are_returned(self, store):
        old = await store.write("u1", "semantic", "a", embedding=vec(20))
        await store.write("u1", "semantic", "b", embedding=vec(20), supersedes=old.id)
        found = await store.search("u1", vec(20), "semantic", 10)
        assert [m.content for m in found] == ["b"]

    async def test_scoped_to_one_user(self, store):
        await store.write("u1", "semantic", "mine", embedding=vec(21))
        await store.write("u2", "semantic", "theirs", embedding=vec(21))
        found = await store.search("u1", vec(21), "semantic", 10)
        assert [m.content for m in found] == ["mine"]

    async def test_scoped_to_one_type(self, store):
        await store.write("u1", "semantic", "sem", embedding=vec(22))
        await store.write("u1", "episodic", "epi", embedding=vec(22))
        found = await store.search("u1", vec(22), "episodic", 10)
        assert [m.content for m in found] == ["epi"]

    async def test_similarity_orders_the_result(self, store):
        await store.write("u1", "semantic", "near", embedding=vec(23))
        await store.write("u1", "semantic", "far", embedding=vec(300))
        found = await store.search("u1", vec(23), "semantic", 10)
        assert found[0].content == "near"

    async def test_recency_can_outrank_similarity(self, store, pool):
        """The reason decay is in SQL and not applied to an already-fetched
        page: a fresher memory has to be able to *enter* the top-k (§8.4)."""
        stale = await store.write("u1", "episodic", "stale", embedding=vec(24))
        await pool.execute(
            "UPDATE memories SET valid_from = now() - interval '400 days'"
            " WHERE id = $1",
            stale.id,
        )
        # Less similar to the query (cosine ~0.707 against `stale`'s 1.0) but
        # written today, so a 30-day half-life should still promote it.
        await store.write("u1", "episodic", "fresh", embedding=partial(24, 25))
        found = await store.search(
            "u1", vec(24), "episodic", 1, recency_halflife_days=30.0
        )
        assert [m.content for m in found] == ["fresh"]

    async def test_without_decay_similarity_wins(self, store, pool):
        """The control for the test above: same two rows, no half-life, and the
        more similar one comes back. Without this, that test would also pass if
        decay were applied unconditionally."""
        stale = await store.write("u1", "episodic", "stale", embedding=vec(24))
        await pool.execute(
            "UPDATE memories SET valid_from = now() - interval '400 days'"
            " WHERE id = $1",
            stale.id,
        )
        await store.write("u1", "episodic", "fresh", embedding=partial(24, 25))
        found = await store.search("u1", vec(24), "episodic", 1)
        assert [m.content for m in found] == ["stale"]

    async def test_confidence_weighting_is_opt_in(self, store):
        """Semantic ranks by similarity * effective_confidence; episodic does
        not weight by confidence at all (§8.4)."""
        await store.write("u1", "semantic", "unsure", embedding=vec(26), confidence=0.1)
        await store.write("u1", "semantic", "sure", embedding=vec(26), confidence=1.0)
        weighted = await store.search(
            "u1", vec(26), "semantic", 1, confidence_weighted=True
        )
        assert [m.content for m in weighted] == ["sure"]

    async def test_k_bounds_the_result(self, store):
        for n in range(5):
            await store.write("u1", "semantic", f"m{n}", embedding=vec(27))
        assert len(await store.search("u1", vec(27), "semantic", 3)) == 3

    async def test_stored_confidence_is_returned_undecayed(self, store):
        """§8.3 decay is the Manager's to apply on read. A store that returned a
        decayed number would make a derived value indistinguishable from the
        stored one."""
        await store.write("u1", "semantic", "x", embedding=vec(28), confidence=0.9)
        found = await store.search("u1", vec(28), "semantic", 1)
        assert found[0].confidence == pytest.approx(0.9)


class TestHistory:
    async def test_oldest_first(self, store):
        original = await store.write("u1", "semantic", "x", embedding=vec(29))
        await store.revert(original.id, 1, actor="a")
        assert [m.revision for m in await store.history(original.id)] == [1, 2]

    async def test_unknown_memory_raises(self, store):
        with pytest.raises(MemoryNotFound):
            await store.history("nope")


class TestEmbeddings:
    async def test_returns_live_vectors_for_the_profile(self, store):
        await store.write("u1", "semantic", "a", embedding=vec(30))
        await store.write("u1", "semantic", "b", embedding=vec(31))
        vectors = await store.embeddings("u1", "semantic")
        assert vectors.shape == (2, DIM)

    async def test_empty_for_a_new_user(self, store):
        assert (await store.embeddings("nobody", "semantic")).size == 0

    async def test_excludes_superseded_rows(self, store):
        old = await store.write("u1", "semantic", "a", embedding=vec(32))
        await store.write("u1", "semantic", "b", embedding=vec(33), supersedes=old.id)
        assert (await store.embeddings("u1", "semantic")).shape == (1, DIM)


class TestLoadsAtSessionStart:
    """M3's exit criterion (PRD §8), against a real database.

    The manager's policy is unit-tested against a fake store in
    `test_memory_manager.py`. What this adds is the seam: real rows, real
    pgvector ranking, real `valid_from` timestamps, and the block that comes
    out the other side. A fake store cannot fail the way a `::vector` cast or
    a decay expression in SQL can.
    """

    async def test_all_three_types_load_within_budget(self, store):
        from app.memory.manager import MemoryManager

        class Embedder:
            model_id, dim = "test", DIM

            def embed(self, texts):
                return np.stack([vec(hash(t) % DIM) for t in texts])

            def embed_query(self, text):
                return self.embed([text])[0]

        embedder = Embedder()
        query = "what moved on auth"
        for mem_type, content in (
            ("semantic", "works in async FastAPI codebases"),
            ("episodic", "asked about auth PRs last Tuesday"),
            ("procedural", "for 'why did X change', search issues before commits"),
        ):
            await store.write(
                "u1",
                mem_type,
                content,
                embedding=embedder.embed([query])[0],
                confidence=0.9,
            )

        block = await MemoryManager(store, embedder).load_context(
            "u1", "sess_1", query, 2000
        )

        assert {m.mem_type for m in block.memories} == {
            "semantic",
            "episodic",
            "procedural",
        }
        assert 0 < block.token_count <= 2000
        assert block.truncated == 0
        assert "[procedural]" in block.text

    async def test_a_user_never_sees_another_users_memories(self, store):
        """The one failure here that is not recoverable by editing a row."""
        from app.memory.manager import MemoryManager

        class Embedder:
            model_id, dim = "test", DIM

            def embed(self, texts):
                return np.stack([vec(1) for _ in texts])

            def embed_query(self, text):
                return self.embed([text])[0]

        await store.write("them", "semantic", "their private fact", embedding=vec(1))
        block = await MemoryManager(store, Embedder()).load_context(
            "me", "sess_1", "anything", 2000
        )
        assert block.memories == []
        assert "private" not in block.text


class TestGovernanceEndpointSurface:
    """PRD §5.5 — provenance on every write, and one-action revert.

    These exercise the store methods `/memory`, `/memory/{id}/history`, and
    `/memory/{id}/revert` are built on. The HTTP layer is a thin projection
    (`app/main.py`); what has to be true is that the data behind it supports
    the governance claim, which is a database property and not a routing one.
    """

    async def test_a_reverted_memory_exposes_its_whole_history(self, store):
        original = await store.write("u1", "semantic", "good", embedding=vec(40))
        await store.write(
            "u1", "semantic", "bad", embedding=vec(41), supersedes=original.id
        )
        # Reverting the superseded memory brings its content back as a new
        # revision without erasing that it was superseded.
        reverted = await store.revert(original.id, 1, actor="arsalon")
        history = await store.history(original.id)

        assert [m.revision for m in history] == [1, 2]
        assert reverted.content == "good"
        assert [row["op"] for row in await store.audit(original.id)] == [
            "revert",
            "supersede",
            "create",
        ]

    async def test_the_audit_trail_names_who_did_what(self, store):
        """An audit row with no actor cannot answer the only question anyone
        asks of one."""
        original = await store.write("u1", "semantic", "x", embedding=vec(42))
        await store.revert(original.id, 1, actor="arsalon")
        trail = await store.audit(original.id)
        assert {row["actor"] for row in trail} == {"agent", "arsalon"}

    async def test_listing_is_scoped_and_filterable_by_type(self, store):
        await store.write("u1", "semantic", "sem", embedding=vec(43))
        await store.write("u1", "episodic", "epi", embedding=vec(44))
        await store.write("u2", "semantic", "theirs", embedding=vec(45))

        assert len(await store.live("u1")) == 2
        assert [m.content for m in await store.live("u1", "semantic")] == ["sem"]
        assert [m.content for m in await store.live("u2")] == ["theirs"]

    async def test_superseded_memories_are_absent_from_the_live_list(self, store):
        """They remain queryable by id and revertable — §8.3's floor removes a
        memory from *loading*, not from existence."""
        old = await store.write("u1", "semantic", "old", embedding=vec(46))
        await store.write("u1", "semantic", "new", embedding=vec(47), supersedes=old.id)
        assert [m.content for m in await store.live("u1")] == ["new"]
        assert len(await store.history(old.id)) == 1
