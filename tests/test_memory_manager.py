"""What loads at session start — TRD §8.3, §8.4, §8.5.

These run against a fake store rather than Postgres. The store's own guarantees
are tested in `test_memory_store.py`; what is under test here is policy —
which memories are chosen, how the budget is split, what the block says, and
what the model is told about provenance. None of that needs a database, and
tying it to one would make the decay assertions depend on `now()`.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from app.interfaces import Memory
from app.memory.manager import (
    BUDGET_SPLIT,
    MIN_EFFECTIVE_CONFIDENCE,
    MemoryManager,
    effective_confidence,
    estimate_tokens,
    render_memory,
)

DIM = 384
NOW = datetime(2026, 8, 26, tzinfo=UTC)


def memory(
    content: str = "a fact",
    *,
    mem_type: str = "semantic",
    confidence: float = 1.0,
    age_days: float = 0.0,
    source_ids: tuple[str, ...] = (),
    session: str | None = None,
    memory_id: str = "mem_1",
) -> Memory:
    return Memory(
        id=memory_id,
        user_id="u1",
        mem_type=mem_type,
        content=content,
        entities=(),
        confidence=confidence,
        revision=1,
        valid_from=NOW - timedelta(days=age_days),
        valid_to=None,
        created_by="extraction",
        source_session_id=session,
        source_ids=source_ids,
        trace_id=None,
    )


class FakeEmbedder:
    model_id = "fake"
    dim = DIM

    def embed(self, texts):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            out[row, hash(text) % DIM] = 1.0
        return out

    def embed_query(self, text):
        return self.embed([text])[0]


class FakeStore:
    """Returns whatever it was handed, per type, in the order given.

    Ranking is the store's job and is tested there. What matters here is that
    the Manager consumes rank order faithfully and asks for the right arms.
    """

    def __init__(self, by_type=None, profile=None):
        self.by_type = by_type or {}
        self.profile = profile if profile is not None else np.empty((0, 0))
        self.calls: list[dict] = []
        self.written: list[dict] = []

    async def search(self, user_id, query_vec, mem_type, k, **kw):
        self.calls.append({"type": mem_type, "k": k, "query_vec": query_vec, **kw})
        return list(self.by_type.get(mem_type, []))[:k]

    async def embeddings(self, user_id, mem_type):
        return self.profile

    async def write(self, user_id, mem_type, content, **kw):
        self.written.append({"user_id": user_id, "type": mem_type, "content": content, **kw})
        return memory(content, mem_type=mem_type)


def manager(store=None, **kw) -> MemoryManager:
    return MemoryManager(store or FakeStore(), FakeEmbedder(), **kw)


class TestEffectiveConfidence:
    """§8.3 — decay applied on read, never persisted."""

    def test_fresh_memory_keeps_its_confidence(self):
        assert effective_confidence(memory(confidence=0.9), now=NOW) == pytest.approx(0.9)

    def test_one_half_life_halves_it(self):
        got = effective_confidence(memory(confidence=0.8, age_days=180), now=NOW)
        assert got == pytest.approx(0.4)

    def test_two_half_lives_quarter_it(self):
        got = effective_confidence(memory(confidence=0.8, age_days=360), now=NOW)
        assert got == pytest.approx(0.2)

    def test_a_future_timestamp_does_not_amplify(self):
        """Clock skew between writer and reader must not manufacture confidence
        above what was stored."""
        got = effective_confidence(memory(confidence=0.5, age_days=-30), now=NOW)
        assert got == pytest.approx(0.5)


class TestRenderMemory:
    """§8.5 — provenance is rendered, not hidden (PRD §5.5)."""

    def test_semantic_shows_decayed_confidence_and_source_count(self):
        line = render_memory(
            memory("works in async codebases", source_ids=("a", "b", "c")), now=NOW
        )
        assert line == "[semantic · conf 1.0 · from 3 sessions] works in async codebases"

    def test_semantic_singular_session(self):
        assert "from 1 session]" in render_memory(memory(source_ids=("a",)), now=NOW)

    def test_semantic_shows_the_decayed_value_not_the_stored_one(self):
        """A model shown `conf 0.9` on a year-old memory would trust a number
        the system itself no longer believes."""
        line = render_memory(memory(confidence=0.9, age_days=180), now=NOW)
        assert "conf 0.5" in line

    def test_episodic_shows_date_and_session(self):
        line = render_memory(
            memory("asked about Depends()", mem_type="episodic", session="sess_a91"),
            now=NOW,
        )
        assert line == "[episodic · 2026-08-26 · sess_a91] asked about Depends()"

    def test_procedural_has_no_decay_or_date(self):
        line = render_memory(memory("search issues first", mem_type="procedural"), now=NOW)
        assert line == "[procedural] search issues first"


class TestLoadContext:
    async def test_all_three_types_load(self):
        """M3's exit criterion, in one assertion."""
        store = FakeStore(
            {
                "semantic": [memory("s", memory_id="m1")],
                "episodic": [memory("e", mem_type="episodic", memory_id="m2")],
                "procedural": [memory("p", mem_type="procedural", memory_id="m3")],
            }
        )
        block = await manager(store).load_context("u1", "sess", "query", 2000, now=NOW)
        assert {m.mem_type for m in block.memories} == {
            "semantic",
            "episodic",
            "procedural",
        }

    async def test_each_type_gets_its_own_share_of_the_budget(self):
        """50/30/20 (§8.4). One ranked pool would let a run of recent episodic
        facts crowd out the standing profile entirely."""
        long_fact = "x" * 4000
        store = FakeStore(
            {t: [memory(long_fact, mem_type=t, memory_id=f"m{t}")] for t in BUDGET_SPLIT}
        )
        block = await manager(store).load_context("u1", "sess", "q", 2000, now=NOW)
        # Only semantic's 50% share (1000 tokens) fits a ~1143-token fact... it
        # does not, so nothing loads and all three are counted as truncated.
        assert block.memories == []
        assert block.truncated == 3

    async def test_budget_is_respected(self):
        facts = [memory("y" * 200, memory_id=f"m{n}") for n in range(20)]
        block = await manager(FakeStore({"semantic": facts})).load_context(
            "u1", "sess", "q", 400, now=NOW
        )
        assert block.token_count <= 400

    async def test_truncated_counts_what_the_budget_dropped(self):
        """§8.4 — the UI shows "8 more not loaded" rather than pretending the
        block is complete."""
        facts = [memory("y" * 400, memory_id=f"m{n}") for n in range(10)]
        block = await manager(FakeStore({"semantic": facts})).load_context(
            "u1", "sess", "q", 1000, now=NOW
        )
        assert block.truncated == 10 - len(block.memories)
        assert block.truncated > 0

    async def test_rank_order_is_preserved_not_reordered_by_length(self):
        """A greedy fill that skipped to a shorter memory further down would
        respect the budget and quietly prefer short memories to relevant ones."""
        facts = [
            memory("z" * 900, memory_id="first"),
            memory("tiny", memory_id="second"),
        ]
        block = await manager(FakeStore({"semantic": facts})).load_context(
            "u1", "sess", "q", 300, now=NOW
        )
        assert block.memories == []

    async def test_low_effective_confidence_is_not_loaded(self):
        """§8.3 — below 0.3 it stops being loaded, but is never deleted."""
        faded = memory(confidence=0.5, age_days=720, memory_id="faded")
        assert effective_confidence(faded, now=NOW) < MIN_EFFECTIVE_CONFIDENCE
        block = await manager(FakeStore({"semantic": [faded]})).load_context(
            "u1", "sess", "q", 2000, now=NOW
        )
        assert block.memories == []

    async def test_the_floor_applies_only_to_semantic(self):
        """Episodic rows carry their own 30-day ranking decay and are not
        filtered by the confidence floor — an old episodic fact is still the
        record of what was asked."""
        old = memory(
            "asked in March", mem_type="episodic", confidence=0.5, age_days=720
        )
        block = await manager(FakeStore({"episodic": [old]})).load_context(
            "u1", "sess", "q", 2000, now=NOW
        )
        assert [m.content for m in block.memories] == ["asked in March"]

    async def test_empty_memory_is_an_empty_block_not_a_crash(self):
        block = await manager().load_context("u1", "sess", "q", 2000, now=NOW)
        assert block.text == ""
        assert block.token_count == 0
        assert block.truncated == 0

    async def test_block_carries_the_preamble(self):
        store = FakeStore({"semantic": [memory("a fact")]})
        block = await manager(store).load_context("u1", "sess", "q", 2000, now=NOW)
        assert "re-verifying it against the tools" in block.text
        assert "[semantic · conf 1.0] a fact" in block.text


class TestArms:
    """§8.4's table: each type gets its own query, k, and decay."""

    async def test_the_three_arms_use_the_specified_k(self):
        store = FakeStore()
        await manager(store).load_context("u1", "sess", "q", 2000, now=NOW)
        by_type = {c["type"]: c for c in store.calls}
        assert by_type["semantic"]["k"] == 20
        assert by_type["episodic"]["k"] == 20
        assert by_type["procedural"]["k"] == 5

    async def test_semantic_is_confidence_weighted_and_episodic_is_not(self):
        store = FakeStore()
        await manager(store).load_context("u1", "sess", "q", 2000, now=NOW)
        by_type = {c["type"]: c for c in store.calls}
        assert by_type["semantic"]["confidence_weighted"] is True
        assert by_type["semantic"]["recency_halflife_days"] == 180.0
        assert by_type["episodic"]["recency_halflife_days"] == 30.0
        assert by_type["episodic"].get("confidence_weighted", False) is False

    async def test_procedural_does_not_decay(self):
        """A recipe does not become less true with age."""
        store = FakeStore()
        await manager(store).load_context("u1", "sess", "q", 2000, now=NOW)
        procedural = next(c for c in store.calls if c["type"] == "procedural")
        assert procedural.get("recency_halflife_days") is None


class TestProfileVector:
    async def test_cold_start_uses_the_query_vector_unchanged(self):
        """A first session has no profile. That is correct behaviour, not a
        special case to guard."""
        store = FakeStore(profile=np.empty((0, 0)))
        mgr = manager(store)
        await mgr.load_context("u1", "sess", "q", 2000, now=NOW)
        semantic = next(c for c in store.calls if c["type"] == "semantic")
        assert np.allclose(semantic["query_vec"], FakeEmbedder().embed_query("q"))

    async def test_profile_blends_into_the_semantic_arm_only(self):
        profile = np.zeros((1, DIM), dtype=np.float32)
        profile[0, 7] = 1.0
        store = FakeStore(profile=profile)
        mgr = manager(store)
        await mgr.load_context("u1", "sess", "q", 2000, now=NOW)
        by_type = {c["type"]: c for c in store.calls}
        assert by_type["semantic"]["query_vec"][7] > 0
        # The episodic arm searches the raw query — §8.4 blends only semantic.
        assert by_type["episodic"]["query_vec"][7] == 0

    async def test_the_blended_vector_is_renormalised(self):
        """The arms are compared by cosine, and the mean of two unit vectors is
        not one."""
        profile = np.zeros((1, DIM), dtype=np.float32)
        profile[0, 7] = 1.0
        store = FakeStore(profile=profile)
        mgr = manager(store)
        await mgr.load_context("u1", "sess", "q", 2000, now=NOW)
        semantic = next(c for c in store.calls if c["type"] == "semantic")
        assert np.linalg.norm(semantic["query_vec"]) == pytest.approx(1.0)


class TestTokenCounting:
    """§8.5 — counted, not estimated. An estimate 15% low turns a budget into a
    suggestion, and the failure is silent."""

    async def test_uses_the_api_when_a_client_is_available(self):
        class Counter:
            def __init__(self):
                self.calls = 0

            class _Messages:
                def __init__(self, outer):
                    self.outer = outer

                async def count_tokens(self, **kw):
                    self.outer.calls += 1
                    return type("R", (), {"input_tokens": 42})()

            @property
            def messages(self):
                return self._Messages(self)

        client = Counter()
        store = FakeStore({"semantic": [memory("a fact")]})
        block = await manager(store, client=client, model="claude-opus-5").load_context(
            "u1", "sess", "q", 2000, now=NOW
        )
        assert block.token_count == 42
        assert client.calls == 1

    async def test_a_counting_failure_does_not_fail_the_turn(self):
        class Broken:
            @property
            def messages(self):
                class M:
                    async def count_tokens(self, **kw):
                        raise RuntimeError("api down")

                return M()

        store = FakeStore({"semantic": [memory("a fact")]})
        block = await manager(store, client=Broken(), model="m").load_context(
            "u1", "sess", "q", 2000, now=NOW
        )
        assert block.token_count > 0

    async def test_the_real_count_trims_an_overshooting_block(self):
        """The estimate packs; the count decides. If the estimate ran low, the
        block is trimmed rather than shipped over budget."""

        class Liar:
            @property
            def messages(self):
                class M:
                    async def count_tokens(self, *, messages, **kw):
                        # Report 10x, so anything packed is over budget.
                        return type("R", (), {"input_tokens": len(messages[0]["content"]) * 10})()

                return M()

        facts = [memory("short", memory_id=f"m{n}") for n in range(3)]
        mgr = manager(FakeStore({"semantic": facts}), client=Liar(), model="m")
        result = await mgr.load_context("u1", "sess", "q", 2000, now=NOW)
        assert result.memories == []
        assert result.truncated >= 3

    def test_the_estimate_is_pessimistic(self):
        """Erring high leaves the block under budget when the real count
        arrives; erring low overshoots silently."""
        text = "a" * 400
        assert estimate_tokens(text) > len(text) / 4


class TestRecord:
    async def test_write_carries_session_and_trace(self):
        """ADR 5's autonomy is only defensible because ADR 4 makes the write
        attributable. These two fields are what make it so."""
        store = FakeStore()
        await manager(store).record(
            "sess_1", "they only care about auth PRs", user_id="u1", trace_id="trace_x"
        )
        written = store.written[0]
        assert written["source_session_id"] == "sess_1"
        assert written["trace_id"] == "trace_x"
        assert written["created_by"] == "agent"

    async def test_a_write_invalidates_the_cached_profile(self):
        """The profile is cached per session (§8.4); a write during the session
        changes what the next load should blend in."""
        store = FakeStore(profile=np.zeros((1, DIM), dtype=np.float32))
        mgr = manager(store)
        await mgr._profile_vector("u1")
        assert "u1" in mgr._profile_cache
        await mgr.record("sess", "a fact", user_id="u1")
        assert "u1" not in mgr._profile_cache
