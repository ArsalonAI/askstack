"""What loads at session start, and what it costs — TRD §8.3, §8.4, §8.5.

This module decides *which* memories reach the prompt and enforces the budget.
It owns no persistence (that is `store.py`) and no extraction (that is M4).

The load is three independent queries, one per memory type, each with its own
ranking rule and its own slice of the budget. They are independent on purpose:
a single ranked pool would let a run of recent episodic facts crowd out every
semantic memory, and the standing profile is the half that makes session two
cheaper than session one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np

from app.interfaces import CreatedBy, Memory, MemoryBlock, MemType

# §8.4. Semantic gets half because it is the session-independent half — the
# part that makes a returning user cheaper than a new one.
BUDGET_SPLIT: dict[MemType, float] = {
    "semantic": 0.50,
    "episodic": 0.30,
    "procedural": 0.20,
}

# §8.3 / §8.4. Semantic facts are standing knowledge and decay slowly;
# episodic ones are situational and decay six times faster. Procedural recipes
# do not decay at all — "search issues before git log" does not become less
# true with age.
CONFIDENCE_HALFLIFE_DAYS = 180.0
EPISODIC_HALFLIFE_DAYS = 30.0

# Below this, a memory stops being loaded. It is not deleted and stays
# queryable and revertable — §8.3.
MIN_EFFECTIVE_CONFIDENCE = 0.3

CANDIDATES_PER_TYPE: dict[MemType, int] = {
    "semantic": 20,
    "episodic": 20,
    "procedural": 5,
}

BLOCK_PREAMBLE = (
    "What you already know about this user and their work, from earlier "
    "sessions. Provenance is shown so you can weigh it: a low-confidence or "
    "old memory is a lead, not a fact. Never restate a status claim from "
    "memory without re-verifying it against the tools — memory records what "
    "was true when it was written, not what is true now."
)


def effective_confidence(memory: Memory, *, now: datetime | None = None) -> float:
    """§8.3, applied on read so it never mutates a row.

    A stored confidence is what the writer believed at `valid_from`. What the
    model should act on is that belief discounted by how long ago it was held,
    and computing it here rather than persisting it means the decay curve can
    change without a migration and without rewriting history.
    """
    now = now or datetime.now(UTC)
    age_days = max((now - memory.valid_from).total_seconds() / 86400.0, 0.0)
    return memory.confidence * 0.5 ** (age_days / CONFIDENCE_HALFLIFE_DAYS)


def render_memory(memory: Memory, *, now: datetime | None = None) -> str:
    """One line of the §8.5 block.

    Provenance is rendered, not hidden. A model handed unlabelled assertions
    treats all of them as ground truth; one that can see `conf 0.4` and a date
    four months old can discount it, which is the difference between memory
    that helps and memory that poisons (PRD §5.5).
    """
    if memory.mem_type == "semantic":
        sessions = len(memory.source_ids)
        provenance = f"semantic · conf {effective_confidence(memory, now=now):.1f}"
        if sessions:
            plural = "s" if sessions != 1 else ""
            provenance += f" · from {sessions} session{plural}"
    elif memory.mem_type == "episodic":
        provenance = f"episodic · {memory.valid_from.date().isoformat()}"
        if memory.source_session_id:
            provenance += f" · {memory.source_session_id}"
    else:
        provenance = "procedural"
    return f"[{provenance}] {memory.content}"


def estimate_tokens(text: str) -> int:
    """A deliberately *pessimistic* character heuristic.

    English runs about 4 characters per token; 3.5 overshoots. That direction
    is chosen on purpose — this estimate does the packing, so erring high
    leaves the block under budget when the real count arrives, whereas erring
    low would silently overshoot and only be caught by the trim loop.
    """
    return math.ceil(len(text) / 3.5)


def _pack(
    candidates: Sequence[Memory], budget: int, *, now: datetime
) -> tuple[list[Memory], int]:
    """Add in rank order until the per-type budget is spent.

    Stops at the first memory that does not fit rather than skipping ahead to a
    shorter one further down. Rank order carries the entire ranking signal, and
    a greedy fill that reordered by length would quietly prefer short memories
    to relevant ones — the budget would be respected and the block would be
    worse, with nothing in the output to say why.
    """
    kept: list[Memory] = []
    spent = 0
    for index, memory in enumerate(candidates):
        cost = estimate_tokens(render_memory(memory, now=now))
        if spent + cost > budget:
            return kept, len(candidates) - index
        kept.append(memory)
        spent += cost
    return kept, 0


class MemoryManager:
    """§3.5. Loads the block; backs the agent-facing `memory_write` tool."""

    def __init__(self, store, embedder, client=None, *, model: str | None = None) -> None:
        self.store = store
        self.embedder = embedder
        # Only used to count tokens. `None` falls back to an estimate and says
        # so — see `_count_tokens`.
        self.client = client
        self.model = model
        self._profile_cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ load

    async def load_context(
        self,
        user_id: str,
        session_id: str,
        query: str,
        budget_tokens: int,
        *,
        now: datetime | None = None,
    ) -> MemoryBlock:
        now = now or datetime.now(UTC)
        query_vec = self.embedder.embed_query(query)

        selected: list[Memory] = []
        truncated = 0
        for mem_type, share in BUDGET_SPLIT.items():
            candidates = await self._candidates(user_id, mem_type, query_vec, now=now)
            kept, dropped = _pack(candidates, int(budget_tokens * share), now=now)
            selected.extend(kept)
            truncated += dropped

        # §8.5: the estimate packs, the real count decides. Packing with
        # `count_tokens` per candidate would be up to 45 API calls per session
        # for a number that only has to be right about the assembled block —
        # so the cheap estimate does the packing and one count verifies it.
        text = self._render_block(selected, now=now)
        token_count = await self._count_tokens(text)
        while selected and token_count > budget_tokens:
            # The estimate is deliberately pessimistic, so this is a guard
            # rather than the normal path. Trims the lowest-ranked memory of
            # the last type, which is the cheapest thing to lose.
            selected.pop()
            truncated += 1
            text = self._render_block(selected, now=now)
            token_count = await self._count_tokens(text)

        return MemoryBlock(
            text=text,
            memories=selected,
            token_count=token_count,
            truncated=truncated,
        )

    async def _candidates(
        self, user_id: str, mem_type: MemType, query_vec: np.ndarray, *, now: datetime
    ) -> list[Memory]:
        """One type's ranked candidates, per the §8.4 table."""
        if mem_type == "semantic":
            memories = await self.store.search(
                user_id,
                await self._blended_query(user_id, query_vec),
                "semantic",
                CANDIDATES_PER_TYPE["semantic"],
                recency_halflife_days=CONFIDENCE_HALFLIFE_DAYS,
                confidence_weighted=True,
            )
            # §8.3's floor. Applied after ranking rather than as a SQL filter
            # because the threshold is on the *derived* value, and pushing a
            # decay formula into a WHERE clause puts it in two places.
            return [
                m for m in memories
                if effective_confidence(m, now=now) >= MIN_EFFECTIVE_CONFIDENCE
            ]
        if mem_type == "episodic":
            return await self.store.search(
                user_id,
                query_vec,
                "episodic",
                CANDIDATES_PER_TYPE["episodic"],
                recency_halflife_days=EPISODIC_HALFLIFE_DAYS,
            )
        return await self.store.search(
            user_id, query_vec, "procedural", CANDIDATES_PER_TYPE["procedural"]
        )

    async def _blended_query(self, user_id: str, query_vec: np.ndarray) -> np.ndarray:
        """§8.4: `mean(embed(query), user_profile_vector)`.

        The profile vector pulls the semantic arm toward what this user
        habitually cares about, so a terse query still surfaces standing
        context. With no profile yet — a first session — this is the query
        vector unchanged, which is the correct cold-start behaviour rather than
        a special case.
        """
        profile = await self._profile_vector(user_id)
        if profile is None:
            return query_vec
        blended = (query_vec + profile) / 2.0
        norm = np.linalg.norm(blended)
        # Re-normalise: the arms are compared by cosine, and the mean of two
        # unit vectors is not one.
        return blended / norm if norm else query_vec

    async def _profile_vector(self, user_id: str) -> np.ndarray | None:
        """Mean embedding of the user's live semantic memories, cached per
        session (§8.4) — it changes only when consolidation writes."""
        if user_id in self._profile_cache:
            cached = self._profile_cache[user_id]
            return cached if cached.size else None
        vectors = await self.store.embeddings(user_id, "semantic")
        profile = vectors.mean(axis=0) if vectors.size else np.empty(0, dtype=np.float32)
        if profile.size:
            norm = np.linalg.norm(profile)
            if norm:
                profile = profile / norm
        self._profile_cache[user_id] = profile
        return profile if profile.size else None

    def _render_block(self, memories: Sequence[Memory], *, now: datetime) -> str:
        if not memories:
            return ""
        lines = [render_memory(m, now=now) for m in memories]
        return f"{BLOCK_PREAMBLE}\n\n" + "\n".join(lines)

    async def _count_tokens(self, text: str) -> int:
        """§8.5: counted, not estimated.

        An estimate that runs 15% low turns a budget into a suggestion, and the
        failure is silent — the block simply grows. When no client is available
        (unit tests, offline evals) this falls back to a deliberately
        *pessimistic* character heuristic, so an untested path overshoots the
        budget rather than blowing through it.
        """
        if not text:
            return 0
        if self.client is None or self.model is None:
            return estimate_tokens(text)
        try:
            counted = await self.client.messages.count_tokens(
                model=self.model, messages=[{"role": "user", "content": text}]
            )
            return counted.input_tokens
        except Exception:  # noqa: BLE001 — a budget is not worth a failed turn
            return estimate_tokens(text)

    # ----------------------------------------------------------------- write

    async def record(
        self,
        session_id: str,
        statement: str,
        *,
        user_id: str,
        mem_type: MemType = "episodic",
        entities: Sequence[str] = (),
        confidence: float = 0.8,
        created_by: CreatedBy = "agent",
        trace_id: str | None = None,
        supersedes: str | None = None,
    ) -> Memory:
        """Backs the agent-facing `memory_write` tool (§3.5).

        Writes go through without an approval gate — ADR 5 — which is only
        defensible because ADR 4 makes them attributable and reversible. Every
        field that makes that true is required here rather than optional:
        `source_session_id` and `trace_id` are what tie a memory back to the
        request that produced it.
        """
        memory = await self.store.write(
            user_id,
            mem_type,
            statement,
            embedding=self.embedder.embed([statement])[0],
            entities=entities,
            confidence=confidence,
            created_by=created_by,
            source_session_id=session_id,
            source_ids=[],
            trace_id=trace_id,
            supersedes=supersedes,
        )
        # A write changes the profile the next load blends in.
        self._profile_cache.pop(user_id, None)
        return memory

    async def search(
        self, user_id: str, query: str, *, mem_type: MemType | None = None, k: int = 5
    ) -> list[Memory]:
        """Backs the agent-facing `memory_search` tool."""
        query_vec = self.embedder.embed_query(query)
        types: tuple[MemType, ...] = (
            (mem_type,) if mem_type else ("semantic", "episodic", "procedural")
        )
        found: list[Memory] = []
        for one in types:
            found.extend(await self.store.search(user_id, query_vec, one, k))
        return found[:k] if mem_type else found
