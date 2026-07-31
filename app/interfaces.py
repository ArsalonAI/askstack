"""The contracts every subsystem codes against — TRD §3.

Nothing here has an implementation, and nothing here imports a concrete class
from another subsystem. That is the whole point of the module: `app/facts`
must never need to import from `app/retrieval` to know what a `Chunk` is.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

import numpy as np

MemType = Literal["episodic", "semantic", "procedural"]
CreatedBy = Literal["extraction", "consolidation", "agent", "human"]
Source = Literal["docs", "code", "issue"]
EntityKind = Literal["pr", "commit", "issue", "release"]


@dataclass(frozen=True)
class Chunk:
    id: str
    source: Source
    path: str
    anchor: str  # heading, symbol, or comment ref
    content: str
    citation: str  # e.g. "code:fastapi/routing.py:L280-L340"
    score: float = 0.0


@dataclass(frozen=True)
class Entity:
    kind: EntityKind
    ref: str  # PR/issue number, commit sha, release tag
    title: str
    author: str
    state: str  # merged | open | closed | published
    at: datetime  # merged_at, closed_at, authored_at, published_at
    citation: str  # e.g. "pr:1234"
    url: str


@dataclass(frozen=True)
class Aggregate:
    """A set answer. `count` is computed in code and never restated by the model."""

    entities: Sequence[Entity]
    count: int
    window: tuple[datetime, datetime] | None
    area: str | None
    rendered: str  # deterministic markdown; handed to the model verbatim


ToolResult = list[Chunk] | Aggregate | Entity


@dataclass(frozen=True)
class Memory:
    id: str
    user_id: str
    mem_type: MemType
    content: str
    entities: Sequence[str]
    confidence: float
    revision: int
    valid_from: datetime
    valid_to: datetime | None
    created_by: CreatedBy
    source_session_id: str | None
    source_ids: Sequence[str]
    trace_id: str | None


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    input_schema: dict
    server: str
    is_synthetic: bool
    score: float = 0.0


@dataclass(frozen=True)
class MemoryBlock:
    text: str
    memories: Sequence[Memory]
    token_count: int
    truncated: int  # how many candidates the budget dropped


@dataclass(frozen=True)
class ConsolidationReport:
    """TRD §8.2 step 5. Not typed in §3, but §8.2 fixes its fields."""

    clusters_formed: int
    memories_written: int
    memories_superseded: int
    facts_skipped: int


class Embedder(Protocol):
    """The swap point for ADR 2. Nothing outside this protocol may reference
    `sentence_transformers`."""

    model_id: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """(n, dim) float32, L2-normalized. Deterministic for identical input."""

    def embed_query(self, text: str) -> np.ndarray:
        """(dim,) float32. Separate method because some models prefix queries.

        `bge-small-en-v1.5` needs the prefix "Represent this sentence for
        searching relevant passages: ". Getting it backwards silently costs
        several points of recall, which is why this is a separate method
        rather than a boolean flag.
        """


class Retriever(Protocol):
    async def search(
        self,
        query: str,
        k: int,
        sources: Sequence[Source] | None = None,
        trace: object | None = None,
    ) -> list[Chunk]:
        """`trace` is the §12 span parent for the `retrieve.*` sub-spans.

        Optional and defaulted: §14.4 requires retrieval metrics to be
        reproducible with no network at all, so an implementation must run
        without an observability backend attached.
        """


class FactsStore(Protocol):
    """The delivery record. Every method is an indexed SQL query — no
    embeddings, no ranking, no model."""

    async def merged_prs(
        self,
        since: datetime,
        until: datetime,
        area: str | None = None,
    ) -> Aggregate: ...

    async def open_issues(
        self,
        label: str | None = None,
        milestone: str | None = None,
        older_than_days: int | None = None,
        *,
        as_of: datetime | None = None,
    ) -> Aggregate: ...

    async def stale_prs(
        self, threshold_days: int, *, as_of: datetime | None = None
    ) -> Aggregate:
        """Open, non-draft, no review activity within the threshold.

        `as_of` is not in TRD §3.3 and needs to be: an age filter measured
        against wall-clock `now()` returns a different set every day, and
        PRD §7.1 requires every golden question to resolve identically against
        one pinned revision.
        """

    async def commits_by_author(
        self,
        since: datetime,
        area: str | None = None,
    ) -> Aggregate: ...

    async def entity(self, kind: EntityKind, ref: str) -> Entity | None:
        """Single lookup. Backs PRD §5.2 verification — 'is this actually merged'."""

    async def release_diff(self, from_tag: str, to_tag: str) -> Aggregate: ...


class MemoryStore(Protocol):
    async def write(
        self,
        user_id: str,
        mem_type: MemType,
        content: str,
        *,
        entities: Sequence[str],
        confidence: float,
        created_by: CreatedBy,
        source_session_id: str | None,
        source_ids: Sequence[str],
        trace_id: str | None,
        supersedes: str | None = None,
    ) -> Memory:
        """Append a new revision. Never mutates an existing row.
        If `supersedes` is set, closes that row's validity window."""

    async def search(
        self,
        user_id: str,
        query_vec: np.ndarray,
        mem_type: MemType,
        k: int,
        *,
        recency_halflife_days: float | None = None,
    ) -> list[Memory]: ...

    async def history(self, memory_id: str) -> list[Memory]:
        """All revisions, oldest first."""

    async def revert(self, memory_id: str, to_revision: int, *, actor: str) -> Memory:
        """Append a new revision whose content equals `to_revision`.
        Writes a memory_audit row. Never deletes."""


class MemoryManager(Protocol):
    async def load_context(
        self,
        user_id: str,
        session_id: str,
        query: str,
        budget_tokens: int,
    ) -> MemoryBlock: ...

    async def record(self, session_id: str, statement: str, **kw) -> Memory:
        """Backs the agent-facing memory_write tool."""


class ToolSelector(Protocol):
    """Three implementations behind one protocol — this is ablation axis C."""

    mode: Literal["semantic", "native", "full"]

    async def select(self, query: str, k: int) -> list[ToolDef]:
        """Returns tools to place in the request's tools[]."""

    def extra_request_params(self) -> dict:
        """Mode-specific additions, e.g. NativeToolSelector's tool-search tool
        and defer_loading flags."""


class Extractor(Protocol):
    async def extract(self, session_id: str) -> list[Memory]: ...


class Consolidator(Protocol):
    async def consolidate(self, user_id: str) -> ConsolidationReport: ...
