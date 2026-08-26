# TRD — askstack

Technical design for the system defined in [`PRD.md`](./PRD.md).

**Status:** Draft · **Last updated:** 2026-07-28

---

## 1. Overview

The PRD states what must be true; this document states how it is made true. Every parameter value, schema, interface, and threshold lives here — the PRD carries none.

**Scope of this document:** system architecture, component interfaces, data model, ingest, retrieval, tool selection, memory lifecycle, prompt assembly, Claude API usage, HTTP/SSE contract, observability, non-functional budgets, eval harness, security, and the decision log.

**The organising idea.** The PRD's six question classes (§5.1 there) split into two kinds. Classes 1–4 — status, change over time, ownership, blockers — have *exactly checkable* answers that are facts about the repository. Classes 5–6 — decision archaeology, scope — are interpretive. That split runs through this entire document: two retrieval substrates (§6), two tool result shapes (§3), two citation kinds (§5.1), and two scorers (§14).

**Out of scope:** product rationale, success criteria, milestones, and risk framing — all in the PRD.

### Glossary

| Term | Meaning |
|---|---|
| **Chunk** | An indexed unit of corpus text with a stable citation ID |
| **Entity** | A first-class repository object — pull request, commit, issue, release |
| **Facts layer** | The relational tables holding entities, queried with SQL rather than embeddings |
| **Area** | A named region of the codebase (`auth`, `routing`), mapped to path globs by a curated file |
| **Aggregate** | A tool result that is a set of entities plus a count, rather than text |
| **Episodic memory** | A structured fact scoped to one past session |
| **Semantic memory** | Session-independent knowledge, produced by consolidating episodic facts |
| **Procedural memory** | Tool definitions and learned task recipes |
| **Memory block** | The token-bounded, provenance-annotated text injected at session start |
| **Revision** | One immutable version of a memory row; memories are append-only |
| **Config hash** | SHA-256 over the ablation flag set, identifying an eval cell |

---

## 2. System architecture

```
                  ┌─────────────────────────────────────────────┐
  POST /chat ────▶│              Orchestrator                   │
                  │        (agent loop, Claude tool runner)     │
                  └──┬──────────┬────────────┬──────────┬───────┘
                     │          │            │          │
           ┌─────────▼─┐  ┌─────▼─────┐ ┌────▼─────┐ ┌──▼────────┐
           │  Memory   │  │  Hybrid   │ │  Facts   │ │  Tool     │
           │  Manager  │  │ Retriever │ │  Store   │ │  Registry │
           └─────┬─────┘  └─────┬─────┘ └────┬─────┘ └──┬────────┘
                 │              │            │          │
                 └──────┬───────┴────────────┴──────────┘
                        ▼
            Postgres 16 + pgvector + tsvector
   ┌────────────────────────────┬────────────────────────────┐
   │  semantic index            │  facts layer               │
   │  chunks · memories         │  pull_requests · commits   │
   │  tool_defs                 │  issues · releases · areas │
   └────────────────────────────┴────────────────────────────┘
                        │
                        ▼
             Langfuse (traces, cost, latency)
```

Two retrieval substrates, one database. The **semantic index** answers interpretive questions by similarity; the **facts layer** answers delivery questions by SQL. Both are populated by the same ingest pass at the same `CORPUS_REF`, so they can never disagree about what exists — a split that drifted would let the agent cite a pull request the facts layer says was never merged.

### 2.1 Request path

```
POST /chat
  ├─ 1. resolve or create session
  ├─ 2. memory.load        → MemoryBlock (≤ budget tokens)     [skipped if MEMORY_ENABLED=false]
  ├─ 3. tools.select       → list[ToolDef]                      [mode per TOOL_RETRIEVAL_MODE]
  ├─ 4. agent loop (Claude tool runner)
  │     ├─ retrieve.dense ∥ retrieve.sparse → retrieve.fuse     [semantic path]
  │     ├─ retrieve.structured                                   [facts path]
  │     ├─ llm.generate    → stream tokens as SSE
  │     ├─ tool.call.<name>
  │     └─ memory.write    (agent-initiated, optional)
  ├─ 5. persist messages
  └─ 6. memory.extract     (background task; end of session or every 10 turns)
```

Steps 2 and 3 are sequential because tool selection embeds the same turn text but does not depend on memory content. Step 4's dense and sparse arms run concurrently via `asyncio.gather`.

**There is no query router.** Which path runs is decided by which tool the model calls, and that is decided by `tools.select`. This adds no component and turns the tool-retrieval thesis into a load-bearing claim rather than a demonstration: if semantic tool retrieval cannot reliably surface `merged_prs` for "what shipped last month", the selection design is wrong and the eval will say so. A separate classifier would be a second thing to train, tune, and drift — see ADR 14.

### 2.2 Component responsibilities

| Component | Module | Owns | Does not own |
|---|---|---|---|
| Orchestrator | `app/orchestrator.py` | Agent loop, SSE emission, session lifecycle | Retrieval logic, memory policy |
| Memory Manager | `app/memory/manager.py` | Which memories load, budget enforcement, block rendering | Memory persistence, extraction |
| Memory Store | `app/memory/store.py` | Versioned persistence, audit rows, revert | What is worth remembering |
| Extractor / Consolidator | `app/memory/lifecycle.py` | Transcript → episodic, episodic → semantic | Runtime loading |
| Hybrid Retriever | `app/retrieval/hybrid.py` | Dense + sparse + fusion over `chunks` | Chunking, embedding model choice |
| Facts Store | `app/facts/store.py` | SQL over entities; date, state, author, area filters | Ranking, relevance |
| Area Resolver | `app/facts/areas.py` | Area name → path globs, from `areas.yaml` | Deciding what an area *should* contain |
| Embedder | `app/retrieval/embedder.py` | Text → vector, batching, model identity | Storage |
| Tool Registry | `app/tools/registry.py` | Tool definitions, execution dispatch, aggregate rendering | Selection strategy |
| Tool Selector | `app/tools/selector.py` | Which tools enter `tools[]` | Tool implementations |

### 2.3 Deployment

Single-process `uvicorn` ASGI app. Batch jobs (`ingest`, `consolidate`) are separate CLI entrypoints under `scripts/`. Postgres and Langfuse run via `docker-compose.yml`. The embedding model loads once at process start into a module-level singleton — it is ~130 MB resident and thread-safe for inference.

---

## 3. Component interfaces

These are the contracts implementation codes against. Defined in `app/interfaces.py` so no module imports a concrete class from another subsystem.

```python
from typing import Protocol, Literal, Sequence
from dataclasses import dataclass
from datetime import datetime
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
    anchor: str            # heading, symbol, or comment ref
    content: str
    citation: str          # e.g. "code:fastapi/routing.py:L280-L340"
    score: float = 0.0


@dataclass(frozen=True)
class Entity:
    kind: EntityKind
    ref: str               # PR/issue number, commit sha, release tag
    title: str
    author: str
    state: str             # merged | open | closed | published
    at: datetime           # merged_at, closed_at, authored_at, published_at
    citation: str          # e.g. "pr:1234"
    url: str


@dataclass(frozen=True)
class Aggregate:
    """A set answer. `count` is computed in code and never restated by the model."""
    entities: Sequence[Entity]
    count: int
    window: tuple[datetime, datetime] | None
    area: str | None
    rendered: str          # deterministic markdown; handed to the model verbatim


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
    truncated: int          # how many candidates the budget dropped
```

### 3.1 `Embedder`

The swap point for ADR 2. Nothing outside this protocol may reference `sentence_transformers`.

```python
class Embedder(Protocol):
    model_id: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """(n, dim) float32, L2-normalized. Deterministic for identical input."""

    def embed_query(self, text: str) -> np.ndarray:
        """(dim,) float32. Separate method because some models prefix queries."""
```

`bge-small-en-v1.5` requires the query prefix `"Represent this sentence for searching relevant passages: "`; `embed_query` applies it and `embed` does not. Getting this backwards silently degrades recall by several points, which is why they are separate methods rather than a boolean flag.

### 3.2 `Retriever`

```python
class Retriever(Protocol):
    async def search(
        self,
        query: str,
        k: int,
        sources: Sequence[Source] | None = None,
    ) -> list[Chunk]: ...


class HybridRetriever(Retriever):
    """Dense ∥ sparse → RRF. Degrades to dense-only when hybrid is disabled."""
    def __init__(self, embedder: Embedder, pool: asyncpg.Pool, hybrid: bool = True): ...
```

### 3.3 `FactsStore`

The delivery record. Every method is an indexed SQL query — no embeddings, no ranking, no model.

```python
class FactsStore(Protocol):
    async def merged_prs(
        self, since: datetime, until: datetime, area: str | None = None,
    ) -> Aggregate: ...

    async def open_issues(
        self, label: str | None = None, milestone: str | None = None,
        older_than_days: int | None = None, *, as_of: datetime | None = None,
    ) -> Aggregate: ...

    async def stale_prs(
        self, threshold_days: int, *, as_of: datetime | None = None,
    ) -> Aggregate:
        """Open, non-draft, no review activity within the threshold."""

    async def commits_by_author(
        self, since: datetime, area: str | None = None,
    ) -> Aggregate: ...

    async def entity(self, kind: EntityKind, ref: str) -> Entity | None:
        """Single lookup. Backs §5.2 verification — 'is this actually merged'."""

    async def release_diff(self, from_tag: str, to_tag: str) -> Aggregate: ...
```

**Every age-relative method takes `as_of`.** "Stale for more than 14 days" measured against wall-clock `now()` returns a different set every day, which makes a class-4 golden question unreproducible and shows up in the eval as a retrieval regression rather than as clock drift. This is the "date-anchored answers rot" risk in PRD §9; the parameter is the mitigation, and the eval runner always passes the question's `as_of`.

**Two filters this corpus cannot exercise.** FastAPI sets no GitHub milestones (0 of 2,325 pull requests, 0 of 3,542 issues) and carries exactly one open issue, having moved support traffic to Discussions years ago. `milestone=` and most `open_issues` shapes are therefore correct code with no data behind them here. Following the precedent PRD §4 sets for sprints, they are excluded from the golden set by construction rather than left to fail silently — class-4 questions lean on stale pull requests and labels instead.

`entity()` is the smallest and most important method here. It is what turns "a design document says we support websockets" into "…and the pull request that implemented it merged on 2026-03-14", which is the PRD §5.2 requirement in one call.

### 3.4 `MemoryStore`

```python
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
        self, user_id: str, query_vec: np.ndarray, mem_type: MemType, k: int,
        *, recency_halflife_days: float | None = None,
    ) -> list[Memory]: ...

    async def history(self, memory_id: str) -> list[Memory]:
        """All revisions, oldest first."""

    async def revert(self, memory_id: str, to_revision: int, *, actor: str) -> Memory:
        """Append a new revision whose content equals `to_revision`.
        Writes a memory_audit row. Never deletes."""
```

Revert appends rather than rolls back. The audit trail must show that a revert happened, which a destructive rollback would erase.

### 3.5 `MemoryManager`

```python
class MemoryManager(Protocol):
    async def load_context(
        self, user_id: str, session_id: str, query: str, budget_tokens: int,
    ) -> MemoryBlock: ...

    async def record(self, session_id: str, statement: str, **kw) -> Memory:
        """Backs the agent-facing memory_write tool."""
```

### 3.6 `ToolSelector`

Three implementations behind one protocol — this is ablation axis C.

```python
class ToolSelector(Protocol):
    mode: Literal["semantic", "native", "full"]

    async def select(self, query: str, k: int) -> list[ToolDef]:
        """Returns tools to place in the request's tools[]."""

    def extra_request_params(self) -> dict:
        """Mode-specific additions, e.g. NativeToolSelector's tool-search tool
        and defer_loading flags."""
```

| Implementation | `select` returns | `extra_request_params` |
|---|---|---|
| `SemanticToolSelector` | top-k by cosine, above floor | `{}` |
| `NativeToolSelector` | **all** tools, each with `defer_loading: True` | adds `tool_search_tool_bm25_20251119` |
| `FullToolSelector` | all tools | `{}` |

`NativeToolSelector` returns everything because Anthropic's server-side tool search does the filtering; `defer_loading` keeps the definitions out of context until the model searches for them. At least one tool must remain non-deferred or the API rejects the request — the selector leaves `search_docs` loaded as the anchor.

### 3.7 Lifecycle jobs

```python
class Extractor(Protocol):
    async def extract(self, session_id: str) -> list[Memory]: ...

class Consolidator(Protocol):
    async def consolidate(self, user_id: str) -> ConsolidationReport: ...
```

---

## 4. Data model

Postgres 16 with `vector` and `pg_trgm`. Migrations via Alembic in `scripts/migrations/`. `scripts/init_db.sql` only creates extensions — it is a bootstrap, superseded by migration `0001`, which lands at M0 because M0's exit criteria is an ingested corpus and ingest needs tables.

```sql
-- ---------------------------------------------------------------- corpus
CREATE TABLE chunks (
    id            TEXT PRIMARY KEY,              -- = citation, stable across re-ingest
    source        TEXT NOT NULL CHECK (source IN ('docs','code','issue')),
    path          TEXT NOT NULL,
    anchor        TEXT NOT NULL,
    content       TEXT NOT NULL,
    content_sha   TEXT NOT NULL,
    token_count   INT  NOT NULL,
    embedding     VECTOR(384) NOT NULL,
    tsv           TSVECTOR NOT NULL,
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX chunks_embedding_idx ON chunks
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX chunks_tsv_idx    ON chunks USING gin (tsv);
CREATE INDEX chunks_source_idx ON chunks (source);          -- source filter pushdown
CREATE INDEX chunks_sha_idx    ON chunks (content_sha);     -- ingest delta detection

-- ---------------------------------------------------------------- facts layer
-- Entities, not text. Every column here exists because some EM question
-- filters or aggregates on it.

CREATE TABLE pull_requests (
    number      INT PRIMARY KEY,
    title       TEXT NOT NULL,
    body        TEXT,
    state       TEXT NOT NULL CHECK (state IN ('open','merged','closed')),
    is_draft    BOOLEAN NOT NULL DEFAULT FALSE,
    author      TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL,
    merged_at   TIMESTAMPTZ,               -- NULL unless state = 'merged'
    closed_at   TIMESTAMPTZ,
    milestone   TEXT,
    additions   INT NOT NULL DEFAULT 0,
    deletions   INT NOT NULL DEFAULT 0,
    url         TEXT NOT NULL
);
-- Partial index: "what merged in a window" is the single hottest query, and
-- merged PRs are a minority of the table.
CREATE INDEX pr_merged_at_idx ON pull_requests (merged_at DESC)
    WHERE state = 'merged';
CREATE INDEX pr_state_idx     ON pull_requests (state, created_at DESC);
CREATE INDEX pr_author_idx    ON pull_requests (author, created_at DESC);
CREATE INDEX pr_milestone_idx ON pull_requests (milestone) WHERE milestone IS NOT NULL;

CREATE TABLE pr_files (
    pr_number  INT NOT NULL REFERENCES pull_requests(number) ON DELETE CASCADE,
    path       TEXT NOT NULL,
    additions  INT NOT NULL DEFAULT 0,
    deletions  INT NOT NULL DEFAULT 0,
    PRIMARY KEY (pr_number, path)
);
CREATE INDEX pr_files_path_idx ON pr_files (path text_pattern_ops);  -- area glob prefix match

CREATE TABLE pr_reviews (
    pr_number    INT NOT NULL REFERENCES pull_requests(number) ON DELETE CASCADE,
    reviewer     TEXT NOT NULL,
    state        TEXT NOT NULL,            -- approved | changes_requested | commented
    submitted_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (pr_number, reviewer, submitted_at)
);
CREATE INDEX pr_reviews_recent_idx ON pr_reviews (pr_number, submitted_at DESC);

CREATE TABLE commits (
    sha         TEXT PRIMARY KEY,
    author      TEXT NOT NULL,
    authored_at TIMESTAMPTZ NOT NULL,
    message     TEXT NOT NULL,
    pr_number   INT REFERENCES pull_requests(number) ON DELETE SET NULL
);
CREATE INDEX commits_author_idx ON commits (author, authored_at DESC);
CREATE INDEX commits_date_idx   ON commits (authored_at DESC);

CREATE TABLE commit_files (
    sha   TEXT NOT NULL REFERENCES commits(sha) ON DELETE CASCADE,
    path  TEXT NOT NULL,
    PRIMARY KEY (sha, path)
);
CREATE INDEX commit_files_path_idx ON commit_files (path text_pattern_ops);

CREATE TABLE issues (
    number        INT PRIMARY KEY,
    title         TEXT NOT NULL,
    body          TEXT,
    state         TEXT NOT NULL CHECK (state IN ('open','closed')),
    author        TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL,
    closed_at     TIMESTAMPTZ,
    closed_by_pr  INT REFERENCES pull_requests(number) ON DELETE SET NULL,
    milestone     TEXT,
    url           TEXT NOT NULL
);
CREATE INDEX issues_state_idx     ON issues (state, created_at);
CREATE INDEX issues_milestone_idx ON issues (milestone) WHERE milestone IS NOT NULL;

CREATE TABLE issue_labels (
    issue_number INT NOT NULL REFERENCES issues(number) ON DELETE CASCADE,
    label        TEXT NOT NULL,
    PRIMARY KEY (issue_number, label)
);
CREATE INDEX issue_labels_label_idx ON issue_labels (label);

CREATE TABLE releases (
    tag          TEXT PRIMARY KEY,
    name         TEXT,
    published_at TIMESTAMPTZ NOT NULL,
    body         TEXT,
    url          TEXT NOT NULL
);
CREATE INDEX releases_date_idx ON releases (published_at DESC);

-- Curated, not derived. See §5.5.
CREATE TABLE areas (
    name       TEXT PRIMARY KEY,
    path_globs TEXT[] NOT NULL,
    description TEXT
);

-- ---------------------------------------------------------------- sessions
CREATE TABLE sessions (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at    TIMESTAMPTZ,
    metadata    JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX sessions_user_idx ON sessions (user_id, started_at DESC);

CREATE TABLE messages (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn        INT  NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('user','assistant','tool')),
    content     JSONB NOT NULL,                  -- Anthropic content blocks, verbatim
    trace_id    TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (session_id, turn, role)
);
CREATE INDEX messages_session_idx ON messages (session_id, turn);

-- ---------------------------------------------------------------- memory
CREATE TABLE memories (
    id                TEXT NOT NULL,             -- stable across revisions
    revision          INT  NOT NULL,
    user_id           TEXT NOT NULL,
    mem_type          TEXT NOT NULL CHECK (mem_type IN ('episodic','semantic','procedural')),
    content           TEXT NOT NULL,
    embedding         VECTOR(384) NOT NULL,
    entities          TEXT[] NOT NULL DEFAULT '{}',
    confidence        REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    valid_from        TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_to          TIMESTAMPTZ,               -- NULL = live
    superseded_by     TEXT,                      -- memories.id that replaced this
    created_by        TEXT NOT NULL CHECK (created_by IN ('extraction','consolidation','agent','human')),
    source_session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
    source_ids        TEXT[] NOT NULL DEFAULT '{}',  -- episodic ids, message ids
    trace_id          TEXT,
    PRIMARY KEY (id, revision)
);

-- The hot path: live memories for one user. Partial index keeps it small
-- even as revision history grows without bound.
CREATE INDEX memories_live_idx ON memories (user_id, mem_type)
    WHERE valid_to IS NULL;
CREATE INDEX memories_embedding_idx ON memories
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX memories_entities_idx ON memories USING gin (entities);
CREATE INDEX memories_session_idx  ON memories (source_session_id);

CREATE TABLE memory_audit (
    id          BIGSERIAL PRIMARY KEY,
    memory_id   TEXT NOT NULL,
    revision    INT  NOT NULL,
    op          TEXT NOT NULL CHECK (op IN ('create','supersede','revert')),
    before      JSONB,
    after       JSONB NOT NULL,
    actor       TEXT NOT NULL,
    trace_id    TEXT,
    at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX memory_audit_memory_idx ON memory_audit (memory_id, at DESC);
CREATE INDEX memory_audit_trace_idx  ON memory_audit (trace_id);

-- ---------------------------------------------------------------- tools
CREATE TABLE tool_defs (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    description   TEXT NOT NULL,
    input_schema  JSONB NOT NULL,
    server        TEXT NOT NULL,
    is_synthetic  BOOLEAN NOT NULL DEFAULT FALSE,
    embedding     VECTOR(384) NOT NULL
);
CREATE INDEX tool_defs_embedding_idx ON tool_defs
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX tool_defs_synthetic_idx ON tool_defs (is_synthetic);

-- ---------------------------------------------------------------- ingest
-- The completion marker §4.2 depends on. `completed_at` is set only once BOTH
-- the facts layer and the semantic index have been written; a partial ingest
-- leaves it NULL and the service refuses to start.
CREATE TABLE ingest_runs (
    id              TEXT PRIMARY KEY,
    corpus_repo     TEXT NOT NULL,
    corpus_ref      TEXT NOT NULL,        -- symbolic ref as requested
    resolved_sha    TEXT NOT NULL,        -- the pinned revision actually ingested
    since           TIMESTAMPTZ,          -- ingest window floor; NULL = full history
    embedding_model TEXT NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ,          -- NULL = partial
    stats           JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX ingest_runs_complete_idx ON ingest_runs (completed_at DESC)
    WHERE completed_at IS NOT NULL;

-- ---------------------------------------------------------------- eval
CREATE TABLE eval_runs (
    id           TEXT PRIMARY KEY,
    config_hash  TEXT NOT NULL,
    git_sha      TEXT NOT NULL,
    metrics      JSONB NOT NULL,
    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at  TIMESTAMPTZ
);
CREATE INDEX eval_runs_config_idx ON eval_runs (config_hash, started_at DESC);
```

### 4.1 Versioning semantics

- `(id, revision)` is the primary key. A memory's identity is `id`; its history is the revision series.
- **Live** = `valid_to IS NULL`. Exactly one revision per `id` may be live; enforced in application code within the write transaction, not by constraint, because the window close and the new insert must be atomic together.
- **Supersession** (one memory replaces another, different `id`): set the old row's `valid_to` and `superseded_by`.
- **Revision** (same memory, updated content, same `id`): close revision *N*, insert *N+1*.
- **Revert to N**: insert revision *max+1* with revision *N*'s content, `created_by='human'`, and an audit row with `op='revert'`.

Nothing is ever deleted or updated in place. Storage cost is bounded by memory volume, which for a single-user demo is trivial; the attribution guarantee is not.

### 4.2 The facts layer is a projection, not a source of truth

Unlike `memories`, the entity tables are **fully rebuildable** from the GitHub API at a given `CORPUS_REF`. They carry no history and no audit trail, and ingest replaces rows rather than versioning them. This is deliberate: GitHub is the system of record, and duplicating its history here would create a second thing that can be stale in a different way from the first.

The practical consequence is that **the facts layer and the semantic index must be rebuilt together**. A run that refreshes `chunks` but not `pull_requests` produces an agent that can find the discussion of a feature but not confirm whether it shipped — the exact failure PRD §5.2 forbids. Ingest writes both in one transaction per entity batch and records a single completion marker; a partial ingest leaves the marker unset and the service refuses to start.

---

## 5. Corpus ingest

`scripts/ingest.py`, driven by `CORPUS_REPO` and `CORPUS_REF`.

### 5.1 Citation ID grammar

Citation IDs are the `chunks.id` primary key, so they must be stable across re-ingest or every re-run orphans the golden set.

There are two kinds. **Span citations** point at retrieved text and are `chunks.id`. **Entity citations** point at a repository object in the facts layer.

```
citation     := span_cite | entity_cite

span_cite    := docs_cite | code_cite | comment_cite | body_cite
docs_cite    := "docs:" path "#" slug             e.g. docs:advanced/events.md#lifespan
code_cite    := "code:" path ":L" start "-L" end  e.g. code:fastapi/routing.py:L280-L340
comment_cite := "issue:" number "#comment-" n     e.g. issue:1234#comment-5
body_cite    := "issue:" number "#body" ["-" n]   e.g. issue:1234#body

entity_cite  := pr_cite | commit_cite | issue_ref | release_cite
pr_cite      := "pr:" number                      e.g. pr:11234
commit_cite  := "commit:" short_sha               e.g. commit:a3f1c9d
issue_ref    := "issue:" number                   e.g. issue:1234
release_cite := "release:" tag                    e.g. release:0.110.0
```

`issue:1234` and `issue:1234#comment-5` are deliberately the same prefix: the bare form is the entity ("this issue is still open"), the fragment form is a span ("here is what Sebastián said about it"). The presence of a fragment is what disambiguates, so a citation parser must check for it before dispatching to the facts layer.

**The issue body is `issue:1234#body`, not the bare form.** An earlier draft gave the body chunk the bare `issue:1234`, which made one string mean both "this issue is still open" and "this paragraph of its description" — precisely the ambiguity the fragment rule exists to prevent. With `#body`, the bare form is *only ever* an entity citation, and every `chunks.id` carries a fragment.

**`docs:path#slug` is not unique on its own, and the ingest must make it so.** A heading slug repeats whenever a document reuses a heading — FastAPI's release notes carry a `### Docs` and a `### Fixes` under every one of 299 versions — and an oversized section emits several parts from one heading. Both cases allocate through a single per-document counter: the first chunk keeps the bare `docs:path#slug`, and each subsequent collision appends `-2`, `-3`, and so on. One allocator rather than two suffix schemes, because two schemes that must never overlap eventually do. Document order is fixed at a pinned `CORPUS_REF`, so the allocation is stable across re-ingest, which is the property this section actually requires.

Line numbers make `code_cite` unstable under upstream edits — accepted, because `CORPUS_REF` pins a ref. Ingest at a different ref invalidates code citations in the golden set; the golden set therefore records the ref it was authored against and the eval runner fails loudly on mismatch rather than silently scoring against shifted lines.

### 5.2 Chunking

**The chunk budget is the embedding model's window, not a round number.** `bge-small-en-v1.5` truncates at 512 tokens: anything longer is embedded from its opening and the remainder is silently discarded. An 800-token target therefore guaranteed loss — measured against the real corpus, 25.4% of chunks exceeded the window, and issue chunks were 42% over, the largest being 75,427 tokens embedded from its first 0.7%. The budget is `512 − 48` for the breadcrumb or header prefix each chunk carries, and **every** splitter overlaps by 96 tokens.

Two properties hold at every boundary. A split lands *between* whole units — paragraphs for prose, statements for code, comments for threads — so no sentence is ever cut mid-way. And where the trailing units fit inside the overlap budget, they are repeated at the head of the next chunk, so a fact that straddles a boundary stays retrievable from both sides.

| Source | Strategy |
|---|---|
| **Docs** | Parse Markdown to a heading tree. One chunk per leaf section; oversized sections split on paragraph boundaries with overlap; sections under 100 tokens merge into the next sibling, which keeps *its own* heading. Chunk text is prefixed with its heading breadcrumb (`# Advanced > Events > Lifespan`) so an isolated chunk retains context. MkDocs explicit anchors (`## Lifespan { #lifespan }`) are used verbatim as the citation slug — the braced anchor is what the published page renders, so a derived slug would point at an anchor that does not exist. |
| **Code** | `ast.parse` per file; one chunk per top-level function, class, and module-level assignment block, plus one per method. The chunk carries the enclosing class name and the full docstring. Oversized functions split with overlap and the `path :: anchor` header repeated on every part; each part reports its **real line range**, so a split function still yields honest citations. Files that fail to parse fall back to overlapping line windows. |
| **Issues** | One chunk per part of the issue body (title + labels + body), cited as `issue:N#body`; one per comment run of at most 5 comments *and* at most one window of tokens. Bot comments are dropped by author denylist. Closed issues only — open issues describe problems, not resolutions, and pollute an answer corpus. |

**Why oversized functions are no longer emitted whole.** The original rule reasoned that splitting a function body destroys what makes it retrievable. That holds against a naive split, but it loses to the embedder: a 4,000-token function emitted whole is *already* invisible past its opening 512 tokens. Splitting with overlap, and repeating the header on every part, keeps each piece both embeddable and identifiable.

### 5.3 Idempotency and delta detection

For each candidate chunk, compute `content_sha = sha256(content)`. Compare against the stored row by `id`:

- absent → insert, embed
- present and `content_sha` matches → skip entirely (no embed call)
- present and differs → re-embed, update in place
- stored but no longer produced → delete

Embedding dominates ingest wall time, so skipping unchanged chunks is what keeps a re-run under a minute instead of half an hour.

### 5.4 GitHub fetch

Four paginated REST endpoints, 100 per page, all cached to `.cache/gh/` keyed by object ID and `updated_at` so re-runs do not re-fetch:

| Endpoint | Populates | Notes |
|---|---|---|
| `/issues?state=all` | `issues`, `issue_labels`, and issue chunks | GitHub returns PRs from this endpoint too — filter on the `pull_request` key or the two tables collide |
| `/pulls?state=all` + `/pulls/{n}/files` + `/pulls/{n}/reviews` | `pull_requests`, `pr_files`, `pr_reviews` | Three calls per PR; this dominates ingest wall time, hence the cache |
| `/commits` + `/commits/{sha}` | `commits`, `commit_files` | The list endpoint omits `files`; only the detail call has them. Commits are immutable, so their cache key needs no version component |
| `/releases` | `releases` | Small, refetched whole each run |

Backoff on 403 secondary rate limits using the `Retry-After` header, and wait for `X-RateLimit-Reset` when the primary budget is spent — a first ingest runs close enough to the 5,000/hour ceiling that failing instead of waiting would leave the corpus half-built.

`GITHUB_TOKEN` needs **read-only access to public repositories and no permissions at all** — a fine-grained token with "Public Repositories (read-only)". The classic `public_repo` scope also works but grants *write* access to every public repository on GitHub, which contradicts the PRD §2 non-goal that askstack never writes to the repository.

Three details that are not obvious from the endpoint list:

- **`additions`/`deletions` are derived, not fetched.** The `/pulls` list omits them and only the per-PR detail call carries them, which would be a fourth call per PR. Summing the `/pulls/{n}/files` rows we already have gives identical numbers for free.
- **A commit finds its PR from its message.** FastAPI squash-merges as `Title (#1234)`, so `commits.pr_number` is parsed rather than looked up, avoiding a call per commit.
- **`issues.closed_by_pr` is parsed from PR bodies.** The timeline API reports it directly at the cost of one call per issue; the closing-keyword regex over bodies we already have is free.

#### Two window floors

`INGEST_SINCE` bounds pull requests and commits. `INGEST_ISSUES_SINCE` bounds issues, and defaults to unbounded. They differ because the two substrates want opposite things:

| Source | Window | Why |
|---|---|---|
| PRs, commits | recent (`2025-01-01`) | The facts layer answers "what shipped last month". Three calls per PR makes depth expensive |
| Issues | full history | Decision archaeology lives in the archive. FastAPI has 3,541 closed issues but only 89 created since 2025 — support traffic moved to Discussions years ago, so windowing issues the same way would starve PRD classes 5–6 |

The consequence is that an issue can be closed by a pull request outside the PR window. Ingest sets `closed_by_pr` only when the PR is actually present and **counts the links it dropped** into `ingest_runs.stats`. A silently dropped foreign key becomes an unexplainable eval gap weeks later.

**Open issues are fetched now.** The chunking rule in §5.2 still indexes only *closed* issues for semantic search — an open issue describes a problem, not a resolution, and pollutes an answer corpus. But open issues are entities in the facts layer, because "what's still open on the v2 milestone" is a class-4 question and requires them.

### 5.5 Area definitions

`areas.yaml`, checked into the repo, loaded into the `areas` table on every ingest:

```yaml
- name: auth
  description: Authentication, authorization, and security utilities
  path_globs:
    - fastapi/security/**
    - tests/test_security*
- name: routing
  description: Request routing, path operations, dependency resolution
  path_globs:
    - fastapi/routing.py
    - fastapi/dependencies/**
```

**This file is hand-authored and it is the weakest link in the ownership answers.** "The payments module" is a human concept with no reliable signal in the file tree; heuristics over directory names and import graphs produce mappings that are wrong in ways nobody notices until an ownership answer names the wrong person. A short curated file is at least honest about being curated — see ADR 16.

Two consequences follow. Every answer that filtered by area **names the globs it resolved**, so a bad mapping is visible in the output rather than silent. And an unrecognised area name is an explicit tool error, never an empty result — "no commits in the payments area" and "there is no area called payments" must not look the same to the manager.

---

## 6. Retrieval design

Two paths. §6.1–6.3 cover the semantic path over `chunks`; §6.4 covers the structured path over the facts layer.

**Why aggregate questions are not a retrieval problem.** "What shipped last month" has an exact answer — the set of pull requests with `state = 'merged'` and `merged_at` in range. Nearest-neighbour search over text returns documents that *discuss* shipping, ranked by similarity, with no notion of completeness. Even a perfect embedding model answers the wrong question. Recall is not approximately right here; it is either the whole set or a bug. This is why classes 1–4 route to SQL and why the structured path has no ranking, no top-k, and no scores — see ADR 13.

### 6.1 Dense arm

pgvector HNSW, cosine distance, 384-d `bge-small-en-v1.5`. Index built with `m=16, ef_construction=64`; queries set `hnsw.ef_search = 100` per session — the default of 40 measurably costs recall at this corpus size, and 100 stays well inside the latency budget.

```sql
SET LOCAL hnsw.ef_search = 100;
SELECT id, source, path, anchor, content, 1 - (embedding <=> $1) AS score
FROM chunks
WHERE ($2::text[] IS NULL OR source = ANY($2))
ORDER BY embedding <=> $1
LIMIT 50;
```

### 6.2 Sparse arm

`tsvector` with the `english` configuration, ranked by `ts_rank_cd`. Code identifiers are the problem: the default parser splits `get_current_user` into one token and stems it badly, and `HTTPException` never matches a query saying "http exception".

The ingest-time `tsv` column is therefore built from the content **plus a decomposed identifier stream**: for every `snake_case` and `CamelCase` identifier, append its parts as separate tokens. `get_current_user` contributes `get current user get_current_user`. This is done in Python at ingest, not in a custom Postgres parser — a custom parser would need a C extension and would not survive a managed-Postgres migration.

```sql
SELECT id, source, path, anchor, content,
       ts_rank_cd(tsv, plainto_tsquery('english', $1)) AS score
FROM chunks
WHERE tsv @@ plainto_tsquery('english', $1)
  AND ($2::text[] IS NULL OR source = ANY($2))
ORDER BY score DESC
LIMIT 50;
```

### 6.3 Fusion

Reciprocal Rank Fusion over the two ranked lists:

```
RRF(d) = Σ_arms  1 / (k + rank_arm(d)),   k = 60
```

`k=60` is the value from the original RRF paper and is not tuned — tuning it against the golden set would be fitting to the eval. Fusion consumes the top 50 from each arm and returns the top 10 to the model. A document present in only one arm still scores, so RRF never punishes a term-only or vector-only match.

Rank-based fusion is used rather than score normalization because cosine similarity and `ts_rank_cd` have no shared scale, and any normalization scheme would need per-corpus calibration. See ADR 3.

`HYBRID_ENABLED=false` skips the sparse arm and returns dense results directly — this is ablation axis A.

### 6.4 Structured path

Plain indexed SQL through `FactsStore` (§3.3). No embeddings, no ranking, no cutoff. Representative:

```sql
-- merged_prs(since, until, area)
SELECT p.number, p.title, p.author, p.merged_at, p.url
FROM pull_requests p
WHERE p.state = 'merged'
  AND p.merged_at >= $1 AND p.merged_at < $2
  AND ($3::text IS NULL OR EXISTS (
        SELECT 1 FROM pr_files f
        JOIN areas a ON a.name = $3
        WHERE f.pr_number = p.number
          AND f.path LIKE ANY (SELECT replace(g, '**', '%') FROM unnest(a.path_globs) g)
      ))
ORDER BY p.merged_at DESC;
```

Two properties that matter more than the SQL itself:

- **No `LIMIT`.** An aggregate answer is the complete set or it is wrong. If a window returns 400 pull requests, the tool returns the count plus a truncated *rendering* while the count stays exact — the model must never see a silently capped list and describe it as "the PRs that shipped".
- **Time expressions are resolved before the query, not inside it.** "Last month" is parsed to a concrete `[since, until)` against the session's `as_of` date and echoed back in the `Aggregate.window` field, so the answer can state the window it actually used. An unresolvable expression is a tool error, not a guess — see §17.

`STRUCTURED_ENABLED=false` disables this path. It is a debugging kill switch, **not an ablation axis**: without it, classes 1–4 are unanswerable, so an "off" arm measures nothing.

---

## 7. Tool retrieval design

### 7.1 Embedding text

A tool's embedding is computed from a rendered template, not from its raw JSON schema:

```
{name}
{description}
Parameters: {param_name}: {param_description}; ...
```

Raw JSON embeds structural noise (`"type": "object"`, `"required"`) that is identical across every tool and dilutes the signal. Parameter *descriptions* carry most of the discriminating information — `path: absolute path to a file in the corpus` is what makes `get_file` match "show me the source of routing.py".

### 7.2 Selection

```sql
SET LOCAL hnsw.ef_search = 100;
SELECT name, description, input_schema, server, is_synthetic,
       1 - (embedding <=> $1) AS score
FROM tool_defs
WHERE 1 - (embedding <=> $1) >= $2      -- TOOL_SIMILARITY_FLOOR, default 0.25
ORDER BY embedding <=> $1
LIMIT $3;                                -- TOOL_RETRIEVAL_K, default 5
```

The floor matters more than `k`. Without it, a query like "hi" returns five arbitrary tools with ~0.05 similarity and the model is invited to call one. With the floor, it returns zero and the model answers conversationally.

### 7.2.1 Aggregation tools

The catalog now spans two result shapes. Search tools return `list[Chunk]`; aggregation tools return `Aggregate` (§3).

| Tool | Backed by | Serves |
|---|---|---|
| `merged_prs(since, until, area?)` | `FactsStore.merged_prs` | class 2 |
| `open_issues(label?, milestone?, older_than_days?)` | `FactsStore.open_issues` | classes 1, 4 |
| `stale_prs(threshold_days)` | `FactsStore.stale_prs` | class 4 |
| `commits_by_author(since, area?)` | `FactsStore.commits_by_author` | class 3 |
| `pr_state(number)` / `issue_state(number)` / `release_info(tag)` | `FactsStore.entity` | classes 1, 6 |
| `release_diff(from_tag, to_tag)` | `FactsStore.release_diff` | class 2 |
| `search_docs` / `search_code` / `search_issues` | `HybridRetriever` | classes 5, 6 |

`release_info` was missing from this table until M2 and the golden set required it: two questions carry `gold_tools: [release_info]`, so tool-selection accuracy was scoring them against a tool no mode could ever surface. The golden set is frozen (PRD §7.1), which makes the catalog the side that moves. It is the third projection of `FactsStore.entity`, alongside the two that were already listed — "was 0.141.0 actually released" is the same single-entity lookup as "did PR 15806 ship".

`open_issues` is the mirror image: it is in the catalog and no golden question exercises it, because M0 found FastAPI has exactly one open issue and that question shape was excluded by construction (§14.1). It still ships — a catalog trimmed to what the eval happens to cover would be fitted to the eval.

**Counts are rendered in code and never generated by the model.** `Aggregate.rendered` is deterministic markdown built by the tool — the count, the window, the resolved area globs, and a bounded table of entities. The model narrates around it and must not restate a number it would have to compute itself. A model asked to summarise forty pull requests will confidently produce a wrong total, and a status tool that miscounts is worse than no tool at all: the manager has no way to tell. See ADR 15.

The system prompt states this constraint explicitly, and the eval's aggregate set-F1 (§14) is computed against the tool's own result rather than the prose, so a model that paraphrases the set inaccurately is caught by citation resolution rather than passing silently.

### 7.2.2 Always-injected tools

`memory_write` and `memory_search` are **always injected** regardless of score. They are agent-infrastructure tools whose relevance is never expressed in the user's query — the user never says "please save this to memory", so semantic retrieval would never surface them, and the write-back loop would silently never fire. This exemption is disclosed in every reported tool-accuracy number: accuracy is computed over the *retrieved* set excluding the two always-on tools.

**At M2 the always-injected set is empty.** Both tools are memory infrastructure and memory arrives at M3, so the M2 tool-accuracy denominator is the whole selected set with nothing excluded. The M3 baseline changes that denominator; §14.3's `milestone` field is what keeps the two from being compared as though they measured the same thing.

### 7.3 Modes

Set by `TOOL_RETRIEVAL_MODE` — ablation axis C.

| Mode | Behavior | Prompt cost |
|---|---|---|
| `semantic` | §7.2 | k definitions |
| `native` | All tools sent with `defer_loading: True`, plus `{"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"}`; the server filters. `search_docs` stays non-deferred (the API rejects a request where every tool is deferred). | names only, until searched |
| `full` | Every definition, every request. Control arm. | all definitions |

### 7.4 Synthetic padding

`scripts/gen_synthetic_tools.py` generates plausible tool definitions from adjacent domains (cloud infra, CI, observability, ticketing) to reach `TOOL_CATALOG_SIZE`. They carry `is_synthetic = TRUE`, are never dispatchable, and the registry raises if one is invoked.

Reporting rules, non-negotiable:

- Every tool-retrieval metric is published with the real/synthetic split alongside it.
- Tool-selection accuracy is additionally computed over **real tools only**, because a gold tool can only ever be real, and a large synthetic pool makes the task look harder than it is in a way that flatters the retriever.

---

## 8. Memory lifecycle

### 8.1 Extraction

Trigger: session end, or every 10 turns for long sessions. Runs as a FastAPI `BackgroundTask` so it never blocks a response.

One Claude call over the transcript with `output_config.format`:

```json
{
  "type": "json_schema",
  "schema": {
    "type": "object",
    "properties": {
      "facts": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "statement":  {"type": "string"},
            "entities":   {"type": "array", "items": {"type": "string"}},
            "kind":       {"type": "string", "enum": ["preference","resolution","failure","context"]},
            "confidence": {"type": "number"},
            "source_message_ids": {"type": "array", "items": {"type": "string"}}
          },
          "required": ["statement","entities","kind","confidence","source_message_ids"],
          "additionalProperties": false
        }
      }
    },
    "required": ["facts"],
    "additionalProperties": false
  }
}
```

Facts with `confidence < 0.5` are discarded. `statement` must be self-contained — the prompt instructs that it will be read without the conversation around it, because a fact reading "they said it doesn't work" is worse than useless in a later session.

**What is worth extracting for this user.** The prompt targets four things a manager's next session benefits from: the workstreams and areas they ask about repeatedly, the people they track, standing constraints they have stated ("I don't care about docs-only PRs"), and **the answer they were given along with its date**. That last one is what makes the flagship scenario work — *"what's changed since we last spoke"* resolves to `merged_prs(since=<prior session timestamp>)`, and the prior timestamp comes from episodic memory rather than being asked for.

Status facts are timestamped and treated as **claims about a moment**, never as standing truth. "The auth migration is three PRs from done" is recorded with its date and re-verified against the facts layer before being restated — otherwise memory becomes a cache of stale status, which is precisely the failure PRD §5.2 exists to prevent.

### 8.2 Consolidation

`scripts/consolidate.py`, nightly and via `POST /admin/consolidate`.

1. Load all live episodic memories for the user.
2. Cluster: agglomerative, cosine distance, average linkage, distance threshold 0.35, with a hard constraint that clustered facts share ≥1 entity. Embedding similarity alone happily merges "prefers async" with "prefers sync"; the entity constraint plus explicit contradiction handling below is what keeps that from becoming a silent corruption.
3. Clusters of ≥3 facts go to one Claude call (structured output) producing `{statement, entities, confidence, contradicts: [memory_id]}`.
4. For each result: write a semantic memory with `created_by='consolidation'` and `source_ids` = the cluster's episodic IDs. For each `contradicts` entry, close that memory's window and set `superseded_by`.
5. Emit a `ConsolidationReport`: clusters formed, memories written, memories superseded, facts skipped.

Episodic memories are **not** deleted after consolidation. They are the evidence for the semantic memory, and `source_ids` is only meaningful if the rows it points to still exist.

### 8.3 Confidence decay

Live semantic memories decay on read, not on a schedule:

```
effective_confidence = stored_confidence * 0.5 ** (age_days / 180)
```

Applied at load time in the Memory Manager, so it never mutates a row. A memory whose effective confidence drops below 0.3 stops being loaded but remains queryable and revertable. Reconsolidation with fresh evidence writes a new revision at full confidence.

### 8.4 Startup load

For each type, with `MEMORY_TOKEN_BUDGET` (default 2000) split 50/30/20 across semantic/episodic/procedural:

| Type | Query |
|---|---|
| Semantic | Cosine over `mean(embed(query), user_profile_vector)`, top 20 candidates, ranked by `similarity * effective_confidence` |
| Episodic | Cosine over `embed(query)`, top 20, ranked by `similarity * 0.5 ** (age_days / 30)` — a 30-day half-life, six times faster than semantic decay, because episodic facts are situational |
| Procedural (recipes) | Cosine, top 5, no decay |

`user_profile_vector` is the mean embedding of the user's live semantic memories, cached per session.

Candidates are added in rank order until the per-type budget is exhausted; `MemoryBlock.truncated` records the drop count so the UI can show "8 more memories not loaded" rather than pretending the block is complete.

### 8.5 Block template

```
[semantic · conf 0.9 · from 3 sessions] User works in async FastAPI codebases, Pydantic v2 idioms.
[episodic · 2026-07-20 · sess_a91] Asked about Depends() scoping with yield; resolved via sub-dependency caching.
[procedural] For "why did X change" questions: search issues first, then git log — docs lag behind.
```

Provenance is rendered, not hidden, so the model can discount a low-confidence or stale memory instead of treating every line as ground truth. Token count is verified with `client.messages.count_tokens` against the assembled block, not estimated — an estimate that runs 15% low turns a budget into a suggestion.

---

## 9. Prompt assembly and caching

Block order is load-bearing. Anything placed before a `cache_control` breakpoint invalidates that cache when it changes.

```
tools[]                    ← deterministic order (sorted by name)
system[0]  stable prompt   ← cache_control: {"type": "ephemeral"}
─────────────────────────────── cache boundary ───────────────────────────────
messages[0] user           ← memory block, rendered as a user-turn preamble
messages[1..] conversation
```

| Element | Volatility | Placement |
|---|---|---|
| Tool definitions | Per-query in `semantic` mode | Before the breakpoint — accepted cost; in `full` and `native` modes the set is constant and caches perfectly |
| System prompt | Never | Before the breakpoint, carries `cache_control` |
| Memory block | Per session | **After** the breakpoint |
| Conversation | Per turn | After |

The memory block deliberately does **not** go in the system prompt. It changes every session, and placing it there would invalidate the cached prefix on every new session — see ADR 8. In `semantic` mode the per-query tool set already limits cache hits within a session; the system prompt still caches across turns, which is where the volume is.

Opus 5's 512-token cache minimum (halved from 1024 on Opus 4.8) means the system prompt caches without padding.

A smoke test in `tests/test_caching.py` asserts `usage.cache_read_input_tokens > 0` on the second turn of a session. A caching claim nobody verifies is usually false.

---

## 10. Claude API usage

**Model:** `claude-opus-5` on every route. Cost is controlled with `output_config.effort`, not by downgrading model tier — see ADR 7.

| Route | Effort | Thinking | Structured output |
|---|---|---|---|
| Agent loop | `high` | adaptive (default) | no |
| Extraction | `low` | adaptive | yes |
| Consolidation | `low` | adaptive | yes |
| LLM judge (eval) | `low` | adaptive | yes |

**Thinking.** Adaptive is the default on Opus 5 — the `thinking` parameter is omitted. `max_tokens` is set to 16000 on non-streaming batch routes and 32000 on the streaming agent loop, sized so thinking plus response fits; `max_tokens` caps thinking *and* output together, so a value tuned on a non-thinking model truncates mid-answer.

**Refusals.** Opus 5 runs cybersecurity classifiers and a corpus containing security-adjacent issue threads will occasionally trip them. Every call site checks `stop_reason` before reading `content`:

```python
resp = client.beta.messages.create(
    model="claude-opus-5",
    max_tokens=16000,
    betas=["server-side-fallback-2026-07-01"],
    fallbacks="default",
    output_config={"effort": effort},
    system=system_blocks,
    messages=messages,
    tools=tools,
)
if resp.stop_reason == "refusal":
    raise RefusalError(resp.stop_details)
```

`fallbacks: "default"` routes a declined request to Anthropic's recommended fallback inside the same call, so a false-positive classifier hit degrades to a slightly different model rather than a failed request.

**Agent loop.** A hand-written loop over `client.beta.messages.stream`, capped at 8 iterations. Tool results are returned as `tool_result` blocks; failures return `is_error: True` with the message rather than raising, so the model can recover. `stop_reason == "pause_turn"` is handled by re-sending, capped at 3 restarts — an unhandled pause silently truncates the answer.

This section specified `client.beta.messages.tool_runner` until M2 built against it. The runner is the right default for a custom-tool agent and it was not chosen here for two specific reasons. First, the loop's structure *is* the API contract: §11.2 requires a typed event at every step, emitted outward mid-flight, and under the runner that means pushing from inside tool callbacks into a queue the SSE generator drains — an indirection that buys nothing when the loop body is the thing being streamed. Second, the Python runner cannot resume `pause_turn` in place; the documented workaround is to mirror the message history and restart the runner, which is more code than the re-send a manual loop already performs. The runner's other advantages — approval gates, result interception — are features this service has no use for.

**Determinism.** `tools[]` is sorted by name before every request. An unsorted set derived from a `dict` or a vector query reorders between runs, which changes the prompt prefix bytes and destroys the cache with no visible symptom other than cost.

---

## 11. API contract

### 11.1 Endpoints

| Method | Path | Body / Query | Response |
|---|---|---|---|
| `POST` | `/chat` | `ChatRequest` | `text/event-stream` |
| `GET` | `/sessions/{id}` | — | `Session` with messages |
| `GET` | `/sessions` | `user_id`, `limit` | `Session[]` |
| `GET` | `/memory` | `user_id`, `type?`, `include_superseded?` | `Memory[]` |
| `GET` | `/memory/{id}/history` | — | `Memory[]`, oldest first |
| `POST` | `/memory/{id}/revert` | `to_revision`, `actor` | `Memory` |
| `POST` | `/admin/consolidate` | `user_id` | `ConsolidationReport` |
| `GET` | `/healthz` | — | `{status, db, embedder, langfuse}` |

```jsonc
// ChatRequest
{
  "user_id":    "string",          // required
  "session_id": "string | null",   // null starts a new session
  "message":    "string"           // required, 1..8000 chars
}
```

### 11.2 SSE event catalog

Every event is `event: <name>` with a JSON `data:` payload. The UI is driven entirely by these — nothing is polled.

| Event | Payload | Emitted |
|---|---|---|
| `session` | `{session_id, trace_id, is_new}` | First, always |
| `memory_loaded` | `{memories: [{id, type, content, confidence, effective_confidence, created_by, source_session_id, revision}], token_count, truncated}` | After step 2 |
| `tools_selected` | `{mode, catalog_size, selected: [{name, score, is_synthetic}], floor}` | After step 3 |
| `retrieval` | `{kind: "semantic", tool, query, chunks: [{citation, source, path, score}], hybrid}` | Per semantic retrieval |
| `retrieval` | `{kind: "structured", tool, window: [from, to] \| null, area_globs: [...] \| null, count, entities: [{citation, kind, ref, title, author, state, at, url}]}` | Per aggregation call |
| `token` | `{text}` | Per text delta |
| `tool_call` | `{name, input, status: "started"\|"ok"\|"error"}` | Per tool invocation |
| `citation` | `{citation, kind: "span"\|"entity", resolved: bool, in_result_set: bool}` | Per citation the model emits |
| `memory_write` | `{id, type, content, confidence}` | When the agent writes memory |
| `done` | `{turn, usage: {input_tokens, output_tokens, cache_read_input_tokens}, cost_usd, latency_ms}` | Last |
| `error` | `{code, message}` | On failure; terminal |

`citation.resolved` and `citation.in_result_set` are computed server-side as the model streams. These are the two fields the gated citation metric reads — the same check runs in production and in eval, so the number CI gates on is the number users experience.

**Both `retrieval` shapes name their tool.** The structured one always did; the semantic one did not, which made a turn's retrievals unattributable as soon as a consumer merged them. Two things need the attribution: §14.1 scores the retrieval whose tool the question asked for, and PRD §5.6's view cannot show which tool produced a result without it.

Resolution is polymorphic over the two citation kinds. A **span** citation resolves if it matches a `chunks.id` and that chunk was in this turn's retrieval results. An **entity** citation resolves if the entity exists in the facts layer and appeared in an aggregation tool result this turn. The second half of each check is what catches the characteristic failure: citing a real pull request the agent never actually looked up, which is indistinguishable from a correct answer without it.

### 11.3 Errors

| Code | HTTP | Meaning |
|---|---|---|
| `session_not_found` | 404 | |
| `memory_not_found` | 404 | |
| `revision_not_found` | 400 | `to_revision` exceeds the memory's history |
| `refusal` | 200 + `error` event | Model declined; `stop_details.category` in message |
| `upstream_unavailable` | 503 | Anthropic or Postgres unreachable |
| `budget_exceeded` | 400 | Message exceeds length limit |

---

## 12. Observability

Langfuse, one trace per `/chat` request. `trace_id` is generated at request entry and written onto `messages.trace_id`, `memories.trace_id`, and `memory_audit.trace_id` — so any memory row can be traced back to the exact request that produced it, which is what makes the ADR-5 autonomy argument hold up.

| Span | Parent | Attributes |
|---|---|---|
| `chat_request` | — | `session_id`, `user_id`, `config_hash`, `git_sha`, `is_new_session`, `as_of` |
| `memory.load` | `chat_request` | `budget_tokens`, `token_count`, `truncated`, `n_loaded` |
| `memory.semantic` | `memory.load` | `n_candidates`, `n_selected`, `mean_confidence` |
| `memory.episodic` | `memory.load` | `n_candidates`, `n_selected`, `mean_age_days` |
| `memory.procedural` | `memory.load` | `n_candidates`, `n_selected` |
| `tools.select` | `chat_request` | `mode`, `catalog_size`, `k`, `floor`, `selected[]`, `scores[]`, `n_synthetic` |
| `agent.loop` | `chat_request` | `n_iterations`, `stop_reason` |
| `retrieve.dense` | `agent.loop` | `k`, `ef_search`, `n_results`, `top_score` |
| `retrieve.sparse` | `agent.loop` | `k`, `n_results`, `top_score` |
| `retrieve.fuse` | `agent.loop` | `rrf_k`, `n_in`, `n_out`, `overlap` |
| `retrieve.structured` | `agent.loop` | `tool`, `window_days`, `area`, `n_globs`, `count`, `truncated_render` |
| `llm.generate` | `agent.loop` | `model`, `effort`, `usage.*`, `cache_read_input_tokens`, `stop_reason` |
| `tool.call.<name>` | `agent.loop` | `is_synthetic`, `duration_ms`, `is_error` |
| `memory.write` | `agent.loop` | `memory_id`, `mem_type`, `created_by`, `confidence` |
| `memory.extract` | — (own trace) | `session_id`, `n_facts`, `n_discarded` |

### Dashboards

1. **Latency by stage** — p50/p95/p99 per span, stacked against the §13 budget.
2. **Cost per query** — split by span, plus cache-read ratio over time.
3. **Tool selection** — score distribution, real/synthetic hit ratio, floor-rejection rate.
4. **Memory health** — writes per session by `created_by`, supersession rate, revert count.

---

## 13. Non-functional requirements

Latency decomposes into the time-to-first-token target rather than being a list of unrelated numbers:

| Stage | p95 budget |
|---|---|
| `memory.load` (3 vector queries + assembly) | 200 ms |
| `tools.select` (1 embed + 1 vector query) | 100 ms |
| `retrieve.*` (dense ∥ sparse, then fuse) | 300 ms |
| Claude first token | 2000 ms |
| Framework and serialization overhead | 400 ms |
| **Time to first token** | **3000 ms** |

| NFR | Target |
|---|---|
| Full response, p95 (agent loop incl. tool calls) | < 25 s |
| Cost per query | < $0.15 p50, < $0.40 p95 |
| Memory context block | ≤ 2000 tokens, hard-enforced |
| Prompt-cache read ratio, warm | ≥ 60% |
| Concurrent sessions without degradation | 5 |
| Full corpus ingest (~40k chunks, laptop CPU) | < 30 min |
| Per-PR eval gate wall time | < 15 min |

Every one of these maps to a span or attribute in §12, so they are observable rather than aspirational. The budget is demo-grade — a single laptop, local embeddings, five concurrent sessions. Production targets would force connection-pool tuning and an embedding service into this document; they are deferred until there is a reason for them.

---

## 14. Eval harness

`evals/runner.py`.

### 14.1 Question schema and scoring by class

Golden-set entries carry a `class` (1–6 per PRD §5.1) and an `as_of` date. The class selects the scorer:

```yaml
- id: q004
  class: 2                      # change over time — exactly checkable
  question: "What shipped in the 30 days before the pin?"
  as_of: "2026-06-01"
  gold_entities: ["pr:11201", "pr:11214", "pr:11230", "pr:11247"]
  gold_tools: [merged_prs]

- id: q031
  class: 5                      # decision archaeology — interpretive
  question: "Why was the sync client dropped?"
  as_of: "2026-06-01"
  gold_chunks: ["issue:9821#comment-12", "docs:advanced/async.md#sync-support"]
  gold_answer_points: ["maintenance burden", "async-first direction"]
  gold_tools: [search_issues]
```

| Class | Scorer | Gated | From |
|---|---|:--:|:--:|
| 1–4 | **Aggregate set-F1** — precision/recall of returned entity set vs. `gold_entities` | ✅ | M1 |
| 5–6 | recall@5, MRR@10 over `gold_chunks` | ✅ | M1 |
| all | tool-selection accuracy, citation resolution | ✅ | M2 |
| all | grounding, coverage (LLM-judged) | report-only | M2 |

**Set-F1 is computed against the tool's result set, not the prose.** The scorer reads the `retrieval` SSE events, so it measures what the system retrieved rather than what the model said about it. Whether the prose faithfully reflects that set is a separate question, answered by citation resolution.

**"The tool's result set" means the retrieval whose tool is in `gold_tools` — not the union of the turn.** A turn may retrieve several times, and which retrievals count decides what the metric measures. Scored over the union it measures how many tools the agent called: over 41 measured turns, one call scored mean 1.000, two 0.635, three 0.079, and four or more 0.007, with the losses entirely in precision. The case that settles it is q006. It checks `pr:15806`, finds it closed unmerged, then separately confirms `release:0.138.0` did ship — the exact behaviour PRD §5.2 exists to require — and the union charged it 0.667 for the second lookup. A metric that penalises the verification the PRD demands is measuring the wrong thing.

Three consequences, all deliberate:

- **A tool called more than once contributes every call.** Two windows of `merged_prs` is one tool used twice, so it cannot be a tool-count artefact.
- **No matching retrieval scores 0.0.** An agent that never called the tool the question asks for did not answer the question, however good its prose. This is what q005 scores, having called nothing at all.
- **The union is still computed and reported, never gated.** It is the evidence for the paragraph above, and when it falls below the gated number it names the questions that retrieved beyond what was asked. Discarding it would leave the choice unfalsifiable.

The rejected third reading was scoring the entities the model *cited*. Measured, it is worse and for a disqualifying reason: on q012 the tool returned exactly the right 40 pull requests and the model cited a handful, scoring 0.234 against the union's 1.000. §7.2.1 has the model narrate around `Aggregate.rendered` rather than restate the set, so that reading punishes the agent for following the design.

**The runner has two shapes, because M1 has no agent.** At M1 there is no orchestrator, no `tools[]`, and no SSE stream to read: the runner calls `FactsStore` and `Retriever` directly. That is enough for the three retrieval metrics — set-F1, recall@5, and MRR@10 need no model in the loop — and it is what PRD §8 asks M1 to commit a baseline for. Tool-selection accuracy and citation resolution are properties of an agent turn and cannot be measured before one exists; they arrive at M2, when the runner switches to driving `POST /chat` and scoring the `retrieval` events described above. The direct-call path is kept after M2 rather than deleted, because it isolates a retrieval regression from an agent regression.

**At M1, set-F1 is a drift tripwire, not a quality metric.** Classes 1–4 dispatch the same `gold_query` that generated the ground truth, so with no agent choosing the query the score is 1.0 by construction. It is still worth running — it fails if a `FactsStore` query changes behaviour or the corpus is re-ingested at a different revision, which is exactly the silent-invalidation risk the next paragraph describes. It becomes a measure of the *system* only at M2, when tool selection decides which query runs and with what arguments. A reader comparing an M1 baseline to an M2 one must not read the drop as a regression, which is why §14.3 stamps the milestone on the file.

**Ground truth for classes 1–4 is generated, not authored.** `evals/build_gold.py` runs the same SQL the tools run against the pinned revision and writes the resulting entity sets into the golden set. This is legitimate precisely because the query is a *fact* about the repository — but it means the scorer and the ground truth share an implementation, so a bug in the SQL would be invisible. Mitigation: the 30 structured questions are spot-checked by hand against the GitHub UI at authoring time, and that check is recorded in the file.

**`as_of` validation.** The runner asserts every question's `as_of` is at or before the pinned `CORPUS_REF` date and aborts the whole run on violation. A question anchored after the pin cannot be answered correctly by any system, and scoring it produces a number that looks like a retrieval regression.

### 14.2 Config hash

An eval cell is identified by SHA-256 over the sorted ablation config:

```python
config = {
    "hybrid": HYBRID_ENABLED,
    "memory": MEMORY_ENABLED,
    "tool_mode": TOOL_RETRIEVAL_MODE,
    "tool_catalog_size": TOOL_CATALOG_SIZE,
    "embedding_model": embedder.model_id,
    "agent_model": AGENT_MODEL,
    "corpus_ref": CORPUS_REF,
}
config_hash = hashlib.sha256(
    json.dumps(config, sort_keys=True).encode()
).hexdigest()[:12]
```

`corpus_ref` and `embedding_model` are in the hash deliberately. Comparing metrics across different corpus refs or embedding models is comparing different experiments, and the hash makes that impossible to do by accident.

`corpus_ref` here is the **resolved** ref — `symbolic@sha`, the same form §14.3 records — read from the latest completed `ingest_runs` row, not the symbolic `CORPUS_REF`. Hashing the symbolic name would defeat the paragraph above: `master` hashes identically across every re-ingest, so two runs against genuinely different corpora would share a cell.

### 14.3 Baseline format

`evals/baselines/main.json`:

```json
{
  "milestone": "M2",
  "config_hash": "a3f1c9d20e84",
  "git_sha": "…",
  "corpus_ref": "master@abc1234",
  "recorded_at": "2026-08-01T00:00:00Z",
  "metrics": {
    "aggregate_set_f1": 0.91,
    "recall_at_5": 0.72,
    "mrr_at_10": 0.61,
    "tool_accuracy_jaccard": 0.84,
    "tool_accuracy_exact": 0.66,
    "citation_resolution": 0.97
  },
  "tolerances": {
    "aggregate_set_f1": 0.02,
    "recall_at_5": 0.02,
    "mrr_at_10": 0.02,
    "tool_accuracy_jaccard": 0.03,
    "tool_accuracy_exact": 0.03,
    "citation_resolution": 0.01
  }
}
```

**A metric with no entry in `tolerances` is report-only and cannot gate.** `aggregate_set_f1_union` is written into `metrics` and deliberately left out of `tolerances`, so the §17 Q10 diagnostic travels with every baseline without ever failing a build. The rule is structural rather than a list the gate has to be kept in sync with: absent tolerance, no gate.

The gate fails if `current < baseline - tolerance` for any gated metric. Improvements never fail. A baseline change requires a PR that edits this file and states why in the description — so a regression cannot be silently absorbed by regenerating the baseline in the same commit that caused it.

**Two baseline files, because §14.1 keeps both runner shapes.** `main.json` is the agent path — the system a user meets, and what the gate compares against. `retrieval.json` is the direct-call path at the same corpus and embedding model, carrying only the three retrieval metrics. When a number moves, the pair says where: both moving is retrieval, `main.json` alone moving is the agent, prompt, or tool layer. One file cannot distinguish those, and at M2 the agent is the noisier of the two.

**`milestone` is part of the identity, alongside `config_hash`.** The M1 baseline carries only `aggregate_set_f1`, `recall_at_5`, and `mrr_at_10`; the M2 baseline adds `tool_accuracy_*` and `citation_resolution` and reinterprets set-F1 per §14.1. Two baselines with different milestones measure different systems and comparing them is meaningless — the same argument §14.2 makes for keeping `corpus_ref` and `embedding_model` inside the hash. A comparison across milestones is an error, not a regression.

### 14.4 Workflows

`.github/workflows/eval-pr.yml` — on pull request:

1. Restore the corpus index from cache keyed on `CORPUS_REF` + ingest-script SHA (re-ingesting per PR would blow the 15-minute budget).
2. Run the default config over all 50 questions.
3. Compare to `main.json`; fail on breach.
4. Post a metrics table as a PR comment either way.

`.github/workflows/eval-nightly.yml` — on schedule:

1. Full 12-cell matrix plus the tool-scaling sweep (15 / 50 / 200 / 500).
2. Cross-session scenario suite, memory on vs. off.
3. Publish a markdown report and a plot artifact. Never blocks.

Only the LLM-judge and agent calls need network. Embeddings are local, so the retrieval metrics are fully deterministic — a recall@5 change is always a real change, never judge noise.

---

## 15. Security

| Concern | Handling |
|---|---|
| `run_snippet` | Subprocess with no network, a read-only bind mount of the corpus checkout, a 5 s CPU limit, and a 256 MB memory cap. Disabled by default via `ENABLE_RUN_SNIPPET=false`. **Not built at M2** — it is absent from the §7.2.1 catalog, and the catalog is authoritative for what the agent can call. The flag and this row describe the sandbox it would need if it is ever added; no golden question requires it. |
| `GITHUB_TOKEN` | A fine-grained token with read-only access to public repositories and **no permissions**. Every ingest call is a GET against public data, so no scope is required; the classic `public_repo` scope would grant write access to every public repo on GitHub and contradict the PRD §2 non-goal. Read at ingest, never passed to the model, never written to a memory row. |
| Secrets in memory | The extraction prompt forbids recording credentials, and a regex denylist (`sk-`, `ghp_`, `AKIA`, bearer patterns) rejects matching statements before write. Memory rows are replayed into future prompts, so a secret written once leaks into every later session. |
| PII in episodic memory | Same path. Single-user demo, no third-party data, but `GET /memory` and revert give the user a delete-equivalent (supersede) for anything recorded. |
| Prompt injection from the corpus | Issue bodies are untrusted text. Retrieved chunks are wrapped in `<document>` tags with an explicit instruction that document content is data, never instructions. Corpus content can never trigger `memory_write` — that tool is only callable from the model's own turn, and written memories carry `created_by='agent'` for audit. |
| Langfuse | Local only, `TELEMETRY_ENABLED=false`. Traces contain user messages; not exposed beyond localhost. |

---

## 16. ADR log

| # | Decision | Rationale | Rejected |
|---|---|---|---|
| 1 | Postgres + pgvector as the single store | One system to run; memory rows and their embeddings mutate in the same transaction | Dedicated vector DB — two systems, no transactional consistency between a memory and its vector |
| 2 | Local `bge-small` embeddings | Free, deterministic, no API key in CI; makes retrieval metrics reproducible | Voyage — better quality but adds a key and per-run cost to every eval; mitigated by the `Embedder` protocol (§3.1) |
| 3 | RRF for fusion, `k=60` untuned | Rank-based fusion needs no score calibration between two differently-scaled arms | Weighted score normalization — requires per-corpus calibration, and tuning `k` on the golden set is fitting to the eval |
| 4 | Append-only memory versioning | Every write is attributable and reversible | In-place update — unattributable, unrevertable, and the direct enabler of the poisoning failure mode |
| 5 | Autonomous writes, no approval gate | Acceptable *because* of ADR 4; provenance plus revert makes autonomy recoverable | Human-in-the-loop gating — safer, but concedes the autonomy claim the project exists to test |
| 6 | Citation *resolution* gates CI; judged *grounding* does not | A nondeterministic judge cannot back a hard gate; resolution is mechanically checkable and catches the failure that matters | Judge-gated CI — flaky builds, and a red build nobody trusts is a gate that gets disabled |
| 7 | `claude-opus-5` everywhere, `effort` as the cost lever | Keeps one variable fixed across ablation cells | Per-route model downgrade — changes two variables at once and muddies attribution |
| 8 | Memory block after the cached prefix | Memory changes per session; in the system prompt it would invalidate the cache every session | Memory in system prompt — cleaner to read, destroys caching |
| 9 | Anthropic's native tool search kept as an ablation arm | Beating full exposure is trivial; beating the first-party baseline is the real result | Semantic vs. full only — a weaker comparison that flatters our implementation |
| 10 | Synthetic tool padding, always labeled | Needed for the scaling curve; unlabeled it would be dishonest measurement | Real tools only (too few to show a curve); unlabeled padding (misleading) |
| 11 | `memory_write`/`memory_search` always injected | Their relevance is never expressed in the user's query, so retrieval would never surface them and the write-back loop would never fire | Pure retrieval — architecturally clean, silently breaks the feature |
| 12 | Identifier decomposition at ingest, in Python | `HTTPException` must match "http exception"; no C extension, survives a managed-Postgres move | Custom Postgres text-search parser — needs a C extension and pins us to self-hosted |
| 13 | A structured facts layer beside the vector index | "What shipped last month" has an exact answer; similarity search returns documents that *discuss* shipping, ranked, with no notion of completeness | Embedding everything and trusting the model to aggregate — fails silently and unfixably, since no embedding quality makes a top-k list into a complete set |
| 14 | Route by tool selection, not a query classifier | Adds no component, and makes the tool-retrieval thesis load-bearing rather than decorative — if selection can't surface `merged_prs`, the eval says so | A separate intent classifier — a second model to train, tune, and drift, with its own silent failure mode |
| 15 | Counts rendered deterministically in code | A model asked to summarise forty PRs produces a confident wrong total, and the manager cannot tell | LLM-summarised aggregates — fluent, unverifiable, and wrong often enough to poison trust in every other answer |
| 16 | `areas.yaml` hand-authored | "The payments module" has no reliable signal in the file tree; a wrong auto-derived mapping corrupts every ownership answer invisibly | Heuristics over directory names or the import graph — plausible-looking and wrong in ways nobody notices |
| 17 | Facts layer rebuilt, not versioned | GitHub is the system of record; a second history here would go stale differently from the first | Versioning entities like `memories` — duplicate state, no added attribution value |

---

## 17. Open technical questions

1. **Time-expression parsing.** "Last month", "since the last release", "this quarter", "recently" must resolve to a concrete range against the session's `as_of`. Options: a rules-based parser (predictable, brittle at the edges), or a structured-output Claude call (flexible, adds a round trip before the real query, and nondeterministic in a gated eval). Leaning rules-based with an explicit error on anything unrecognised, since a silently-wrong window produces a confidently-wrong status report. Undecided.
2. **Class-6 verification: automatic or model-driven?** "Do we support websockets?" needs a merge-state check on whatever the semantic path surfaced. Either the search tool auto-joins to the facts layer and returns state alongside every chunk, or the model is expected to call `pr_state` itself. Auto-join is reliable but couples the two paths and costs on every search; model-driven is clean but will sometimes be skipped — which is exactly the PRD §5.2 failure. Probably auto-join; needs measurement.
3. **Chunk-level vs. document-level retrieval for issues.** A long thread's resolution often sits in the last comment while the query matches the first. Parent-document expansion after fusion may beat per-comment chunks — this needed an A/B at M1, and M1 measured the motivating half of it. Over the 20 interpretive questions at the pinned revision, exact chunk recall@5 is **0.350** while *parent-document* recall@5 — did the right issue thread or doc surface at all — is **0.900**. The retriever finds the right conversation and the wrong comment. q043 is the clean case: 8 of the top 10 are `issue:98` chunks, the thread the answer lives in, but the gold `issue:98#comment-0` sits at rank 17. The ranking signal is sound and the granularity is wrong, which is the strongest available argument for expansion: fuse at comment level, expand to the thread, re-rank within it. Still undecided pending the A/B, but the decision now has a number to beat — 0.350, the committed M1 baseline. Two questions (q044, q047) miss even at the parent level; both ask a general capability question whose answer sits in a specifically-titled thread, and expansion will not help them.
4. **Aggregate truncation threshold.** At what result count does `Aggregate.rendered` stop listing entities and summarise? Too low and the model can't cite specifics; too high and a 400-PR window floods the context. Unmeasured.
5. **Consolidation clustering threshold (0.35).** Picked by inspection, not measured. Should be swept once there is real episodic volume, with the caveat that sweeping it against the golden set is not valid — needs a held-out memory-quality rubric.
6. **`user_profile_vector` staleness.** Cached per session; a long session that writes new semantic memories will retrieve against a stale profile. Probably immaterial at demo scale, but unverified.
7. **Cache behavior in `semantic` mode.** Per-query tool sets sit before the cache breakpoint, so within-session cache hits may fall well short of the 60% target in that mode specifically. If measurement confirms it, the fix is moving tools after the breakpoint via mid-conversation tool changes (`mid-conversation-tool-changes-2026-07-01`) — deferred until measured.
8. **Frontend stack for the M6 transparency view.** A React SPA or server-rendered templates. The only hard constraint is §11.2: the view is driven entirely by SSE and nothing is polled, so anything that consumes `EventSource` and holds per-turn state across memories, tools, retrieval, and citations qualifies. Deferring rather than deciding — the contract arrives at M2 and has not been exercised by a real client, and picking a stack before then would be choosing without the one piece of information that matters. PRD §5.6 and the M6 milestone are deliberately stack-agnostic and need no amendment either way. Undecided.
9. **`TOOL_SIMILARITY_FLOOR` does not separate anything with `bge-small`.** §7.2 justifies the floor with "a query like 'hi' returns five arbitrary tools with ~0.05 similarity"; measured against the real catalog, it does not. Over the 50 golden questions and 5 conversational non-questions, cosine similarity to a tool embedding falls in a narrow, heavily-overlapping band: gold tools median **0.556** (min 0.424), non-gold tools median **0.510** (max 0.702), junk queries median **0.452** (max 0.556). Junk's ceiling is gold's median. No absolute threshold separates them — the default 0.25 admits everything, and a floor high enough to reject "hi" would reject half the gold tools with it. The premise was a wider similarity range than `bge-small-en-v1.5` produces between any two English strings. Three ways out, none measured: a relative floor (keep tools within δ of the top score), a better-spread embedding model behind the ADR 2 swap point, or accepting that `k` alone does the work and deleting the floor. Left as specced and reported honestly in the M2 baseline rather than tuned, because tuning it against the golden set is fitting to the eval. Undecided.
10. **Set-F1 over the union of a turn's retrievals measures tool-call count, not correctness.** ~~Undecided.~~ **Resolved** — §14.1 now scores the retrieval whose tool is in `gold_tools`, and the union is reported beside it as `aggregate_set_f1_union`, never gated. The evidence and the two rejected readings are recorded there. Decided before the M2 baseline existed, on the q006 argument rather than on which reading scored higher; the union, the reading it replaces, was the highest of the three at 0.721.

11. **recall@5 over the union of a turn's searches hides retrievals the system actually made.** The interpretive analogue of Q10, and measured rather than argued by analogy. recall@5 reads the concatenation of every semantic `retrieval` event in call order, so its window is the *first* search's top 5 and every later search lands outside it. Over the 20 interpretive questions in the M2 baseline, **6 turns retrieved a gold chunk and ranked it below 5** — q032 at rank 10, q034 at 11, q035 at 6 and 15, q036 at 9, q043 at 18, q046 at 9 and 14. Every one made two or more searches; unions ran 17 to 54 chunks. All six score recall@5 = 0.00 with recall over the full union of 0.50 to 1.00. That is 6 of the 16 interpretive zeros, and it is a measurement artefact rather than a retrieval failure: the agent found the passage and the metric could not see it.

    This does not make 0.175 wrong to gate on — it is the honest number for the metric as defined, and it is what the M2 baseline records. It makes the *definition* wrong, in the same way and for a sharper reason than Q10. The fix is the parallel one: score the search whose tool is in `gold_tools`, which §11.2's `tool` field now makes possible. It is not applied here because Q10 was decided before its baseline existed and this one cannot be — changing it now would move a gated metric in the same commit that records it, which is exactly what §14.3 forbids. It belongs in a PR of its own that states the expected rise.

    **The evidence needs one fresh agent run.** Per-search breakdowns are recorded from this commit forward, but the 50 rows already cached were written without them and the `cache_format` guard deliberately does not reject those rows — they are valid measurements of what the baseline reports, and invalidating a paid run to collect a diagnostic would be the wrong trade. Measuring this requires `evals/runner.py --agent --fresh`, at the cost of a full run. Undecided.
12. **`tool_accuracy_exact` is near-zero by construction in `semantic` mode.** The selector returns exactly `TOOL_RETRIEVAL_K` tools (default 5) while a golden question's `gold_tools` is usually one. An exact set match is therefore almost never possible, and the metric measures `k` rather than the selector. Jaccard degrades gracefully and is the meaningful one; `full` mode makes exact match structurally impossible in the same way. Either the metric needs a top-1 or recall@k formulation, or it should be reported only for modes where the returned-set size can vary. Undecided.

