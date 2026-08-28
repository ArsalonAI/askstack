"""The service — TRD §2.3, §11.1.

Single-process uvicorn ASGI app. The connection pool and the embedding model
are process-wide singletons held on the lifespan: the model is ~130 MB resident
and loading it per request would dominate every latency number in §13.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import asyncpg
from anthropic import AsyncAnthropic
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.facts.store import PostgresFactsStore
from app.memory.consolidation import Consolidator
from app.memory.lifecycle import Extractor
from app.memory.manager import MemoryManager, effective_confidence
from app.memory.store import (
    MemoryNotFound,
    PostgresMemoryStore,
    RevisionNotFound,
)
from app.orchestrator import Orchestrator, sse
from app.retrieval.embedder import get_embedder
from app.retrieval.hybrid import HybridRetriever
from app.tools.registry import ToolRegistry
from app.tools.selector import build_selector

log = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    user_id: str
    session_id: str | None = None
    message: str = Field(min_length=1, max_length=8000)
    # Not in §11.1's ChatRequest. The eval needs every question answered against
    # the date it was anchored to, and a window measured from wall-clock now is
    # not reproducible against a pinned revision (PRD §9).
    as_of: datetime | None = None


async def _corpus_as_of(pool: asyncpg.Pool) -> datetime:
    """The pinned revision's date, used as the default `as_of`.

    Wall-clock now would let a window run past the revision the corpus was
    ingested at, silently under-reporting rather than erroring.
    """
    async with pool.acquire() as conn:
        pinned = await conn.fetchval(
            "SELECT max(greatest(merged_at, closed_at, created_at)) FROM pull_requests"
        )
    return pinned or datetime.now(UTC)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.pool = await asyncpg.create_pool(settings.database_url, min_size=2, max_size=10)
    app.state.embedder = get_embedder()
    app.state.client = AsyncAnthropic(api_key=settings.anthropic_api_key or None)
    app.state.as_of = await _corpus_as_of(app.state.pool)
    log.info("corpus pinned at %s", app.state.as_of)
    try:
        yield
    finally:
        await app.state.pool.close()


app = FastAPI(title="askstack", lifespan=lifespan)


def _orchestrator(as_of: datetime | None = None) -> Orchestrator:
    pool = app.state.pool
    retriever = HybridRetriever(pool, app.state.embedder)
    # `None` when MEMORY_ENABLED is false, which drops both memory tools from
    # the catalog rather than offering tools that would fail — the ablation's
    # off arm has to measure a system without memory (§7.3).
    store = PostgresMemoryStore(pool)
    memory = (
        MemoryManager(
            store,
            app.state.embedder,
            app.state.client,
            model=settings.agent_model,
        )
        if settings.memory_enabled
        else None
    )
    # §8.1 runs as a background task off the orchestrator, not a FastAPI
    # BackgroundTask: the SSE response is a long-lived stream, and a FastAPI
    # background task only fires once that stream closes — which for a client
    # that disconnects early is never.
    extractor = (
        Extractor(pool, store, app.state.embedder, app.state.client, settings)
        if settings.memory_enabled
        else None
    )
    # One connection for the facts layer per request-scoped orchestrator; the
    # retriever takes the pool because §2.1 runs its two arms concurrently.
    registry = ToolRegistry(
        PostgresFactsStore(pool),
        retriever,
        top_k=settings.retrieval_top_k,
        memory=memory,
    )
    selector = build_selector(
        settings.tool_retrieval_mode,
        # The registry's view, not the raw catalog: with memory off it omits
        # both memory tools, and the selector must not offer what dispatch
        # would refuse.
        registry.definitions(),
        pool=pool,
        embedder=app.state.embedder,
        floor=settings.tool_similarity_floor,
    )
    return Orchestrator(
        pool,
        registry,
        selector,
        app.state.client,
        settings,
        as_of=as_of or app.state.as_of,
        memory=memory,
        extractor=extractor,
    )


UI = Path(__file__).resolve().parents[1] / "ui" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def serve_ui() -> str:
    """PRD §5.6's transparency view — TRD §17 Q8, resolved.

    One static file, no build step. That question was left open at M2 for lack
    of information: "a React SPA or server-rendered templates", pending a real
    client exercising the §11.2 contract. Two now have, and what they showed is
    that the choice was never between those two.

    `EventSource` — which §17 Q8 named as the constraint — **cannot be used at
    all**, because it only issues GET requests and §11.1 makes `/chat` a POST
    with a JSON body. Every client has to read the stream off `fetch` and split
    SSE frames by hand regardless of framework, so a framework buys nothing
    here: the view has no routing, no forms beyond one input, and no client
    state that outlives a turn.

    Read from disk per request rather than cached at import. The file is a few
    kilobytes, this is a single-process app, and editing the view without
    restarting the server is worth more than the read.
    """
    return UI.read_text()


@app.get("/healthz")
async def healthz() -> dict:
    """§11.1. Reports what is actually reachable, not that the process is up."""
    try:
        async with app.state.pool.acquire() as conn:
            chunks = await conn.fetchval("SELECT count(*) FROM chunks")
        db = {"ok": True, "chunks": chunks}
    except Exception as exc:  # noqa: BLE001
        db = {"ok": False, "error": str(exc)}
    return {
        "status": "ok" if db["ok"] else "degraded",
        "db": db,
        "embedder": settings.embedding_model,
        "model": settings.agent_model,
        "anthropic_key": bool(settings.anthropic_api_key),
        "corpus_as_of": app.state.as_of.isoformat(),
        "tool_mode": settings.tool_retrieval_mode,
        "langfuse": bool(settings.langfuse_public_key),
    }


@app.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    orchestrator = _orchestrator(request.as_of)

    async def stream() -> AsyncIterator[str]:
        async for event, data in orchestrator.run(
            request.user_id, request.session_id, request.message, as_of=request.as_of
        ):
            yield sse(event, data)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/sessions/{session_id}/end")
async def end_session(session_id: str) -> dict:
    """§8.1's session-end extraction trigger.

    Not in §11.1's original endpoint table, and it has to be: extraction's two
    triggers are "every 10 turns" and "session end", and nothing in a stateless
    HTTP API tells the server a session is over. Without this, a three-turn
    session — the shape PRD §7.2's whole cross-session suite is built from —
    never extracts anything, and memory stays empty for exactly the case it
    exists to serve.
    """
    report = await _orchestrator().end_session(session_id)
    if report is None:
        return {"session_id": session_id, "extracted": 0, "already_ended": True}
    return {
        "session_id": session_id,
        "extracted": report.written,
        "considered": report.considered,
        "discarded_low_confidence": report.discarded_low_confidence,
    }


def _memory_json(memory) -> dict:
    return {
        "id": memory.id,
        "revision": memory.revision,
        "user_id": memory.user_id,
        "type": memory.mem_type,
        "content": memory.content,
        "entities": list(memory.entities),
        "confidence": round(memory.confidence, 3),
        "effective_confidence": round(effective_confidence(memory), 3),
        "created_by": memory.created_by,
        "source_session_id": memory.source_session_id,
        "source_ids": list(memory.source_ids),
        "valid_from": memory.valid_from.isoformat(),
        "valid_to": memory.valid_to.isoformat() if memory.valid_to else None,
        "trace_id": memory.trace_id,
    }


@app.get("/memory")
async def list_memory(user_id: str, type: str | None = None) -> dict:
    """§11.1. Every live memory for one user.

    `effective_confidence` is rendered alongside the stored value rather than
    instead of it (§8.3). A UI showing only the decayed number cannot explain
    why a memory stopped being loaded, and one showing only the stored number
    contradicts what the agent actually saw.
    """
    store = PostgresMemoryStore(app.state.pool)
    memories = await store.live(user_id, type)
    return {"user_id": user_id, "memories": [_memory_json(m) for m in memories]}


@app.get("/memory/{memory_id}/history")
async def memory_history(memory_id: str) -> dict:
    """§11.1 — all revisions, oldest first.

    The read side of ADR 4. Nothing is ever edited in place, so this is the
    complete record of what a memory has ever said, which is what makes the
    autonomy of ADR 5 recoverable rather than merely asserted.
    """
    store = PostgresMemoryStore(app.state.pool)
    try:
        revisions = await store.history(memory_id)
    except MemoryNotFound:
        return {"error": {"code": "memory_not_found", "message": memory_id}}
    return {
        "id": memory_id,
        "revisions": [_memory_json(m) for m in revisions],
        "audit": [
            {
                "op": row["op"],
                "revision": row["revision"],
                "actor": row["actor"],
                "at": row["at"].isoformat(),
                "trace_id": row["trace_id"],
            }
            for row in await store.audit(memory_id)
        ],
    }


class RevertRequest(BaseModel):
    to_revision: int = Field(ge=1)
    actor: str = Field(min_length=1, max_length=200)


@app.post("/memory/{memory_id}/revert")
async def revert_memory(memory_id: str, request: RevertRequest) -> dict:
    """§11.1, PRD §5.5's one-action rollback.

    Appends a new revision carrying the old content; it does not roll back.
    A destructive rollback would erase the evidence that a revert happened,
    which is the one thing the audit trail has to show.
    """
    store = PostgresMemoryStore(app.state.pool)
    try:
        reverted = await store.revert(
            memory_id, request.to_revision, actor=request.actor
        )
    except MemoryNotFound:
        return {"error": {"code": "memory_not_found", "message": memory_id}}
    except RevisionNotFound:
        # §11.3 gives these different codes on purpose: a missing memory is a
        # 404 and a missing revision is a 400, and collapsing them would make
        # "you asked for revision 99" indistinguishable from "no such memory".
        return {
            "error": {
                "code": "revision_not_found",
                "message": f"{memory_id} has no revision {request.to_revision}",
            }
        }
    return _memory_json(reverted)


@app.post("/admin/consolidate")
async def admin_consolidate(user_id: str) -> dict:
    """§11.1, §8.2. Episodic → semantic for one user.

    Awaited rather than backgrounded, unlike extraction. Nobody is waiting on a
    streamed answer here, and a caller who asked for consolidation wants the
    `ConsolidationReport` — clusters formed, memories written, memories
    superseded — not an acknowledgement that something will happen later.
    """
    if not settings.memory_enabled:
        return {"error": {"code": "memory_disabled", "message": "MEMORY_ENABLED=false"}}
    report = await Consolidator(
        PostgresMemoryStore(app.state.pool),
        app.state.embedder,
        app.state.client,
        settings,
    ).consolidate(user_id)
    return {
        "user_id": user_id,
        "clusters_formed": report.clusters_formed,
        "memories_written": report.memories_written,
        "memories_superseded": report.memories_superseded,
        "facts_skipped": report.facts_skipped,
    }


@app.get("/sessions/{session_id}")
async def session(session_id: str) -> dict:
    async with app.state.pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM sessions WHERE id = $1", session_id)
        if row is None:
            return {"error": {"code": "session_not_found"}}
        messages = await conn.fetch(
            "SELECT turn, role, content, created_at FROM messages"
            " WHERE session_id = $1 ORDER BY turn, role",
            session_id,
        )
    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "started_at": row["started_at"].isoformat(),
        "messages": [dict(m) for m in messages],
    }
