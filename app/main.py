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

import asyncpg
from anthropic import AsyncAnthropic
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.facts.store import PostgresFactsStore
from app.orchestrator import Orchestrator, sse
from app.retrieval.embedder import get_embedder
from app.retrieval.hybrid import HybridRetriever
from app.tools.registry import CATALOG, ToolRegistry
from app.tools.selector import FullToolSelector

log = logging.getLogger(__name__)

# Ablation axis C. Modes are added here as they land; an unimplemented mode
# fails loudly rather than falling back, because a run that reports `semantic`
# in its config hash while actually broadcasting every tool would poison the
# comparison the whole thesis rests on (§7.3).
SELECTORS = {"full": FullToolSelector}


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
    # One connection for the facts layer per request-scoped orchestrator; the
    # retriever takes the pool because §2.1 runs its two arms concurrently.
    registry = ToolRegistry(
        PostgresFactsStore(pool), retriever, top_k=settings.retrieval_top_k
    )
    if settings.tool_retrieval_mode not in SELECTORS:
        raise NotImplementedError(
            f"TOOL_RETRIEVAL_MODE={settings.tool_retrieval_mode!r} is not implemented "
            f"yet. Available: {', '.join(sorted(SELECTORS))}."
        )
    selector_cls = SELECTORS[settings.tool_retrieval_mode]
    return Orchestrator(
        pool,
        registry,
        selector_cls(CATALOG),
        app.state.client,
        settings,
        as_of=as_of or app.state.as_of,
    )


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
