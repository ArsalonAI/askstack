"""Which tool definitions enter `tools[]` — TRD §7.3, ablation axis C.

Three implementations behind one protocol. This module decides *which* tools
the model can see; `app/tools/registry.py` owns what they do. The split is the
whole ablation: swapping the selector changes prompt size and selection
accuracy without touching a single tool implementation.

`full` is the control arm and lands first — it needs no tool embeddings, so the
agent loop can be built and demonstrated before the semantic path exists. The
thesis is that `semantic` matches it on accuracy at a fraction of the prompt,
and a control arm you cannot run is not a comparison.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import asyncpg

from app.interfaces import Embedder, ToolDef
from app.tools.registry import ALWAYS_INJECTED

# §7.2. The floor matters more than k: without it a query like "hi" returns
# five arbitrary tools at ~0.05 similarity and the model is invited to call
# one. With it, nothing is returned and the model answers conversationally.
SELECT_SQL = """
SELECT name, description, input_schema, server, is_synthetic,
       1 - (embedding <=> $1::vector) AS score
FROM tool_defs
WHERE 1 - (embedding <=> $1::vector) >= $2
ORDER BY embedding <=> $1::vector
LIMIT $3
"""

EF_SEARCH = 100

# Anthropic's own tool search, kept as an ablation arm (ADR 9). Beating full
# exposure is trivial; beating the first-party baseline is the real result.
NATIVE_SEARCH_TOOL = {
    "type": "tool_search_tool_bm25_20251119",
    "name": "tool_search_tool_bm25",
}


def embedding_text(tool: ToolDef) -> str:
    """§7.1's rendered template — never the raw JSON Schema.

    Raw JSON embeds structural noise (`"type": "object"`, `"required"`) that is
    identical across every tool and dilutes the signal. Parameter *descriptions*
    carry most of what discriminates one tool from another.
    """
    properties = tool.input_schema.get("properties", {})
    params = "; ".join(
        f"{name}: {spec.get('description', '')}".strip()
        for name, spec in sorted(properties.items())
    )
    lines = [tool.name, tool.description]
    if params:
        lines.append(f"Parameters: {params}")
    return "\n".join(lines)


def to_api_tools(tools: Sequence[ToolDef], *, deferred: Sequence[str] = ()) -> list[dict]:
    """`ToolDef` → the Anthropic `tools[]` wire shape.

    Sorted by name, always (§10). The list is built from a vector query in
    `semantic` mode and from a dict elsewhere; either can reorder between runs,
    and a reordered prefix invalidates the prompt cache with no symptom other
    than the bill.
    """
    deferred_names = set(deferred)
    payload = []
    for tool in sorted(tools, key=lambda t: t.name):
        entry: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        if tool.name in deferred_names:
            entry["defer_loading"] = True
        payload.append(entry)
    return payload


def always_injected(catalog: Sequence[ToolDef]) -> list[ToolDef]:
    """The tools that bypass selection entirely — §7.2.2, ADR 11.

    Their relevance is never expressed in the user's query. Nobody types
    "please save this to memory", so semantic retrieval would never surface
    `memory_write` and the write-back loop would silently never fire — the
    feature would be architecturally clean and functionally absent.

    Intersected with the caller's catalog rather than asserted, so a run with
    memory disabled (whose registry omits both tools) injects nothing and the
    off arm measures a system without memory.
    """
    return [t for t in catalog if t.name in ALWAYS_INJECTED]


def _with_always_injected(
    selected: Sequence[ToolDef], catalog: Sequence[ToolDef]
) -> list[ToolDef]:
    chosen = {t.name for t in selected}
    extra = [t for t in always_injected(catalog) if t.name not in chosen]
    return sorted([*selected, *extra], key=lambda t: t.name)


class FullToolSelector:
    """Every definition, every request — §7.3's control arm.

    Deliberately the most expensive arm in prompt tokens and the baseline the
    other two have to beat. It is also the only arm whose tool set is constant,
    so it caches perfectly (§9) — which is worth stating plainly, because it
    means `semantic` is not strictly cheaper and the measurement has to show
    where the crossover is.
    """

    mode = "full"

    def __init__(self, catalog: Sequence[ToolDef]) -> None:
        self._catalog = tuple(catalog)

    async def select(self, query: str, k: int) -> list[ToolDef]:
        # Neither argument is used, and that is the point of the arm.
        return sorted(self._catalog, key=lambda t: t.name)

    def extra_request_params(self) -> dict:
        return {}


class SemanticToolSelector:
    """Top-k over `tool_defs` above a similarity floor — §7.2, the thesis arm.

    Tool definitions live in the same pgvector index as everything else, so
    "which tools are relevant to this question" is the same operation as "which
    passages are relevant to this question". That is the claim being tested: it
    should hold selection accuracy roughly flat while prompt size stays flat
    too, as the catalog grows from tens of tools to hundreds.
    """

    mode = "semantic"

    def __init__(
        self,
        catalog: Sequence[ToolDef],
        pool: asyncpg.Pool,
        embedder: Embedder,
        *,
        floor: float = 0.25,
    ) -> None:
        self._by_name = {tool.name: tool for tool in catalog}
        self.pool = pool
        self.embedder = embedder
        self.floor = floor

    async def select(self, query: str, k: int) -> list[ToolDef]:
        vector = await asyncio.to_thread(self.embedder.embed_query, query)
        literal = "[" + ",".join(f"{v:.7g}" for v in vector) + "]"
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # `SET LOCAL` outside a transaction is a silent no-op that
                # leaves ef_search at the default and costs recall.
                await conn.execute(f"SET LOCAL hnsw.ef_search = {EF_SEARCH}")
                rows = await conn.fetch(SELECT_SQL, literal, self.floor, k)

        selected = []
        for row in rows:
            # The catalog is the source of truth for what is dispatchable; the
            # table only decides *which* of them the model gets to see. A row
            # with no matching implementation is a stale seed, not a tool.
            tool = self._by_name.get(row["name"])
            if tool is None:
                continue
            selected.append(
                ToolDef(
                    name=tool.name,
                    description=tool.description,
                    input_schema=tool.input_schema,
                    server=tool.server,
                    is_synthetic=row["is_synthetic"],
                    score=float(row["score"]),
                )
            )
        # §7.2.2. Added after the top-k, not counted against it: `k` bounds how
        # many tools retrieval chose, and spending one of those slots on a tool
        # that was never a candidate would make the arm look worse than it is.
        return _with_always_injected(selected, self._by_name.values())

    def extra_request_params(self) -> dict:
        return {}


class NativeToolSelector:
    """Provider-side filtering — §7.3, ADR 9.

    Every tool is sent with `defer_loading`, and Anthropic's BM25 tool-search
    tool decides what actually enters context. This is the arm that makes the
    result honest: beating full exposure is trivial, and beating the model
    provider's own implementation is the claim worth making.
    """

    mode = "native"

    # The API rejects a request in which every tool is deferred, so one stays
    # loaded. `search_docs` is the least specific tool in the catalog and the
    # cheapest to have present unnecessarily.
    ALWAYS_LOADED = "search_docs"

    def __init__(self, catalog: Sequence[ToolDef]) -> None:
        self._catalog = tuple(catalog)

    async def select(self, query: str, k: int) -> list[ToolDef]:
        return sorted(self._catalog, key=lambda t: t.name)

    def extra_request_params(self) -> dict:
        # The memory tools are never deferred. Deferring them would put them
        # behind a BM25 search the model has no reason to run — nobody phrases
        # a question so that "memory_write" is the matching tool — which is
        # ADR 11's failure mode reintroduced by the provider's mechanism
        # instead of ours.
        loaded = {self.ALWAYS_LOADED, *(t.name for t in always_injected(self._catalog))}
        return {
            "defer": [t.name for t in self._catalog if t.name not in loaded],
            "extra_tools": [NATIVE_SEARCH_TOOL],
        }


def build_selector(
    mode: str,
    catalog: Sequence[ToolDef],
    *,
    pool: asyncpg.Pool | None = None,
    embedder: Embedder | None = None,
    floor: float = 0.25,
):
    """Ablation axis C, resolved once at request construction.

    An unknown mode raises rather than falling back: a run that reports
    `semantic` in its config hash while actually broadcasting every tool would
    poison the comparison the whole thesis rests on.
    """
    if mode == "full":
        return FullToolSelector(catalog)
    if mode == "native":
        return NativeToolSelector(catalog)
    if mode == "semantic":
        if pool is None or embedder is None:
            raise ValueError("semantic tool selection needs a pool and an embedder")
        return SemanticToolSelector(catalog, pool, embedder, floor=floor)
    raise NotImplementedError(f"unknown TOOL_RETRIEVAL_MODE {mode!r}")
