#!/usr/bin/env python3
"""The tool-scaling curve — PRD §5.4, TRD §7.3. Costs nothing to run.

    python evals/sweep.py                    # the full curve, no API key needed
    python evals/sweep.py --sizes 13 50      # a subset
    python evals/sweep.py --json evals/baselines/sweep.json

**This measures the thesis without an agent, and that is a design decision
rather than a budget compromise.** PRD §5.4 claims that retrieved tools beat
broadcasting them as the catalog grows. That claim has exactly two testable
halves, and neither needs a model to answer:

- *Does selection still find the right tool at 500 tools?* The selector is a
  pgvector query over `tool_defs` using a local embedding model. Running it
  through an agent adds a paid round trip and answers the same question with
  more noise.
- *What does the prompt cost at 500 tools?* Tool definitions are serialised
  deterministically (§10 sorts them by name), so the token count is a property
  of the catalog, not of a conversation.

The half that genuinely needs an agent — whether the model *answers better*
with a well-chosen tool set — is measured once by the golden-set baseline, not
at every point on a curve. Paying per catalog size for it would cost roughly
$170 a run and re-measure the agent four times to learn one thing about the
selector.

**`native` mode is excluded from the accuracy curve and reported as prompt
cost only.** Its filtering happens server-side inside Anthropic's tool-search
tool (ADR 9), so there is no local selection to score; scoring it would mean
paying for an agent run per catalog size, which is the cost this module exists
to avoid. Its prompt-size column is still exact and still worth plotting.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import asyncpg
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.retrieval.embedder import get_embedder  # noqa: E402
from app.tools.registry import ALWAYS_INJECTED, CATALOG  # noqa: E402
from app.tools.selector import (  # noqa: E402
    NATIVE_SEARCH_TOOL,
    SemanticToolSelector,
    to_api_tools,
)

QUESTIONS = Path(__file__).parent / "golden" / "questions.yaml"
DEFAULT_SIZES = (len(CATALOG), 50, 200, 500)

# Opus 5 input rate. Prompt cost is reported per *request*, not per turn: a
# turn makes several, and multiplying here would bake in an agent-shaped
# assumption this module is deliberately free of.
PRICE_IN_PER_TOKEN = 5 / 1_000_000


def jaccard(got: set[str], want: set[str]) -> float:
    if not got and not want:
        return 1.0
    union = got | want
    return len(got & want) / len(union) if union else 0.0


def recall(got: set[str], want: set[str]) -> float:
    """Did the selector surface the gold tool at all?

    The metric the curve actually needs. Jaccard falls as `k` grows relative to
    `gold_tools` whether or not selection got better, which is §17 Q12's
    complaint about exact match in a different costume. Recall asks the
    question PRD §5.4 poses — *is the right tool still findable* — and is
    comparable across catalog sizes because it does not depend on `k`.
    """
    return len(got & want) / len(want) if want else 1.0


def prompt_tokens(tools) -> int:
    """Serialised size of a tool payload, in tokens.

    Estimated at 3.5 characters per token rather than counted through the API.
    `count_tokens` is free but needs a key and a network round trip, and this
    module's whole point is that the curve reproduces offline. The estimate is
    deliberately pessimistic (English runs nearer 4), so the reported cost is
    an upper bound and the *shape* of the curve — which is the claim — is
    unaffected by the constant.
    """
    return int(len(json.dumps(to_api_tools(tools))) / 3.5)


async def measure(conn, embedder, questions, size: int) -> dict:
    """One point on the curve. No model call anywhere in here."""
    rows = await conn.fetch(
        "SELECT name, is_synthetic FROM tool_defs ORDER BY is_synthetic, name"
    )
    catalog_size = len(rows)
    real_names = {r["name"] for r in rows if not r["is_synthetic"]}

    selector = SemanticToolSelector(
        CATALOG, _PoolShim(conn), embedder, floor=settings.tool_similarity_floor
    )

    recalls: list[float] = []
    jaccards: list[float] = []
    crowded_out = 0          # top-k slots taken by synthetic tools
    topk_total = 0
    offered: list[int] = []  # real tools that actually survived to the model

    for question in questions:
        gold = set(question.get("gold_tools") or [])
        if not gold:
            continue
        chosen = await selector.select(question["question"], settings.tool_retrieval_k)
        # §7.2.2 — the always-injected pair is present regardless of query and
        # measures nothing about the selector.
        names = {t.name for t in chosen if t.name not in ALWAYS_INJECTED}
        recalls.append(recall(names, gold))
        jaccards.append(jaccard(names, gold))
        offered.append(len(names))

        # The same query the selector just ran, read raw. Counting synthetics
        # *after* selection always yields zero — `SemanticToolSelector` drops
        # rows with no dispatchable implementation, so a synthetic that won a
        # top-k slot vanishes before it can be counted. The slot is still
        # spent: the model is offered fewer than k tools, and the gold tool it
        # displaced is simply absent. That displacement is the mechanism behind
        # the recall curve, and measuring it post-filter hides it completely.
        raw = await _raw_topk(conn, embedder, question["question"])
        topk_total += len(raw)
        crowded_out += sum(1 for r in raw if r["is_synthetic"])

    # ADR 10: every reported number splits real from synthetic.
    return {
        "requested_size": size,
        "catalog_size": catalog_size,
        "real_tools": len(real_names),
        "synthetic_tools": catalog_size - len(real_names),
        "questions": len(recalls),
        "tool_recall": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
        "tool_jaccard": round(sum(jaccards) / len(jaccards), 4) if jaccards else 0.0,
        # What fraction of the top-k the distractors took. This is the driver
        # of the recall column, not a side statistic.
        "crowd_out_rate": round(crowded_out / topk_total, 4) if topk_total else 0.0,
        "tools_offered": round(sum(offered) / len(offered), 2) if offered else 0.0,
        "prompt_tokens": {
            "semantic": prompt_tokens(sorted(CATALOG, key=lambda t: t.name)[
                : settings.tool_retrieval_k
            ]),
            "full": _full_tokens(catalog_size),
            "native": _native_tokens(catalog_size),
        },
    }


def _per_tool_tokens() -> float:
    ordered = sorted(CATALOG, key=lambda t: t.name)
    return prompt_tokens(ordered) / len(ordered)


def _full_tokens(catalog_size: int) -> int:
    """Every definition, every request — §7.3's control arm."""
    return int(_per_tool_tokens() * catalog_size)


def _native_tokens(catalog_size: int) -> int:
    """Names only until the model searches, plus the provider's search tool.

    Deferred tools still occupy the request: the API is told they exist. The
    name-and-type stub is roughly a quarter of a full definition, and the
    server-side search tool is a fixed addition.
    """
    stub = _per_tool_tokens() * 0.25
    return int(stub * catalog_size + len(json.dumps(NATIVE_SEARCH_TOOL)) / 3.5)


async def _raw_topk(conn, embedder, query: str):
    """The selector's own SQL, unfiltered — see the call site for why."""
    from app.tools.selector import EF_SEARCH, SELECT_SQL

    vector = embedder.embed_query(query)
    literal = "[" + ",".join(f"{v:.7g}" for v in vector) + "]"
    async with conn.transaction():
        await conn.execute(f"SET LOCAL hnsw.ef_search = {EF_SEARCH}")
        return await conn.fetch(
            SELECT_SQL, literal, settings.tool_similarity_floor,
            settings.tool_retrieval_k,
        )


class _PoolShim:
    """`SemanticToolSelector` expects a pool; this sweep holds one connection.

    Sharing the connection is what keeps the sweep a single transaction-free
    read loop instead of 200 acquire/release cycles.
    """

    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        conn = self._conn

        class _Ctx:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


def print_curve(points: list[dict]) -> None:
    print(
        f"\n{'catalog':>8}{'real':>6}{'synth':>7}"
        f"{'recall':>9}{'jaccard':>9}{'crowd-out':>11}{'offered':>9}"
        f"{'semantic':>10}{'native':>9}{'full':>9}"
    )
    for p in points:
        t = p["prompt_tokens"]
        print(
            f"{p['catalog_size']:>8}{p['real_tools']:>6}{p['synthetic_tools']:>7}"
            f"{p['tool_recall']:>9.3f}{p['tool_jaccard']:>9.3f}"
            f"{p['crowd_out_rate']:>11.3f}{p['tools_offered']:>9.2f}"
            f"{t['semantic']:>10}{t['native']:>9}{t['full']:>9}"
        )

    first, last = points[0], points[-1]
    grew = last["catalog_size"] / max(first["catalog_size"], 1)
    print(f"\n  catalog grew {grew:.0f}x ({first['catalog_size']} → {last['catalog_size']})")
    print(
        f"  semantic prompt   {first['prompt_tokens']['semantic']:>6} → "
        f"{last['prompt_tokens']['semantic']:>6} tokens  "
        f"({last['prompt_tokens']['semantic'] / max(first['prompt_tokens']['semantic'], 1):.1f}x)"
    )
    print(
        f"  full prompt       {first['prompt_tokens']['full']:>6} → "
        f"{last['prompt_tokens']['full']:>6} tokens  "
        f"({last['prompt_tokens']['full'] / max(first['prompt_tokens']['full'], 1):.1f}x)"
    )
    delta = (last["prompt_tokens"]["full"] - last["prompt_tokens"]["semantic"])
    print(
        f"  at {last['catalog_size']} tools, semantic saves {delta:,} input tokens per "
        f"request (${delta * PRICE_IN_PER_TOKEN:.4f})"
    )
    print(
        f"  tool recall       {first['tool_recall']:.3f} → {last['tool_recall']:.3f}"
        f"   ({last['tool_recall'] - first['tool_recall']:+.3f})"
    )
    print(
        f"  crowd-out rate    {first['crowd_out_rate']:.3f} → {last['crowd_out_rate']:.3f}"
        f"   (share of the top-{settings.tool_retrieval_k} taken by distractors)"
    )
    print(
        f"  real tools shown  {first['tools_offered']:.2f} → {last['tools_offered']:.2f}"
        f"   (of k={settings.tool_retrieval_k})"
    )


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--json", type=Path, help="write the curve here")
    args = parser.parse_args(argv)

    # questions.yaml is a bare list, not a mapping with a `questions` key.
    questions = yaml.safe_load(QUESTIONS.read_text())
    embedder = get_embedder()
    conn = await asyncpg.connect(settings.database_url)
    points: list[dict] = []
    try:
        for size in args.sizes:
            # Re-pad in place. Sizes are prefix-stable (see the generator), so
            # the 50-tool point is the first 50 of the 500 rather than a
            # different sample — otherwise the curve confounds size with
            # composition.
            await _repad(conn, embedder, size)
            point = await measure(conn, embedder, questions, size)
            points.append(point)
            print(
                f"  size={point['catalog_size']:>4} recall={point['tool_recall']:.3f}",
                file=sys.stderr,
                flush=True,
            )
    finally:
        # Restore the catalog the rest of the system expects. The sweep borrows
        # `tool_defs` and pads it to 500; leaving it padded means the next
        # agent run selects from a catalog its own config hash does not
        # describe. `evals/runner.py` now refuses such a run outright, but the
        # borrower should put the table back rather than relying on the next
        # caller to notice.
        try:
            await _repad(conn, embedder, settings.tool_catalog_size or len(CATALOG))
            print(
                f"  restored tool_defs to {settings.tool_catalog_size or len(CATALOG)}",
                file=sys.stderr,
            )
        finally:
            await conn.close()

    print_curve(points)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "embedding_model": settings.embedding_model,
                    "tool_retrieval_k": settings.tool_retrieval_k,
                    "similarity_floor": settings.tool_similarity_floor,
                    "points": points,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {args.json}")
    return 0


async def _repad(conn, embedder, size: int) -> None:
    from app.tools.selector import embedding_text
    from scripts.gen_synthetic_tools import _vector_literal, generate

    await conn.execute("DELETE FROM tool_defs WHERE is_synthetic")
    tools = generate(max(size - len(CATALOG), 0))
    if not tools:
        return
    vectors = embedder.embed([embedding_text(t) for t in tools])
    for tool, vector in zip(tools, vectors, strict=True):
        await conn.execute(
            "INSERT INTO tool_defs"
            " (id, name, description, input_schema, server, is_synthetic, embedding)"
            " VALUES ($1, $2, $3, $4::jsonb, $5, TRUE, $6::vector)",
            f"syn_{tool.name}",
            tool.name,
            tool.description,
            json.dumps(tool.input_schema),
            tool.server,
            _vector_literal(vector),
        )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
