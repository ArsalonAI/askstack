#!/usr/bin/env python
"""Synthetic tool padding for the scaling curve — TRD §7.4, ADR 10.

    python scripts/gen_synthetic_tools.py --size 200     # pad to 200 tools
    python scripts/gen_synthetic_tools.py --size 0       # remove all padding

**Generated from templates, not from a model.** §7.4 asks for "plausible tool
definitions from adjacent domains" and says nothing about how they are made. A
Claude call per tool would cost real money for output nobody reads, and — worse
— would make the catalog non-reproducible: the scaling curve is a claim about
how selection accuracy behaves at 500 tools, and that claim is only checkable
if the 500 tools are the same 500 next time. Templates are deterministic,
free, and re-runnable with no API key, which is the same property §14.4 wants
from the retrieval metrics.

**They must be plausible, not random.** Padding with obvious noise would make
the curve flattering: a selector only has to beat gibberish to look good at
scale. These are drawn from domains a real engineering agent would plausibly
have wired up — cloud infra, CI, observability, ticketing, on-call — and
several deliberately shadow the real catalog's vocabulary ("list", "search",
"status", "recent") so that the distractors compete for the same queries.

They carry `is_synthetic = TRUE`, are never dispatchable (`ToolRegistry` raises
on one), and every reported number splits real from synthetic — ADR 10.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.interfaces import ToolDef  # noqa: E402
from app.retrieval.embedder import get_embedder  # noqa: E402
from app.tools.registry import CATALOG  # noqa: E402
from app.tools.selector import embedding_text  # noqa: E402

# Each domain contributes an object vocabulary and a house style. The verbs are
# shared, which is the point: "list_recent_deployments" and the real
# "merged_prs" are both "recent things, listed", so the selector has to
# discriminate on subject rather than on shape.
DOMAINS: dict[str, dict] = {
    "cloud": {
        "blurb": "cloud infrastructure",
        "objects": [
            ("instance", "compute instance"),
            ("bucket", "object storage bucket"),
            ("cluster", "managed Kubernetes cluster"),
            ("volume", "attached block volume"),
            ("load_balancer", "load balancer"),
            ("dns_record", "DNS record"),
            ("secret", "stored secret"),
            ("vpc", "virtual private cloud"),
            ("image", "machine image"),
            ("snapshot", "volume snapshot"),
        ],
    },
    "ci": {
        "blurb": "continuous integration",
        "objects": [
            ("build", "CI build"),
            ("pipeline", "CI pipeline"),
            ("artifact", "build artifact"),
            ("runner", "CI runner"),
            ("deployment", "deployment"),
            ("job", "pipeline job"),
            ("cache_entry", "build cache entry"),
            ("workflow", "workflow definition"),
        ],
    },
    "observability": {
        "blurb": "metrics, logs, and traces",
        "objects": [
            ("metric", "time-series metric"),
            ("alert", "alerting rule"),
            ("dashboard", "dashboard"),
            ("log_stream", "log stream"),
            ("trace", "distributed trace"),
            ("span", "trace span"),
            ("slo", "service level objective"),
            ("error_group", "grouped error"),
        ],
    },
    "ticketing": {
        "blurb": "project and ticket tracking",
        "objects": [
            ("ticket", "tracked ticket"),
            ("epic", "epic"),
            ("sprint", "sprint"),
            ("board", "project board"),
            ("comment", "ticket comment"),
            ("label", "ticket label"),
            ("assignee", "ticket assignee"),
        ],
    },
    "oncall": {
        "blurb": "incident response and on-call",
        "objects": [
            ("incident", "declared incident"),
            ("page", "on-call page"),
            ("rotation", "on-call rotation"),
            ("postmortem", "incident postmortem"),
            ("escalation", "escalation policy"),
            ("maintenance_window", "scheduled maintenance window"),
        ],
    },
    "warehouse": {
        "blurb": "the analytics warehouse",
        "objects": [
            ("table", "warehouse table"),
            ("view", "materialized view"),
            ("query_run", "executed query"),
            ("schema", "warehouse schema"),
            ("column", "table column"),
            ("partition", "table partition"),
            ("sync", "ingestion sync"),
        ],
    },
    "compliance": {
        "blurb": "security and compliance",
        "objects": [
            ("policy", "compliance policy"),
            ("audit_event", "audit-log event"),
            ("access_grant", "access grant"),
            ("vulnerability", "reported vulnerability"),
            ("certificate", "TLS certificate"),
            ("review", "access review"),
            ("exception", "policy exception"),
        ],
    },
}

# Verb → (name template, description template, schema builder key).
VERBS: list[tuple[str, str, str, str]] = [
    ("list", "list_{obj}s", "List every {desc} in the {blurb} account, optionally "
     "filtered by status. Returns the complete set with a count.", "filter"),
    ("get", "get_{obj}", "Fetch one {desc} by id and return its current "
     "configuration and status.", "id"),
    ("search", "search_{obj}s", "Search {desc}s by free-text query across their "
     "names, tags, and descriptions. Returns ranked matches.", "query"),
    ("recent", "recent_{obj}_changes", "List changes to {desc}s over a time "
     "window. Call this for 'what changed' questions about {blurb}.", "window"),
    ("status", "{obj}_status", "Report the current health and status of one "
     "{desc}, including recent state transitions.", "id"),
    ("owner", "{obj}_owner", "Report which team or person owns a given {desc}.",
     "id"),
    ("count", "count_{obj}s", "Count {desc}s matching a status filter. Returns "
     "an exact total for {blurb} reporting.", "filter"),
    ("history", "{obj}_history", "Return the change history of one {desc}, "
     "oldest first, with the actor for each change.", "id"),
    ("compare", "compare_{obj}s", "Compare two {desc}s field by field and "
     "report the differences.", "id"),
    ("tags", "{obj}_tags", "List the tags and metadata attached to one {desc}.",
     "id"),
]

_STR = {"type": "string"}


def _schema(kind: str, desc: str) -> dict:
    fields = {
        "filter": ({"status": {**_STR, "description": f"Restrict to {desc}s in this status."}}, []),
        "id": ({"id": {**_STR, "description": f"Identifier of the {desc}."}}, ["id"]),
        "query": ({"query": {**_STR, "description": f"Free-text query over {desc}s."}}, ["query"]),
        "window": (
            {"window": {**_STR, "description": "Time window, e.g. 'last 30 days'."}},
            ["window"],
        ),
    }[kind]
    return {
        "type": "object",
        "properties": fields[0],
        "required": fields[1],
        "additionalProperties": False,
    }


def generate(count: int) -> list[ToolDef]:
    """`count` synthetic tools, deterministically.

    Iterates verb-major so that a smaller catalog is a strict prefix of a
    larger one — the 50-tool point on the curve is the first 50 of the 500, not
    a different sample. Without that the curve would confound catalog *size*
    with catalog *composition*, and a bump at 200 could be either.
    """
    out: list[ToolDef] = []
    real = {tool.name for tool in CATALOG}
    for verb_index, (_, name_tpl, desc_tpl, schema_kind) in enumerate(VERBS):
        for domain, spec in DOMAINS.items():
            for obj, desc in spec["objects"]:
                if len(out) >= count:
                    return out
                name = name_tpl.format(obj=obj)
                if name in real or any(t.name == name for t in out):
                    continue
                out.append(
                    ToolDef(
                        name=name,
                        description=desc_tpl.format(desc=desc, blurb=spec["blurb"]),
                        input_schema=_schema(schema_kind, desc),
                        server=domain,
                        is_synthetic=True,
                    )
                )
        del verb_index
    if len(out) < count:
        raise ValueError(
            f"the templates yield {len(out)} distinct tools; {count} requested. "
            "Add domains or objects rather than duplicating names — a catalog "
            "with repeated tools measures deduplication, not scale."
        )
    return out


def _vector_literal(vector) -> str:
    return "[" + ",".join(f"{v:.7g}" for v in vector) + "]"


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--size", type=int, required=True,
        help="total catalog size including the real tools. 0 removes all padding.",
    )
    args = parser.parse_args(argv)

    padding = max(args.size - len(CATALOG), 0)
    tools = generate(padding)
    print(f"real={len(CATALOG)} synthetic={len(tools)} total={len(CATALOG) + len(tools)}")

    conn = await asyncpg.connect(settings.database_url)
    try:
        await conn.execute("DELETE FROM tool_defs WHERE is_synthetic")
        if not tools:
            print("padding removed")
            return 0

        embedder = get_embedder()
        vectors = embedder.embed([embedding_text(t) for t in tools])
        async with conn.transaction():
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
        print(f"seeded {len(tools)} synthetic tools")
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
