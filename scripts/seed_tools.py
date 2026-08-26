#!/usr/bin/env python
"""Embed the tool catalog into `tool_defs` — TRD §7.1, §7.2.

    python scripts/seed_tools.py            # upsert the catalog
    python scripts/seed_tools.py --check    # report drift, write nothing

Semantic tool selection queries this table, so a stale row is a silently
wrong ablation arm: a tool whose description changed but whose embedding did
not will keep being retrieved for the old phrasing. `--check` exists so CI can
say so rather than discovering it in a metric.

Synthetic padding (§7.4) is M5's scaling curve and is not generated here.
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
from app.retrieval.embedder import get_embedder  # noqa: E402
from app.tools.registry import CATALOG  # noqa: E402
from app.tools.selector import embedding_text  # noqa: E402


def _vector_literal(vector) -> str:
    return "[" + ",".join(f"{v:.7g}" for v in vector) + "]"


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report drift, write nothing")
    args = parser.parse_args(argv)

    texts = [embedding_text(tool) for tool in CATALOG]
    conn = await asyncpg.connect(settings.database_url)
    try:
        stored = {
            row["name"]: row["description"]
            for row in await conn.fetch("SELECT name, description FROM tool_defs")
        }
        catalog = {tool.name: tool.description for tool in CATALOG}
        missing = sorted(set(catalog) - set(stored))
        orphaned = sorted(set(stored) - set(catalog))
        changed = sorted(n for n in set(catalog) & set(stored) if catalog[n] != stored[n])

        for label, names in (
            ("missing", missing),
            ("changed", changed),
            ("orphaned", orphaned),
        ):
            if names:
                print(f"{label}: {', '.join(names)}")

        if args.check:
            if missing or changed or orphaned:
                print(
                    f"\n{len(missing) + len(changed) + len(orphaned)} tool(s) out of "
                    "sync. Run scripts/seed_tools.py.",
                    file=sys.stderr,
                )
                return 1
            print(f"{len(CATALOG)} tools in sync; nothing to write")
            return 0

        vectors = get_embedder().embed(texts)
        async with conn.transaction():
            await conn.executemany(
                """
                INSERT INTO tool_defs
                    (id, name, description, input_schema, server, is_synthetic, embedding)
                VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7::vector)
                ON CONFLICT (name) DO UPDATE SET
                    description = EXCLUDED.description,
                    input_schema = EXCLUDED.input_schema,
                    server = EXCLUDED.server,
                    is_synthetic = EXCLUDED.is_synthetic,
                    embedding = EXCLUDED.embedding
                """,
                [
                    (
                        f"tool_{tool.name}",
                        tool.name,
                        tool.description,
                        json.dumps(tool.input_schema),
                        tool.server,
                        tool.is_synthetic,
                        _vector_literal(vector),
                    )
                    for tool, vector in zip(CATALOG, vectors, strict=True)
                ],
            )
            if orphaned:
                # A tool removed from the catalog but left in the table stays
                # selectable and dispatches to nothing.
                await conn.execute(
                    "DELETE FROM tool_defs WHERE name = ANY($1::text[]) AND NOT is_synthetic",
                    orphaned,
                )
        print(f"seeded {len(CATALOG)} tools" + (f", removed {len(orphaned)}" if orphaned else ""))
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
