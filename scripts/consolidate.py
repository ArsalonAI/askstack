#!/usr/bin/env python
"""Episodic → semantic consolidation — TRD §8.2.

    python scripts/consolidate.py --user arsalon
    python scripts/consolidate.py --user arsalon --dry-run   # cluster, don't call

`--dry-run` is the one worth reaching for first. Clustering is free and local;
the Claude call per cluster is not. Seeing which memories group together — and
which are rejected for sharing no entity — tells you whether the threshold is
doing anything sensible before you pay to consolidate them.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.memory.consolidation import (  # noqa: E402
    MIN_CLUSTER_SIZE,
    Consolidator,
    cluster_memories,
    render_cluster,
)
from app.memory.store import PostgresMemoryStore  # noqa: E402
from app.retrieval.embedder import get_embedder  # noqa: E402


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True, help="user_id to consolidate")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="show the clusters and stop; makes no model calls and writes nothing",
    )
    args = parser.parse_args(argv)

    pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=4)
    if pool is None:
        print("could not connect to the database", file=sys.stderr)
        return 2
    try:
        store = PostgresMemoryStore(pool)
        episodic, vectors = await store.live_with_vectors(args.user, "episodic")
        print(f"{len(episodic)} live episodic memories for {args.user!r}")
        if len(episodic) < MIN_CLUSTER_SIZE:
            print(f"fewer than {MIN_CLUSTER_SIZE}; nothing to consolidate")
            return 0

        clusters = cluster_memories(episodic, vectors)
        clustered = {mid for c in clusters for mid in c.ids}
        print(
            f"{len(clusters)} cluster(s), "
            f"{len(episodic) - len(clustered)} memory(ies) unclustered"
        )
        for index, cluster in enumerate(clusters, start=1):
            shared = ", ".join(sorted(cluster.shared_entities)) or "—"
            print(f"\n  cluster {index}  ({len(cluster.memories)} memories, shared: {shared})")
            for line in render_cluster(cluster).splitlines():
                print(f"    {line}")

        if args.dry_run:
            print("\ndry run; no model calls, nothing written")
            return 0
        if not clusters:
            return 0
        if not settings.anthropic_api_key:
            print("\nconsolidation needs ANTHROPIC_API_KEY", file=sys.stderr)
            return 2

        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        report = await Consolidator(
            store, get_embedder(), client, settings
        ).consolidate(args.user)
        print(
            f"\nclusters={report.clusters_formed} written={report.memories_written} "
            f"superseded={report.memories_superseded} skipped={report.facts_skipped}"
        )
        return 0
    finally:
        await pool.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
