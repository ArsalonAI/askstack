#!/usr/bin/env python
"""Corpus ingest — TRD §5.

Thin by design: everything substantive lives in `app/ingest/` and `app/facts/`
so it can be tested without a network.

    python scripts/ingest.py --facts-only --limit 50    # bounded smoke run
    python scripts/ingest.py                            # full run

The two window floors differ on purpose. The facts layer answers "what shipped
last month" and wants recency; the interpretive corpus answers "why did we drop
the sync client" and wants depth. FastAPI has 3,541 closed issues but only 89
created since 2025 — support traffic moved to Discussions years ago — so one
window cannot serve both. PRs cost three API calls each and issues cost one,
which is what makes the split affordable.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.facts.areas import load_areas_file, sync_areas  # noqa: E402
from app.ingest import chunking, facts  # noqa: E402
from app.ingest.checkout import WalkStats, ensure_checkout, walk_sources  # noqa: E402
from app.ingest.github import GitHubClient  # noqa: E402
from app.ingest.index import sync_chunks  # noqa: E402

log = logging.getLogger("ingest")

CHECKOUT_ROOT = Path(".cache/corpus")


def _floor(value) -> datetime | None:
    if value is None:
        return None
    return datetime.combine(value, datetime.min.time(), tzinfo=UTC)


async def ingest_facts(
    conn: asyncpg.Connection,
    gh: GitHubClient,
    *,
    pr_since: datetime | None,
    issue_since: datetime | None,
    limit: int | None,
) -> dict[str, int]:
    counts = {
        "pull_requests": 0,
        "pr_files": 0,
        "pr_reviews": 0,
        "issues": 0,
        "commits": 0,
        "releases": 0,
    }
    # issue number -> the PR whose body claims to close it
    closing: dict[int, int] = {}

    log.info("pull requests since %s", pr_since or "the beginning")
    async for pr in gh.pull_requests(pr_since):
        number, updated = pr["number"], pr["updated_at"]
        files = await gh.pull_request_files(number, updated)
        reviews = await gh.pull_request_reviews(number, updated)
        async with conn.transaction():
            await facts.upsert_pull_request(conn, pr, files, reviews)
        for issue_number in facts.closed_issue_refs(pr.get("body")):
            closing[issue_number] = number
        counts["pull_requests"] += 1
        counts["pr_files"] += len(files)
        counts["pr_reviews"] += len(reviews)
        if counts["pull_requests"] % 50 == 0:
            log.info("  %d pull requests", counts["pull_requests"])
        if limit and counts["pull_requests"] >= limit:
            log.info("  stopping at --limit %d", limit)
            break

    log.info("issues since %s", issue_since or "the beginning")
    async for issue in gh.issues(issue_since):
        async with conn.transaction():
            await facts.upsert_issue(conn, issue)
        counts["issues"] += 1
        if counts["issues"] % 200 == 0:
            log.info("  %d issues", counts["issues"])
        if limit and counts["issues"] >= limit:
            log.info("  stopping at --limit %d", limit)
            break

    dropped = await facts.link_closing_prs(conn, closing)
    counts["closing_links_dropped"] = dropped
    if dropped:
        log.info("%d closing links dropped (PR outside the window)", dropped)

    log.info("commits since %s", pr_since or "the beginning")
    async for commit in gh.commits(pr_since):
        detail = await gh.commit_detail(commit["sha"])
        async with conn.transaction():
            await facts.upsert_commit(conn, detail, detail.get("files", []))
        counts["commits"] += 1
        if counts["commits"] % 100 == 0:
            log.info("  %d commits", counts["commits"])
        if limit and counts["commits"] >= limit:
            log.info("  stopping at --limit %d", limit)
            break

    for release in await gh.releases():
        async with conn.transaction():
            await facts.upsert_release(conn, release)
        counts["releases"] += 1

    return counts


async def build_chunks(
    conn: asyncpg.Connection, gh: GitHubClient, sha: str, *, limit: int | None
) -> list[chunking.RawChunk]:
    """Docs and code from the pinned checkout, issues from the facts layer."""
    checkout = await ensure_checkout(gh._client, gh.repo, sha, CHECKOUT_ROOT)
    chunks: list[chunking.RawChunk] = []
    walk_stats = WalkStats()
    files = 0

    for source, path, absolute in walk_sources(checkout, walk_stats):
        try:
            text = absolute.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if not text.strip():
            continue
        chunks.extend(
            chunking.chunk_docs(path, text)
            if source == "docs"
            else chunking.chunk_code(path, text)
        )
        files += 1
        if files % 500 == 0:
            log.info("  %d files -> %d chunks", files, len(chunks))
    log.info("%d files -> %d docs/code chunks", files, len(chunks))

    # §5.2 indexes closed issues only: an open issue describes a problem, not a
    # resolution, and pollutes an answer corpus. They stay in the facts layer.
    rows = await conn.fetch(
        """
        SELECT i.number, i.title, i.body,
               coalesce(array_agg(l.label) FILTER (WHERE l.label IS NOT NULL), '{}') AS labels
        FROM issues i
        LEFT JOIN issue_labels l ON l.issue_number = i.number
        WHERE i.state = 'closed'
        GROUP BY i.number
        ORDER BY i.number DESC
        """
        + (f" LIMIT {int(limit)}" if limit else "")
    )
    log.info("%d closed issues", len(rows))

    for index, row in enumerate(rows, 1):
        comments = await gh.issue_comments(row["number"])
        chunks.extend(
            chunking.chunk_issue(
                row["number"], row["title"], row["body"], row["labels"], comments
            )
        )
        if index % 250 == 0:
            log.info("  %d/%d issues -> %d chunks", index, len(rows), len(chunks))

    return chunks, walk_stats


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--facts-only",
        action="store_true",
        help="skip the semantic index; leaves the run marker incomplete",
    )
    parser.add_argument("--limit", type=int, help="cap items per source (smoke runs)")
    parser.add_argument("--no-cache", action="store_true", help="ignore .cache/gh")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )
    # httpx logs every request at INFO, which buries our own progress lines.
    logging.getLogger("httpx").setLevel(logging.DEBUG if args.verbose else logging.WARNING)

    if not settings.github_token:
        log.error("GITHUB_TOKEN is empty; unauthenticated ingest caps at 60 req/hour")
        return 2

    pr_since = _floor(settings.ingest_since)
    issue_since = _floor(settings.ingest_issues_since)
    run_id = f"ing_{uuid.uuid4().hex[:12]}"

    conn = await asyncpg.connect(settings.database_url)
    try:
        async with GitHubClient(
            token=settings.github_token,
            repo=settings.corpus_repo,
            use_cache=not args.no_cache,
        ) as gh:
            sha, committed_at = await gh.resolve_ref(settings.corpus_ref)
            log.info("pinned %s@%s (%s)", settings.corpus_repo, sha[:12], committed_at.date())

            await facts.start_run(
                conn,
                run_id=run_id,
                corpus_repo=settings.corpus_repo,
                corpus_ref=settings.corpus_ref,
                resolved_sha=sha,
                since=pr_since,
                embedding_model=settings.embedding_model,
            )

            areas = load_areas_file(settings.areas_file)
            log.info("loaded %d areas from %s", await sync_areas(conn, areas), settings.areas_file)

            counts = await ingest_facts(
                conn,
                gh,
                pr_since=pr_since,
                issue_since=issue_since,
                limit=args.limit,
            )

            stats = {
                **counts,
                "areas": len(areas),
                "issue_since": issue_since.isoformat() if issue_since else None,
                "fetch": gh.stats.as_dict(),
            }

            if args.facts_only:
                # The marker stays unset: TRD §4.2 requires both substrates
                # before a run counts as complete, and a service that started
                # against a facts-only corpus could find a discussion but not
                # confirm whether it shipped.
                await facts.record_stats(conn, run_id, stats)
                log.info("facts-only run %s recorded, marker left incomplete", run_id)
            else:
                from app.retrieval.embedder import get_embedder

                chunks, walk_stats = await build_chunks(conn, gh, sha, limit=args.limit)
                index_stats = await sync_chunks(
                    conn, get_embedder(), chunks, prune=not args.limit
                )
                stats["index"] = index_stats.as_dict()
                stats["corpus"] = walk_stats.as_dict()
                stats["fetch"] = gh.stats.as_dict()
                await facts.complete_run(conn, run_id, stats)
                log.info("run %s complete", run_id)

            log.info("%s", stats)
    finally:
        await conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
