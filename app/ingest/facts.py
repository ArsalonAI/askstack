"""GitHub payloads -> the facts layer — TRD §4.2, §5.4.

The facts layer is a projection, not a source of truth (ADR 17): rows are
replaced rather than versioned, and the whole thing is rebuildable from the
API at a given `CORPUS_REF`. So every write here is an upsert, and none of it
carries history.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import asyncpg

# "Closes #1234", "fixes #1234", "Resolved #1234" — GitHub's own closing-keyword
# set. Parsing PR bodies is free; the timeline API that reports this directly
# would cost one extra call per issue.
CLOSES_RE = re.compile(
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*:?\s+#(\d+)", re.IGNORECASE
)
# FastAPI squash-merges as "Title (#1234)", which is how a commit finds its PR
# without a second API call.
COMMIT_PR_RE = re.compile(r"\(#(\d+)\)\s*$")

GHOST = "ghost"  # GitHub returns a null user for deleted accounts


def _login(payload: dict | None) -> str:
    if not payload:
        return GHOST
    return payload.get("login") or GHOST


def _ts(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _milestone(payload: dict | None) -> str | None:
    return payload.get("title") if payload else None


def pr_state(payload: dict) -> str:
    """GitHub reports open|closed plus a `merged_at`; the schema wants the
    three-way distinction, because reporting a merely-closed PR as shipped work
    is the defining failure of this product (PRD §5.2)."""
    if payload.get("merged_at"):
        return "merged"
    return "closed" if payload["state"] == "closed" else "open"


def closed_issue_refs(body: str | None) -> set[int]:
    return {int(m) for m in CLOSES_RE.findall(body or "")}


def commit_pr_number(message: str) -> int | None:
    match = COMMIT_PR_RE.search(message.splitlines()[0])
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------- upserts


async def upsert_pull_request(
    conn: asyncpg.Connection, payload: dict, files: list[dict], reviews: list[dict]
) -> None:
    number = payload["number"]
    # The list endpoint omits additions/deletions -- only the per-PR detail
    # call carries them, and TRD §5.4 budgets three calls per PR, not four.
    # Summing the file rows we already fetched gives the same numbers free.
    additions = sum(f.get("additions", 0) for f in files)
    deletions = sum(f.get("deletions", 0) for f in files)

    await conn.execute(
        """
        INSERT INTO pull_requests (
            number, title, body, state, is_draft, author, created_at,
            merged_at, closed_at, milestone, additions, deletions, url
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
        ON CONFLICT (number) DO UPDATE SET
            title = EXCLUDED.title, body = EXCLUDED.body, state = EXCLUDED.state,
            is_draft = EXCLUDED.is_draft, author = EXCLUDED.author,
            created_at = EXCLUDED.created_at, merged_at = EXCLUDED.merged_at,
            closed_at = EXCLUDED.closed_at, milestone = EXCLUDED.milestone,
            additions = EXCLUDED.additions, deletions = EXCLUDED.deletions,
            url = EXCLUDED.url
        """,
        number,
        payload["title"],
        payload.get("body"),
        pr_state(payload),
        bool(payload.get("draft")),
        _login(payload.get("user")),
        _ts(payload["created_at"]),
        _ts(payload.get("merged_at")),
        _ts(payload.get("closed_at")),
        _milestone(payload.get("milestone")),
        additions,
        deletions,
        payload["html_url"],
    )

    # Children are replaced wholesale: a file removed from a PR must disappear,
    # and an upsert alone would leave it behind.
    await conn.execute("DELETE FROM pr_files WHERE pr_number = $1", number)
    if files:
        await conn.executemany(
            "INSERT INTO pr_files (pr_number, path, additions, deletions) "
            "VALUES ($1,$2,$3,$4) ON CONFLICT DO NOTHING",
            [
                (number, f["filename"], f.get("additions", 0), f.get("deletions", 0))
                for f in files
            ],
        )

    await conn.execute("DELETE FROM pr_reviews WHERE pr_number = $1", number)
    rows = [
        (number, _login(r.get("user")), (r.get("state") or "").lower(), _ts(r["submitted_at"]))
        for r in reviews
        if r.get("submitted_at")
    ]
    if rows:
        await conn.executemany(
            "INSERT INTO pr_reviews (pr_number, reviewer, state, submitted_at) "
            "VALUES ($1,$2,$3,$4) ON CONFLICT DO NOTHING",
            rows,
        )


async def upsert_issue(conn: asyncpg.Connection, payload: dict) -> None:
    number = payload["number"]
    await conn.execute(
        """
        INSERT INTO issues (
            number, title, body, state, author, created_at, closed_at, milestone, url
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
        ON CONFLICT (number) DO UPDATE SET
            title = EXCLUDED.title, body = EXCLUDED.body, state = EXCLUDED.state,
            author = EXCLUDED.author, created_at = EXCLUDED.created_at,
            closed_at = EXCLUDED.closed_at, milestone = EXCLUDED.milestone,
            url = EXCLUDED.url
        """,
        number,
        payload["title"],
        payload.get("body"),
        payload["state"],
        _login(payload.get("user")),
        _ts(payload["created_at"]),
        _ts(payload.get("closed_at")),
        _milestone(payload.get("milestone")),
        payload["html_url"],
    )

    await conn.execute("DELETE FROM issue_labels WHERE issue_number = $1", number)
    labels = {label["name"] for label in payload.get("labels", []) if label.get("name")}
    if labels:
        await conn.executemany(
            "INSERT INTO issue_labels (issue_number, label) VALUES ($1,$2) "
            "ON CONFLICT DO NOTHING",
            [(number, label) for label in sorted(labels)],
        )


async def link_closing_prs(conn: asyncpg.Connection, links: dict[int, int]) -> int:
    """Set `issues.closed_by_pr` from the PR bodies that claimed to close them.

    Returns the number of links dropped because the closing PR falls outside
    the PR ingest window. That count is recorded on the run: a silently dropped
    foreign key becomes an unexplainable eval gap three weeks later.
    """
    if not links:
        return 0
    rows = await conn.fetch(
        """
        UPDATE issues SET closed_by_pr = v.pr
        FROM (SELECT * FROM unnest($1::int[], $2::int[]) AS t(issue, pr)) AS v
        WHERE issues.number = v.issue
          AND issues.state = 'closed'
          AND EXISTS (SELECT 1 FROM pull_requests p WHERE p.number = v.pr)
        RETURNING issues.number
        """,
        list(links.keys()),
        list(links.values()),
    )
    return len(links) - len(rows)


async def upsert_commit(conn: asyncpg.Connection, payload: dict, files: list[dict]) -> None:
    sha = payload["sha"]
    commit = payload["commit"]
    message = commit["message"]
    pr_number = commit_pr_number(message)

    # The squash-merge reference can point outside the PR window, and the FK
    # would reject it. Drop the link rather than the commit.
    if pr_number is not None:
        known = await conn.fetchval(
            "SELECT 1 FROM pull_requests WHERE number = $1", pr_number
        )
        if not known:
            pr_number = None

    await conn.execute(
        """
        INSERT INTO commits (sha, author, authored_at, message, pr_number)
        VALUES ($1,$2,$3,$4,$5)
        ON CONFLICT (sha) DO UPDATE SET
            author = EXCLUDED.author, authored_at = EXCLUDED.authored_at,
            message = EXCLUDED.message, pr_number = EXCLUDED.pr_number
        """,
        sha,
        _login(payload.get("author")) if payload.get("author") else commit["author"]["name"],
        _ts(commit["author"]["date"]),
        message,
        pr_number,
    )

    await conn.execute("DELETE FROM commit_files WHERE sha = $1", sha)
    paths = {f["filename"] for f in files}
    if paths:
        await conn.executemany(
            "INSERT INTO commit_files (sha, path) VALUES ($1,$2) ON CONFLICT DO NOTHING",
            [(sha, path) for path in sorted(paths)],
        )


async def upsert_release(conn: asyncpg.Connection, payload: dict) -> None:
    await conn.execute(
        """
        INSERT INTO releases (tag, name, published_at, body, url)
        VALUES ($1,$2,$3,$4,$5)
        ON CONFLICT (tag) DO UPDATE SET
            name = EXCLUDED.name, published_at = EXCLUDED.published_at,
            body = EXCLUDED.body, url = EXCLUDED.url
        """,
        payload["tag_name"],
        payload.get("name"),
        _ts(payload.get("published_at") or payload.get("created_at")),
        payload.get("body"),
        payload["html_url"],
    )


# ---------------------------------------------------------- run bookkeeping


async def start_run(
    conn: asyncpg.Connection,
    *,
    run_id: str,
    corpus_repo: str,
    corpus_ref: str,
    resolved_sha: str,
    since: datetime | None,
    embedding_model: str,
) -> None:
    """Recorded before any fetching. `completed_at` stays NULL until both the
    facts layer and the semantic index are written (TRD §4.2)."""
    await conn.execute(
        """
        INSERT INTO ingest_runs (
            id, corpus_repo, corpus_ref, resolved_sha, since, embedding_model
        ) VALUES ($1,$2,$3,$4,$5,$6)
        """,
        run_id,
        corpus_repo,
        corpus_ref,
        resolved_sha,
        since,
        embedding_model,
    )


async def complete_run(
    conn: asyncpg.Connection, run_id: str, stats: dict[str, Any]
) -> None:
    import json

    await conn.execute(
        "UPDATE ingest_runs SET completed_at = now(), stats = $2 WHERE id = $1",
        run_id,
        json.dumps(stats),
    )


async def record_stats(conn: asyncpg.Connection, run_id: str, stats: dict[str, Any]) -> None:
    """Write stats without marking the run complete — used by `--facts-only`,
    which deliberately leaves the semantic index unbuilt."""
    import json

    await conn.execute(
        "UPDATE ingest_runs SET stats = $2 WHERE id = $1", run_id, json.dumps(stats)
    )
