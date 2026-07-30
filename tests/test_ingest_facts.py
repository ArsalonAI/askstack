"""Facts-layer parsing and upserts, against fixtures rather than the network.

The parsing tests are pure. The upsert tests use a real Postgres because the
things worth catching here — the closing-PR foreign key, wholesale child
replacement — are constraint behaviour, not Python behaviour.
"""

import asyncpg
import pytest
from alembic import command

from app.ingest import facts

# ------------------------------------------------------------------ parsing


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"state": "closed", "merged_at": "2026-01-01T00:00:00Z"}, "merged"),
        ({"state": "closed", "merged_at": None}, "closed"),
        ({"state": "open", "merged_at": None}, "open"),
        # A merged PR whose `state` GitHub still reports as open should never
        # read as unmerged — PRD §5.2 turns on exactly this distinction.
        ({"state": "open", "merged_at": "2026-01-01T00:00:00Z"}, "merged"),
    ],
)
def test_pr_state(payload, expected):
    assert facts.pr_state(payload) == expected


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("Closes #123", {123}),
        ("fixes #1 and Resolves #2", {1, 2}),
        ("closed #42\nfixed #43", {42, 43}),
        ("Fixes: #7", {7}),
        ("see #99 for context", set()),  # a bare mention is not a closing link
        (None, set()),
    ],
)
def test_closed_issue_refs(body, expected):
    assert facts.closed_issue_refs(body) == expected


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Fix routing bug (#11234)", 11234),
        ("Fix routing bug (#11234)\n\nlonger body (#999)", 11234),
        ("Merge pull request #123 from foo/bar", None),  # not the squash form
        ("no reference at all", None),
    ],
)
def test_commit_pr_number(message, expected):
    assert facts.commit_pr_number(message) == expected


def test_deleted_authors_become_ghost():
    assert facts._login(None) == facts.GHOST
    assert facts._login({"login": None}) == facts.GHOST
    assert facts._login({"login": "tiangolo"}) == "tiangolo"


# ------------------------------------------------------------------ upserts


def _pr(number: int, **kw) -> dict:
    return {
        "number": number,
        "title": f"PR {number}",
        "body": kw.get("body"),
        "state": kw.get("state", "closed"),
        "draft": False,
        "user": {"login": "tiangolo"},
        "created_at": "2026-01-01T00:00:00Z",
        "merged_at": kw.get("merged_at", "2026-01-02T00:00:00Z"),
        "closed_at": "2026-01-02T00:00:00Z",
        "milestone": None,
        "html_url": f"https://github.com/fastapi/fastapi/pull/{number}",
    }


def _issue(number: int, state: str = "closed") -> dict:
    return {
        "number": number,
        "title": f"Issue {number}",
        "body": "something broke",
        "state": state,
        "user": {"login": "someone"},
        "created_at": "2025-12-01T00:00:00Z",
        "closed_at": "2026-01-02T00:00:00Z" if state == "closed" else None,
        "milestone": None,
        "labels": [{"name": "bug"}, {"name": "bug"}, {"name": "confirmed"}],
        "html_url": f"https://github.com/fastapi/fastapi/issues/{number}",
    }


@pytest.fixture
async def conn(alembic_config, test_database):
    command.downgrade(alembic_config, "base")
    command.upgrade(alembic_config, "head")
    connection = await asyncpg.connect(test_database)
    try:
        yield connection
    finally:
        await connection.close()
        command.downgrade(alembic_config, "base")


async def test_pull_request_upsert_is_idempotent(conn):
    files = [{"filename": "fastapi/routing.py", "additions": 10, "deletions": 2}]
    reviews = [
        {"user": {"login": "reviewer"}, "state": "APPROVED", "submitted_at": "2026-01-01T12:00:00Z"}
    ]
    for _ in range(2):
        await facts.upsert_pull_request(conn, _pr(1), files, reviews)

    row = await conn.fetchrow("SELECT * FROM pull_requests WHERE number = 1")
    assert row["state"] == "merged"
    # Derived from the file rows, since the list endpoint omits them.
    assert (row["additions"], row["deletions"]) == (10, 2)
    assert await conn.fetchval("SELECT count(*) FROM pr_files") == 1
    assert await conn.fetchval("SELECT count(*) FROM pr_reviews") == 1
    assert await conn.fetchval("SELECT state FROM pr_reviews") == "approved"


async def test_removed_files_disappear_on_reingest(conn):
    await facts.upsert_pull_request(
        conn,
        _pr(1),
        [
            {"filename": "a.py", "additions": 1, "deletions": 0},
            {"filename": "b.py", "additions": 1, "deletions": 0},
        ],
        [],
    )
    await facts.upsert_pull_request(
        conn, _pr(1), [{"filename": "a.py", "additions": 1, "deletions": 0}], []
    )
    paths = [r["path"] for r in await conn.fetch("SELECT path FROM pr_files")]
    assert paths == ["a.py"]


async def test_issue_labels_deduplicate(conn):
    await facts.upsert_issue(conn, _issue(500))
    labels = [r["label"] for r in await conn.fetch("SELECT label FROM issue_labels ORDER BY 1")]
    assert labels == ["bug", "confirmed"]


async def test_closing_link_set_when_the_pr_is_in_the_window(conn):
    await facts.upsert_pull_request(conn, _pr(1, body="Closes #500"), [], [])
    await facts.upsert_issue(conn, _issue(500))
    dropped = await facts.link_closing_prs(conn, {500: 1})
    assert dropped == 0
    assert await conn.fetchval("SELECT closed_by_pr FROM issues WHERE number = 500") == 1


async def test_closing_link_dropped_and_counted_when_the_pr_is_outside_the_window(conn):
    """The split window means an issue can be closed by a PR we never fetched.
    That must null the link and be counted, not raise a foreign-key error."""
    await facts.upsert_issue(conn, _issue(500))
    dropped = await facts.link_closing_prs(conn, {500: 9999})
    assert dropped == 1
    assert await conn.fetchval("SELECT closed_by_pr FROM issues WHERE number = 500") is None


async def test_open_issues_are_never_linked(conn):
    await facts.upsert_pull_request(conn, _pr(1, body="Closes #501"), [], [])
    await facts.upsert_issue(conn, _issue(501, state="open"))
    assert await facts.link_closing_prs(conn, {501: 1}) == 1
    assert await conn.fetchval("SELECT closed_by_pr FROM issues WHERE number = 501") is None


async def test_commit_drops_a_pr_reference_outside_the_window(conn):
    commit = {
        "sha": "a" * 40,
        "author": {"login": "tiangolo"},
        "commit": {
            "message": "Fix routing (#9999)",
            "author": {"name": "Sebastian", "date": "2026-01-02T00:00:00Z"},
        },
    }
    await facts.upsert_commit(conn, commit, [{"filename": "fastapi/routing.py"}])
    row = await conn.fetchrow("SELECT * FROM commits")
    assert row["pr_number"] is None
    assert await conn.fetchval("SELECT count(*) FROM commit_files") == 1


async def test_commit_links_to_a_pr_that_is_present(conn):
    await facts.upsert_pull_request(conn, _pr(11234), [], [])
    commit = {
        "sha": "b" * 40,
        "author": None,
        "commit": {
            "message": "Fix routing (#11234)",
            "author": {"name": "Someone Deleted", "date": "2026-01-02T00:00:00Z"},
        },
    }
    await facts.upsert_commit(conn, commit, [])
    row = await conn.fetchrow("SELECT * FROM commits")
    assert row["pr_number"] == 11234
    assert row["author"] == "Someone Deleted"  # falls back to the git author
