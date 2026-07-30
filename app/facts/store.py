"""The delivery record — TRD §3.3, §6.4.

Every method here is an indexed SQL query. No embeddings, no ranking, no
model. That is the whole point of ADR 13: "what shipped last month" has an
exact answer, and similarity search returns documents that *discuss* shipping,
ranked, with no notion of completeness.

Counts are computed here and rendered here (ADR 15). A model asked to
summarise forty pull requests produces a confident wrong total and the manager
cannot tell, so `Aggregate.rendered` is handed to the model verbatim.
"""

from __future__ import annotations

from datetime import UTC, datetime

import asyncpg

from app.facts.areas import Area, resolve
from app.interfaces import Aggregate, Entity, EntityKind

# TRD §17.4 is explicit that this is unmeasured: too low and the model cannot
# cite specifics, too high and a 400-PR window floods the context. 50 is a
# starting point to be swept, not a tuned value.
RENDER_LIMIT = 50


def _entity(kind: EntityKind, row: asyncpg.Record) -> Entity:
    if kind == "pr":
        return Entity(
            kind="pr",
            ref=str(row["number"]),
            title=row["title"],
            author=row["author"],
            state=row["state"],
            at=row["merged_at"] or row["closed_at"] or row["created_at"],
            citation=f"pr:{row['number']}",
            url=row["url"],
        )
    if kind == "issue":
        return Entity(
            kind="issue",
            ref=str(row["number"]),
            title=row["title"],
            author=row["author"],
            state=row["state"],
            at=row["closed_at"] or row["created_at"],
            citation=f"issue:{row['number']}",
            url=row["url"],
        )
    if kind == "commit":
        return Entity(
            kind="commit",
            ref=row["sha"][:7],
            title=row["message"].splitlines()[0],
            author=row["author"],
            state="merged",
            at=row["authored_at"],
            citation=f"commit:{row['sha'][:7]}",
            url=f"https://github.com/fastapi/fastapi/commit/{row['sha']}",
        )
    return Entity(
        kind="release",
        ref=row["tag"],
        title=row["name"] or row["tag"],
        author="",
        state="published",
        at=row["published_at"],
        citation=f"release:{row['tag']}",
        url=row["url"],
    )


def _render(
    heading: str,
    entities: list[Entity],
    *,
    area: Area | None = None,
    extra: list[str] | None = None,
) -> str:
    """Deterministic markdown. The count is stated once, by us, up front."""
    lines = [f"**{heading}: {len(entities)}**"]
    if area is not None:
        # §5.5: every answer filtered by area names the globs it resolved, so
        # a bad curated mapping is visible in the output rather than silent.
        lines.append(f"_area `{area.name}` resolved to: {', '.join(area.path_globs)}_")
    lines.extend(extra or [])
    if not entities:
        lines.append("\n(none)")
        return "\n".join(lines)

    lines.append("")
    for entity in entities[:RENDER_LIMIT]:
        when = entity.at.date().isoformat() if entity.at else "—"
        author = f" — @{entity.author}" if entity.author else ""
        lines.append(f"- `{entity.citation}` {when}{author} — {entity.title}")
    if len(entities) > RENDER_LIMIT:
        lines.append(f"- …and {len(entities) - RENDER_LIMIT} more (not listed)")
    return "\n".join(lines)


async def _area(conn: asyncpg.Connection, name: str | None) -> Area | None:
    """Resolve an area, or None. An unknown name raises `UnknownArea` from
    `resolve` — §5.5 requires that, never an empty result."""
    return await resolve(conn, name) if name else None


class PostgresFactsStore:
    """`FactsStore` over the entity tables."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self.conn = conn

    async def merged_prs(
        self, since: datetime, until: datetime, area: str | None = None
    ) -> Aggregate:
        resolved = await _area(self.conn, area)
        rows = await self.conn.fetch(
            """
            SELECT DISTINCT p.number, p.title, p.author, p.state,
                   p.merged_at, p.closed_at, p.created_at, p.url
            FROM pull_requests p
            LEFT JOIN pr_files f ON f.pr_number = p.number
            WHERE p.state = 'merged'
              AND p.merged_at >= $1 AND p.merged_at < $2
              AND ($3::text[] IS NULL OR f.path LIKE ANY($3))
            ORDER BY p.merged_at DESC
            """,
            since,
            until,
            resolved.sql_prefixes() if resolved else None,
        )
        entities = [_entity("pr", r) for r in rows]
        window = f"{since.date()} to {until.date()}"
        return Aggregate(
            entities=entities,
            count=len(entities),
            window=(since, until),
            area=area,
            rendered=_render(f"Pull requests merged {window}", entities, area=resolved),
        )

    async def open_issues(
        self,
        label: str | None = None,
        milestone: str | None = None,
        older_than_days: int | None = None,
        *,
        as_of: datetime | None = None,
    ) -> Aggregate:
        """`as_of` for the same reason as `stale_prs`: an age filter measured
        against wall-clock now is not reproducible against a pinned revision."""
        cutoff = as_of or datetime.now(UTC)
        rows = await self.conn.fetch(
            """
            SELECT DISTINCT i.number, i.title, i.author, i.state,
                   i.closed_at, i.created_at, i.url
            FROM issues i
            LEFT JOIN issue_labels l ON l.issue_number = i.number
            WHERE i.state = 'open'
              AND ($1::text IS NULL OR l.label = $1)
              AND ($2::text IS NULL OR i.milestone = $2)
              AND ($3::int IS NULL
                   OR i.created_at < $4::timestamptz - ($3::int * interval '1 day'))
            ORDER BY i.created_at
            """,
            label,
            milestone,
            older_than_days,
            cutoff,
        )
        entities = [_entity("issue", r) for r in rows]
        filters = [f"label={label}" if label else "", f"milestone={milestone}" if milestone else ""]
        suffix = " ".join(f for f in filters if f)
        return Aggregate(
            entities=entities,
            count=len(entities),
            window=None,
            area=None,
            rendered=_render(f"Open issues {suffix}".strip(), entities),
        )

    async def stale_prs(
        self, threshold_days: int, *, as_of: datetime | None = None
    ) -> Aggregate:
        """Open, non-draft, no review activity within the threshold.

        `as_of` defaults to wall-clock now, but the eval must always pass it.
        "Stale for more than 14 days" measured against `now()` returns a
        different set every day, and PRD §7.1 requires every golden question to
        resolve identically against one pinned revision. This is the "date
        anchored answers rot" risk in PRD §9, and the parameter is the
        mitigation.
        """
        cutoff = as_of or datetime.now(UTC)
        rows = await self.conn.fetch(
            """
            SELECT p.number, p.title, p.author, p.state,
                   p.merged_at, p.closed_at, p.created_at, p.url
            FROM pull_requests p
            WHERE p.state = 'open'
              AND NOT p.is_draft
              AND p.created_at < $2::timestamptz - ($1::int * interval '1 day')
              AND NOT EXISTS (
                  SELECT 1 FROM pr_reviews r
                  WHERE r.pr_number = p.number
                    AND r.submitted_at > $2::timestamptz - ($1::int * interval '1 day')
              )
            ORDER BY p.created_at
            """,
            threshold_days,
            cutoff,
        )
        entities = [_entity("pr", r) for r in rows]
        return Aggregate(
            entities=entities,
            count=len(entities),
            window=None,
            area=None,
            rendered=_render(
                f"Open pull requests with no review activity in {threshold_days} days",
                entities,
            ),
        )

    async def commits_by_author(
        self, since: datetime, area: str | None = None
    ) -> Aggregate:
        resolved = await _area(self.conn, area)
        rows = await self.conn.fetch(
            """
            SELECT DISTINCT c.sha, c.author, c.authored_at, c.message
            FROM commits c
            LEFT JOIN commit_files f ON f.sha = c.sha
            WHERE c.authored_at >= $1
              AND ($2::text[] IS NULL OR f.path LIKE ANY($2))
            ORDER BY c.authored_at DESC
            """,
            since,
            resolved.sql_prefixes() if resolved else None,
        )
        entities = [_entity("commit", r) for r in rows]

        # The ownership question is "who", so the tally is the answer and the
        # commit list is the evidence. Computed here for the same reason as
        # every other count (ADR 15).
        tally: dict[str, int] = {}
        for entity in entities:
            tally[entity.author] = tally.get(entity.author, 0) + 1
        ranked = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))
        breakdown = ["", "By author:"] + [f"- @{a}: {n}" for a, n in ranked]

        return Aggregate(
            entities=entities,
            count=len(entities),
            window=(since, datetime.now(UTC)),
            area=area,
            rendered=_render(
                f"Commits since {since.date()}", entities, area=resolved, extra=breakdown
            ),
        )

    async def entity(self, kind: EntityKind, ref: str) -> Entity | None:
        """Single lookup. Backs PRD §5.2 verification — "is this actually merged".

        `ref` is a string for every kind (the protocol takes PR numbers,
        commit SHAs, and release tags through one parameter), so the numeric
        kinds convert here. A non-numeric PR ref is a caller bug, not a miss:
        returning None would read as "that PR does not exist".
        """
        if kind in ("pr", "issue"):
            try:
                number = int(ref)
            except ValueError as exc:
                raise ValueError(f"{kind} ref must be numeric, got {ref!r}") from exc
            table = "pull_requests" if kind == "pr" else "issues"
            columns = (
                "number, title, author, state, merged_at, closed_at, created_at, url"
                if kind == "pr"
                else "number, title, author, state, closed_at, created_at, url"
            )
            row = await self.conn.fetchrow(
                f"SELECT {columns} FROM {table} WHERE number = $1", number
            )
        elif kind == "commit":
            row = await self.conn.fetchrow(
                "SELECT sha, author, authored_at, message FROM commits"
                " WHERE sha LIKE $1 || '%'",
                ref,
            )
        elif kind == "release":
            row = await self.conn.fetchrow(
                "SELECT tag, name, published_at, url FROM releases WHERE tag = $1", ref
            )
        else:
            raise ValueError(f"unknown entity kind {kind!r}")
        return _entity(kind, row) if row else None

    async def release_diff(self, from_tag: str, to_tag: str) -> Aggregate:
        bounds = await self.conn.fetch(
            "SELECT tag, published_at FROM releases WHERE tag = ANY($1::text[])",
            [from_tag, to_tag],
        )
        found = {r["tag"]: r["published_at"] for r in bounds}
        missing = [t for t in (from_tag, to_tag) if t not in found]
        if missing:
            # Same principle as an unknown area (§5.5): "nothing shipped
            # between these tags" and "one of those tags does not exist" must
            # not look the same to the manager.
            raise ValueError(f"unknown release tag(s): {', '.join(missing)}")

        start, end = sorted([found[from_tag], found[to_tag]])
        aggregate = await self.merged_prs(start, end)
        return Aggregate(
            entities=aggregate.entities,
            count=aggregate.count,
            window=(start, end),
            area=None,
            rendered=_render(
                f"Merged between {from_tag} and {to_tag}", list(aggregate.entities)
            ),
        )
