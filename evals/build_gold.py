#!/usr/bin/env python
"""Generate ground truth for the exactly-checkable questions — TRD §14.1.

Classes 1-4 have answers that are *facts about the repository at the pinned
revision*, so their ground truth is computed rather than authored: each
question carries a `gold_query`, which is dispatched through the same
`FactsStore` the agent's tools will use.

    python evals/build_gold.py --check    # validate, write nothing
    python evals/build_gold.py            # regenerate gold_entities.yaml

§14.1 is candid that this makes the scorer and the ground truth share an
implementation, so a bug in the SQL would be invisible. That is why the
generated answers are cross-checked against GitHub's search API by hand and the
check recorded in `evals/golden/README.md` — see the verification log there.

The generated file is deliberately *not* written back into `questions.yaml`.
The freeze rule (PRD §7.1) protects what humans authored; entity sets are a
pure function of (query spec, pinned corpus) and are regenerable at will.
Keeping them in separate files makes that a file boundary rather than a
convention nobody enforces.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import asyncpg
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.facts.store import PostgresFactsStore  # noqa: E402

GOLDEN = Path(__file__).parent / "golden"
QUESTIONS = GOLDEN / "questions.yaml"
HELDOUT = GOLDEN / "heldout.yaml"
GENERATED = GOLDEN / "gold_entities.yaml"
HELDOUT_GENERATED = GOLDEN / "heldout_entities.yaml"

EXACT_CLASSES = {1, 2, 3, 4}
INTERPRETIVE_CLASSES = {5, 6}

# Methods a question may name. Anything else is a typo, not a feature.
DISPATCH = {
    "merged_prs": ("since", "until", "area"),
    "stale_prs": ("threshold_days",),
    "commits_by_author": ("since", "area"),
    "open_issues": ("label", "milestone", "older_than_days"),
    "release_diff": ("from_tag", "to_tag"),
    "entity": ("kind", "ref"),
}

# Age-relative methods must be anchored, or the answer drifts daily and the
# eval reads clock drift as a retrieval regression (PRD §9).
NEEDS_AS_OF = {"stale_prs", "open_issues"}


class ValidationError(Exception):
    pass


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time(), tzinfo=UTC)
    return datetime.fromisoformat(str(value)).replace(tzinfo=UTC)


def load_questions(path: Path) -> list[dict]:
    raw = yaml.safe_load(path.read_text()) or []
    if not isinstance(raw, list):
        raise ValidationError(f"{path} must be a list of questions")
    return raw


async def pin_info(conn: asyncpg.Connection) -> dict:
    row = await conn.fetchrow(
        "SELECT resolved_sha, since, corpus_repo FROM ingest_runs"
        " WHERE completed_at IS NOT NULL ORDER BY started_at DESC LIMIT 1"
    )
    if row is None:
        raise ValidationError(
            "no completed ingest run. The golden set is anchored to a pinned "
            "revision; scoring against a partial corpus is meaningless."
        )
    sha_date = await conn.fetchval(
        "SELECT max(greatest(merged_at, closed_at, created_at)) FROM pull_requests"
    )
    return {
        "resolved_sha": row["resolved_sha"],
        "pr_floor": row["since"],
        "corpus_repo": row["corpus_repo"],
        "pin_date": sha_date,
    }


async def validate(conn: asyncpg.Connection, questions: list[dict], pin: dict) -> list[str]:
    """Every check that can be made without running a query."""
    errors: list[str] = []
    seen: set[str] = set()

    chunk_ids = {
        r["id"]
        for r in await conn.fetch(
            "SELECT id FROM chunks WHERE id = ANY($1::text[])",
            [c for q in questions for c in q.get("gold_chunks", [])],
        )
    }

    for question in questions:
        qid = question.get("id", "<missing id>")
        where = f"{qid}"

        if qid in seen:
            errors.append(f"{where}: duplicate id")
        seen.add(qid)

        klass = question.get("class")
        if klass not in EXACT_CLASSES | INTERPRETIVE_CLASSES:
            errors.append(f"{where}: class must be 1-6, got {klass!r}")
            continue

        if not question.get("question", "").strip():
            errors.append(f"{where}: empty question text")
        if not question.get("gold_tools"):
            errors.append(f"{where}: gold_tools is required (tool-selection accuracy)")

        as_of_raw = question.get("as_of")
        if not as_of_raw:
            errors.append(f"{where}: as_of is required — every question is date-anchored")
            continue
        as_of = _as_datetime(as_of_raw)

        # §14.1: a question anchored after the pin cannot be answered correctly
        # by any system, and scoring it looks like a retrieval regression.
        if pin["pin_date"] and as_of > pin["pin_date"]:
            errors.append(
                f"{where}: as_of {as_of.date()} is after the pin "
                f"({pin['pin_date'].date()})"
            )

        if klass in EXACT_CLASSES:
            spec = question.get("gold_query")
            if not spec:
                errors.append(f"{where}: class {klass} requires gold_query")
                continue
            method = spec.get("method")
            if method not in DISPATCH:
                errors.append(f"{where}: unknown gold_query method {method!r}")
                continue
            unknown = set(spec.get("args", {})) - set(DISPATCH[method])
            if unknown:
                errors.append(f"{where}: {method} got unknown args {sorted(unknown)}")

            # Classes 1-4 read the facts layer, which is windowed. A question
            # anchored before the floor would score against a truncated corpus.
            floor = pin["pr_floor"]
            if floor and as_of < floor:
                errors.append(
                    f"{where}: as_of {as_of.date()} precedes the PR ingest floor "
                    f"({floor.date()})"
                )
            for key in ("since", "until"):
                if key in spec.get("args", {}):
                    bound = _as_datetime(spec["args"][key])
                    if floor and bound < floor:
                        errors.append(
                            f"{where}: gold_query.{key} {bound.date()} precedes the "
                            f"PR ingest floor ({floor.date()})"
                        )
        else:
            if not question.get("gold_chunks"):
                errors.append(f"{where}: class {klass} requires gold_chunks")
            if not question.get("gold_answer_points"):
                errors.append(f"{where}: class {klass} requires gold_answer_points")
            for citation in question.get("gold_chunks", []):
                if citation not in chunk_ids:
                    errors.append(f"{where}: gold_chunk {citation!r} is not in the index")

    return errors


async def run_query(store: PostgresFactsStore, question: dict) -> list[str]:
    """Dispatch one `gold_query` and return its entity citations."""
    spec = question["gold_query"]
    method = spec["method"]
    args = dict(spec.get("args") or {})
    as_of = _as_datetime(question["as_of"])

    for key in ("since", "until"):
        if key in args:
            args[key] = _as_datetime(args[key])
    if method in NEEDS_AS_OF:
        args["as_of"] = as_of

    result = await getattr(store, method)(**args)
    if result is None:  # entity() miss — a legitimate "no, that never shipped"
        return []
    if hasattr(result, "entities"):
        return [e.citation for e in result.entities]
    return [result.citation]


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate, write nothing")
    parser.add_argument("--heldout", action="store_true", help="operate on heldout.yaml")
    args = parser.parse_args(argv)

    source = HELDOUT if args.heldout else QUESTIONS
    target = HELDOUT_GENERATED if args.heldout else GENERATED
    if not source.is_file():
        print(f"{source} does not exist", file=sys.stderr)
        return 2

    questions = load_questions(source)
    conn = await asyncpg.connect(settings.database_url)
    try:
        pin = await pin_info(conn)
        errors = await validate(conn, questions, pin)
        if errors:
            print(f"{len(errors)} problem(s) in {source.name}:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            return 1

        counts = {k: 0 for k in sorted(EXACT_CLASSES | INTERPRETIVE_CLASSES)}
        for question in questions:
            counts[question["class"]] += 1
        print(f"{source.name}: {len(questions)} questions, by class {counts}")
        print(f"pin: {pin['corpus_repo']}@{pin['resolved_sha'][:12]}")

        store = PostgresFactsStore(conn)
        generated: dict[str, list[str]] = {}
        empty: list[str] = []
        for question in questions:
            if question["class"] not in EXACT_CLASSES:
                continue
            citations = await run_query(store, question)
            generated[question["id"]] = citations
            if not citations and question.get("expect_empty") is not True:
                empty.append(question["id"])

        if empty:
            # Not fatal -- "did X ship?" can legitimately answer no -- but an
            # unintended empty set silently scores as a perfect miss forever.
            print(
                f"warning: {len(empty)} question(s) produced an empty answer set: "
                f"{', '.join(empty)}. Set `expect_empty: true` if intended.",
                file=sys.stderr,
            )

        if args.check:
            print("check passed; nothing written")
            return 0

        payload = {
            "_generated_by": "evals/build_gold.py — do not hand-edit",
            "_corpus_repo": pin["corpus_repo"],
            "_resolved_sha": pin["resolved_sha"],
            "gold_entities": generated,
        }
        target.write_text(yaml.safe_dump(payload, sort_keys=True, width=100))
        total = sum(len(v) for v in generated.values())
        print(f"wrote {target.name}: {len(generated)} questions, {total} entities")
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
