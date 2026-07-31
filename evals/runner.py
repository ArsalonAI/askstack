#!/usr/bin/env python
"""Score the golden set — TRD §14.

    python evals/runner.py                            # metrics table
    python evals/runner.py --json evals/baselines/main.json
    python evals/runner.py --question q033 --explain   # inspect one question

**This is the M1 shape of the runner.** There is no orchestrator yet, so it
calls `FactsStore` and `Retriever` directly rather than driving `POST /chat` and
reading `retrieval` SSE events. That is enough for the three retrieval metrics;
tool-selection accuracy and citation resolution are properties of an agent turn
and arrive at M2. §14.1 spells out the split.

Two things about the numbers, both stated in §14.1 and worth repeating where
someone reads the output:

- **Set-F1 is a drift tripwire at M1, not a quality metric.** Classes 1-4
  dispatch the same `gold_query` that generated the ground truth, so with no
  agent choosing the query it is 1.0 by construction. It fails only if a
  `FactsStore` query changed behaviour or the corpus was re-ingested at a
  different revision — which is exactly what it is here to catch.
- **recall@5 and MRR@10 are real measurements.** `gold_chunks` were chosen by
  reading issue threads, never by querying the retriever, so they do not
  measure themselves.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import asyncpg
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.facts.store import PostgresFactsStore  # noqa: E402
from app.interfaces import Chunk  # noqa: E402
from app.retrieval.embedder import get_embedder  # noqa: E402
from app.retrieval.hybrid import HybridRetriever  # noqa: E402
from evals.build_gold import (  # noqa: E402
    EXACT_CLASSES,
    GENERATED,
    INTERPRETIVE_CLASSES,
    QUESTIONS,
    ValidationError,
    load_questions,
    pin_info,
    run_query,
    validate,
)

# §14.3. Only the three retrieval metrics exist at M1; the M2 baseline adds
# tool_accuracy_* and citation_resolution.
MILESTONE = "M1"
TOLERANCES = {
    "aggregate_set_f1": 0.02,
    "recall_at_5": 0.02,
    "mrr_at_10": 0.02,
}

RECALL_K = 5
MRR_K = 10
# Retrieve enough for the deepest metric; recall@5 is a prefix of the same list.
SEARCH_K = MRR_K


def set_f1(predicted: list[str], gold: list[str]) -> float:
    """Set-F1 over entity citations.

    Empty-vs-empty is 1.0, not 0.0. "Which auth PRs merged in the last three
    months" has the correct answer *none*, and a scorer that reads that as a
    total miss punishes the one behaviour §5.2 most wants — see `expect_empty`
    in the held-out set.
    """
    got, want = set(predicted), set(gold)
    if not got and not want:
        return 1.0
    if not got or not want:
        return 0.0
    hits = len(got & want)
    if not hits:
        return 0.0
    precision = hits / len(got)
    recall = hits / len(want)
    return 2 * precision * recall / (precision + recall)


def recall_at_k(retrieved: list[str], gold: list[str], k: int = RECALL_K) -> float:
    want = set(gold)
    if not want:
        return 0.0
    return len(want & set(retrieved[:k])) / len(want)


def mrr_at_k(retrieved: list[str], gold: list[str], k: int = MRR_K) -> float:
    """Reciprocal rank of the *first* gold chunk. 0.0 if none in the top k."""
    want = set(gold)
    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in want:
            return 1.0 / rank
    return 0.0


@dataclass
class Result:
    qid: str
    klass: int
    metrics: dict[str, float]
    predicted: list[str] = field(default_factory=list)
    gold: list[str] = field(default_factory=list)
    retrieved: list[Chunk] = field(default_factory=list)


async def score_exact(store: PostgresFactsStore, question: dict, gold: list[str]) -> Result:
    predicted = await run_query(store, question)
    return Result(
        qid=question["id"],
        klass=question["class"],
        metrics={"set_f1": set_f1(predicted, gold)},
        predicted=predicted,
        gold=gold,
    )


async def score_interpretive(retriever: HybridRetriever, question: dict) -> Result:
    gold = list(question.get("gold_chunks") or [])
    chunks = await retriever.search(question["question"], SEARCH_K)
    ids = [c.id for c in chunks]
    return Result(
        qid=question["id"],
        klass=question["class"],
        metrics={
            "recall_at_5": recall_at_k(ids, gold),
            "mrr_at_10": mrr_at_k(ids, gold),
        },
        predicted=ids,
        gold=gold,
        retrieved=chunks,
    )


def config_hash(pin: dict) -> tuple[str, dict]:
    """§14.2. `corpus_ref` is the resolved ref, not the symbolic name."""
    config = {
        "hybrid": settings.hybrid_enabled,
        "memory": settings.memory_enabled,
        "tool_mode": settings.tool_retrieval_mode,
        "tool_catalog_size": settings.tool_catalog_size,
        "embedding_model": settings.embedding_model,
        "agent_model": settings.agent_model,
        "corpus_ref": f"{settings.corpus_ref}@{pin['resolved_sha'][:12]}",
    }
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    return digest, config


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(results: list[Result]) -> dict[str, float]:
    exact = [r for r in results if r.klass in EXACT_CLASSES]
    interp = [r for r in results if r.klass in INTERPRETIVE_CLASSES]
    return {
        "aggregate_set_f1": _mean([r.metrics["set_f1"] for r in exact]),
        "recall_at_5": _mean([r.metrics["recall_at_5"] for r in interp]),
        "mrr_at_10": _mean([r.metrics["mrr_at_10"] for r in interp]),
    }


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[1],
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def report(results: list[Result], metrics: dict[str, float], digest: str, pin: dict) -> None:
    print(f"\ncorpus   {pin['corpus_repo']}@{pin['resolved_sha'][:12]}")
    print(f"config   {digest}   hybrid={settings.hybrid_enabled} "
          f"embed={settings.embedding_model}")
    print(f"scored   {len(results)} questions, milestone {MILESTONE}\n")

    print(f"{'class':>5}  {'n':>3}  {'set-F1':>7}  {'recall@5':>9}  {'MRR@10':>7}")
    for klass in sorted(EXACT_CLASSES | INTERPRETIVE_CLASSES):
        rows = [r for r in results if r.klass == klass]
        if not rows:
            continue
        if klass in EXACT_CLASSES:
            cells = f"{_mean([r.metrics['set_f1'] for r in rows]):>7.3f}  {'—':>9}  {'—':>7}"
        else:
            cells = (
                f"{'—':>7}  {_mean([r.metrics['recall_at_5'] for r in rows]):>9.3f}  "
                f"{_mean([r.metrics['mrr_at_10'] for r in rows]):>7.3f}"
            )
        print(f"{klass:>5}  {len(rows):>3}  {cells}")

    print(
        f"\n  aggregate_set_f1  {metrics['aggregate_set_f1']:.3f}"
        "   (drift tripwire at M1 — §14.1)"
    )
    print(f"  recall_at_5       {metrics['recall_at_5']:.3f}")
    print(f"  mrr_at_10         {metrics['mrr_at_10']:.3f}")

    # Structural 1.0 is expected; anything less means the facts layer or the
    # corpus moved under the committed ground truth, which invalidates it.
    drifted = [r for r in results if r.klass in EXACT_CLASSES and r.metrics["set_f1"] < 1.0]
    if drifted:
        print(
            f"\n  WARNING: {len(drifted)} structured question(s) no longer reproduce "
            f"their ground truth: {', '.join(r.qid for r in drifted)}."
            "\n  Either FactsStore changed behaviour or the corpus was re-ingested. "
            "Re-run evals/build_gold.py --check."
        )

    weakest = sorted(
        (r for r in results if r.klass in INTERPRETIVE_CLASSES),
        key=lambda r: (r.metrics["recall_at_5"], r.metrics["mrr_at_10"]),
    )[:5]
    if weakest:
        print("\n  weakest interpretive questions (--explain <id> to inspect):")
        for r in weakest:
            print(
                f"    {r.qid}  recall@5={r.metrics['recall_at_5']:.2f} "
                f"mrr@10={r.metrics['mrr_at_10']:.2f}"
            )
    print()


def explain(question: dict, result: Result) -> None:
    """One question in full. The manual-inspection path, in place of a UI."""
    print(f"\n{result.qid}  class {result.klass}   as_of {question['as_of']}")
    print(f"  {question['question']}\n")

    if result.klass in EXACT_CLASSES:
        spec = question["gold_query"]
        print(f"  gold_query   {spec['method']}({spec.get('args') or {}})")
        print(f"  set-F1       {result.metrics['set_f1']:.3f}\n")
        got, want = set(result.predicted), set(result.gold)
        for citation in sorted(got | want):
            mark = "  " if citation in got and citation in want else (
                "+ " if citation in got else "- "
            )
            print(f"    {mark}{citation}")
        if not (got | want):
            print("    (both empty — scored 1.0)")
        print("\n    + returned but not in gold · - in gold but not returned")
        print()
        return

    gold = set(result.gold)
    print(f"  recall@5     {result.metrics['recall_at_5']:.3f}")
    print(f"  mrr@10       {result.metrics['mrr_at_10']:.3f}")
    print(f"  gold_chunks  {', '.join(result.gold)}\n")
    for rank, chunk in enumerate(result.retrieved, start=1):
        mark = "★" if chunk.id in gold else " "
        excerpt = " ".join(chunk.content.split())[:100]
        print(f"  {mark} {rank:>2}  {chunk.score:.5f}  {chunk.citation}")
        print(f"           {excerpt}")
    missed = [c for c in result.gold if c not in {c.id for c in result.retrieved}]
    if missed:
        print(f"\n  not retrieved in top {SEARCH_K}: {', '.join(missed)}")
    print()


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write a §14.3 baseline to this path")
    parser.add_argument("--question", help="score only this question id")
    parser.add_argument("--explain", action="store_true", help="print the full detail")
    args = parser.parse_args(argv)

    questions = load_questions(QUESTIONS)
    if not GENERATED.is_file():
        print(f"{GENERATED} is missing. Run evals/build_gold.py first.", file=sys.stderr)
        return 2
    generated = (yaml.safe_load(GENERATED.read_text()) or {}).get("gold_entities", {})

    pool = await asyncpg.create_pool(settings.database_url, min_size=2, max_size=4)
    if pool is None:
        print("could not connect to the database", file=sys.stderr)
        return 2
    try:
        async with pool.acquire() as conn:
            pin = await pin_info(conn)
            # §14.1: a question anchored after the pin cannot be answered
            # correctly by any system, so the whole run aborts rather than
            # reporting a number that reads as a retrieval regression.
            errors = await validate(conn, questions, pin)
        if errors:
            print(f"{len(errors)} problem(s) in {QUESTIONS.name}:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            return 1

        selected = questions
        if args.question:
            selected = [q for q in questions if q["id"] == args.question]
            if not selected:
                print(f"no question with id {args.question!r}", file=sys.stderr)
                return 2

        missing = [
            q["id"]
            for q in selected
            if q["class"] in EXACT_CLASSES and q["id"] not in generated
        ]
        if missing:
            raise ValidationError(
                f"no ground truth for {', '.join(missing)}. "
                "Run evals/build_gold.py first."
            )

        retriever = HybridRetriever(pool, get_embedder())
        results: list[Result] = []
        async with pool.acquire() as conn:
            store = PostgresFactsStore(conn)
            for question in selected:
                if question["class"] in EXACT_CLASSES:
                    results.append(
                        await score_exact(store, question, generated[question["id"]])
                    )
                else:
                    results.append(await score_interpretive(retriever, question))

        if args.explain:
            by_id = {q["id"]: q for q in selected}
            for result in results:
                explain(by_id[result.qid], result)
            return 0

        metrics = aggregate(results)
        digest, _ = config_hash(pin)
        report(results, metrics, digest, pin)

        if args.json:
            if args.question:
                # A baseline written from one question would gate the whole
                # suite on a single number.
                print("--json requires the full set; drop --question", file=sys.stderr)
                return 2
            payload = {
                "milestone": MILESTONE,
                "config_hash": digest,
                "git_sha": _git_sha(),
                "corpus_ref": f"{settings.corpus_ref}@{pin['resolved_sha'][:12]}",
                "recorded_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "metrics": {k: round(v, 4) for k, v in metrics.items()},
                "tolerances": TOLERANCES,
            }
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(payload, indent=2) + "\n")
            print(f"wrote {args.json}")
        return 0
    finally:
        await pool.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
