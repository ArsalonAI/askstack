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
    as_datetime,
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
    "tool_accuracy_jaccard": 0.03,
    "tool_accuracy_exact": 0.03,
    "citation_resolution": 0.01,
}

# Bumped whenever a scoring rule changes, which invalidates every cached row
# written under the old one. `2` is §14.1's gold-tool-matched set-F1 replacing
# the union — see §17 Q10.
CACHE_FORMAT = 2

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
    # Populated only on the M2 agent path: what the selector offered, what the
    # model actually called, and every citation it wrote.
    agent: dict | None = None


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


def jaccard(got: set[str], want: set[str]) -> float:
    if not got and not want:
        return 1.0
    union = got | want
    return len(got & want) / len(union) if union else 0.0


def gold_tool_entities(
    retrievals: list[dict], gold_tools: set[str]
) -> list[str]:
    """The entities retrieved by the tool the question asks for — §14.1.

    Scored over the retrieval whose tool is in `gold_tools`, not over the union
    of the turn. The union measures how many tools the agent called: with it,
    q006 scored 0.667 for checking `pr:15806` and then separately confirming
    `release:0.138.0` had shipped, which is exactly the verification PRD §5.2
    requires. A metric that penalises that is measuring the wrong thing.

    A tool called more than once contributes all its calls — the agent may
    legitimately need two windows — because those are the *same* tool and so
    cannot be a tool-count artefact.

    No matching retrieval yields `[]`, which scores 0.0 against a non-empty
    gold set. That is the intended reading: an agent that never called the
    right tool did not answer the question, however good its prose was.
    """
    return list(
        dict.fromkeys(
            citation
            for r in retrievals
            if r["tool"] in gold_tools
            for citation in r["entities"]
        )
    )


async def score_agent(orchestrator, question: dict, gold_entities: list[str]) -> Result:
    """One agent turn, scored off the SSE events — §14.1 at M2.

    Set-F1 reads the `retrieval` events rather than the prose, so it measures
    what the system actually retrieved rather than what the model said about
    it. Whether the prose reflects that set faithfully is the separate question
    citation resolution answers.
    """
    klass = question["class"]
    as_of = as_datetime(question["as_of"])

    selected: list[str] = []
    # Per-retrieval rather than flattened, because §14.1 scores the retrieval
    # whose tool matches and the union is no longer recoverable once merged.
    # Kept in the cache so a later reading can be compared without re-running
    # fifty paid turns to get the breakdown back.
    retrievals: list[dict] = []
    chunks: list[str] = []
    citations: list[dict] = []
    tool_calls: list[str] = []
    error: dict | None = None
    usage: dict = {}

    async for event, data in orchestrator.run(
        "eval", None, question["question"], as_of=as_of
    ):
        if event == "tools_selected":
            selected = [t["name"] for t in data["selected"]]
        elif event == "retrieval":
            if data["kind"] == "structured":
                retrievals.append(
                    {
                        "tool": data["tool"],
                        "entities": [e["citation"] for e in data["entities"]],
                    }
                )
            else:
                chunks.extend(c["citation"] for c in data["chunks"])
        elif event == "citation":
            citations.append(data)
        elif event == "tool_call" and data["status"] != "started":
            tool_calls.append(data["name"])
        elif event == "error":
            error = data
        elif event == "done":
            usage = data

    gold_tools = set(question.get("gold_tools") or [])
    metrics: dict[str, float] = {
        "tool_jaccard": jaccard(set(selected), gold_tools),
        "tool_exact": float(set(selected) == gold_tools),
    }
    if citations:
        # §11.2: both halves. A citation that resolves to a real entity the
        # turn never looked up is the failure this metric exists to catch.
        metrics["citation_resolution"] = sum(
            bool(c["resolved"] and c["in_result_set"]) for c in citations
        ) / len(citations)

    if klass in EXACT_CLASSES:
        predicted = gold_tool_entities(retrievals, gold_tools)
        metrics["set_f1"] = set_f1(predicted, gold_entities)
        # Reported, never gated. §17 Q10 is settled but its evidence should stay
        # visible: when the two diverge the agent made calls beyond the one the
        # question asked for, which is worth seeing even though it no longer
        # costs anything.
        union = list(dict.fromkeys(c for r in retrievals for c in r["entities"]))
        metrics["set_f1_union"] = set_f1(union, gold_entities)
        gold = gold_entities
    else:
        predicted = list(dict.fromkeys(chunks))
        gold = list(question.get("gold_chunks") or [])
        metrics["recall_at_5"] = recall_at_k(predicted, gold)
        metrics["mrr_at_10"] = mrr_at_k(predicted, gold)

    return Result(
        qid=question["id"],
        klass=klass,
        metrics=metrics,
        predicted=predicted,
        gold=list(gold),
        agent={
            "selected": selected,
            "called": tool_calls,
            "gold_tools": sorted(gold_tools),
            "retrievals": retrievals,
            "citations": citations,
            "error": error,
            "usage": usage,
        },
    )


def _load_cache(path: Path) -> dict[str, Result]:
    """Reusable results only. A row from an older scorer is not one.

    `predicted` and `metrics` are stored already-scored, so a row written before
    a scoring change would be replayed under the old definition and averaged in
    beside freshly-scored ones — a baseline half-computed each way, with nothing
    in the output to say so. Stale rows are dropped and re-run instead; the cost
    of re-running them is the price of the change, and it is visible.
    """
    if not path.is_file():
        return {}
    cached, stale = {}, 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("cache_format") != CACHE_FORMAT:
            stale += 1
            continue
        cached[row["qid"]] = Result(
            qid=row["qid"],
            klass=row["klass"],
            metrics=row["metrics"],
            predicted=row["predicted"],
            gold=row["gold"],
            agent=row["agent"],
        )
    if stale:
        print(
            f"  discarding {stale} cached result(s) from an older scorer "
            f"(cache_format != {CACHE_FORMAT}); they will be re-run",
            file=sys.stderr,
        )
    return cached


async def run_agent_path(
    pool,
    retriever,
    questions: list[dict],
    generated: dict,
    *,
    concurrency: int = 4,
    cache_path: Path | None = None,
) -> list[Result]:
    """Drive a real agent turn per question — §14.1's M2 shape.

    Each question gets its own orchestrator and its own session: a shared
    session would let turn N read turn N-1's history, and questions scored
    against context from unrelated questions are not the questions the golden
    set froze.

    Results are appended to `cache_path` as they land and reused on the next
    run. Fifty agent turns cost real money and take long enough to be killed by
    something — a timeout, a laptop lid — and losing 40 finished answers to
    re-measure the last 10 is the kind of waste that discourages running the
    eval at all.
    """
    from anthropic import AsyncAnthropic

    from app.orchestrator import Orchestrator
    from app.tools.registry import CATALOG, ToolRegistry
    from app.tools.selector import build_selector

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    semaphore = asyncio.Semaphore(concurrency)
    cached = _load_cache(cache_path) if cache_path else {}
    pending = [q for q in questions if q["id"] not in cached]
    if cached:
        print(
            f"  reusing {len(cached)} cached result(s); {len(pending)} to run",
            file=sys.stderr,
        )
    done = 0
    write_lock = asyncio.Lock()

    async def one(question: dict) -> Result:
        nonlocal done
        # No outer `pool.acquire()`. `PostgresFactsStore` takes the pool
        # directly, and holding one connection for a whole turn deadlocks: the
        # orchestrator needs connections *inside* it — session, history, the
        # two retriever arms concurrently, the selector, the citation checks —
        # and with concurrency == pool size none can ever be granted.
        async with semaphore:
            orchestrator = Orchestrator(
                pool,
                ToolRegistry(
                    PostgresFactsStore(pool), retriever, top_k=settings.retrieval_top_k
                ),
                build_selector(
                    settings.tool_retrieval_mode,
                    CATALOG,
                    pool=pool,
                    embedder=retriever.embedder,
                    floor=settings.tool_similarity_floor,
                ),
                client,
                settings,
            )
            result = await score_agent(
                orchestrator, question, generated.get(question["id"], [])
            )
        done += 1
        print(f"  [{done}/{len(pending)}] {result.qid}", file=sys.stderr, flush=True)
        # An errored turn is never cached. It is not a measurement — a 400 from
        # an exhausted credit balance would otherwise be frozen into the cache
        # and reused as a zero on every later run.
        if cache_path and not result.agent.get("error"):
            async with write_lock:
                with cache_path.open("a") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "cache_format": CACHE_FORMAT,
                                "qid": result.qid,
                                "klass": result.klass,
                                "metrics": result.metrics,
                                "predicted": result.predicted,
                                "gold": result.gold,
                                "agent": result.agent,
                            },
                            default=str,
                        )
                        + "\n"
                    )
        return result

    fresh = list(await asyncio.gather(*(one(q) for q in pending)))
    return [*cached.values(), *fresh]


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
    metrics = {
        "aggregate_set_f1": _mean([r.metrics["set_f1"] for r in exact]),
        "recall_at_5": _mean([r.metrics["recall_at_5"] for r in interp]),
        "mrr_at_10": _mean([r.metrics["mrr_at_10"] for r in interp]),
    }
    # The agent-turn metrics exist only on the M2 path (§14.1).
    # `aggregate_set_f1_union` carries no tolerance and so never reaches the
    # gate — it is the §17 Q10 diagnostic, not a second headline number.
    for key, source in (
        ("aggregate_set_f1_union", "set_f1_union"),
        ("tool_accuracy_jaccard", "tool_jaccard"),
        ("tool_accuracy_exact", "tool_exact"),
        ("citation_resolution", "citation_resolution"),
    ):
        values = [r.metrics[source] for r in results if source in r.metrics]
        if values:
            metrics[key] = _mean(values)
    return metrics


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


def report(
    results: list[Result],
    metrics: dict[str, float],
    digest: str,
    pin: dict,
    *,
    milestone: str = MILESTONE,
) -> None:
    agent = milestone == "M2"
    print(f"\ncorpus   {pin['corpus_repo']}@{pin['resolved_sha'][:12]}")
    print(f"config   {digest}   hybrid={settings.hybrid_enabled} "
          f"embed={settings.embedding_model}")
    if agent:
        print(f"agent    {settings.agent_model} effort={settings.agent_effort} "
              f"tools={settings.tool_retrieval_mode} k={settings.tool_retrieval_k}")
    print(f"scored   {len(results)} questions, milestone {milestone}\n")

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

    note = "" if agent else "   (drift tripwire at M1 — §14.1)"
    print(f"\n  aggregate_set_f1      {metrics['aggregate_set_f1']:.3f}{note}")
    print(f"  recall_at_5           {metrics['recall_at_5']:.3f}")
    print(f"  mrr_at_10             {metrics['mrr_at_10']:.3f}")
    for key in ("tool_accuracy_jaccard", "tool_accuracy_exact", "citation_resolution"):
        if key in metrics:
            print(f"  {key:<21} {metrics[key]:.3f}")
    if "aggregate_set_f1_union" in metrics:
        union = metrics["aggregate_set_f1_union"]
        print(
            f"\n  set-F1 over the union of the turn  {union:.3f}   "
            f"(report-only — §17 Q10)"
        )
        if union < metrics["aggregate_set_f1"]:
            extra = [
                r
                for r in results
                if r.klass in EXACT_CLASSES
                and r.metrics.get("set_f1_union", 1.0) < r.metrics["set_f1"]
            ]
            print(
                f"  {len(extra)} question(s) retrieved beyond the tool asked for: "
                f"{', '.join(r.qid for r in extra[:8])}"
                f"{'...' if len(extra) > 8 else ''}"
            )

    if not agent:
        # Structural 1.0 is expected on the M1 path; anything less means the
        # facts layer or the corpus moved under the committed ground truth.
        drifted = [
            r for r in results if r.klass in EXACT_CLASSES and r.metrics["set_f1"] < 1.0
        ]
        if drifted:
            print(
                f"\n  WARNING: {len(drifted)} structured question(s) no longer reproduce "
                f"their ground truth: {', '.join(r.qid for r in drifted)}."
                "\n  Either FactsStore changed behaviour or the corpus was re-ingested. "
                "Re-run evals/build_gold.py --check."
            )
    else:
        errors = [r for r in results if (r.agent or {}).get("error")]
        if errors:
            print(f"\n  {len(errors)} question(s) errored — these are NOT measurements:")
            for r in errors[:5]:
                print(
                    f"    {r.qid}  {r.agent['error']['code']}: "
                    f"{r.agent['error']['message'][:90]}"
                )

        cost = sum((r.agent or {}).get("usage", {}).get("cost_usd", 0.0) for r in results)
        latencies = sorted(
            (r.agent or {}).get("usage", {}).get("latency_ms", 0) for r in results
        )
        if latencies and latencies[-1]:
            p95 = latencies[int(0.95 * (len(latencies) - 1))]
            print(f"\n  cost ${cost:.2f} total, ${cost / max(len(results), 1):.3f}/question")
            print(f"  latency p50 {latencies[len(latencies) // 2] / 1000:.1f}s "
                  f"p95 {p95 / 1000:.1f}s   (§13 budget: 25s p95)")

        missed = [r for r in results if r.metrics.get("tool_jaccard", 1.0) == 0.0]
        if missed:
            print(f"\n  {len(missed)} question(s) where no gold tool was selected:")
            for r in missed[:8]:
                print(f"    {r.qid}  gold={r.agent['gold_tools']} selected={r.agent['selected']}")

        silent = [
            r
            for r in results
            if r.agent and not r.agent["called"] and not r.agent.get("error")
        ]
        if silent:
            # Answering without calling anything means nothing was retrieved,
            # so set-F1 is 0 however good the prose was.
            print(f"\n  {len(silent)} question(s) answered with no tool call: "
                  f"{', '.join(r.qid for r in silent)}")

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
    parser.add_argument(
        "--agent",
        action="store_true",
        help="drive the agent and score its SSE events (M2). Costs money.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4, help="parallel agent turns (--agent only)"
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(".cache/agent_results.jsonl"),
        help="append/reuse per-question agent results so a killed run resumes",
    )
    parser.add_argument(
        "--fresh", action="store_true", help="ignore and overwrite the result cache"
    )
    args = parser.parse_args(argv)

    if args.agent and not settings.anthropic_api_key:
        print("--agent needs ANTHROPIC_API_KEY", file=sys.stderr)
        return 2

    questions = load_questions(QUESTIONS)
    if not GENERATED.is_file():
        print(f"{GENERATED} is missing. Run evals/build_gold.py first.", file=sys.stderr)
        return 2
    generated = (yaml.safe_load(GENERATED.read_text()) or {}).get("gold_entities", {})

    # An agent turn holds several connections at once — the two retriever arms
    # run concurrently (§2.1) on top of session, history, selector, and
    # citation lookups — so the pool has to outsize the concurrency by a good
    # margin or the run starves rather than slows.
    pool_size = max(4, args.concurrency * 4) if args.agent else 4
    pool = await asyncpg.create_pool(
        settings.database_url, min_size=2, max_size=pool_size
    )
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

        if args.agent:
            cache_path = args.cache
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if args.fresh and cache_path.exists():
                cache_path.unlink()
            results = await run_agent_path(
                pool,
                retriever,
                selected,
                generated,
                concurrency=args.concurrency,
                cache_path=cache_path,
            )
        else:
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

        milestone = "M2" if args.agent else MILESTONE
        results.sort(key=lambda r: r.qid)
        metrics = aggregate(results)
        digest, _ = config_hash(pin)
        report(results, metrics, digest, pin, milestone=milestone)

        if args.json:
            if args.question:
                # A baseline written from one question would gate the whole
                # suite on a single number.
                print("--json requires the full set; drop --question", file=sys.stderr)
                return 2
            failed = [r for r in results if (r.agent or {}).get("error")]
            if failed:
                # A metric averaged over questions that never reached the model
                # is not a measurement of anything, and committing it as a
                # baseline would gate every later run against noise.
                print(
                    f"\nrefusing to write a baseline: {len(failed)} question(s) errored "
                    f"({', '.join(r.qid for r in failed[:5])}"
                    f"{'...' if len(failed) > 5 else ''}).\n"
                    "Fix the cause and re-run — completed questions are cached and "
                    "will not be re-charged.",
                    file=sys.stderr,
                )
                return 1
            payload = {
                "milestone": milestone,
                "config_hash": digest,
                "git_sha": _git_sha(),
                "corpus_ref": f"{settings.corpus_ref}@{pin['resolved_sha'][:12]}",
                "recorded_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "metrics": {k: round(v, 4) for k, v in metrics.items()},
                "tolerances": {k: TOLERANCES[k] for k in metrics if k in TOLERANCES},
            }
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(payload, indent=2) + "\n")
            print(f"wrote {args.json}")
        return 0
    finally:
        await pool.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
