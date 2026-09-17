#!/usr/bin/env python
"""The memory ablation on the golden set, with bootstrap intervals — PRD §7.3.

    python evals/ablation.py                  # run both arms, then analyse
    python evals/ablation.py --analyse-only   # re-analyse the cached arms, free
    python evals/ablation.py --json results/ablation_2026-09-17.json

Both arms are driven through `evals/runner.py --agent`, unchanged, as two
subprocesses that differ only in `MEMORY_ENABLED`. Reusing the runner rather
than re-implementing a turn means the ablation scores exactly what the gate
scores. The arms run *concurrently* rather than one after the other: latency
here is dominated by generation speed (§13), and running the off arm an hour
after the on arm would let API load drift masquerade as a memory effect.

Each arm writes its own cache. The runner keys cached rows by question id
alone, so a shared cache would hand the off arm the on arm's answers and
report a null result with complete confidence.

**What this can and cannot measure.** Every golden question runs as its own
`eval_{qid}` user with no prior session (§14.1), so the memory block is empty
in the on arm by construction. The golden-set comparison therefore prices
memory — two always-injected tools, a memory load, the model's own
`memory_search` calls — and cannot value it. Memory's benefit is what the
cross-session suite measures (§7.2), and its committed result is re-analysed
here with the same bootstrap so the two sit side by side.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.tools.registry import ALWAYS_INJECTED  # noqa: E402
from evals.build_gold import EXACT_CLASSES, QUESTIONS, load_questions  # noqa: E402
from evals.runner import _load_cache  # noqa: E402

ARMS = {"on": True, "off": False}
CACHE_DIR = ROOT / ".cache" / "ablation"
SCENARIOS = ROOT / "evals" / "baselines" / "scenarios.json"

N_BOOT = 10_000
SEED = 20260917
CI = 0.95


def cache_path(arm: str) -> Path:
    return CACHE_DIR / f"memory_{arm}.jsonl"


# --- per-question rows ------------------------------------------------------


def succeeded(row) -> bool:
    """One pass/fail per question, so the two question groups can share a rate.

    Classes 1-4 have exact ground truth: success is set-F1 of 1.0, the right
    entities and nothing else. Classes 5-6 are interpretive and have no exact
    answer to match: success is citing at least one gold passage in the answer
    — grounded in the right source, which is the §5.2 property, rather than
    merely having retrieved it somewhere in the turn.
    """
    if row.klass in EXACT_CLASSES:
        return row.metrics["set_f1"] == 1.0
    gold = set(row.gold)
    return any(c["citation"] in gold for c in row.agent["citations"])


def to_record(row) -> dict:
    usage = row.agent["usage"]
    return {
        "qid": row.qid,
        "class": row.klass,
        "set_f1": row.metrics.get("set_f1"),
        "success": succeeded(row),
        "latency_ms": usage["latency_ms"],
        "cost_usd": usage["cost_usd"],
        "output_tokens": usage.get("usage", {}).get("output_tokens"),
        "memory_tools_offered": bool(set(row.agent["selected"]) & set(ALWAYS_INJECTED)),
        "memory_tools_called": sorted(set(row.agent["called"]) & set(ALWAYS_INJECTED)),
    }


def load_arm(arm: str) -> list[dict]:
    """The arm's fifty rows, refusing anything that would misreport it."""
    rows = _load_cache(cache_path(arm))
    records = [to_record(r) for r in sorted(rows.values(), key=lambda r: r.qid)]
    # The arm is checked from the data, not trusted from the filename: with
    # memory on the registry always offers both memory tools, and with it off
    # it never does. A mislabelled or mixed cache fails here.
    wrong = [r["qid"] for r in records if r["memory_tools_offered"] != ARMS[arm]]
    if wrong:
        raise SystemExit(
            f"memory_{arm}: {len(wrong)} row(s) were not run with memory {arm} "
            f"({', '.join(wrong[:5])}). Delete {cache_path(arm)} and re-run."
        )
    return records


# --- bootstrap --------------------------------------------------------------


def percentile(values: list[float], q: float) -> float:
    """Linear interpolation between closest ranks, as numpy's default does."""
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    return percentile(values, 0.5)


def interval(samples: list[float]) -> list[float]:
    alpha = (1 - CI) / 2
    return [percentile(samples, alpha), percentile(samples, 1 - alpha)]


def paired_bootstrap(
    on: list[dict],
    off: list[dict],
    metric,
    *,
    strata=None,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    """Point estimates and percentile intervals for each arm and the difference.

    **Paired**: one resample of question indices is applied to both arms, so
    the interval on the difference reflects how the arms disagree on the *same*
    questions rather than the much larger spread between easy and hard
    questions. Unpaired intervals would be honest and several times wider.

    **Stratified** when `strata` is given: indices are resampled within each
    stratum, so every replicate keeps the set's 30/20 split between exact and
    interpretive questions instead of drifting towards whichever is easier.

    `metric` maps a list of records to one number, which lets the same routine
    serve a mean, a rate and a median.
    """
    assert [r["qid"] for r in on] == [r["qid"] for r in off], "arms are not paired"
    rng = random.Random(seed)
    groups: dict[object, list[int]] = {}
    for i, record in enumerate(on):
        groups.setdefault(strata(record) if strata else None, []).append(i)

    on_s, off_s, diff_s = [], [], []
    for _ in range(n_boot):
        idx = [rng.choice(members) for members in groups.values() for _ in members]
        a = metric([on[i] for i in idx])
        b = metric([off[i] for i in idx])
        on_s.append(a)
        off_s.append(b)
        diff_s.append(a - b)

    point_on, point_off = metric(on), metric(off)
    return {
        "on": {"estimate": point_on, "ci95": interval(on_s)},
        "off": {"estimate": point_off, "ci95": interval(off_s)},
        "diff_on_minus_off": {"estimate": point_on - point_off, "ci95": interval(diff_s)},
    }


def summarise(on: list[dict], off: list[dict]) -> dict:
    exact = [i for i, r in enumerate(on) if r["class"] in EXACT_CLASSES]
    stratum = lambda r: r["class"] in EXACT_CLASSES  # noqa: E731

    return {
        "set_f1": {
            "definition": "mean set-F1 over the exact-answer questions (classes 1-4)",
            "n": len(exact),
            **paired_bootstrap(
                [on[i] for i in exact],
                [off[i] for i in exact],
                lambda rs: mean([r["set_f1"] for r in rs]),
            ),
        },
        "success_rate": {
            "definition": (
                "classes 1-4: set-F1 = 1.0; classes 5-6: at least one gold passage cited"
            ),
            "n": len(on),
            **paired_bootstrap(
                on, off, lambda rs: mean([float(r["success"]) for r in rs]), strata=stratum
            ),
        },
        "latency_p50_s": {
            "definition": "median end-to-end turn latency, seconds",
            "n": len(on),
            **paired_bootstrap(
                on, off, lambda rs: median([r["latency_ms"] for r in rs]) / 1000,
                strata=stratum,
            ),
        },
        "latency_p95_s": {
            "definition": "95th-percentile turn latency, seconds (interval is wide at n=50)",
            "n": len(on),
            **paired_bootstrap(
                on, off, lambda rs: percentile([r["latency_ms"] for r in rs], 0.95) / 1000,
                strata=stratum,
            ),
        },
        "cost_per_query_usd": {
            "definition": "mean API cost per question, USD",
            "n": len(on),
            **paired_bootstrap(
                on, off, lambda rs: mean([r["cost_usd"] for r in rs]), strata=stratum
            ),
        },
    }


def cross_session() -> dict:
    """The committed §7.2 result under the same bootstrap, paired by scenario."""
    raw = json.loads(SCENARIOS.read_text())
    by_arm = {
        arm: sorted((s for s in raw["scenarios"] if s["arm"] == arm), key=lambda s: s["id"])
        for arm in ARMS
    }
    on = [{"qid": s["id"], **s} for s in by_arm["on"]]
    off = [{"qid": s["id"], **s} for s in by_arm["off"]]
    return {
        "source": str(SCENARIOS.relative_to(ROOT)),
        "recorded_at": raw["recorded_at"],
        "task_success": {
            "definition": "scenario's final session answered correctly",
            "n": len(on),
            **paired_bootstrap(on, off, lambda rs: mean([float(r["succeeded"]) for r in rs])),
        },
        "cost_per_scenario_usd": {
            "definition": "mean API cost per scenario (three sessions), USD",
            "n": len(on),
            **paired_bootstrap(on, off, lambda rs: mean([r["cost_usd"] for r in rs])),
        },
    }


def incomplete(on: list[dict], off: list[dict], expected: list[str]) -> str | None:
    """Why the arms cannot be analysed, or `None` if they can.

    Checked against the golden set, not merely against each other. Arms that
    both stopped at the same question -- an exhausted credit balance does
    exactly that, since they run side by side -- agree with each other
    perfectly and would be reported as a fifty-question result.
    """
    want = set(expected)
    for arm, records in (("on", on), ("off", off)):
        missing = want - {r["qid"] for r in records}
        if missing:
            return (
                f"memory {arm} is missing {len(missing)} of {len(want)} questions "
                f"({', '.join(sorted(missing)[:5])}{'...' if len(missing) > 5 else ''}). "
                "Errored turns are never cached; re-run, and finished questions "
                "will not be re-charged."
            )
    return None


# --- running ----------------------------------------------------------------


async def eval_memory_owners() -> set[str]:
    import asyncpg

    conn = await asyncpg.connect(settings.database_url)
    try:
        rows = await conn.fetch(
            "SELECT DISTINCT user_id FROM memories WHERE user_id LIKE 'eval_q%'"
        )
        return {r["user_id"] for r in rows}
    finally:
        await conn.close()


def leaking(owners: set[str], expected: list[str], cached: set[str]) -> list[str]:
    """Questions still to run whose user already has memories.

    Only those can leak. A golden-set user with memories from an earlier
    attempt would be answering a question it has, in effect, already seen --
    leakage in memory's favour. But the on arm is *supposed* to be able to
    write memories, and one written while answering a question that is now
    cached is never read again: no later question runs as that user.
    """
    return sorted(q for q in expected if q not in cached and f"eval_{q}" in owners)


async def run_arms(concurrency: int, fresh: bool) -> int:
    expected = [q["id"] for q in load_questions(QUESTIONS)]
    cached = set() if fresh else {r.qid for r in _load_cache(cache_path("on")).values()}
    blocked = leaking(await eval_memory_owners(), expected, cached)
    if blocked:
        print(
            f"refusing to run: {', '.join(blocked)} still to run in the on arm, but "
            "their eval users already hold memories from a previous attempt. "
            "Inspect and delete them first.",
            file=sys.stderr,
        )
        return 1

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    procs = {}
    for arm, enabled in ARMS.items():
        cmd = [
            sys.executable, str(ROOT / "evals" / "runner.py"), "--agent",
            "--cache", str(cache_path(arm)), "--concurrency", str(concurrency),
        ]
        if fresh:
            cmd.append("--fresh")
        env = {**os.environ, "MEMORY_ENABLED": "true" if enabled else "false"}
        log = (CACHE_DIR / f"memory_{arm}.log").open("w")
        procs[arm] = (await asyncio.create_subprocess_exec(
            *cmd, env=env, cwd=ROOT, stdout=log, stderr=asyncio.subprocess.STDOUT
        ), log)
        print(f"  started memory={arm} (log: {CACHE_DIR / f'memory_{arm}.log'})")

    failed = []
    for arm, (proc, log) in procs.items():
        code = await proc.wait()
        log.close()
        print(f"  memory={arm} exited {code}")
        if code:
            failed.append(arm)
    return 1 if failed else 0


# --- output -----------------------------------------------------------------


def _fmt(cell: dict, fmt: str) -> str:
    lo, hi = cell["ci95"]
    return f"{cell['estimate']:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"


def table(golden: dict, sessions: dict) -> str:
    # n is read from the data, never written into the label: a label that says
    # fifty over a partial run is how a partial run gets quoted.
    rows = [
        (f"Set-F1, exact questions (n={golden['set_f1']['n']})", golden["set_f1"], ".3f"),
        (f"Success rate (n={golden['success_rate']['n']})", golden["success_rate"], ".2f"),
        ("Latency p50, s", golden["latency_p50_s"], ".1f"),
        ("Latency p95, s", golden["latency_p95_s"], ".1f"),
        ("Cost / query, $", golden["cost_per_query_usd"], ".3f"),
        (
            f"Cross-session task success (n={sessions['task_success']['n']})",
            sessions["task_success"],
            ".2f",
        ),
        ("Cross-session cost / scenario, $", sessions["cost_per_scenario_usd"], ".3f"),
    ]
    lines = [
        "| Metric | Memory off | Memory on | Δ (on − off) |",
        "|---|---:|---:|---:|",
    ]
    for name, cell, fmt in rows:
        diff = cell["diff_on_minus_off"]
        lo, hi = diff["ci95"]
        lines.append(
            f"| {name} | {_fmt(cell['off'], fmt)} | {_fmt(cell['on'], fmt)} "
            f"| {diff['estimate']:+{fmt}} [{lo:+{fmt}}, {hi:+{fmt}}] |"
        )
    return "\n".join(lines)


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyse-only", action="store_true", help="skip the paid run")
    parser.add_argument("--fresh", action="store_true", help="discard both arms' caches")
    parser.add_argument("--concurrency", type=int, default=2, help="per arm")
    parser.add_argument("--json", type=Path, help="write raw results here")
    args = parser.parse_args(argv)

    if not args.analyse_only:
        code = asyncio.run(run_arms(args.concurrency, args.fresh))
        if code:
            return code

    on, off = load_arm("on"), load_arm("off")
    problem = incomplete(on, off, [q["id"] for q in load_questions(QUESTIONS)])
    if problem:
        print(problem, file=sys.stderr)
        return 1

    golden = summarise(on, off)
    sessions = cross_session()
    print(table(golden, sessions))

    if args.json:
        payload = {
            "experiment": "memory on/off",
            "recorded_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "git_sha": _git_sha(),
            "config": {
                "agent_model": settings.agent_model,
                "agent_effort": settings.agent_effort,
                "tool_retrieval_mode": settings.tool_retrieval_mode,
                "tool_retrieval_k": settings.tool_retrieval_k,
                "hybrid_enabled": settings.hybrid_enabled,
                "embedding_model": settings.embedding_model,
                "concurrency_per_arm": args.concurrency,
            },
            "bootstrap": {
                "resamples": N_BOOT,
                "seed": SEED,
                "interval": "percentile, 95%",
                "paired": True,
                "stratified_by": "exact (classes 1-4) vs interpretive (classes 5-6)",
            },
            "golden_set": golden,
            "cross_session": sessions,
            "total_cost_usd": round(sum(r["cost_usd"] for r in on + off), 4),
            "questions": {"on": on, "off": off},
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.json.relative_to(ROOT) if args.json.is_absolute() else args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
