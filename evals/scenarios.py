#!/usr/bin/env python3
"""Cross-session scenarios — PRD §7.2, the suite that decides whether memory paid.

The golden set measures one turn at a time and so cannot see the thing this
project is actually claiming. A stateless agent answers "what merged in auth in
June" perfectly well; what it cannot do is answer "what's changed since we last
spoke" without asking who "we" are and when "last" was. This suite runs the
same three-session check-in with memory on and with memory off, and reports the
difference.

The headline metric is **turns to success** (PRD §6). It is mechanical: the
harness sends the session's prompt, and if the agent asks a question instead of
calling a tool, the harness sends the scripted `clarification` and counts
another turn. Memory-on should need one turn where memory-off needs two. No
judge is involved in any number here.

    python evals/scenarios.py --check          # validate the suite, no model
    python evals/scenarios.py                  # both arms; costs money
    python evals/scenarios.py --arm on         # one arm
    python evals/scenarios.py --scenario s01
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import asyncpg
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings, settings  # noqa: E402
from app.facts.store import PostgresFactsStore  # noqa: E402
from app.retrieval.embedder import get_embedder  # noqa: E402
from app.retrieval.hybrid import HybridRetriever  # noqa: E402

SUITE = Path(__file__).parent / "scenarios" / "scenarios.yaml"

# How many clarifications the harness will offer before calling the task
# failed. Two turns is the interesting boundary: one means memory resolved the
# reference, two means it had to be told. More than that is a failure, not a
# slower success — a manager who has to explain themselves three times has
# stopped using the tool.
MAX_TURNS = 3


class SuiteError(ValueError):
    """The suite is malformed. Raised rather than skipped — a scenario that
    silently does not run measures nothing and looks like a pass."""


@dataclass
class TurnResult:
    prompt: str
    session_id: str | None = None
    called: list[str] = field(default_factory=list)
    areas: list[str] = field(default_factory=list)
    windows: list[list[str]] = field(default_factory=list)
    cost_usd: float = 0.0
    latency_ms: int = 0
    memories_loaded: int = 0
    memory_writes: int = 0
    error: dict | None = None


@dataclass
class SessionResult:
    index: int
    session_id: str | None
    turns: list[TurnResult] = field(default_factory=list)
    # Memories §8.1 extracted when this session ended. The number that explains
    # whether the *next* session had anything to load.
    extracted: int = 0
    # Rolled up on the cache path, where per-turn results are not retained.
    memories_loaded: int = 0

    @property
    def cost_usd(self) -> float:
        return sum(t.cost_usd for t in self.turns)


@dataclass
class ScenarioResult:
    scenario_id: str
    arm: str
    sessions: list[SessionResult] = field(default_factory=list)
    succeeded: bool = False
    turns_to_success: int | None = None
    # A cached row carries the scored outcome and the cost, not the per-turn
    # detail. Reporting `mem_loaded=0` for one would state as a measurement
    # something the cache simply does not hold.
    from_cache: bool = False

    @property
    def cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.sessions)

    @property
    def error(self) -> dict | None:
        for session in self.sessions:
            for turn in session.turns:
                if turn.error:
                    return turn.error
        return None


# Bumped when a scoring rule changes, exactly as in `evals/runner.py`.
CACHE_FORMAT = 1


def cache_key(scenario_id: str, arm: str) -> str:
    return f"{scenario_id}/{arm}"


def load_cache(path: Path | None) -> dict[str, ScenarioResult]:
    """Completed scenarios from an earlier run.

    `evals/runner.py` has cached per question since M2 for a reason this
    harness then rediscovered the hard way: a run killed partway through
    throws away everything it already paid for. Three sessions per scenario
    with extraction between them takes long enough that something *will*
    interrupt it — a timeout, a laptop lid, a killed background task — and
    losing eight completed scenarios to re-measure the remaining twelve is the
    kind of waste that stops people running the eval at all.

    An errored scenario is never cached, for the same reason the report refuses
    to be written from one: it is not a measurement, and freezing it would turn
    a billing failure into a permanent zero.
    """
    if path is None or not path.is_file():
        return {}
    cached, stale = {}, 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("cache_format") != CACHE_FORMAT:
            stale += 1
            continue
        result = ScenarioResult(
            scenario_id=row["scenario_id"],
            arm=row["arm"],
            succeeded=row["succeeded"],
            turns_to_success=row["turns_to_success"],
            from_cache=True,
        )
        session = SessionResult(index=0, session_id=None)
        session.turns = [TurnResult(prompt="", cost_usd=row["cost_usd"])]
        session.extracted = row.get("extracted", 0)
        session.memories_loaded = row.get("memories_loaded", 0)
        result.sessions = [session]
        cached[cache_key(result.scenario_id, result.arm)] = result
    if stale:
        print(f"  discarding {stale} cached scenario(s) from an older scorer",
              file=sys.stderr)
    return cached


def append_cache(path: Path | None, result: ScenarioResult) -> None:
    if path is None or result.error:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "cache_format": CACHE_FORMAT,
                    "scenario_id": result.scenario_id,
                    "arm": result.arm,
                    "succeeded": result.succeeded,
                    "turns_to_success": result.turns_to_success,
                    "cost_usd": round(result.cost_usd, 6),
                    "extracted": sum(s.extracted for s in result.sessions),
                    "memories_loaded": sum(
                        t.memories_loaded for s in result.sessions for t in s.turns
                    ),
                }
            )
            + "\n"
        )


def load_suite(path: Path = SUITE) -> dict:
    raw = yaml.safe_load(path.read_text()) or {}
    scenarios = raw.get("scenarios") or []
    if not scenarios:
        raise SuiteError(f"{path.name} defines no scenarios")
    validate(scenarios)
    return raw


def validate(scenarios: list[dict]) -> None:
    """Structural checks that need no model and no database.

    Every one of these is a way for a scenario to run, produce a number, and
    measure nothing — which is worse than failing to run at all.
    """
    seen: set[str] = set()
    for scenario in scenarios:
        sid = scenario.get("id")
        if not sid:
            raise SuiteError("a scenario has no id")
        if sid in seen:
            raise SuiteError(f"duplicate scenario id {sid!r}")
        seen.add(sid)

        sessions = scenario.get("sessions") or []
        if len(sessions) != 3:
            # PRD §7.2 fixes this at three. Two cannot establish *and* narrow
            # before the elliptical turn, and the third session is the whole
            # experiment.
            raise SuiteError(f"{sid}: expected 3 sessions, got {len(sessions)}")
        for index, session in enumerate(sessions):
            if not session.get("prompt"):
                raise SuiteError(f"{sid}: session {index + 1} has no prompt")

        final = sessions[-1]
        expect = final.get("expect") or {}
        if not expect.get("tool"):
            raise SuiteError(f"{sid}: the final session needs expect.tool")
        if not final.get("clarification"):
            # Without a scripted clarification the memory-off arm has no way to
            # recover, so it scores 0 by construction and the comparison
            # flatters memory instead of measuring it.
            raise SuiteError(f"{sid}: the final session needs a clarification")


def _is_elliptical(prompt: str, expect: dict) -> bool:
    """Does the final prompt actually depend on the earlier sessions?

    A scenario whose third prompt names its own subject would be answerable
    cold, and memory-on versus memory-off would measure nothing. Checked rather
    than trusted, because it is easy to fix a flaky scenario by quietly making
    the prompt more explicit.
    """
    lowered = prompt.lower()
    for constraint in (expect.get("area"), expect.get("since")):
        if constraint and str(constraint).lower() in lowered:
            return False
    return True


async def run_scenario(
    scenario: dict,
    *,
    pool,
    retriever,
    client,
    arm_settings: Settings,
    as_of: datetime,
) -> ScenarioResult:
    """Three sessions under one user, with the arm's settings.

    The user id carries across the three sessions and is unique to the
    scenario and arm — that continuity is the thing under test, and sharing a
    user across scenarios would let s01's memories answer s07's question.
    """
    result = ScenarioResult(
        scenario_id=scenario["id"], arm=arm_settings_name(arm_settings)
    )
    user_id = f"mgr_{scenario['id']}_{result.arm}"
    sessions = scenario["sessions"]
    expect = sessions[-1].get("expect") or {}

    for index, session in enumerate(sessions, start=1):
        is_final = index == len(sessions)
        session_result = SessionResult(index=index, session_id=None)

        # `session_id` starts as None on every index, so each is a *new*
        # conversation. That is what makes this a cross-session test rather
        # than one long one: the third session cannot read the first two from
        # history, only from memory.
        session_id: str | None = None
        prompt = session["prompt"]

        for turn_number in range(1, MAX_TURNS + 1):
            turn = await _one_turn(
                user_id,
                session_id,
                prompt,
                as_of,
                pool=pool,
                retriever=retriever,
                client=client,
                arm_settings=arm_settings,
            )
            session_result.turns.append(turn)
            session_id = turn.session_id or session_id
            session_result.session_id = session_id

            if turn.error:
                result.sessions.append(session_result)
                return result

            # Only the final session is scored. The first two exist to put
            # something in memory worth having.
            if not is_final:
                break
            if _satisfies(turn, expect):
                result.succeeded = True
                result.turns_to_success = turn_number
                break

            clarification = session.get("clarification")
            if turn_number >= MAX_TURNS or not clarification:
                break
            # The agent asked instead of answering. Answer it, and charge the
            # scenario another turn — this is the measurement.
            prompt = clarification

        # §8.1's session-end trigger, awaited so session N's memories exist
        # before session N+1 loads its block. This is the whole mechanism the
        # suite measures: without it a three-turn check-in extracts nothing and
        # both arms are identical by construction.
        session_result.extracted = await _end_session(
            session_id,
            pool=pool,
            retriever=retriever,
            client=client,
            arm_settings=arm_settings,
        )
        result.sessions.append(session_result)

    return result


async def _end_session(
    session_id: str | None, *, pool, retriever, client, arm_settings
) -> int:
    if session_id is None or not arm_settings.memory_enabled:
        return 0
    from app.memory.lifecycle import Extractor
    from evals.runner import build_memory

    manager = build_memory(pool, retriever.embedder, client, arm_settings)
    if manager is None:
        return 0
    extractor = Extractor(
        pool, manager.store, retriever.embedder, client, arm_settings
    )
    async with pool.acquire() as conn:
        closed = await conn.fetchval(
            "UPDATE sessions SET ended_at = now()"
            " WHERE id = $1 AND ended_at IS NULL RETURNING id",
            session_id,
        )
    if closed is None:
        return 0
    report = await extractor.extract(session_id)
    return report.written


def arm_settings_name(arm: Settings) -> str:
    return "on" if arm.memory_enabled else "off"


def _satisfies(turn: TurnResult, expect: dict) -> bool:
    """Did this turn actually do the task?

    Deliberately strict about the constraint and lenient about everything else.
    Calling `merged_prs` for the wrong area is not a partial success — it is a
    confidently wrong status report, which is the PRD §5.2 failure.
    """
    if expect["tool"] not in turn.called:
        return False
    if area := expect.get("area"):
        return area in turn.areas
    return True


async def _one_turn(
    user_id: str,
    session_id: str | None,
    prompt: str,
    as_of: datetime,
    *,
    pool,
    retriever,
    client,
    arm_settings,
) -> TurnResult:
    from app.orchestrator import Orchestrator
    from app.tools.registry import ToolRegistry
    from app.tools.selector import build_selector
    from evals.runner import build_memory

    registry = ToolRegistry(
        PostgresFactsStore(pool),
        retriever,
        top_k=arm_settings.retrieval_top_k,
        memory=build_memory(pool, retriever.embedder, client, arm_settings),
    )
    orchestrator = Orchestrator(
        pool,
        registry,
        build_selector(
            arm_settings.tool_retrieval_mode,
            registry.definitions(),
            pool=pool,
            embedder=retriever.embedder,
            floor=arm_settings.tool_similarity_floor,
        ),
        client,
        arm_settings,
        as_of=as_of,
        memory=registry.memory,
    )

    turn = TurnResult(prompt=prompt)
    async for event, data in orchestrator.run(user_id, session_id, prompt, as_of=as_of):
        if event == "session":
            turn.session_id = data["session_id"]
        elif event == "memory_loaded":
            turn.memories_loaded = len(data["memories"])
        elif event == "memory_write":
            turn.memory_writes += 1
        elif event == "tool_call" and data["status"] == "ok":
            turn.called.append(data["name"])
        elif event == "retrieval" and data["kind"] == "structured":
            if data.get("area"):
                turn.areas.append(data["area"])
            if data.get("window"):
                turn.windows.append(data["window"])
        elif event == "error":
            turn.error = data
        elif event == "done":
            turn.cost_usd = data.get("cost_usd", 0.0)
            turn.latency_ms = data.get("latency_ms", 0)
    return turn


def report(results: list[ScenarioResult]) -> dict:
    """PRD §7.2's three numbers, per arm."""
    arms: dict[str, dict] = {}
    for arm in sorted({r.arm for r in results}):
        rows = [r for r in results if r.arm == arm]
        done = [r for r in rows if r.succeeded]
        arms[arm] = {
            "scenarios": len(rows),
            "task_success": len(done) / len(rows) if rows else 0.0,
            # Averaged over *completed* tasks only. Including failures would
            # mix "took two turns" with "never got there", and the second is
            # not a slower version of the first.
            "turns_to_success": (
                sum(r.turns_to_success for r in done) / len(done) if done else None
            ),
            "cost_per_completed_task": (
                sum(r.cost_usd for r in done) / len(done) if done else None
            ),
            "total_cost_usd": sum(r.cost_usd for r in rows),
            "errored": sum(1 for r in rows if r.error),
        }
    return arms


def _or_dash(value: float | None, fmt: str) -> str:
    """A missing number prints as an em dash, never as 0.

    Zero turns to success would be the best possible score, and an arm that
    completed nothing must not be able to top the table.
    """
    return "—" if value is None else format(value, fmt)


def print_report(results: list[ScenarioResult], arms: dict) -> None:
    print(f"\n{'arm':>5}  {'n':>3}  {'success':>8}  {'turns':>7}  {'$/task':>8}  {'total':>8}")
    for arm, metrics in arms.items():
        turns = _or_dash(metrics["turns_to_success"], ".2f")
        per = _or_dash(metrics["cost_per_completed_task"], ".3f")
        per = per if per == "—" else f"${per}"
        print(
            f"{arm:>5}  {metrics['scenarios']:>3}  {metrics['task_success']:>8.2f}  "
            f"{turns:>7}  {per:>8}  ${metrics['total_cost_usd']:.2f}"
        )

    if {"on", "off"} <= arms.keys():
        on, off = arms["on"], arms["off"]
        print("\n  memory on vs off")
        print(f"    task success      {off['task_success']:.2f} → {on['task_success']:.2f}")
        if on["turns_to_success"] and off["turns_to_success"]:
            saved = off["turns_to_success"] - on["turns_to_success"]
            direction = "fewer" if saved > 0 else "more"
            print(
                f"    turns to success  {off['turns_to_success']:.2f} → "
                f"{on['turns_to_success']:.2f}  "
                f"({abs(saved):.2f} {direction} turns with memory)"
            )

    print("\n  per scenario")
    for result in sorted(results, key=lambda r: (r.scenario_id, r.arm)):
        mark = "ok " if result.succeeded else "MISS"
        turns = result.turns_to_success or "—"
        loaded = sum(t.memories_loaded for s in result.sessions for t in s.turns) or sum(
            s.memories_loaded for s in result.sessions
        )
        writes = sum(t.memory_writes for s in result.sessions for t in s.turns)
        extracted = sum(s.extracted for s in result.sessions)
        cached = "  (cached)" if result.from_cache else ""
        print(
            f"    {result.scenario_id}  {result.arm:<3}  {mark}  turns={turns}  "
            f"mem_loaded={loaded}  extracted={extracted}  agent_writes={writes}  "
            f"${result.cost_usd:.3f}{cached}"
        )
        if result.error:
            print(f"          error: {result.error}")
        elif not result.succeeded and result.from_cache:
            print("          (cached; per-turn detail not retained — rerun with --fresh)")
        elif not result.succeeded:
            # A miss with no detail is unactionable: "the agent did not do the
            # task" does not say whether it called nothing, called the wrong
            # tool, or resolved the wrong area — three different bugs.
            for session in result.sessions:
                for attempt in session.turns:
                    called = ", ".join(attempt.called) or "(no tool call)"
                    areas = f" areas={attempt.areas}" if attempt.areas else ""
                    print(
                        f"          s{session.index}: {attempt.prompt[:56]!r}"
                        f"\n                 → {called}{areas}"
                    )


def check(raw: dict) -> int:
    scenarios = raw["scenarios"]
    print(f"scenarios.yaml: {len(scenarios)} scenarios, {len(scenarios) * 3} sessions")

    problems: list[str] = []
    for scenario in scenarios:
        final = scenario["sessions"][-1]
        if not _is_elliptical(final["prompt"], final.get("expect") or {}):
            problems.append(
                f"  {scenario['id']}: the final prompt names its own constraint, so "
                f"it is answerable without memory and measures nothing"
            )
    if problems:
        print("\nproblems:")
        print("\n".join(problems))
        return 1
    print("every final prompt depends on the earlier sessions; check passed")
    return 0


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate only; no model")
    parser.add_argument("--arm", choices=["on", "off"], help="run one arm")
    parser.add_argument("--scenario", help="run one scenario id")
    parser.add_argument("--json", type=Path, help="write the report here")
    parser.add_argument(
        "--cache", type=Path, default=Path(".cache/scenario_results.jsonl"),
        help="reuse completed scenarios so a killed run resumes",
    )
    parser.add_argument(
        "--fresh", action="store_true", help="ignore and overwrite the cache"
    )
    args = parser.parse_args(argv)

    raw = load_suite()
    if args.check:
        return check(raw)

    if not settings.anthropic_api_key:
        print("needs ANTHROPIC_API_KEY", file=sys.stderr)
        return 2

    scenarios = raw["scenarios"]
    if args.scenario:
        scenarios = [s for s in scenarios if s["id"] == args.scenario]
        if not scenarios:
            print(f"no scenario {args.scenario!r}", file=sys.stderr)
            return 2

    as_of = datetime.fromisoformat(str(raw["as_of"])).replace(tzinfo=UTC)
    arms = [args.arm] if args.arm else ["off", "on"]

    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    pool = await asyncpg.create_pool(settings.database_url, min_size=2, max_size=16)
    if pool is None:
        print("could not connect to the database", file=sys.stderr)
        return 2

    if args.fresh and args.cache and args.cache.exists():
        args.cache.unlink()
    cached = load_cache(args.cache)
    if cached:
        print(f"  resuming: {len(cached)} scenario(s) already measured",
              file=sys.stderr)

    results: list[ScenarioResult] = []
    try:
        retriever = HybridRetriever(pool, get_embedder())
        for arm in arms:
            arm_settings = settings.model_copy(update={"memory_enabled": arm == "on"})
            print(f"\n=== memory {arm} ===", file=sys.stderr)
            for scenario in scenarios:
                key = cache_key(scenario["id"], arm)
                if key in cached:
                    results.append(cached[key])
                    print(f"  {scenario['id']} cached", file=sys.stderr, flush=True)
                    continue
                result = await run_scenario(
                    scenario,
                    pool=pool,
                    retriever=retriever,
                    client=client,
                    arm_settings=arm_settings,
                    as_of=as_of,
                )
                results.append(result)
                append_cache(args.cache, result)
                mark = "ok" if result.succeeded else "MISS"
                print(
                    f"  {result.scenario_id} {mark} turns={result.turns_to_success} "
                    f"${result.cost_usd:.3f}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        await pool.close()

    metrics = report(results)
    print_report(results, metrics)

    if args.json:
        # The same guard `evals/runner.py` already has, and for the same reason
        # — learned twice because it was not carried across.
        #
        # An errored scenario is not a failed scenario. When this run exhausted
        # its credit balance partway through, every memory-on scenario errored
        # and the report recorded task_success 0.00 for `on` against 1.00 for
        # `off`: a file saying memory made the system strictly worse, caused
        # entirely by billing. Committed, it would have been the headline
        # number for the architecture this project exists to test.
        failed = [r for r in results if r.error]
        if failed:
            print(
                f"\nrefusing to write a report: {len(failed)} scenario(s) errored "
                f"({', '.join(f'{r.scenario_id}/{r.arm}' for r in failed[:5])}"
                f"{'...' if len(failed) > 5 else ''}).\n"
                "An errored arm is not a measured arm — writing this would "
                "record an infrastructure failure as an architectural result.",
                file=sys.stderr,
            )
            return 1
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "recorded_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "as_of": as_of.isoformat(),
                    "arms": metrics,
                    "scenarios": [
                        {
                            "id": r.scenario_id,
                            "arm": r.arm,
                            "succeeded": r.succeeded,
                            "turns_to_success": r.turns_to_success,
                            "cost_usd": round(r.cost_usd, 4),
                        }
                        for r in results
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
