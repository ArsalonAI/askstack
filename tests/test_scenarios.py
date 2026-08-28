"""The cross-session suite — PRD §7.2.

Nothing here calls a model. What is tested is the harness's arithmetic and the
suite's structure, because every failure mode in this file is silent: a
scenario that runs and measures nothing still prints a number, and a
turns-to-success that averages the wrong set still looks like a result.

The suite's *value* — whether memory actually lowers turns to success — is not
testable without spending money, and is what `python evals/scenarios.py`
reports.
"""

import pytest

from evals.scenarios import (
    MAX_TURNS,
    ScenarioResult,
    SessionResult,
    SuiteError,
    TurnResult,
    _is_elliptical,
    _satisfies,
    load_suite,
    report,
    validate,
)


def turn(called=(), areas=(), cost=0.1) -> TurnResult:
    return TurnResult(
        prompt="p", called=list(called), areas=list(areas), cost_usd=cost
    )


def scenario_result(arm, *, succeeded, turns=None, cost=0.3) -> ScenarioResult:
    result = ScenarioResult(scenario_id="s01", arm=arm)
    result.succeeded = succeeded
    result.turns_to_success = turns
    session = SessionResult(index=3, session_id="sess")
    session.turns = [turn(cost=cost)]
    result.sessions = [session]
    return result


def valid_scenario(**over) -> dict:
    base = {
        "id": "s01",
        "sessions": [
            {"prompt": "one"},
            {"prompt": "two"},
            {
                "prompt": "what's changed since we last spoke?",
                "clarification": "the auth area",
                "expect": {"tool": "merged_prs", "area": "auth"},
            },
        ],
    }
    return {**base, **over}


class TestTheCommittedSuite:
    """The real file, not a fixture. PRD §7.2 fixes its shape."""

    def test_it_loads_and_validates(self):
        assert len(load_suite()["scenarios"]) == 10

    def test_every_scenario_has_three_sessions(self):
        for scenario in load_suite()["scenarios"]:
            assert len(scenario["sessions"]) == 3, scenario["id"]

    def test_every_final_prompt_is_elliptical(self):
        """The whole experiment. A third prompt that names its own area is
        answerable cold, and both arms would score the same — a null result
        caused by the scenario, not by the system."""
        for scenario in load_suite()["scenarios"]:
            final = scenario["sessions"][-1]
            assert _is_elliptical(final["prompt"], final.get("expect") or {}), (
                f"{scenario['id']}: final prompt names its own constraint"
            )

    def test_every_clarification_supplies_what_the_prompt_omits(self):
        """The memory-off arm's only route to success. If the clarification
        does not actually name the constraint, that arm fails for the wrong
        reason and the comparison overstates memory."""
        for scenario in load_suite()["scenarios"]:
            final = scenario["sessions"][-1]
            area = (final.get("expect") or {}).get("area")
            if area:
                assert area in final["clarification"].lower(), scenario["id"]

    def test_scenario_ids_are_unique(self):
        ids = [s["id"] for s in load_suite()["scenarios"]]
        assert len(ids) == len(set(ids))


class TestValidate:
    def test_accepts_a_well_formed_scenario(self):
        validate([valid_scenario()])

    def test_rejects_a_duplicate_id(self):
        with pytest.raises(SuiteError, match="duplicate"):
            validate([valid_scenario(), valid_scenario()])

    def test_rejects_the_wrong_session_count(self):
        """Two sessions cannot establish *and* narrow before the elliptical
        turn, so the third session — the experiment — has nothing to rely on."""
        short = valid_scenario()
        short["sessions"] = short["sessions"][:2]
        with pytest.raises(SuiteError, match="3 sessions"):
            validate([short])

    def test_rejects_a_final_session_with_no_expectation(self):
        loose = valid_scenario()
        loose["sessions"][-1]["expect"] = {}
        with pytest.raises(SuiteError, match="expect.tool"):
            validate([loose])

    def test_rejects_a_final_session_with_no_clarification(self):
        """Without one the memory-off arm scores 0 by construction, which
        flatters memory rather than measuring it."""
        loose = valid_scenario()
        del loose["sessions"][-1]["clarification"]
        with pytest.raises(SuiteError, match="clarification"):
            validate([loose])


class TestIsElliptical:
    def test_a_prompt_naming_its_area_is_not(self):
        assert not _is_elliptical(
            "what merged in auth recently?", {"area": "auth"}
        )

    def test_a_prompt_referring_back_is(self):
        assert _is_elliptical("what's changed since we last spoke?", {"area": "auth"})

    def test_case_is_ignored(self):
        assert not _is_elliptical("What about AUTH?", {"area": "auth"})


class TestSatisfies:
    def test_the_expected_tool_must_be_called(self):
        assert _satisfies(turn(called=["merged_prs"]), {"tool": "merged_prs"})
        assert not _satisfies(turn(called=["stale_prs"]), {"tool": "merged_prs"})

    def test_the_area_constraint_must_be_resolved(self):
        """Calling the right tool for the wrong area is not a partial success —
        it is a confidently wrong status report, the PRD §5.2 failure."""
        expect = {"tool": "merged_prs", "area": "auth"}
        assert _satisfies(turn(called=["merged_prs"], areas=["auth"]), expect)
        assert not _satisfies(turn(called=["merged_prs"], areas=["docs"]), expect)
        assert not _satisfies(turn(called=["merged_prs"]), expect)

    def test_no_area_constraint_means_the_tool_is_enough(self):
        assert _satisfies(turn(called=["release_diff"]), {"tool": "release_diff"})

    def test_a_turn_that_called_nothing_fails(self):
        """The mechanical stand-in for "the agent asked instead of answering"."""
        assert not _satisfies(turn(), {"tool": "merged_prs"})


class TestReport:
    def test_turns_to_success_averages_completed_tasks_only(self):
        """Mixing a failure in as a large number would make "never got there"
        read as "took a while", which are not the same outcome."""
        arms = report(
            [
                scenario_result("on", succeeded=True, turns=1),
                scenario_result("on", succeeded=True, turns=3),
                scenario_result("on", succeeded=False, turns=None),
            ]
        )
        assert arms["on"]["turns_to_success"] == pytest.approx(2.0)
        assert arms["on"]["task_success"] == pytest.approx(2 / 3)

    def test_no_completed_task_reports_none_not_zero(self):
        """Zero turns to success would be the best possible score."""
        arms = report([scenario_result("off", succeeded=False)])
        assert arms["off"]["turns_to_success"] is None
        assert arms["off"]["cost_per_completed_task"] is None

    def test_total_cost_counts_failures_too(self):
        """A failed task still cost money, and the budget is real."""
        arms = report(
            [
                scenario_result("on", succeeded=True, turns=1, cost=0.2),
                scenario_result("on", succeeded=False, cost=0.5),
            ]
        )
        assert arms["on"]["total_cost_usd"] == pytest.approx(0.7)
        assert arms["on"]["cost_per_completed_task"] == pytest.approx(0.2)

    def test_arms_are_reported_separately(self):
        arms = report(
            [
                scenario_result("on", succeeded=True, turns=1),
                scenario_result("off", succeeded=True, turns=2),
            ]
        )
        assert arms["on"]["turns_to_success"] == 1
        assert arms["off"]["turns_to_success"] == 2


class TestHarnessBounds:
    def test_max_turns_leaves_room_for_one_clarification(self):
        """One turn means memory resolved it; two means it had to be told.
        A bound below 2 could never observe the difference."""
        assert MAX_TURNS >= 2


class TestReportRefusesErroredRuns:
    """An errored arm is not a measured arm.

    This run exhausted its credit balance partway through: every memory-on
    scenario errored, and the harness wrote a report recording task_success
    0.00 for `on` against 1.00 for `off` — a file asserting that memory made
    the system strictly worse, caused entirely by billing. `evals/runner.py`
    already refused to write a baseline under exactly these conditions; the
    guard had simply not been carried across.
    """

    def test_an_errored_scenario_is_visible_on_the_result(self):
        errored = scenario_result("on", succeeded=False)
        errored.sessions[0].turns[0].error = {"code": "upstream_unavailable"}
        assert errored.error is not None

    def test_error_is_none_on_a_clean_run(self):
        assert scenario_result("on", succeeded=True, turns=1).error is None

    def test_a_failed_task_is_not_an_errored_one(self):
        """The distinction the guard rests on: an agent that ran and did not
        do the task is a measurement; an agent that never reached the model is
        not."""
        missed = scenario_result("off", succeeded=False)
        assert missed.error is None

    def test_errored_scenarios_are_counted_per_arm(self):
        errored = scenario_result("on", succeeded=False)
        errored.sessions[0].turns[0].error = {"code": "upstream_unavailable"}
        arms = report([errored, scenario_result("off", succeeded=True, turns=2)])
        assert arms["on"]["errored"] == 1
        assert arms["off"]["errored"] == 0
