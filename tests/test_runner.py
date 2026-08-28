"""Scorers — TRD §14.1.

Pure functions over lists, so none of this needs a database or a model. That
is deliberate: these are the numbers CI gates on, and a scorer bug would be
invisible behind a plausible-looking metric.
"""

import json

import pytest

from app.tools.registry import ALWAYS_INJECTED
from evals.runner import (
    CACHE_FORMAT,
    _load_cache,
    gold_tool_entities,
    jaccard,
    mrr_at_k,
    recall_at_k,
    set_f1,
)


class TestSetF1:
    def test_exact_match(self):
        assert set_f1(["pr:1", "pr:2"], ["pr:2", "pr:1"]) == 1.0

    def test_empty_vs_empty_is_perfect(self):
        """"Which auth PRs merged in the last three months" can correctly answer
        *none*. Scoring that 0.0 would punish the one behaviour §5.2 wants —
        the held-out set's `expect_empty` questions depend on this."""
        assert set_f1([], []) == 1.0

    def test_empty_prediction_against_nonempty_gold(self):
        assert set_f1([], ["pr:1"]) == 0.0

    def test_nonempty_prediction_against_empty_gold(self):
        """Inventing entities where the answer is none is the §5.2 failure."""
        assert set_f1(["pr:1"], []) == 0.0

    def test_disjoint(self):
        assert set_f1(["pr:1"], ["pr:2"]) == 0.0

    def test_partial(self):
        # 1 hit, precision 1/2, recall 1/3 -> F1 = 2·(1/2·1/3)/(1/2+1/3) = 0.4
        assert set_f1(["pr:1", "pr:9"], ["pr:1", "pr:2", "pr:3"]) == 0.4

    def test_duplicates_do_not_inflate(self):
        assert set_f1(["pr:1", "pr:1"], ["pr:1"]) == 1.0


class TestGoldToolEntities:
    """§14.1 — which of a turn's retrievals set-F1 is scored over.

    The union reading these replace measured tool-call count: q006 lost a third
    of its score for performing the §5.2 verification the PRD requires.
    """

    def test_only_the_gold_tool_counts(self):
        retrievals = [
            {"tool": "pr_state", "entities": ["pr:15806"]},
            {"tool": "merged_prs", "entities": [f"pr:{n}" for n in range(500)]},
        ]
        assert gold_tool_entities(retrievals, {"pr_state"}) == ["pr:15806"]

    def test_verification_lookups_are_free(self):
        """q006's shape: check the PR, then separately confirm the release."""
        retrievals = [
            {"tool": "pr_state", "entities": ["pr:15806"]},
            {"tool": "release_info", "entities": ["release:0.138.0"]},
        ]
        predicted = gold_tool_entities(retrievals, {"pr_state"})
        assert set_f1(predicted, ["pr:15806"]) == 1.0

    def test_one_tool_called_twice_contributes_both(self):
        """Two windows is one tool used twice — not a tool-count artefact."""
        retrievals = [
            {"tool": "merged_prs", "entities": ["pr:1"]},
            {"tool": "merged_prs", "entities": ["pr:2"]},
        ]
        assert gold_tool_entities(retrievals, {"merged_prs"}) == ["pr:1", "pr:2"]

    def test_deduplicates_across_calls(self):
        retrievals = [
            {"tool": "merged_prs", "entities": ["pr:1", "pr:2"]},
            {"tool": "merged_prs", "entities": ["pr:2", "pr:3"]},
        ]
        assert gold_tool_entities(retrievals, {"merged_prs"}) == ["pr:1", "pr:2", "pr:3"]

    def test_never_calling_the_gold_tool_scores_zero(self):
        """q005 called nothing at all. An agent that never ran the right query
        did not answer the question, however good its prose."""
        retrievals = [{"tool": "merged_prs", "entities": ["pr:1"]}]
        predicted = gold_tool_entities(retrievals, {"issue_state"})
        assert predicted == []
        assert set_f1(predicted, ["issue:11143"]) == 0.0

    def test_no_retrievals_at_all(self):
        assert gold_tool_entities([], {"pr_state"}) == []


class TestCacheFormat:
    """A cached row is a *scored* row, so it belongs to one scorer version."""

    def _row(self, qid, **over):
        row = {
            "cache_format": CACHE_FORMAT,
            "qid": qid,
            "klass": 1,
            "metrics": {"set_f1": 1.0},
            "predicted": ["pr:1"],
            "gold": ["pr:1"],
            "agent": {"retrievals": [], "error": None},
        }
        return {**row, **over}

    def test_current_rows_load(self, tmp_path):
        path = tmp_path / "c.jsonl"
        path.write_text(json.dumps(self._row("q001")) + "\n")
        assert set(_load_cache(path)) == {"q001"}

    def test_rows_from_an_older_scorer_are_dropped(self, tmp_path):
        """Replaying a union-scored row beside a gold-tool-scored one would
        average a baseline half-computed each way, with nothing saying so."""
        path = tmp_path / "c.jsonl"
        path.write_text(
            json.dumps(self._row("q001", cache_format=CACHE_FORMAT - 1)) + "\n"
            + json.dumps({k: v for k, v in self._row("q002").items()
                          if k != "cache_format"}) + "\n"
            + json.dumps(self._row("q003")) + "\n"
        )
        assert set(_load_cache(path)) == {"q003"}

    def test_missing_file_is_empty_not_an_error(self, tmp_path):
        assert _load_cache(tmp_path / "nope.jsonl") == {}


class TestRecallAtK:
    def test_counts_only_the_first_k(self):
        retrieved = ["a", "b", "c", "d", "e", "gold"]
        assert recall_at_k(retrieved, ["gold"], k=5) == 0.0
        assert recall_at_k(retrieved, ["gold"], k=6) == 1.0

    def test_fraction_of_gold_found(self):
        assert recall_at_k(["a", "b"], ["a", "z"], k=5) == 0.5

    def test_empty_gold_is_zero_not_a_crash(self):
        # Distinct from set-F1: an interpretive question with no gold chunks is
        # a malformed question, not a legitimate empty answer. build_gold's
        # validate rejects it; this only guarantees no ZeroDivisionError.
        assert recall_at_k(["a"], [], k=5) == 0.0


class TestJaccard:
    """Tool-selection accuracy — §14.1. Jaccard rather than exact match is what
    degrades gracefully when the selector returns k tools and gold names one."""

    def test_identical_sets(self):
        assert jaccard({"pr_state"}, {"pr_state"}) == 1.0

    def test_disjoint_sets(self):
        assert jaccard({"pr_state"}, {"merged_prs"}) == 0.0

    def test_gold_present_among_extras(self):
        # The shape of every `semantic` result: 1 gold tool inside 5 selected.
        assert jaccard({"a", "b", "c", "d", "e"}, {"a"}) == 0.2

    def test_both_empty_is_perfect(self):
        assert jaccard(set(), set()) == 1.0

    def test_empty_prediction_against_gold(self):
        assert jaccard(set(), {"pr_state"}) == 0.0


class TestMRRAtK:
    def test_reciprocal_of_first_hit(self):
        assert mrr_at_k(["a", "b", "gold"], ["gold"], k=10) == 1 / 3

    def test_first_gold_wins_not_the_best_one(self):
        assert mrr_at_k(["g2", "g1"], ["g1", "g2"], k=10) == 1.0

    def test_beyond_k_scores_zero(self):
        retrieved = [f"c{i}" for i in range(10)] + ["gold"]
        assert mrr_at_k(retrieved, ["gold"], k=10) == 0.0

    def test_no_hit(self):
        assert mrr_at_k(["a", "b"], ["gold"], k=10) == 0.0


class TestAlwaysInjectedExclusion:
    """§7.2.2 — tool accuracy is computed over the retrieved set, excluding the
    always-on tools.

    Without this, memory shipping at M3 would show up as a tool-accuracy drop
    caused entirely by two tools being present in every request. The metric
    would move because the system gained a feature, not because the selector
    got worse, and the M2→M3 comparison would read as a regression.
    """

    def test_the_memory_tools_are_the_always_injected_set(self):
        assert set(ALWAYS_INJECTED) == {"memory_search", "memory_write"}

    def test_excluding_them_leaves_the_selector_measured_alone(self):
        selected = {"merged_prs", "pr_state", "memory_search", "memory_write"}
        scored = {t for t in selected if t not in ALWAYS_INJECTED}
        assert scored == {"merged_prs", "pr_state"}

    def test_a_perfect_selection_still_scores_1_with_memory_present(self):
        """The case that would silently break: gold is one tool, the selector
        found exactly it, and two always-on tools drag jaccard to 0.33."""
        selected = {"pr_state", "memory_search", "memory_write"}
        scored = {t for t in selected if t not in ALWAYS_INJECTED}
        assert jaccard(scored, {"pr_state"}) == 1.0
        assert jaccard(selected, {"pr_state"}) == pytest.approx(1 / 3)


class TestGenerationRate:
    """§13's latency budget is a statement about output volume.

    Measured on the M4 run, latency is 0.6s fixed + 17.3ms per output token —
    generation is essentially the whole turn. The rate is what converts a
    seconds budget into a token ceiling, which is the number anyone can act on.
    """

    def test_it_is_tokens_over_seconds(self):
        from evals.runner import generation_rate

        assert generation_rate([1000], [60]) == pytest.approx(60.0)

    def test_it_takes_the_median_not_the_ratio_of_sums(self):
        """One very long turn would otherwise dominate and report the slowest
        case as the typical one."""
        from evals.runner import generation_rate

        # Two fast turns at 100 tok/s, one slow outlier at 10 tok/s.
        rate = generation_rate([1000, 1000, 10_000], [100, 100, 100])
        assert rate == pytest.approx(100.0)

    def test_zero_length_turns_do_not_divide_by_zero(self):
        from evals.runner import generation_rate

        assert generation_rate([0, 1000], [0, 50]) == pytest.approx(50.0)

    def test_an_empty_run_reports_zero_rather_than_raising(self):
        from evals.runner import generation_rate

        assert generation_rate([], []) == 0.0

    def test_mismatched_inputs_raise_rather_than_silently_truncating(self):
        """`zip` without `strict` would pair a latency with another turn's
        token count and report a plausible wrong rate."""
        from evals.runner import generation_rate

        with pytest.raises(ValueError):
            generation_rate([1000, 2000], [50])

    def test_the_budget_constant_matches_the_spec(self):
        from evals.runner import LATENCY_BUDGET_MS

        assert LATENCY_BUDGET_MS == 25_000  # §13: "Full response, p95 < 25 s"
