"""Golden-set invariants — PRD §7.1, TRD §14.1.

These run without a database: they check the committed YAML is well-formed and
internally consistent. The checks that need the corpus — every `gold_chunk`
resolves, every `as_of` sits inside the ingest window — live in
`evals/build_gold.py --check`, which runs against the pinned revision.
"""

import itertools
from datetime import date
from pathlib import Path

import pytest
import yaml

GOLDEN = Path(__file__).resolve().parents[1] / "evals" / "golden"
QUESTIONS = yaml.safe_load((GOLDEN / "questions.yaml").read_text())
HELDOUT = yaml.safe_load((GOLDEN / "heldout.yaml").read_text())
ALL = QUESTIONS + HELDOUT

PIN_DATE = date(2026, 7, 29)
EXACT = {1, 2, 3, 4}
INTERPRETIVE = {5, 6}
KNOWN_METHODS = {
    "merged_prs",
    "stale_prs",
    "commits_by_author",
    "open_issues",
    "release_diff",
    "entity",
}


def _as_date(value) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def test_the_set_is_the_size_the_prd_specifies():
    assert len(QUESTIONS) == 50
    assert len(HELDOUT) == 15


def test_roughly_thirty_are_exactly_checkable():
    """PRD §7.1: ~30 with exact ground truth, ~20 scored against source spans.
    The split is what lets §6 gate on set-F1 rather than on a judge."""
    exact = sum(1 for q in QUESTIONS if q["class"] in EXACT)
    assert exact == 30
    assert len(QUESTIONS) - exact == 20


@pytest.mark.parametrize("question", ALL, ids=lambda q: q["id"])
def test_question_schema(question):
    assert question["class"] in EXACT | INTERPRETIVE
    assert question["question"].strip()
    assert question["gold_tools"], "tool-selection accuracy needs an expected tool"

    if question["class"] in EXACT:
        spec = question["gold_query"]
        assert spec["method"] in KNOWN_METHODS
        assert "gold_chunks" not in question, "exact questions generate their answers"
        # Ground truth is generated; hand-writing it here would defeat the point.
        assert "gold_entities" not in question
    else:
        assert question["gold_chunks"], "interpretive questions cite source spans"
        assert question["gold_answer_points"], "coverage needs expected points"
        assert "gold_query" not in question


@pytest.mark.parametrize("question", ALL, ids=lambda q: q["id"])
def test_every_question_is_anchored_at_or_before_the_pin(question):
    """§14.1: a question anchored after the pin cannot be answered correctly by
    any system, and scoring it produces what looks like a retrieval regression."""
    assert _as_date(question["as_of"]) <= PIN_DATE


def test_ids_are_unique_across_both_files():
    ids = [q["id"] for q in ALL]
    duplicates = {i for i in ids if ids.count(i) > 1}
    assert not duplicates, duplicates


def test_held_out_questions_reuse_nothing_from_the_golden_set():
    """A held-out set that overlaps the tuning set does not measure overfitting."""
    golden_queries = {
        yaml.safe_dump(q["gold_query"], sort_keys=True)
        for q in QUESTIONS
        if q["class"] in EXACT
    }
    heldout_queries = {
        yaml.safe_dump(q["gold_query"], sort_keys=True)
        for q in HELDOUT
        if q["class"] in EXACT
    }
    assert not golden_queries & heldout_queries

    golden_chunks = {c for q in QUESTIONS for c in q.get("gold_chunks", [])}
    heldout_chunks = {c for q in HELDOUT for c in q.get("gold_chunks", [])}
    assert not golden_chunks & heldout_chunks


def test_no_two_exact_questions_share_a_query():
    """Two questions with identical ground truth double-count whatever they
    test. A 7-day and a 14-day stale threshold returned the same 59 pull
    requests here, because every open one predates both."""
    specs = [
        (q["id"], yaml.safe_dump(q["gold_query"], sort_keys=True))
        for q in QUESTIONS
        if q["class"] in EXACT
    ]
    collisions = [
        (a, b) for (a, sa), (b, sb) in itertools.combinations(specs, 2) if sa == sb
    ]
    assert not collisions, collisions


def test_generated_answers_are_not_committed_into_the_frozen_files():
    """The freeze rule protects hand-authored content. Generated entity sets
    live in their own file so regenerating never rewrites it."""
    for question in ALL:
        assert "gold_entities" not in question


def test_the_freeze_rule_is_stated_in_the_files_themselves():
    """A rule that lives only in the README is a rule people edit files
    without reading."""
    for name in ("questions.yaml", "heldout.yaml"):
        text = (GOLDEN / name).read_text()
        assert "FREEZE RULE" in text or "SEALED UNTIL M5" in text


def test_generated_entities_match_the_questions_that_need_them():
    generated = yaml.safe_load((GOLDEN / "gold_entities.yaml").read_text())
    entities = generated["gold_entities"]
    expected = {q["id"] for q in QUESTIONS if q["class"] in EXACT}
    assert set(entities) == expected
    assert generated["_resolved_sha"].startswith("95f8322ee1dc")


def test_only_deliberately_empty_answers_are_empty():
    """An accidental empty gold set scores as a permanent perfect miss."""
    generated = yaml.safe_load((GOLDEN / "gold_entities.yaml").read_text())
    by_id = {q["id"]: q for q in QUESTIONS}
    for qid, citations in generated["gold_entities"].items():
        if not citations:
            assert by_id[qid].get("expect_empty") is True, f"{qid} is unexpectedly empty"
