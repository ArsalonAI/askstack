"""Time-expression resolution — TRD §6.4, §17.1.

The expected ranges below are the `gold_query` args of the windowed golden
questions, verbatim. That is the point of the test: the grammar is only useful
if the expression a model would plausibly emit resolves to the same window the
frozen ground truth was generated from. A parser that is self-consistently
wrong passes any test written against itself.
"""

from datetime import UTC, datetime

import pytest

from app.tools.windows import UnresolvableWindow, resolve

AS_OF = datetime(2026, 7, 29, tzinfo=UTC)


def window(expression: str) -> tuple[str, str]:
    since, until = resolve(expression, AS_OF)
    return since.date().isoformat(), until.date().isoformat()


@pytest.mark.parametrize(
    ("expression", "since", "until", "question"),
    [
        ("last 7 days", "2026-07-22", "2026-07-29", "q009"),
        ("last 30 days", "2026-06-29", "2026-07-29", "q010"),
        ("2026-06", "2026-06-01", "2026-07-01", "q011"),
        ("2026-Q1", "2026-01-01", "2026-04-01", "q012"),
        ("2026-H1", "2026-01-01", "2026-07-01", "q016"),
        ("2025", "2025-01-01", "2026-01-01", "q017"),
        ("2026", "2026-01-01", "2026-07-29", "q022"),
        ("last month", "2026-06-29", "2026-07-29", "q021"),
        ("since 2026-01-01", "2026-01-01", "2026-07-29", "q018"),
        ("since 2026-06-01", "2026-06-01", "2026-07-29", "q020"),
        ("2026-05", "2026-05-01", "2026-06-01", "h005"),
        ("2026-Q2", "2026-04-01", "2026-07-01", "h006"),
        ("last 14 days", "2026-07-15", "2026-07-29", "h008"),
        ("since 2026-04-01", "2026-04-01", "2026-07-29", "h010"),
        ("last 3 months", "2026-04-29", "2026-07-29", "h011"),
    ],
)
def test_matches_golden_set_windows(expression, since, until, question):
    assert window(expression) == (since, until), question


def test_months_are_calendar_months_not_thirty_days():
    """h011 is 2026-04-29, not 2026-04-30. The two differ by a day often
    enough to change a merged-PR set at the window edge."""
    assert window("last 3 months")[0] == "2026-04-29"


def test_month_arithmetic_clamps_to_a_short_month():
    assert resolve("last 1 month", datetime(2026, 3, 31, tzinfo=UTC))[0].date().isoformat() == (
        "2026-02-28"
    )


def test_unfinished_period_is_capped_at_as_of():
    """"2026" asked in July 2026 ends today, not in December — a window must
    never claim to cover time the corpus cannot contain."""
    assert window("2026")[1] == "2026-07-29"
    assert window("2026-Q4")[1] == "2026-07-29"


def test_completed_period_is_not_capped():
    assert window("2025")[1] == "2026-01-01"


@pytest.mark.parametrize(
    "expression",
    [
        "whenever",
        "recently",
        "last sprint",
        "sprint 4",
        "2026-Q5",
        "2026-13",
        "last 0 days",
        "2026-05-01..2026-04-01",  # inverted
        "since yesterday",
        "",
    ],
)
def test_unrecognised_expressions_raise(expression):
    """§6.4: an unresolvable expression is a tool error, never a guess. A
    silently-wrong window produces a confidently-wrong status report."""
    with pytest.raises(UnresolvableWindow):
        resolve(expression, AS_OF)


def test_error_names_the_expression_and_the_grammar():
    with pytest.raises(UnresolvableWindow) as exc:
        resolve("last sprint", AS_OF)
    assert "last sprint" in str(exc.value)
    assert "yyyy" in str(exc.value).lower()


def test_release_form_is_deferred_to_the_registry():
    """The pure parser has no facts layer, so it cannot date a tag. It must
    reject rather than silently ignore the anchor."""
    with pytest.raises(UnresolvableWindow):
        resolve("since release 0.138.0", AS_OF)


def test_naive_as_of_is_treated_as_utc():
    assert resolve("last 7 days", datetime(2026, 7, 29))[0].tzinfo is UTC


def test_explicit_range():
    assert window("2026-01-01..2026-04-01") == ("2026-01-01", "2026-04-01")
