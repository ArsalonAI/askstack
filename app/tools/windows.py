"""Time expressions → a concrete `[since, until)` — TRD §6.4, §17.1.

§6.4 requires the window to be resolved *before* the query and echoed back in
`Aggregate.window`, so an answer can state the range it actually used. §17.1
left the parser undecided between rules and a structured-output Claude call.
This is the rules-based arm, and the reason is the failure mode rather than the
cost: a model that quietly resolves "the first quarter" to the wrong three
months produces a confidently-wrong status report, and nothing downstream can
tell. An expression this module does not recognise raises — a tool error the
model can see and correct, never a guess.

The model is not asked to do date arithmetic. It maps English onto the grammar
below ("the first quarter of 2026" → `2026-Q1`), which is a translation it is
good at; the arithmetic happens here against the session's `as_of`.

    last 7 days | last 3 months | last month | last 2 years
    2025 | 2026-06 | 2026-Q1 | 2026-H1
    2026-01-01..2026-04-01
    since 2026-04-01
"""

from __future__ import annotations

import calendar
import re
from datetime import UTC, datetime, timedelta

# `since release <tag>` is resolved by the registry, which has the facts layer
# to look the tag's publication date up with. Recognised here only so the error
# message can say what the caller should have done.
RELEASE_WINDOW = re.compile(r"^\s*since\s+release\s+(\S+)\s*$", re.I)

# The count is optional so "last month" resolves rather than erroring — the
# model emits the bare form often, and rejecting it costs a round trip to say
# something the grammar can express unambiguously.
_LAST = re.compile(r"^\s*(?:last|past)\s+(\d+\s+)?(day|week|month|year)s?\s*$", re.I)
_SINCE = re.compile(r"^\s*since\s+(\d{4}-\d{2}-\d{2})\s*$", re.I)
_RANGE = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})\s*\.\.\s*(\d{4}-\d{2}-\d{2})\s*$")
_YEAR = re.compile(r"^\s*(\d{4})\s*$")
_MONTH = re.compile(r"^\s*(\d{4})-(\d{2})\s*$")
_QUARTER = re.compile(r"^\s*(\d{4})-Q([1-4])\s*$", re.I)
_HALF = re.compile(r"^\s*(\d{4})-H([12])\s*$", re.I)

GRAMMAR = (
    "last [n] days|weeks|months|years · <yyyy> · <yyyy>-<mm> · <yyyy>-Q1..Q4 · "
    "<yyyy>-H1|H2 · <yyyy-mm-dd>..<yyyy-mm-dd> · since <yyyy-mm-dd> · "
    "since release <tag>"
)


class UnresolvableWindow(ValueError):
    """The expression is not in the grammar.

    Deliberately not a fallback to "everything" or "the last 30 days": a window
    nobody chose is indistinguishable in the answer from one the manager asked
    for.
    """

    def __init__(self, expression: str) -> None:
        super().__init__(
            f"cannot resolve the time window {expression!r}. Use one of: {GRAMMAR}"
        )
        self.expression = expression


def _at(year: int, month: int, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def _minus_months(moment: datetime, months: int) -> datetime:
    """Calendar months, not 30-day approximations.

    "the last three months" from 29 July is 29 April, not 30 April. The two
    differ by a day often enough to change a merged-PR set at a window edge.
    """
    total = moment.month - 1 - months
    year = moment.year + total // 12
    month = total % 12 + 1
    day = min(moment.day, calendar.monthrange(year, month)[1])
    return moment.replace(year=year, month=month, day=day)


def resolve(expression: str, as_of: datetime) -> tuple[datetime, datetime]:
    """`[since, until)` against the session's `as_of`.

    `until` is always capped at `as_of`. A calendar period that has not finished
    yet — "2026" asked in July 2026 — ends today rather than in December, so the
    window never claims to cover time the corpus cannot have.
    """
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=UTC)

    if RELEASE_WINDOW.match(expression):
        raise UnresolvableWindow(expression)  # registry resolves this form

    if match := _LAST.match(expression):
        count = int(match.group(1)) if match.group(1) else 1
        unit = match.group(2).lower()
        if count < 1:
            raise UnresolvableWindow(expression)
        if unit == "day":
            since = as_of - timedelta(days=count)
        elif unit == "week":
            since = as_of - timedelta(weeks=count)
        elif unit == "month":
            since = _minus_months(as_of, count)
        else:
            since = _minus_months(as_of, count * 12)
        return since, as_of

    if match := _SINCE.match(expression):
        return datetime.fromisoformat(match.group(1)).replace(tzinfo=UTC), as_of

    if match := _RANGE.match(expression):
        since = datetime.fromisoformat(match.group(1)).replace(tzinfo=UTC)
        until = datetime.fromisoformat(match.group(2)).replace(tzinfo=UTC)
        if until <= since:
            raise UnresolvableWindow(expression)
        return since, min(until, as_of)

    if match := _YEAR.match(expression):
        year = int(match.group(1))
        return _at(year, 1), min(_at(year + 1, 1), as_of)

    if match := _MONTH.match(expression):
        year, month = int(match.group(1)), int(match.group(2))
        if not 1 <= month <= 12:
            raise UnresolvableWindow(expression)
        end = _at(year + 1, 1) if month == 12 else _at(year, month + 1)
        return _at(year, month), min(end, as_of)

    if match := _QUARTER.match(expression):
        year, quarter = int(match.group(1)), int(match.group(2))
        start_month = 3 * (quarter - 1) + 1
        end = _at(year + 1, 1) if quarter == 4 else _at(year, start_month + 3)
        return _at(year, start_month), min(end, as_of)

    if match := _HALF.match(expression):
        year, half = int(match.group(1)), int(match.group(2))
        if half == 1:
            return _at(year, 1), min(_at(year, 7), as_of)
        return _at(year, 7), min(_at(year + 1, 1), as_of)

    raise UnresolvableWindow(expression)
