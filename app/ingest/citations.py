"""Citation ID grammar — TRD §5.1.

Citation IDs are the `chunks.id` primary key, so they must be stable across
re-ingest: every re-run that changed them would orphan the golden set.

Two kinds. **Span citations** point at retrieved text and are `chunks.id`.
**Entity citations** point at a repository object in the facts layer.

    citation     := span_cite | entity_cite

    span_cite    := docs_cite | code_cite | comment_cite | body_cite
    docs_cite    := "docs:" path "#" slug
    code_cite    := "code:" path ":L" start "-L" end
    comment_cite := "issue:" number "#comment-" n
    body_cite    := "issue:" number "#body" ["-" n]

    entity_cite  := pr_cite | commit_cite | issue_ref | release_cite
    pr_cite      := "pr:" number
    commit_cite  := "commit:" short_sha
    issue_ref    := "issue:" number
    release_cite := "release:" tag
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

CitationKind = Literal[
    "docs", "code", "comment", "body", "pr", "commit", "issue", "release"
]

DOCS_RE = re.compile(r"^docs:(?P<path>[^#]+)#(?P<slug>.+)$")
CODE_RE = re.compile(
    r"^code:(?P<path>.+):L(?P<start>\d+)-L(?P<end>\d+)(#part-(?P<part>\d+))?$"
)
COMMENT_RE = re.compile(r"^issue:(?P<number>\d+)#comment-(?P<index>\d+)$")
BODY_RE = re.compile(r"^issue:(?P<number>\d+)#body(-(?P<index>\d+))?$")
PR_RE = re.compile(r"^pr:(?P<number>\d+)$")
COMMIT_RE = re.compile(r"^commit:(?P<sha>[0-9a-f]{7,40})$")
ISSUE_RE = re.compile(r"^issue:(?P<number>\d+)$")
RELEASE_RE = re.compile(r"^release:(?P<tag>.+)$")

SHORT_SHA = 7

# Span kinds resolve against `chunks`; entity kinds resolve against the facts
# layer. Dispatching on the wrong one is the bug §5.1 warns about.
SPAN_KINDS: frozenset[CitationKind] = frozenset({"docs", "code", "comment", "body"})


class InvalidCitation(ValueError):
    pass


@dataclass(frozen=True)
class Citation:
    kind: CitationKind
    raw: str
    path: str | None = None
    slug: str | None = None
    start: int | None = None
    end: int | None = None
    number: int | None = None
    index: int | None = None
    sha: str | None = None
    tag: str | None = None

    @property
    def is_span(self) -> bool:
        return self.kind in SPAN_KINDS


def parse(raw: str) -> Citation:
    """Parse a citation ID.

    Order matters. `issue:1234#comment-5` must be tested before `issue:1234`,
    because the bare form is the *entity* ("this issue is still open") and the
    fragment form is a *span* ("here is what was said about it"). §5.1 calls
    this out explicitly: a parser must check for the fragment before
    dispatching to the facts layer.
    """
    if m := COMMENT_RE.match(raw):
        return Citation(
            "comment", raw, number=int(m["number"]), index=int(m["index"])
        )
    if m := BODY_RE.match(raw):
        return Citation(
            "body", raw, number=int(m["number"]), index=int(m["index"] or 0)
        )
    if m := ISSUE_RE.match(raw):
        return Citation("issue", raw, number=int(m["number"]))
    if m := DOCS_RE.match(raw):
        return Citation("docs", raw, path=m["path"], slug=m["slug"])
    if m := CODE_RE.match(raw):
        return Citation(
            "code",
            raw,
            path=m["path"],
            start=int(m["start"]),
            end=int(m["end"]),
            index=int(m["part"] or 0),
        )
    if m := PR_RE.match(raw):
        return Citation("pr", raw, number=int(m["number"]))
    if m := COMMIT_RE.match(raw):
        return Citation("commit", raw, sha=m["sha"])
    if m := RELEASE_RE.match(raw):
        return Citation("release", raw, tag=m["tag"])
    raise InvalidCitation(f"not a citation: {raw!r}")


def slugify(heading: str) -> str:
    """Heading text -> the `#slug` half of a docs citation.

    Matches the usual Markdown anchor convention: lowercase, non-alphanumerics
    collapsed to hyphens. Stability matters more than fidelity here — a slug
    that changes shape between releases orphans every golden-set citation that
    used it.
    """
    normalized = unicodedata.normalize("NFKD", heading)
    ascii_only = normalized.encode("ascii", "ignore").decode()
    return re.sub(r"-{2,}", "-", re.sub(r"[^a-z0-9]+", "-", ascii_only.lower())).strip("-")


def docs(path: str, heading: str) -> str:
    return f"docs:{path}#{slugify(heading)}"


def code(path: str, start: int, end: int) -> str:
    return f"code:{path}:L{start}-L{end}"


def comment(number: int, index: int) -> str:
    return f"issue:{number}#comment-{index}"


def body(number: int, index: int = 0) -> str:
    """The issue body as a *span*.

    Deliberately not the bare `issue:1234`: that form is the entity citation,
    and having one string mean both "this issue is still open" and "this
    paragraph of its description" is exactly the ambiguity this grammar exists
    to avoid.
    """
    return f"issue:{number}#body" if index == 0 else f"issue:{number}#body-{index}"


def issue(number: int) -> str:
    return f"issue:{number}"


def pr(number: int) -> str:
    return f"pr:{number}"


def commit(sha: str) -> str:
    return f"commit:{sha[:SHORT_SHA]}"


def release(tag: str) -> str:
    return f"release:{tag}"
