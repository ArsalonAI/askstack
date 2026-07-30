"""Citation grammar — TRD §5.1."""

import pytest

from app.ingest import citations


@pytest.mark.parametrize(
    ("raw", "kind", "is_span"),
    [
        ("docs:advanced/events.md#lifespan", "docs", True),
        ("code:fastapi/routing.py:L280-L340", "code", True),
        ("issue:1234#comment-5", "comment", True),
        ("pr:11234", "pr", False),
        ("commit:a3f1c9d", "commit", False),
        ("issue:1234", "issue", False),
        ("release:0.110.0", "release", False),
    ],
)
def test_every_form_in_the_grammar_parses(raw, kind, is_span):
    parsed = citations.parse(raw)
    assert parsed.kind == kind
    assert parsed.is_span is is_span
    assert parsed.raw == raw


def test_the_fragment_decides_issue_entity_versus_span():
    """§5.1: the bare form is the entity ("this issue is still open"), the
    fragment form is a span ("here is what was said about it"). A parser that
    checks the bare pattern first sends every comment to the facts layer."""
    entity = citations.parse("issue:1234")
    span = citations.parse("issue:1234#comment-5")
    assert entity.kind == "issue" and not entity.is_span
    assert span.kind == "comment" and span.is_span
    assert span.number == 1234 and span.index == 5


def test_code_citation_carries_its_span():
    parsed = citations.parse("code:fastapi/routing.py:L280-L340")
    assert (parsed.path, parsed.start, parsed.end) == ("fastapi/routing.py", 280, 340)


def test_docs_path_may_contain_colons_and_slashes():
    parsed = citations.parse("docs:advanced/settings.md#env-vars")
    assert parsed.path == "advanced/settings.md"
    assert parsed.slug == "env-vars"


@pytest.mark.parametrize(
    "raw", ["", "nonsense", "pr:", "pr:abc", "code:x.py:L1", "commit:xyz"]
)
def test_malformed_citations_raise(raw):
    with pytest.raises(citations.InvalidCitation):
        citations.parse(raw)


@pytest.mark.parametrize(
    ("heading", "slug"),
    [
        ("Lifespan", "lifespan"),
        ("Sub-dependencies", "sub-dependencies"),
        ("What's `Depends()` for?", "what-s-depends-for"),
        ("Créer une application", "creer-une-application"),
        ("  Trailing space  ", "trailing-space"),
    ],
)
def test_slugify(heading, slug):
    assert citations.slugify(heading) == slug


def test_builders_round_trip_through_the_parser():
    """Every ID we emit must parse back — the golden set stores these verbatim
    and the eval's citation-resolution metric depends on it."""
    built = [
        citations.docs("advanced/events.md", "Lifespan"),
        citations.code("fastapi/routing.py", 280, 340),
        citations.comment(1234, 5),
        citations.issue(1234),
        citations.pr(11234),
        citations.commit("a3f1c9d20e84bb1a"),
        citations.release("0.110.0"),
    ]
    for raw in built:
        assert citations.parse(raw).raw == raw


def test_commit_citations_are_shortened_consistently():
    assert citations.commit("a3f1c9d20e84bb1a") == "commit:a3f1c9d"
    assert citations.commit("a3f1c9d") == "commit:a3f1c9d"


def test_split_code_parts_stay_inside_the_grammar():
    """A token-split line can leave two chunks on the same line range, so the
    allocator appends a part suffix. A bare `-2` would produce
    `code:x.py:L2-L2-2`, which does not parse."""
    parsed = citations.parse("code:big.py:L2-L2#part-2")
    assert parsed.kind == "code"
    assert (parsed.path, parsed.start, parsed.end, parsed.index) == ("big.py", 2, 2, 2)
    assert parsed.is_span
    assert citations.parse("code:big.py:L2-L9").index == 0
