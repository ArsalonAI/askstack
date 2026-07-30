"""Chunking strategies — TRD §5.2."""

import pytest

from app.ingest import chunking
from app.ingest.chunking import chunk_code, chunk_docs, chunk_issue

# ------------------------------------------------------------------- docs

def _filler(topic: str) -> str:
    """Prose comfortably over the hundred-token floor, so the section stands
    on its own rather than exercising the merge rule."""
    return " ".join([f"This paragraph describes {topic} in some detail."] * 30)


MARKDOWN = f"""\
# Advanced

{_filler("the advanced guide")}

## Events

{_filler("application lifecycle events")}

### Lifespan

{_filler("the lifespan handler that runs on startup and shutdown")}
"""


def test_leaf_sections_become_chunks_with_breadcrumbs():
    chunks = chunk_docs("advanced/events.md", MARKDOWN)
    anchors = [c.anchor for c in chunks]
    assert "Advanced > Events > Lifespan" in anchors
    lifespan = next(c for c in chunks if c.anchor.endswith("Lifespan"))
    # §5.2: the breadcrumb is prefixed so an isolated chunk retains context.
    assert lifespan.content.startswith("# Advanced > Events > Lifespan")
    assert lifespan.id == "docs:advanced/events.md#lifespan"


def test_short_sections_merge_into_the_next_sibling():
    """§5.2 says the short section merges *into the next sibling*, so the
    following section's heading wins. Keeping the short one's identity would
    file a whole subtree under a one-line intro heading."""
    markdown = f"# Tiny\n\nOne line.\n\n## Real\n\n{_filler('the real section')}\n"
    chunks = chunk_docs("x.md", markdown)
    assert len(chunks) == 1
    assert "One line." in chunks[0].content
    assert chunks[0].anchor == "Tiny > Real"
    assert chunks[0].id == "docs:x.md#real"


def test_a_trailing_short_section_is_kept_not_dropped():
    chunks = chunk_docs("x.md", MARKDOWN + "\n## Footnote\n\nBrief.\n")
    assert any("Brief." in c.content for c in chunks)


def test_headings_inside_code_fences_do_not_start_sections():
    """A `# comment` on the first line of a Python example would otherwise
    split the section it is documenting."""
    markdown = MARKDOWN + "\n```python\n# Not a heading\nx = 1\n```\n"
    anchors = [c.anchor for c in chunk_docs("x.md", markdown)]
    assert not any("Not a heading" in a for a in anchors)


def test_mkdocs_explicit_anchors_win_over_derived_slugs():
    """FastAPI writes `## Lifespan { #lifespan }`. The braced anchor is what
    the published page renders, so a citation must use it — otherwise the ID
    points at an anchor that does not exist on the page a manager would open.
    Deriving a slug from the raw heading also doubled it: `#lifespan-lifespan`."""
    markdown = f"# Lifespan Events {{ #lifespan-events }}\n\n{_filler('events')}\n"
    (chunk,) = chunk_docs("docs/en/docs/advanced/events.md", markdown)
    assert chunk.id == "docs:docs/en/docs/advanced/events.md#lifespan-events"
    assert chunk.anchor == "Lifespan Events"
    assert "{" not in chunk.anchor


def test_headings_without_an_explicit_anchor_still_get_a_slug():
    markdown = f"## Plain Heading\n\n{_filler('things')}\n"
    (chunk,) = chunk_docs("x.md", markdown)
    assert chunk.id == "docs:x.md#plain-heading"


def test_repeated_headings_get_distinct_ids():
    """FastAPI's release notes repeat `### Docs` under every version, so
    `docs:path#slug` is not unique on its own."""
    markdown = "".join(
        f"## {version}\n\n### Docs\n\n{_filler('changes')}\n\n"
        for version in ("0.110.0", "0.109.0", "0.108.0")
    )
    ids = [c.id for c in chunk_docs("release-notes.md", markdown)]
    assert len(ids) == len(set(ids)), ids
    assert ids[0] == "docs:release-notes.md#docs"
    assert ids[1] == "docs:release-notes.md#docs-2"


def test_oversized_sections_split_with_distinct_ids():
    paragraph = " ".join(["word"] * 300)
    markdown = "# Big\n\n" + "\n\n".join([paragraph] * 6)
    chunks = chunk_docs("big.md", markdown)
    assert len(chunks) > 1
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(c.token_count <= chunking.MAX_TOKENS * 1.5 for c in chunks)


def test_an_unsplit_section_keeps_the_bare_citation():
    """Golden-set entries reference `docs:path#slug`; a suffix would orphan them."""
    (chunk,) = [c for c in chunk_docs("x.md", MARKDOWN) if c.anchor == "Advanced"]
    assert chunk.id == "docs:x.md#advanced"


# ------------------------------------------------------------------- code

SOURCE = '''\
"""Module docstring."""

DEFAULT_LIMIT = 10


def get_current_user(token: str) -> str:
    """Resolve a user."""
    return token


class APIRouter:
    """Routes requests."""

    def add_api_route(self, path: str) -> None:
        """Register a route."""
        self.routes.append(path)
'''


def test_top_level_functions_and_classes_each_become_a_chunk():
    anchors = {c.anchor for c in chunk_code("fastapi/routing.py", SOURCE)}
    assert {"get_current_user", "APIRouter", "APIRouter.add_api_route"} <= anchors


def test_methods_carry_their_enclosing_class():
    chunks = chunk_code("fastapi/routing.py", SOURCE)
    method = next(c for c in chunks if c.anchor == "APIRouter.add_api_route")
    # The class name is in the text, not just the anchor: a method body needs
    # the context that makes it findable.
    assert "APIRouter.add_api_route" in method.content
    assert "Register a route." in method.content


def test_code_chunk_ids_are_line_spans():
    chunk = next(c for c in chunk_code("a.py", SOURCE) if c.anchor == "get_current_user")
    assert chunk.id.startswith("code:a.py:L")
    from app.ingest.citations import parse

    parsed = parse(chunk.id)
    assert parsed.start < parsed.end


def test_unparseable_files_fall_back_to_windows():
    chunks = chunk_code("broken.py", "def oops(:\n  this is not python\n" * 50)
    assert chunks
    assert all(c.source == "code" for c in chunks)
    assert all(c.anchor.startswith("L") for c in chunks)


def test_module_level_assignments_are_indexed():
    anchors = {c.anchor for c in chunk_code("a.py", SOURCE)}
    assert any(a.startswith("L") for a in anchors), "DEFAULT_LIMIT should be chunked"


# ----------------------------------------------------------------- issues


def _comment(login: str, body: str) -> dict:
    return {"user": {"login": login}, "body": body}


def test_issue_body_and_comment_threads():
    comments = [_comment("someone", f"comment {i}") for i in range(12)]
    chunks = chunk_issue(1234, "Sync client", "Should we drop it?", ["question"], comments)

    # The body is a span citation, never the bare `issue:1234` -- that form is
    # the entity, and one string cannot mean both.
    assert chunks[0].id == "issue:1234#body"
    assert "labels: question" in chunks[0].content
    # 12 comments -> threads of at most 5 -> 3 threads.
    assert [c.id for c in chunks[1:]] == [
        "issue:1234#comment-0",
        "issue:1234#comment-1",
        "issue:1234#comment-2",
    ]


def test_bot_comments_are_dropped():
    comments = [
        _comment("github-actions[bot]", "beep boop"),
        _comment("dependabot[bot]", "bumping"),
        _comment("tiangolo", "we decided to drop it"),
    ]
    chunks = chunk_issue(1, "T", "b", [], comments)
    thread = chunks[1]
    assert "we decided to drop it" in thread.content
    assert "beep boop" not in thread.content


def test_empty_comments_do_not_create_chunks():
    chunks = chunk_issue(1, "T", "b", [], [_comment("a", "   "), _comment("b", "")])
    assert len(chunks) == 1  # the body chunk only


def test_comment_authors_are_attributed():
    """Decision archaeology cites who argued what; an unattributed thread
    cannot answer "why did we drop the sync client"."""
    chunks = chunk_issue(1, "T", "b", [], [_comment("tiangolo", "because async")])
    assert "@tiangolo: because async" in chunks[1].content


@pytest.mark.parametrize("labels", [[], ["bug"], ["bug", "confirmed"]])
def test_labels_render_deterministically(labels):
    a = chunk_issue(1, "T", "b", labels, [])[0].content
    b = chunk_issue(1, "T", "b", list(reversed(labels)), [])[0].content
    assert a == b


def test_content_sha_is_stable():
    a = chunk_issue(1, "T", "b", ["x"], [])[0]
    b = chunk_issue(1, "T", "b", ["x"], [])[0]
    assert a.content_sha == b.content_sha
    assert a.content_sha != chunk_issue(1, "T", "different", ["x"], [])[0].content_sha


# ------------------------------------------------- embedder window (§5.2)


def test_no_chunk_exceeds_the_embedder_window():
    """bge-small-en truncates at 512 tokens. A chunk above that is embedded
    from its opening and the rest is silently discarded — at the old 800-token
    target, a quarter of the index was partly invisible to the dense arm."""
    long_doc = "# Big { #big }\n\n" + "\n\n".join([_filler("things")] * 20)
    long_code = "def huge():\n" + "\n".join(f"    x{i} = {i} + 1" for i in range(2000))
    long_issue_body = "\n\n".join([_filler("a problem")] * 20)
    long_comments = [
        {"user": {"login": "a"}, "body": _filler("an argument")} for _ in range(12)
    ]

    produced = [
        *chunk_docs("big.md", long_doc),
        *chunk_code("huge.py", long_code),
        *chunk_issue(1, "T", long_issue_body, [], long_comments),
        *chunk_code("broken.py", "def oops(:\n" + "  bad syntax here\n" * 2000),
    ]
    assert produced
    over = [(c.id, c.token_count) for c in produced if c.token_count > chunking.EMBED_WINDOW]
    assert not over, f"chunks exceed the 512-token window: {over[:5]}"


def test_splits_carry_an_overlapping_tail():
    """A fact straddling a boundary stays retrievable from both sides. Docs
    already overlapped; code and issues did not until the 512-token change."""
    paragraphs = [f"Paragraph number {i} discusses one specific thing." for i in range(200)]
    parts = chunking._split_text(
        "\n\n".join(paragraphs), chunking.MAX_TOKENS, chunking.OVERLAP_TOKENS
    )
    assert len(parts) > 1
    shared = [p for p in paragraphs if sum(1 for part in parts if p in part) > 1]
    assert shared, "no paragraph appears in two consecutive parts"


def test_boundaries_never_fall_mid_unit():
    """The stronger guarantee: even when a unit is too large to carry forward
    as overlap, a split lands *between* whole paragraphs, so no sentence is
    ever cut in half."""
    paragraphs = [_filler(f"topic {i}") for i in range(8)]
    parts = chunking._split_text(
        "\n\n".join(paragraphs), chunking.MAX_TOKENS, chunking.OVERLAP_TOKENS
    )
    assert len(parts) > 1
    for paragraph in paragraphs:
        assert any(paragraph in part for part in parts), "a paragraph was cut apart"


def test_split_code_reports_real_line_numbers():
    """A split function yields honest citations rather than a suffix on the
    whole span, so every part still resolves to the lines it contains."""
    from app.ingest.citations import parse

    source = "def huge():\n" + "\n".join(f"    x{i} = {i}" for i in range(1500))
    chunks = chunk_code("huge.py", source)
    assert len(chunks) > 1
    spans = [(parse(c.id).start, parse(c.id).end) for c in chunks]
    assert all(s < e for s, e in spans)
    assert spans == sorted(spans)
    total = len(source.splitlines())
    assert all(e <= total for _, e in spans)


def test_long_issue_bodies_split_with_body_citations():
    chunks = chunk_issue(9, "T", "\n\n".join([_filler("detail")] * 20), [], [])
    ids = [c.id for c in chunks]
    assert ids[0] == "issue:9#body"
    assert ids[1] == "issue:9#body-1"
    assert len(set(ids)) == len(ids)


def test_class_chunk_stops_at_the_first_method():
    """Emitting the whole class alongside its methods indexed every method
    twice — `fastapi/applications.py` is one 4,700-line class, so the class
    chunks duplicated the entire file and collided with method line spans."""
    source = (
        "class Big:\n"
        '    """Docstring."""\n'
        "    LIMIT = 10\n"
        "\n"
        "    def alpha(self):\n"
        + "".join(f"        a{i} = {i}\n" for i in range(60))
        + "\n    def beta(self):\n"
        + "".join(f"        b{i} = {i}\n" for i in range(60))
    )
    chunks = chunk_code("big.py", source)
    header = [c for c in chunks if c.anchor == "Big"]
    assert len(header) == 1
    assert "Docstring." in header[0].content
    assert "LIMIT = 10" in header[0].content
    # The method bodies belong to the methods, not to the class chunk.
    assert "a0 = 0" not in header[0].content
    assert "b0 = 0" not in header[0].content
    assert {"Big.alpha", "Big.beta"} <= {c.anchor for c in chunks}


def test_code_chunk_ids_are_unique_within_a_file():
    """A citation collision silently drops content, and the failure is
    invisible until an eval question cannot be answered."""
    source = "class C:\n" + "".join(
        f"    def m{i}(self):\n" + "".join(f"        x{j} = {j}\n" for j in range(40))
        for i in range(6)
    )
    ids = [c.id for c in chunk_code("c.py", source)]
    assert len(ids) == len(set(ids)), [i for i in ids if ids.count(i) > 1]


def test_text_with_no_split_points_still_fits_the_window():
    """Real issue bodies contain pasted blobs — a single 28k-token 'paragraph'
    with no blank lines and no newlines. Earlier fixtures always had structure
    to split on, so they never exercised this and 1,391 real chunks stayed
    over the window."""
    # NOT `"x" * 200_000`: wordpiece collapses an over-long word to a single
    # [UNK], so that blob counted as ~1 token and the assertion was vacuous.
    # Real tokens, no paragraph breaks, no line breaks.
    blob = " ".join(f"token{i}" for i in range(20_000))
    assert chunking.count_tokens(blob) > 10_000, "fixture must actually be huge"
    for chunks in (
        chunk_issue(1, "T", blob, [], []),
        chunk_issue(2, "T", "intro\n\n" + blob + "\n\noutro", [], []),
        chunk_issue(3, "T", "b", [], [{"user": {"login": "a"}, "body": blob}]),
        chunk_code("blob.py", f"data = '{blob}'\n"),
        chunk_docs("blob.md", f"# H\n\n{blob}\n"),
    ):
        assert chunks
        over = [(c.id, c.token_count) for c in chunks if c.token_count > chunking.EMBED_WINDOW]
        assert not over, f"over window: {over[:3]}"


def test_token_split_lines_keep_truthful_line_numbers():
    """A token-split line reports the source line it came from, so citations
    stay honest even where the 1:1 line mapping breaks down."""
    from app.ingest.citations import parse

    huge = " ".join(f"v{i}" for i in range(20_000))
    source = "a = 1\n" + f"b = '{huge}'\n" + "c = 3\n"
    chunks = chunk_code("big.py", source)
    total = len(source.splitlines())
    for chunk in chunks:
        parsed = parse(chunk.id)
        assert 1 <= parsed.start <= parsed.end <= total


def test_overlap_tail_is_dropped_when_the_next_unit_would_not_fit():
    """`_pack` checked the size before flushing but not after carrying the
    overlap forward, so an oversized unit landed on top of a 95-token tail and
    blew the window by 200 tokens."""
    small = "short paragraph here"
    big = " ".join(f"word{i}" for i in range(400))
    units = [small, small, small, big]
    parts = chunking._pack(units, "\n\n", chunking.MAX_TOKENS, chunking.OVERLAP_TOKENS)
    for part in parts:
        joined = "\n\n".join(units[i] for i in part)
        assert chunking.count_tokens(joined) <= chunking.MAX_TOKENS or len(part) == 1
