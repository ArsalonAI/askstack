"""Corpus chunking — TRD §5.2.

Three strategies, one per source, because the thing that makes a chunk
retrievable differs by kind: a docs section needs its heading context, a
function needs to stay whole, and an issue thread needs its resolution.

Token counts use the embedding model's own tokenizer rather than a word-count
approximation — the 800-token split threshold only means anything if it is
measured in the units the model actually sees.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache

from app.ingest import citations

log = logging.getLogger(__name__)

# bge-small-en-v1.5 truncates at 512 tokens. A chunk larger than that is
# embedded from its opening and the rest is silently discarded -- at 800
# tokens, a quarter of the index was partly invisible to the dense arm. So the
# budget is the model's window minus room for the breadcrumb/header prefix
# each chunk carries.
EMBED_WINDOW = 512
# Absorbs joiner tokens and the drift a tokenizer decode/encode round trip can
# introduce. A fixed prefix allowance was wrong: headers vary from a short
# `# path :: name` to a deep breadcrumb, and the long ones pushed chunks to 581
# tokens. Callers subtract their *actual* header cost via `content_budget`.
SAFETY_MARGIN = 16
MAX_TOKENS = EMBED_WINDOW - SAFETY_MARGIN
# Every splitter overlaps, so a fact that straddles a boundary stays
# retrievable from both sides rather than being cut in half.
OVERLAP_TOKENS = 96
MIN_SECTION_TOKENS = 100
MAX_COMMENTS_PER_THREAD = 5

# §5.2: "Bot comments are dropped by author allowlist." Inverted here to a
# denylist of the bots FastAPI actually runs, because an allowlist of humans
# would need updating every time a new contributor comments.
BOT_AUTHORS = frozenset(
    {
        "github-actions[bot]",
        "dependabot[bot]",
        "pre-commit-ci[bot]",
        "codecov[bot]",
        "codecov-commenter",
        "sonarcloud[bot]",
    }
)

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
# MkDocs attr_list explicit anchors: `## Lifespan { #lifespan }`. The braced
# anchor is what the published page actually renders, so it wins over a slug
# derived from the heading text -- otherwise a citation points at an anchor
# that does not exist on the page a manager would open.
ANCHOR_RE = re.compile(r"\{\s*#(?P<anchor>[A-Za-z0-9_-]+)\s*\}\s*$")


@dataclass(frozen=True)
class RawChunk:
    """A chunk before it has an embedding."""

    id: str
    source: str
    path: str
    anchor: str
    content: str
    token_count: int

    @property
    def content_sha(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer

    from app.config import settings

    return AutoTokenizer.from_pretrained(settings.embedding_model)


def count_tokens(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False))


def content_budget(header: str) -> int:
    """Tokens left for content once `header` has been paid for.

    Every chunk repeats a header — a breadcrumb, a `path :: anchor` line — and
    it counts against the same 512-token window. Charging a flat allowance
    instead of the real cost is what left chunks at 581 tokens.
    """
    return max(64, MAX_TOKENS - count_tokens(header))


# ------------------------------------------------------------------- docs


@dataclass
class _Section:
    breadcrumb: list[str]
    heading: str
    lines: list[str]
    anchor: str | None = None  # explicit `{ #anchor }`, when the page sets one

    @property
    def body(self) -> str:
        return "\n".join(self.lines).strip()


def _heading_parts(text: str) -> tuple[str, str | None]:
    """`Lifespan { #lifespan }` -> ("Lifespan", "lifespan")."""
    match = ANCHOR_RE.search(text)
    if not match:
        return text.strip(), None
    return text[: match.start()].strip(), match.group("anchor")


def _split_sections(markdown: str) -> list[_Section]:
    """Markdown -> a flat list of leaf sections, each carrying its breadcrumb.

    Fenced code blocks are skipped when looking for headings: a `# comment` on
    the first line of a Python example would otherwise start a new section.
    """
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []
    current = _Section([], "", [])
    in_fence = False

    for line in markdown.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
        match = None if in_fence else HEADING_RE.match(line)
        if match:
            if current.body or current.heading:
                sections.append(current)
            level = len(match.group(1))
            heading, anchor = _heading_parts(match.group(2))
            while stack and stack[-1][0] >= level:
                stack.pop()
            breadcrumb = [h for _, h in stack] + [heading]
            stack.append((level, heading))
            current = _Section(breadcrumb, heading, [], anchor)
        else:
            current.lines.append(line)

    if current.body or current.heading:
        sections.append(current)
    return [s for s in sections if s.body]


def _pack(
    units: list[str], joiner: str, max_tokens: int, overlap: int
) -> list[list[int]]:
    """Greedily pack `units` into parts, carrying an overlapping tail forward.

    Returns index lists rather than text so callers that need to know *which*
    units landed in each part — code, which reports real line numbers — can.

    The overlap is what makes a boundary non-destructive: a sentence, a
    statement, or an argument that straddles a split stays retrievable from
    both sides instead of being cut in half.
    """
    parts: list[list[int]] = []
    buffer: list[int] = []
    buffer_tokens = 0
    joiner_tokens = count_tokens(joiner) if joiner.strip() else 1

    for index, unit in enumerate(units):
        tokens = count_tokens(unit)
        if buffer and buffer_tokens + tokens + joiner_tokens > max_tokens:
            parts.append(buffer)
            tail: list[int] = []
            tail_tokens = 0
            for previous in reversed(buffer):
                previous_tokens = count_tokens(units[previous])
                if tail_tokens + previous_tokens > overlap:
                    break
                tail.insert(0, previous)
                tail_tokens += previous_tokens
            # The tail is only worth carrying if the incoming unit still fits
            # behind it. Skipping this re-check let a 465-token unit land on
            # top of a 95-token tail and blow the window by 200 tokens.
            if tail_tokens + tokens + joiner_tokens > max_tokens:
                tail, tail_tokens = [], 0
            buffer, buffer_tokens = tail, tail_tokens
        # A single unit larger than the whole budget still has to go somewhere;
        # it becomes its own part and is truncated by the model rather than
        # dropped. Splitting mid-token would corrupt code and prose alike.
        buffer.append(index)
        buffer_tokens += tokens + joiner_tokens

    if buffer:
        parts.append(buffer)
    return parts or [[]]


def _hard_split(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Last resort: split on token boundaries.

    Only reached by text with no internal structure to split on — a minified
    blob, a base64 payload, a single enormous line pasted into an issue. The
    "boundaries land between whole units" guarantee cannot hold when there are
    no units, so the fallback is to at least stay inside the embedder window
    rather than hand it something it will silently truncate.
    """
    tokenizer = _tokenizer()
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return [text]
    stride = max(1, max_tokens - overlap)
    return [
        tokenizer.decode(ids[start : start + max_tokens])
        for start in range(0, len(ids), stride)
    ]


def _split_text(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Split on paragraph boundaries, then lines, then tokens.

    Each fallback applies to the *individual* unit that overflows, not to the
    whole text. Only checking `len(units) == 1` left a single huge paragraph
    intact whenever it had smaller siblings, which is exactly what real issue
    bodies look like — 1,391 chunks were still over the window.
    """
    units = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not units:
        return [text]

    expanded: list[str] = []
    for unit in units:
        if count_tokens(unit) <= max_tokens:
            expanded.append(unit)
            continue
        lines = [line for line in unit.splitlines() if line.strip()]
        for line in lines if len(lines) > 1 else [unit]:
            if count_tokens(line) <= max_tokens:
                expanded.append(line)
            else:
                expanded.extend(_hard_split(line, max_tokens, overlap))

    return [
        "\n\n".join(expanded[i] for i in part)
        for part in _pack(expanded, "\n\n", max_tokens, overlap)
    ]


def _slug_suffix(base: str, n: int) -> str:
    return f"{base}-{n}"


def _part_suffix(base: str, n: int) -> str:
    """Code citations end in a line range, so a bare `-2` would not parse.
    The fragment form keeps the ID inside the §5.1 grammar."""
    return f"{base}#part-{n}"


def _unique(base: str, used: set[str], suffix=_slug_suffix) -> str:
    """Allocate a collision-free citation ID within one document.

    `docs:path#slug` is not unique on its own: FastAPI's release notes repeat
    `### Docs` and `### Fixes` under every version, and an oversized section
    also emits several parts. Both cases funnel through here, so uniqueness is
    guaranteed by construction rather than by two schemes that must not
    overlap. Document order is fixed at a pinned ref, so the IDs are stable
    across re-ingest — which is the property §5.1 actually requires.
    """
    if base not in used:
        used.add(base)
        return base
    n = 2
    while suffix(base, n) in used:
        n += 1
    allocated = suffix(base, n)
    used.add(allocated)
    return allocated


def chunk_docs(path: str, markdown: str) -> list[RawChunk]:
    sections = _split_sections(markdown)
    chunks: list[RawChunk] = []
    pending: _Section | None = None
    used: set[str] = set()

    for section in sections:
        # §5.2: sections under 100 tokens merge into the next sibling. A
        # two-line section retrieves as noise on its own.
        #
        # The *next* sibling's heading and breadcrumb win, because that is what
        # "merge into" means: the short section's text flows into the following
        # one, not the other way round. Keeping the short section's identity
        # would file a whole subtree under a one-line intro heading.
        if pending is not None:
            section = _Section(
                section.breadcrumb,
                section.heading,
                [*pending.lines, "", *section.lines],
            )
            pending = None
        if count_tokens(section.body) < MIN_SECTION_TOKENS:
            pending = section
            continue
        chunks.extend(_emit_docs_section(path, section, used))

    if pending is not None:  # trailing short section: keep it rather than drop it
        chunks.extend(_emit_docs_section(path, pending, used))
    return chunks


def _emit_docs_section(path: str, section: _Section, used: set[str]) -> Iterator[RawChunk]:
    breadcrumb = " > ".join(section.breadcrumb) or path
    # An explicit `{ #anchor }` is the anchor the published page renders, so it
    # is used verbatim; only headings without one get a derived slug.
    base_id = (
        f"docs:{path}#{section.anchor}"
        if section.anchor
        else citations.docs(path, section.heading or path)
    )
    header = f"# {breadcrumb}\n\n"
    parts = _split_text(section.body, content_budget(header), OVERLAP_TOKENS)

    for part in parts:
        # The breadcrumb is prefixed so an isolated chunk retains its context.
        content = f"{header}{part}"
        yield RawChunk(
            # The first part of the first section with this heading keeps the
            # bare `docs:path#slug` form, which is what golden-set entries
            # reference; everything after it gets a numeric suffix.
            id=_unique(base_id, used),
            source="docs",
            path=path,
            anchor=breadcrumb,
            content=content,
            token_count=count_tokens(content),
        )


# ------------------------------------------------------------------- code


def chunk_code(path: str, source: str) -> list[RawChunk]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        log.debug("%s failed to parse; falling back to windows", path)
        return _window_fallback(path, source)

    lines = source.splitlines()
    chunks: list[RawChunk] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            chunks.extend(_chunk_class(path, node, lines))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Assign | ast.AnnAssign):
            chunks.extend(_chunk_node(path, node, lines, qualifier=None))

    chunks = [c for c in chunks if c.content.strip()] or _window_fallback(path, source)

    # Belt and braces. Line spans should be unique now that a class chunk stops
    # at its first method, but a citation collision silently drops content, and
    # the failure is invisible until an eval question cannot be answered.
    used: set[str] = set()
    return [
        c
        if (allocated := _unique(c.id, used, _part_suffix)) == c.id
        else replace(c, id=allocated)
        for c in chunks
    ]


def _first_method_line(node: ast.ClassDef) -> int | None:
    """Line where the class's first method begins, decorators included."""
    starts = [
        min([child.lineno, *(d.lineno for d in child.decorator_list)])
        for child in node.body
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    return min(starts) if starts else None


def _chunk_class(path: str, node: ast.ClassDef, lines: list[str]) -> Iterator[RawChunk]:
    """The class *header* becomes one chunk; each method becomes its own.

    The header is the signature, docstring, and class-level assignments — not
    the method bodies. Emitting the whole class alongside its methods indexed
    every method twice: `fastapi/applications.py` is a single 4,700-line class,
    so the class chunks duplicated the entire file, inflating the index and
    filling top-k with near-identical hits. It also collided outright, because
    a split part of the class body can cover exactly the same line range as a
    method, producing two different chunks with one citation.

    Both are still reachable: a query for the class name finds the header with
    its docstring, and a query for a method finds the method.
    """
    end = _first_method_line(node)
    header_end = (end - 1) if end else (getattr(node, "end_lineno", node.lineno))
    if header_end >= node.lineno:
        yield from _chunk_span(path, node.lineno, header_end, lines, anchor=node.name)
    for child in node.body:
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            yield from _chunk_node(path, child, lines, qualifier=node.name)


def _chunk_node(
    path: str, node: ast.AST, lines: list[str], *, qualifier: str | None
) -> list[RawChunk]:
    """One node -> one chunk, or several overlapping ones if it is large.

    §5.2 originally emitted oversized functions whole, on the grounds that
    splitting a body destroys what makes it retrievable. That reasoning holds
    against a naive split, but a whole 4,000-token function is worse: the
    embedder sees only its first 512 tokens, so most of the body is invisible
    to the dense arm anyway. Splitting *with overlap*, and repeating the
    `path :: anchor` header on every part, keeps each piece both embeddable and
    identifiable.
    """
    start = node.lineno
    end = getattr(node, "end_lineno", start) or start
    name = getattr(node, "name", None) or f"L{start}"
    anchor = f"{qualifier}.{name}" if qualifier else name
    return list(_chunk_span(path, start, end, lines, anchor=anchor))


def _chunk_span(
    path: str, start: int, end: int, lines: list[str], *, anchor: str
) -> Iterator[RawChunk]:
    """Emit one chunk per overlapping part of `lines[start-1:end]`."""
    # The enclosing class name goes in the text, not just the anchor, so a
    # method body carries the context that makes it findable. Repeated on
    # every part, so a split function stays identifiable.
    header = f"# {path} :: {anchor}"
    budget = content_budget(header + "\n")
    body_lines, line_numbers = _expand_lines(lines[start - 1 : end], start, budget)

    for part in _pack(body_lines, "\n", budget, OVERLAP_TOKENS):
        if not part:
            continue
        content = header + "\n" + "\n".join(body_lines[i] for i in part)
        if not content.strip():
            continue
        # Real line numbers per part, so a split function still yields honest
        # citations rather than a suffix on the whole span. `line_numbers` maps
        # back to source lines, which no longer align 1:1 once an oversized
        # line has been token-split into several pieces.
        yield RawChunk(
            id=citations.code(path, line_numbers[part[0]], line_numbers[part[-1]]),
            source="code",
            path=path,
            anchor=anchor,
            content=content,
            token_count=count_tokens(content),
        )


def _expand_lines(
    lines: list[str], first_lineno: int, budget: int
) -> tuple[list[str], list[int]]:
    """Token-split any line that exceeds the window on its own.

    A single line can blow the budget — a long literal, a generated table, a
    minified blob checked into the tree. Returns the expanded pieces alongside
    the source line number each one came from, so citations stay truthful.
    """
    pieces: list[str] = []
    numbers: list[int] = []
    for offset, line in enumerate(lines):
        lineno = first_lineno + offset
        if count_tokens(line) <= budget:
            pieces.append(line)
            numbers.append(lineno)
        else:
            for piece in _hard_split(line, budget, OVERLAP_TOKENS):
                pieces.append(piece)
                numbers.append(lineno)
    return pieces, numbers


def _window_fallback(path: str, source: str) -> list[RawChunk]:
    """Overlapping line windows, for files `ast.parse` rejects."""
    pieces, numbers = _expand_lines(source.splitlines(), 1, MAX_TOKENS)
    chunks: list[RawChunk] = []
    for part in _pack(pieces, "\n", MAX_TOKENS, OVERLAP_TOKENS):
        if not part:
            continue
        start, end = numbers[part[0]], numbers[part[-1]]
        content = "\n".join(pieces[i] for i in part)
        if not content.strip():
            continue
        chunks.append(
            RawChunk(
                id=citations.code(path, start, end),
                source="code",
                path=path,
                anchor=f"L{start}-L{end}",
                content=content,
                token_count=count_tokens(content),
            )
        )
    return chunks


# ------------------------------------------------------------------ issues


def chunk_issue(
    number: int,
    title: str,
    body: str | None,
    labels: Sequence[str],
    comments: Sequence[dict],
) -> list[RawChunk]:
    """One chunk for the issue body, one per run of up to five comments.

    Callers pass closed issues only (§5.2): an open issue describes a problem
    rather than a resolution, and pollutes an answer corpus.
    """
    path = f"issues/{number}"
    label_line = f"labels: {', '.join(sorted(labels))}" if labels else ""
    header = "\n".join(filter(None, [f"# {title}", label_line]))
    chunks: list[RawChunk] = []

    # The body is a *span* citation (`issue:N#body`), never the bare
    # `issue:N` -- that form is the entity, and one string meaning both "this
    # issue is open" and "this paragraph of its description" is the ambiguity
    # §5.1 exists to prevent.
    body_parts = _split_text(
        (body or "").strip(), content_budget(header + "\n"), OVERLAP_TOKENS
    )
    for index, part in enumerate(body_parts):
        content = f"{header}\n{part}".strip()
        if not content:
            continue
        chunks.append(
            RawChunk(
                id=citations.body(number, index),
                source="issue",
                path=path,
                anchor=title if index == 0 else f"{title} (body {index + 1})",
                content=content,
                token_count=count_tokens(content),
            )
        )

    human = [
        c
        for c in comments
        if ((c.get("user") or {}).get("login") or "") not in BOT_AUTHORS
        and (c.get("body") or "").strip()
    ]

    # Threads are bounded by comment count *and* token budget. Five comments
    # is usually a coherent exchange, but five long ones overflow the embedder
    # window, and a thread nobody can retrieve is not evidence of anything.
    thread_header = f"# {title}\n\n"
    thread_budget = content_budget(thread_header)
    thread_index = 0
    for start in range(0, len(human), MAX_COMMENTS_PER_THREAD):
        run = human[start : start + MAX_COMMENTS_PER_THREAD]
        # A single comment can exceed the window on its own -- a pasted
        # traceback, a full config file, a 28,000-token log dump. Splitting the
        # *run* was not enough; the individual comment has to be split too, and
        # each piece keeps its author so attribution survives.
        rendered: list[str] = []
        origins: list[int] = []
        for offset, comment_payload in enumerate(run):
            author = (comment_payload.get("user") or {}).get("login") or "unknown"
            text = comment_payload["body"].strip()
            for piece in _split_text(text, thread_budget - 8, OVERLAP_TOKENS):
                rendered.append(f"@{author}: {piece}")
                origins.append(offset)

        for part in _pack(rendered, "\n\n", thread_budget, OVERLAP_TOKENS):
            if not part:
                continue
            content = thread_header + "\n\n".join(rendered[i] for i in part)
            first = start + origins[part[0]] + 1
            last = start + origins[part[-1]] + 1
            chunks.append(
                RawChunk(
                    id=citations.comment(number, thread_index),
                    source="issue",
                    path=path,
                    anchor=f"{title} (comments {first}-{last})",
                    content=content,
                    token_count=count_tokens(content),
                )
            )
            thread_index += 1

    return chunks
