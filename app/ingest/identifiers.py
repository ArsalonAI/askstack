"""Identifier decomposition for the sparse arm — TRD §6.2, ADR 12.

Postgres's `english` text-search parser handles prose well and code badly. It
takes `get_current_user` as a single token and stems it into something a query
saying "get current user" will never match, and `HTTPException` never matches
"http exception".

So the `tsv` column is built from the chunk content *plus* a decomposed
identifier stream: every `snake_case` and `CamelCase` identifier contributes
its parts as separate tokens alongside the original. `get_current_user` yields
`get current user get_current_user`.

Done in Python at ingest rather than in a custom Postgres parser: a custom
parser needs a C extension and would not survive a move to managed Postgres.
"""

from __future__ import annotations

import re

# A token that could be an identifier: letters/digits/underscores, starting
# with a letter or underscore, and long enough to be worth decomposing.
IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")

# Split CamelCase and PascalCase, keeping acronym runs together:
# "HTTPException" -> ["HTTP", "Exception"], "getURL" -> ["get", "URL"].
CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")

MIN_PART = 2


def split_identifier(identifier: str) -> list[str]:
    """`get_current_user` -> ['get', 'current', 'user'].

    Returns an empty list when the identifier has no internal structure, so a
    plain English word costs nothing.
    """
    parts: list[str] = []
    for chunk in identifier.split("_"):
        if not chunk:
            continue
        parts.extend(CAMEL_RE.findall(chunk))
    parts = [p.lower() for p in parts if len(p) >= MIN_PART]
    if len(parts) < 2:
        return []
    return parts


def decompose(text: str) -> str:
    """The extra token stream to append to a chunk's `tsv` input.

    Deduplicated and sorted: identical content must always produce an identical
    `tsv`, or delta detection (§5.3) would rewrite unchanged rows forever.
    """
    tokens: set[str] = set()
    for match in IDENTIFIER_RE.finditer(text):
        identifier = match.group()
        parts = split_identifier(identifier)
        if parts:
            tokens.update(parts)
            tokens.add(identifier.lower())
    return " ".join(sorted(tokens))


def tsv_input(text: str) -> str:
    """Content plus its decomposed identifiers, ready for `to_tsvector`."""
    extra = decompose(text)
    return f"{text}\n{extra}" if extra else text
