"""The corpus working tree at a pinned revision.

Docs and code chunks need file contents, which the REST endpoints in §5.4 do
not provide. Fetching them per-blob would be ~3,100 API calls; the tarball
endpoint is one call for the whole tree, and extracting it by SHA gives the
read-only checkout §15 already assumes exists.
"""

from __future__ import annotations

import logging
import re
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

DOC_SUFFIXES = (".md",)
CODE_SUFFIXES = (".py",)

# Directories with nothing an engineering manager would ask about.
SKIP_DIRS = frozenset({".git", ".github", "__pycache__", "node_modules", "site"})

# FastAPI ships docs/en plus a dozen literal translations. Indexing them is
# actively harmful, not merely wasteful: bge-small-en produces near-meaningless
# vectors for Spanish prose, and every translated page embeds the *same*
# English Python code blocks, so the sparse arm matches them and a top-10 comes
# back as one page in ten languages. Nothing is lost -- a translation carries
# no content docs/en does not.
# Matches the locale directories FastAPI actually ships -- de, es, ja, zh,
# zh-hant, ... -- while leaving content paths like `docs/advanced/` alone.
TRANSLATED_DOCS_RE = re.compile(r"^docs/(?!en/)[a-z]{2}(-[a-z]+)?/")
DOCS_LANGUAGE = "en"


@dataclass
class WalkStats:
    """What the walk deliberately left out, so corpus scope is auditable
    rather than buried in a constant."""

    indexed: int = 0
    excluded: dict[str, int] = field(default_factory=dict)

    def _drop(self, reason: str) -> None:
        self.excluded[reason] = self.excluded.get(reason, 0) + 1

    def as_dict(self) -> dict[str, object]:
        return {
            "files_indexed": self.indexed,
            "files_excluded": self.excluded,
            "docs_language": DOCS_LANGUAGE,
        }


async def ensure_checkout(
    client: httpx.AsyncClient, repo: str, sha: str, root: Path
) -> Path:
    """Extract `repo` at `sha` under `root`, reusing it if already present."""
    target = root / sha
    if (target / ".complete").is_file():
        log.info("reusing checkout %s", target)
        return target

    target.mkdir(parents=True, exist_ok=True)
    archive = target.with_suffix(".tar.gz")
    log.info("downloading %s@%s", repo, sha[:12])

    async with client.stream("GET", f"/repos/{repo}/tarball/{sha}") as response:
        response.raise_for_status()
        with archive.open("wb") as handle:
            async for block in response.aiter_bytes(1 << 20):
                handle.write(block)

    with tarfile.open(archive) as tar:
        # The tarball wraps everything in a `<owner>-<repo>-<sha>/` directory;
        # strip it so paths match the ones GitHub reports for pr_files.
        members = tar.getmembers()
        prefix = members[0].name.split("/")[0] + "/" if members else ""
        for member in members:
            if not member.name.startswith(prefix):
                continue
            member.name = member.name[len(prefix) :]
            if member.name:
                tar.extract(member, target, filter="data")

    archive.unlink(missing_ok=True)
    (target / ".complete").touch()
    return target


def walk_sources(
    checkout: Path, stats: WalkStats | None = None
) -> Iterator[tuple[str, str, Path]]:
    """Yield `(source, repo_relative_path, absolute_path)` for indexable files."""
    stats = stats if stats is not None else WalkStats()
    for path in sorted(checkout.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(checkout)
        posix = relative.as_posix()

        if SKIP_DIRS & set(relative.parts):
            stats._drop("skipped_dir")
            continue
        if TRANSLATED_DOCS_RE.match(posix):
            stats._drop("translated_docs")
            continue

        if path.suffix in DOC_SUFFIXES:
            stats.indexed += 1
            yield "docs", posix, path
        elif path.suffix in CODE_SUFFIXES:
            stats.indexed += 1
            yield "code", posix, path
        else:
            stats._drop("unsupported_suffix")
