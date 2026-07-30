"""Corpus walk and its exclusions."""

from app.ingest.checkout import WalkStats, walk_sources


def _tree(root, paths):
    for path in paths:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x")
    return root


def test_english_docs_are_indexed_and_translations_are_not(tmp_path):
    """82% of FastAPI's doc files are literal translations. bge-small-en gives
    them near-meaningless vectors, and they carry the same English code blocks,
    so the sparse arm matches them and a top-10 returns one page ten times."""
    root = _tree(
        tmp_path,
        [
            "docs/en/docs/index.md",
            "docs/es/docs/index.md",
            "docs/pt/docs/index.md",
            "docs/zh-hant/docs/index.md",
            "fastapi/routing.py",
        ],
    )
    found = {path for _, path, _ in walk_sources(root)}
    assert found == {"docs/en/docs/index.md", "fastapi/routing.py"}


def test_non_language_docs_paths_survive(tmp_path):
    """`docs/advanced/` is content, not a locale — a looser pattern would eat it."""
    root = _tree(tmp_path, ["docs/advanced/events.md", "docs/index.md"])
    found = {path for _, path, _ in walk_sources(root)}
    assert found == {"docs/advanced/events.md", "docs/index.md"}


def test_sources_are_classified(tmp_path):
    root = _tree(tmp_path, ["docs/en/x.md", "fastapi/routing.py", "README.md"])
    kinds = {path: source for source, path, _ in walk_sources(root)}
    assert kinds == {"docs/en/x.md": "docs", "README.md": "docs", "fastapi/routing.py": "code"}


def test_skip_dirs_and_unsupported_suffixes_are_dropped(tmp_path):
    root = _tree(
        tmp_path,
        [".github/workflows/ci.yml", "fastapi/__pycache__/x.py", "logo.png", "a.py"],
    )
    assert {path for _, path, _ in walk_sources(root)} == {"a.py"}


def test_exclusions_are_counted(tmp_path):
    """Recorded on the ingest run so corpus scope is auditable rather than
    buried in a constant."""
    root = _tree(tmp_path, ["docs/en/a.md", "docs/es/a.md", "docs/fr/a.md", "logo.png"])
    stats = WalkStats()
    list(walk_sources(root, stats))
    assert stats.indexed == 1
    assert stats.excluded["translated_docs"] == 2
    assert stats.excluded["unsupported_suffix"] == 1
    assert stats.as_dict()["docs_language"] == "en"
