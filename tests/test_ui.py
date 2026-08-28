"""The transparency view — PRD §5.6, TRD §11.2, §17 Q8.

A static file cannot be unit-tested the way a module can, and mostly should not
be. What *is* worth asserting is the contract between it and the server,
because that contract breaks silently: add an SSE event to §11.2, forget the
view, and the pane it belongs in simply never updates. Nothing errors.

The rendering itself is exercised by opening it.
"""

import re
from pathlib import Path

import pytest

from app.main import UI

# §11.2's catalog. `token` and `error` are handled in the stream loop rather
# than the event switch, so both spellings are searched for.
SSE_EVENTS = (
    "session",
    "memory_loaded",
    "tools_selected",
    "retrieval",
    "token",
    "tool_call",
    "citation",
    "memory_write",
    "done",
    "error",
)


@pytest.fixture(scope="module")
def source() -> str:
    return UI.read_text()


def test_the_file_the_service_serves_exists(source):
    assert UI.name == "index.html"
    assert source.lstrip().startswith("<!doctype html>")


@pytest.mark.parametrize("event", SSE_EVENTS)
def test_every_sse_event_is_handled(source, event):
    """The silent failure this guards: a new event lands in §11.2, the view
    does not know about it, and its pane stops updating with no error."""
    assert f'"{event}"' in source, f"the view ignores the `{event}` event"


def test_the_view_renders_all_four_things_5_6_requires(source):
    """§5.6: "which memories were loaded and why, which tools were selected out
    of the full catalog, which sources were consulted, and which of those the
    answer actually cited"."""
    for required in ("Memory loaded", "Tools selected", "Sources consulted"):
        assert required in source
    # "out of the full catalog" — a selected count with no denominator does not
    # answer the question §5.4 is about.
    assert "catalog_size" in source


def test_citation_verdicts_come_from_the_server_not_the_client(source):
    """§11.2 computes `resolved` and `in_result_set` server-side, and the same
    check runs in production and in eval. A view that re-derived them would be
    showing a second opinion."""
    assert "data.resolved && data.in_result_set" in source


def test_it_does_not_poll(source):
    """§11.2: "the UI is driven entirely by these — nothing is polled"."""
    assert "setInterval" not in source
    assert "setTimeout" not in source


def test_it_splits_sse_frames_on_the_blank_line(source):
    r"""Splitting on "\n" alone tears a frame in half whenever a network chunk
    boundary lands mid-event, which is intermittent and looks like data loss."""
    assert '"\\n\\n"' in source


def test_no_external_resources(source):
    """No CDN, no fonts, no build step — the view has to work from a checkout
    with the service running and nothing else."""
    assert "http://" not in source.replace("http://localhost", "")
    assert "https://" not in source
    assert "<script src" not in source


class TestCitationPattern:
    """The view highlights citations inside streamed prose. Its regex has to
    match the §5.1 grammar the server actually emits, or the answer renders
    with none of them marked."""

    @pytest.fixture
    def pattern(self, source) -> re.Pattern:
        raw = re.search(r"const CITE = /(.+?)/g;", source).group(1)
        return re.compile(raw.replace("(?:", "(?:"))

    @pytest.mark.parametrize(
        "citation",
        [
            "pr:15806",
            "issue:1234",
            "commit:a3f1c9d",
            "release:0.141.0",
            "docs:advanced/events.md#lifespan",
            "code:fastapi/routing.py:L280-L340",
            "issue:1234#comment-5",
            "issue:1234#body",
        ],
    )
    def test_it_matches_every_citation_form(self, pattern, citation):
        assert pattern.fullmatch(citation), citation

    def test_it_stops_at_sentence_punctuation(self, pattern):
        """"…shipped in pr:15806." must not swallow the full stop into the
        citation, or the highlighted token stops matching the server's."""
        assert pattern.search("shipped in pr:15806.").group(0) == "pr:15806"
        assert pattern.search("(see pr:15806)").group(0) == "pr:15806"
        assert pattern.search("pr:15806, and").group(0) == "pr:15806"

    def test_it_ignores_bare_prose(self, pattern):
        assert not pattern.search("the pull request shipped")


def test_the_readme_layout_mentions_the_view():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    assert "ui/" in readme
