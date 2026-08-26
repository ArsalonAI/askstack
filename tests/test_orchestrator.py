"""Orchestrator internals — TRD §10, §11.2.

The agent loop itself is exercised end to end by `evals/runner.py --agent`.
What is unit-tested here is the bookkeeping the loop depends on and that fails
quietly when wrong: which citations count as "looked up this turn", what a
`retrieval` event carries per substrate, what gets persisted for replay, and
how a turn is costed.
"""

from datetime import UTC, datetime

import pytest

from app.interfaces import Aggregate, Chunk, Entity
from app.orchestrator import (
    SDK_ONLY_BLOCK_FIELDS,
    Turn,
    _retrieval_event,
    _wire_block,
    replayable,
    sse,
)
from app.tools.registry import ToolOutcome


def entity(ref="16105", kind="pr", state="merged") -> Entity:
    return Entity(
        kind=kind,
        ref=ref,
        title="Fix background tasks",
        author="tiangolo",
        state=state,
        at=datetime(2026, 7, 20, tzinfo=UTC),
        citation=f"{kind}:{ref}",
        url="https://github.com/fastapi/fastapi/pull/16105",
    )


def chunk(cid="issue:98#comment-0") -> Chunk:
    return Chunk(
        id=cid, source="issue", path="issues/98", anchor="comment-0",
        content="text", citation=cid, score=0.42,
    )


def aggregate(entities=()) -> Aggregate:
    return Aggregate(
        entities=tuple(entities),
        count=len(entities),
        window=(datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 7, 29, tzinfo=UTC)),
        area="auth",
        rendered="**rendered**",
    )


def outcome(name, result) -> ToolOutcome:
    return ToolOutcome(name=name, rendered="x", result=result, input={"query": "q"})


class TestTurnResultSet:
    """§11.2's second half — the load-bearing one. A citation resolves only if
    the turn actually looked the thing up."""

    def test_aggregate_contributes_every_entity(self):
        turn = Turn()
        turn.record(outcome("merged_prs", aggregate([entity("1"), entity("2")])))
        assert turn.entity_results == {"pr:1", "pr:2"}

    def test_single_entity_lookup_contributes_itself(self):
        turn = Turn()
        turn.record(outcome("pr_state", entity()))
        assert turn.entity_results == {"pr:16105"}

    def test_chunks_contribute_to_the_span_set_not_the_entity_set(self):
        """Spans resolve against `chunks`; entities against the facts layer.
        Mixing them makes every citation look resolved."""
        turn = Turn()
        turn.record(outcome("search_issues", [chunk()]))
        assert turn.span_results == {"issue:98#comment-0"}
        assert turn.entity_results == set()

    def test_a_failed_tool_contributes_nothing(self):
        turn = Turn()
        turn.record(ToolOutcome(name="pr_state", rendered="err", result=None, is_error=True))
        assert not turn.entity_results and not turn.span_results


class TestCost:
    def test_cached_reads_are_cheaper_than_fresh_input(self):
        cached, fresh = Turn(), Turn()
        cached.usage = {"input_tokens": 0, "cache_read_input_tokens": 10_000}
        fresh.usage = {"input_tokens": 10_000, "cache_read_input_tokens": 0}
        assert cached.cost_usd < fresh.cost_usd

    def test_usage_accumulates_across_loop_iterations(self):
        class Usage:
            input_tokens = 100
            output_tokens = 50
            cache_read_input_tokens = 0
            cache_creation_input_tokens = 0

        turn = Turn()
        turn.add_usage(Usage())
        turn.add_usage(Usage())
        assert turn.usage["input_tokens"] == 200

    def test_missing_usage_fields_do_not_crash(self):
        turn = Turn()
        turn.add_usage(object())
        assert turn.cost_usd == 0.0


class TestRetrievalEvent:
    def test_structured_shape_carries_the_window_and_count(self):
        event = _retrieval_event(outcome("merged_prs", aggregate([entity()])))
        assert event["kind"] == "structured"
        assert event["tool"] == "merged_prs"
        assert event["count"] == 1
        assert event["area"] == "auth"
        assert event["window"][0].startswith("2026-01-01")

    def test_semantic_shape_carries_citations_and_scores(self):
        event = _retrieval_event(outcome("search_issues", [chunk()]))
        assert event["kind"] == "semantic"
        assert event["chunks"][0]["citation"] == "issue:98#comment-0"
        assert event["chunks"][0]["score"] == pytest.approx(0.42)

    def test_both_shapes_name_their_tool(self):
        """§11.2. Without this a turn's retrievals are unattributable once a
        consumer merges them — which is what §14.1 scores and what PRD §5.6's
        view has to show."""
        for produced in (
            outcome("search_issues", [chunk()]),
            outcome("merged_prs", aggregate([entity()])),
            outcome("pr_state", entity()),
        ):
            assert _retrieval_event(produced)["tool"] == produced.name

    def test_a_single_entity_still_emits_an_event(self):
        """`pr_state` is how §5.2 verification happens; a turn that verified
        and emitted nothing would score as never having checked."""
        event = _retrieval_event(outcome("pr_state", entity()))
        assert event["count"] == 1
        assert event["entities"][0]["state"] == "merged"

    def test_an_error_emits_nothing(self):
        assert _retrieval_event(ToolOutcome(name="pr_state", rendered="e", result=None)) is None


class TestWireBlocks:
    """Replay fidelity. `messages.content` is documented as verbatim, and
    verbatim has to mean the API accepts it back on the next turn."""

    class Block:
        def __init__(self, payload):
            self._payload = payload

        def model_dump(self, mode=None):
            return dict(self._payload)

    def test_sdk_only_fields_are_dropped(self):
        block = self.Block({"type": "text", "text": "hi", "parsed_output": {"a": 1}})
        assert _wire_block(block) == {"type": "text", "text": "hi"}

    def test_nulls_are_dropped(self):
        block = self.Block({"type": "text", "text": "hi", "citations": None})
        assert "citations" not in _wire_block(block)

    def test_tool_use_blocks_keep_the_fields_replay_needs(self):
        block = self.Block(
            {"type": "tool_use", "id": "toolu_1", "name": "pr_state", "input": {"number": 1}}
        )
        assert _wire_block(block) == {
            "type": "tool_use", "id": "toolu_1", "name": "pr_state",
            "input": {"number": 1},
        }

    def test_the_denylist_is_not_empty(self):
        """If this ever empties, the 400 that motivated it comes back."""
        assert SDK_ONLY_BLOCK_FIELDS


class TestSSEFraming:
    def test_frame_shape(self):
        assert sse("token", {"text": "hi"}) == 'event: token\ndata: {"text": "hi"}\n\n'

    def test_datetimes_serialise(self):
        frame = sse("retrieval", {"at": datetime(2026, 7, 29, tzinfo=UTC)})
        assert "2026-07-29" in frame

    def test_payload_is_one_line(self):
        """A newline inside `data:` would split the frame and desync every
        client reading the stream."""
        body = sse("token", {"text": "a\nb"}).split("data: ")[1]
        assert body.count("\n") == 2  # the two that terminate the frame


class TestMemoryRetrievalEvents:
    """Memory is neither retrieval substrate."""

    @pytest.mark.parametrize("name", ["memory_search", "memory_write"])
    def test_memory_tools_emit_no_retrieval_event(self, name):
        """A remembered pull request must not enter this turn's result set. If
        it did, §11.2's second citation check would start passing for entities
        the agent never actually looked up — which is the exact failure that
        check exists to catch."""
        assert _retrieval_event(outcome(name, [chunk()])) is None


class TestMemoryBlockPlacement:
    """ADR 8 and §9. The block's *position* is the whole decision, and getting
    it wrong costs a cache hit on every session with no other symptom."""

    def test_the_system_prompt_never_carries_the_block(self):
        """In the system prompt the block would invalidate the cached prefix on
        every new session. The breakpoint is on system[0] and must stay stable
        across sessions."""
        from app.orchestrator import SYSTEM_PROMPT

        assert "[semantic" not in SYSTEM_PROMPT
        assert "[episodic" not in SYSTEM_PROMPT
        assert "{" not in SYSTEM_PROMPT  # no interpolation slot at all


class TestReplayableHistory:
    """A stored turn replayed as history must be a valid request.

    Found by the cross-session suite on its first run, not by a unit test: a
    `tool_use` block was persisted verbatim (correct, §4) with no `tool_result`
    to follow it, so every second turn that had called a tool returned 400.
    Single-turn evals never saw it, and the caching test's second turn happened
    not to call a tool.
    """

    def test_tool_use_is_dropped_on_replay(self):
        content = [
            {"type": "text", "text": "Looking that up."},
            {"type": "tool_use", "id": "toolu_1", "name": "merged_prs", "input": {}},
        ]
        assert replayable(content) == [{"type": "text", "text": "Looking that up."}]

    def test_tool_result_is_dropped_too(self):
        """The other half of the pair. A result with no preceding use is the
        same error from the other side."""
        content = [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "x"}]
        assert replayable(content) == []

    def test_text_only_content_is_untouched(self):
        content = [{"type": "text", "text": "PR 15806 did not ship."}]
        assert replayable(content) == content

    def test_a_plain_string_message_passes_through(self):
        """User turns are persisted as bare strings, not block lists."""
        assert replayable("what shipped last month?") == "what shipped last month?"

    def test_thinking_blocks_are_dropped_too(self):
        """The second half of the same bug, and the reason this is an allowlist.

        Removing `tool_use` from a message that also thought counts as
        *modifying* the thinking block, which the API rejects on its own — so
        dropping one without the other just trades a paired-block error for a
        thinking-block error. The first fix did exactly that.
        """
        content = [
            {"type": "thinking", "thinking": "..."},
            {"type": "text", "text": "hi"},
            {"type": "tool_use", "id": "t", "name": "merged_prs", "input": {}},
        ]
        assert replayable(content) == [{"type": "text", "text": "hi"}]
