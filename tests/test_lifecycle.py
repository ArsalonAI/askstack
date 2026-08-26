"""Transcript → episodic memory — TRD §8.1.

Extraction is the one place a model writes directly into durable state, which
makes the filters around it load-bearing rather than defensive. What is tested
here is everything between the transcript and the row: what reaches the model,
what is admitted, what provenance is attached, and — most importantly — that a
failure never escapes into a request that already succeeded.

The prompt's *quality* is not testable here and is measured by the cross-session
suite (PRD §7.2), which is what extraction exists to make non-trivial.
"""

import json

import numpy as np

from app.memory.lifecycle import (
    EXTRACTION_SCHEMA,
    FACT_KINDS,
    MIN_CONFIDENCE,
    TURNS_PER_EXTRACTION,
    Extractor,
    admissible,
    render_transcript,
    should_extract,
)

DIM = 384


def fact(**over) -> dict:
    base = {
        "statement": "This manager scopes questions to the auth area.",
        "entities": [],
        "kind": "preference",
        "confidence": 0.9,
        "source_message_ids": ["msg_1"],
    }
    return {**base, **over}


class FakeEmbedder:
    model_id, dim = "fake", DIM

    def embed(self, texts):
        return np.zeros((len(texts), DIM), dtype=np.float32)

    def embed_query(self, text):
        return self.embed([text])[0]


class FakeStore:
    def __init__(self):
        self.writes: list[dict] = []

    async def write(self, user_id, mem_type, content, **kw):
        self.writes.append(
            {"user_id": user_id, "type": mem_type, "content": content, **kw}
        )
        return object()


class FakePool:
    """Returns one transcript. `fetch` is the only method the extractor uses."""

    def __init__(self, rows):
        self.rows = rows

    async def fetch(self, *args):
        return self.rows


def row(msg_id="msg_1", role="user", text="What merged in auth in June?",
        user_id="u1", trace_id="trace_1"):
    return {
        "id": msg_id,
        "role": role,
        "content": json.dumps([{"type": "text", "text": text}]),
        "user_id": user_id,
        "trace_id": trace_id,
    }


class FakeClient:
    """Stands in for AsyncAnthropic. Records the request; returns `facts`."""

    def __init__(self, facts=None, *, raises=None, stop_reason="end_turn", text=None):
        self.facts = facts if facts is not None else []
        self.raises = raises
        self.stop_reason = stop_reason
        self.text = text
        self.requests: list[dict] = []
        outer = self

        class _Messages:
            async def create(self, **kw):
                outer.requests.append(kw)
                if outer.raises:
                    raise outer.raises
                payload = (
                    outer.text
                    if outer.text is not None
                    else json.dumps({"facts": outer.facts})
                )
                block = type("B", (), {"type": "text", "text": payload})()
                return type(
                    "R",
                    (),
                    {
                        "content": [block],
                        "stop_reason": outer.stop_reason,
                        "stop_details": None,
                    },
                )()

        self.messages = _Messages()


class Settings:
    agent_model = "claude-opus-5"
    batch_effort = "low"


def extractor(rows=None, client=None, store=None) -> Extractor:
    return Extractor(
        FakePool(rows if rows is not None else [row()]),
        store or FakeStore(),
        FakeEmbedder(),
        client or FakeClient([fact()]),
        Settings(),
    )


class TestSchema:
    """The schema is what makes the output constrained rather than suggested."""

    def test_objects_forbid_extra_properties(self):
        """Without `additionalProperties: false` the constraint is advisory."""
        assert EXTRACTION_SCHEMA["additionalProperties"] is False
        item = EXTRACTION_SCHEMA["properties"]["facts"]["items"]
        assert item["additionalProperties"] is False

    def test_every_property_is_required(self):
        item = EXTRACTION_SCHEMA["properties"]["facts"]["items"]
        assert set(item["required"]) == set(item["properties"])

    def test_kind_is_an_enum_of_the_four_documented_kinds(self):
        item = EXTRACTION_SCHEMA["properties"]["facts"]["items"]
        assert item["properties"]["kind"]["enum"] == list(FACT_KINDS)
        assert set(FACT_KINDS) == {"preference", "resolution", "failure", "context"}


class TestRenderTranscript:
    def test_marks_each_message_with_its_id(self):
        """The `[id]` markers are where `source_message_ids` comes from. Without
        them the model has nothing to cite and would invent provenance."""
        rendered = render_transcript(
            [{"id": "msg_7", "role": "user", "content": [{"type": "text", "text": "hi"}]}]
        )
        assert rendered == "[msg_7] user: hi"

    def test_tool_blocks_are_excluded(self):
        """A fact extracted from a tool result is a fact about the *corpus*, not
        about this user — and the facts layer answers those better and does not
        go stale."""
        rendered = render_transcript(
            [
                {
                    "id": "msg_1",
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Looking."},
                        {"type": "tool_use", "id": "t", "name": "merged_prs", "input": {}},
                        {"type": "thinking", "thinking": "hmm"},
                    ],
                }
            ]
        )
        assert rendered == "[msg_1] assistant: Looking."

    def test_a_message_with_no_text_is_dropped_entirely(self):
        rendered = render_transcript(
            [{"id": "m", "role": "assistant", "content": [{"type": "tool_use"}]}]
        )
        assert rendered == ""

    def test_plain_string_content_is_handled(self):
        """User turns are persisted as bare strings, not block lists."""
        assert render_transcript(
            [{"id": "m", "role": "user", "content": "what shipped?"}]
        ) == "[m] user: what shipped?"


class TestAdmissible:
    def test_the_confidence_floor(self):
        assert admissible(fact(confidence=MIN_CONFIDENCE))
        assert not admissible(fact(confidence=MIN_CONFIDENCE - 0.01))

    def test_an_empty_statement_is_rejected(self):
        """The schema guarantees a string, not a *useful* one."""
        assert not admissible(fact(statement="   "))

    def test_confidence_outside_the_unit_interval_is_rejected(self):
        """`memories.confidence` has a CHECK constraint; catching it here turns
        a failed write at the end of a paid call into a dropped fact."""
        assert not admissible(fact(confidence=1.5))
        assert not admissible(fact(confidence=-0.1))

    def test_non_numeric_confidence_does_not_raise(self):
        assert not admissible(fact(confidence="high"))

    def test_missing_confidence_does_not_raise(self):
        broken = fact()
        del broken["confidence"]
        assert not admissible(broken)


class TestExtract:
    async def test_writes_an_episodic_memory_per_admitted_fact(self):
        store = FakeStore()
        report = await extractor(client=FakeClient([fact(), fact()]), store=store).extract(
            "sess_1"
        )
        assert report.written == 2
        assert all(w["type"] == "episodic" for w in store.writes)

    async def test_provenance_is_attached_to_every_write(self):
        """PRD §5.5. `created_by`, the session, the trace, and the source
        messages are what make a bad memory traceable to what produced it."""
        store = FakeStore()
        await extractor(
            client=FakeClient([fact(source_message_ids=["msg_1", "msg_2"])]), store=store
        ).extract("sess_1")
        written = store.writes[0]
        assert written["created_by"] == "extraction"
        assert written["source_session_id"] == "sess_1"
        assert written["trace_id"] == "trace_1"
        assert written["source_ids"] == ["msg_1", "msg_2"]
        assert written["user_id"] == "u1"

    async def test_low_confidence_facts_are_discarded_and_counted(self):
        report = await extractor(
            client=FakeClient([fact(), fact(confidence=0.2)])
        ).extract("sess_1")
        assert report.written == 1
        assert report.discarded_low_confidence == 1
        assert report.considered == 2

    async def test_an_empty_session_makes_no_model_call(self):
        """No transcript, no call. A paid request that can only return an empty
        list is pure waste."""
        client = FakeClient([fact()])
        report = await extractor(rows=[], client=client).extract("sess_1")
        assert client.requests == []
        assert report.written == 0

    async def test_a_transcript_of_only_tool_blocks_makes_no_model_call(self):
        client = FakeClient([fact()])
        rows = [
            {
                "id": "m",
                "role": "assistant",
                "content": json.dumps([{"type": "tool_use"}]),
                "user_id": "u1",
                "trace_id": "t",
            }
        ]
        await extractor(rows=rows, client=client).extract("sess_1")
        assert client.requests == []

    async def test_extracting_nothing_is_a_valid_outcome(self):
        report = await extractor(client=FakeClient([])).extract("sess_1")
        assert report.written == 0
        assert report.considered == 0


class TestTheRequest:
    async def test_uses_structured_output_with_the_schema(self):
        client = FakeClient([fact()])
        await extractor(client=client).extract("sess_1")
        output_config = client.requests[0]["output_config"]
        assert output_config["format"]["type"] == "json_schema"
        assert output_config["format"]["schema"] is EXTRACTION_SCHEMA

    async def test_runs_at_batch_effort_not_agent_effort(self):
        """§10 — extraction is a batch route and runs at `low`."""
        client = FakeClient([fact()])
        await extractor(client=client).extract("sess_1")
        assert client.requests[0]["output_config"]["effort"] == "low"

    async def test_sends_no_sampling_parameters(self):
        """`temperature`/`top_p`/`top_k` are rejected on Opus 5."""
        client = FakeClient([fact()])
        await extractor(client=client).extract("sess_1")
        sent = client.requests[0]
        assert not {"temperature", "top_p", "top_k"} & set(sent)

    async def test_the_transcript_reaches_the_prompt(self):
        client = FakeClient([fact()])
        await extractor(client=client).extract("sess_1")
        prompt = client.requests[0]["messages"][0]["content"]
        assert "What merged in auth in June?" in prompt
        assert "[msg_1]" in prompt


class TestFailuresNeverEscape:
    """Extraction runs after the user already has their answer. Every failure
    here must cost the *next* session some context, never surface as an error
    on a request that already succeeded."""

    async def test_an_api_error_returns_an_empty_report(self):
        report = await extractor(
            client=FakeClient(raises=RuntimeError("api down"))
        ).extract("sess_1")
        assert report.written == 0

    async def test_a_refusal_is_not_parsed_as_output(self):
        """§10 — check `stop_reason` before reading content. A refusal returns
        200 with empty content, and indexing it would raise."""
        client = FakeClient([], stop_reason="refusal")
        report = await extractor(client=client).extract("sess_1")
        assert report.written == 0

    async def test_unparseable_output_returns_an_empty_report(self):
        report = await extractor(client=FakeClient(text="not json")).extract("sess_1")
        assert report.written == 0

    async def test_output_missing_the_facts_key_returns_empty(self):
        report = await extractor(client=FakeClient(text='{"other": 1}')).extract("s")
        assert report.written == 0


class TestShouldExtract:
    """§8.1's mid-session trigger. Turns are 0-indexed."""

    def test_fires_after_the_tenth_completed_turn(self):
        assert should_extract(TURNS_PER_EXTRACTION - 1)

    def test_does_not_fire_on_the_first_turn(self):
        """Extracting after one exchange spends a call on almost nothing."""
        assert not should_extract(0)

    def test_fires_once_per_interval(self):
        fires = [t for t in range(40) if should_extract(t)]
        assert fires == [9, 19, 29, 39]


class TestPromptContract:
    """The prompt is prose, but three of its instructions are load-bearing
    enough that losing them silently would be a correctness regression."""

    def test_it_demands_self_contained_statements(self):
        from app.memory.lifecycle import EXTRACTION_PROMPT

        assert "self-contained" in EXTRACTION_PROMPT

    def test_it_forbids_recording_status_as_standing_truth(self):
        """The §5.2 failure this whole system exists to prevent: a stale status
        claim restated months later as though it were current."""
        from app.memory.lifecycle import EXTRACTION_PROMPT

        assert "never a standing truth" in EXTRACTION_PROMPT

    def test_it_states_the_confidence_floor_the_code_enforces(self):
        """If the prompt and the filter disagree, the model pads toward a
        threshold that silently discards its output."""
        from app.memory.lifecycle import EXTRACTION_PROMPT

        assert str(MIN_CONFIDENCE) in EXTRACTION_PROMPT
