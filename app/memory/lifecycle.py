"""Transcript → episodic memory — TRD §8.1.

This is the half of memory that M3 left missing. The Manager can load memories
and the agent can write one when it thinks to, but `memory_write` is optional
by design (§2.1 step 4) and fires rarely: across the twelve sessions of the
first cross-session smoke run the agent wrote exactly one memory, in the last
session. Nothing was ever available to load. Extraction is what makes the
memory non-empty without depending on the agent noticing.

One Claude call per session, structured output, `low` effort (§10). Runs as a
background task so it never blocks a response — a manager waiting on their
answer should not pay for the bookkeeping that helps their *next* session.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.interfaces import Memory

log = logging.getLogger(__name__)

# §8.1. Below this a fact is discarded rather than stored at low confidence:
# the confidence decay in §8.3 already starts a memory sliding toward the 0.3
# floor, so anything admitted under 0.5 is close to unusable on arrival and
# only adds noise to the block's token budget.
MIN_CONFIDENCE = 0.5

# §8.1's trigger. Long sessions extract mid-flight so a manager who never ends
# a session still accumulates memory.
TURNS_PER_EXTRACTION = 10

FACT_KINDS = ("preference", "resolution", "failure", "context")

# §8.1, verbatim. `additionalProperties: false` and a complete `required` list
# are what make the output actually constrained rather than merely suggested.
EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "entities": {"type": "array", "items": {"type": "string"}},
                    "kind": {"type": "string", "enum": list(FACT_KINDS)},
                    "confidence": {"type": "number"},
                    "source_message_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "statement",
                    "entities",
                    "kind",
                    "confidence",
                    "source_message_ids",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["facts"],
    "additionalProperties": False,
}

EXTRACTION_PROMPT = """\
You are reading one session between an engineering manager and an assistant \
that answers questions about a code repository's delivery state. Extract facts \
worth carrying into this manager's *next* session.

Extract four things and nothing else:

1. **Workstreams and areas they ask about repeatedly.** "Asks about the auth \
area" is worth remembering; "asked a question" is not.
2. **People they track.** Contributors they name, follow up on, or compare.
3. **Standing constraints they state.** "I don't care about docs-only PRs", \
"only routing work matters to me". These are preferences and they persist.
4. **The answer they were given, along with its date.** This is what makes \
"what's changed since we last spoke" resolvable without asking them when we \
last spoke.

Rules that matter more than coverage:

- **Every statement must be self-contained.** It will be read months later with \
none of this conversation around it. "They said it doesn't work" is worse than \
useless. Name the subject, the area, and the date.
- **A status claim is a claim about a moment, never a standing truth.** Write \
"As of {as_of}, the auth migration was three PRs from done" — never "the auth \
migration is three PRs from done". Status goes stale and a memory that reads as \
permanent will be restated as current months later, which is the exact failure \
this system exists to prevent.
- **Confidence is how durably true the fact is**, not how clearly it was said. \
A stated preference is high. An inferred interest is low. Anything below 0.5 \
will be discarded, so do not pad.
- **`entities` are citations** — `pr:15806`, `issue:1234`, `release:0.141.0`, \
`commit:a3f1c9d` — for the objects the fact is about. Empty is fine.
- **`source_message_ids`** are the message ids the fact came from, drawn from \
the transcript's `[id]` markers. This is the provenance trail; do not invent one.

Extract nothing if the session carries nothing durable. An empty list is a \
correct answer and a better one than four vague facts.

Session date: {as_of}

<transcript>
{transcript}
</transcript>
"""


@dataclass(frozen=True)
class ExtractionReport:
    """What one extraction pass did. Returned rather than logged so a caller
    (the eval, an admin endpoint) can assert on it."""

    session_id: str
    considered: int
    written: int
    discarded_low_confidence: int
    memories: Sequence[Memory] = ()

    @property
    def skipped(self) -> int:
        return self.considered - self.written


def render_transcript(messages: Sequence[dict]) -> str:
    """The transcript the model reads.

    Only user and assistant *text* survives. Tool calls and their results are
    the system's own working — the manager never saw them, and a fact extracted
    from a tool result is a fact about the corpus rather than about this user,
    which the facts layer already answers better and more reliably.
    """
    lines: list[str] = []
    for message in messages:
        text = _text_of(message["content"])
        if not text.strip():
            continue
        lines.append(f"[{message['id']}] {message['role']}: {text}")
    return "\n\n".join(lines)


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def admissible(fact: dict) -> bool:
    """§8.1's floor, plus the shape checks the schema cannot express.

    `output_config.format` guarantees the *shape*; it cannot guarantee a
    non-empty statement or a confidence inside [0, 1], and the `memories`
    table's CHECK constraint on confidence would turn either into a failed
    write at the end of a paid call.
    """
    statement = str(fact.get("statement") or "").strip()
    if not statement:
        return False
    try:
        confidence = float(fact["confidence"])
    except (KeyError, TypeError, ValueError):
        return False
    return MIN_CONFIDENCE <= confidence <= 1.0


class Extractor:
    """§3.7. Transcript in, episodic memories out."""

    def __init__(self, pool, store, embedder, client, settings) -> None:
        self.pool = pool
        self.store = store
        self.embedder = embedder
        self.client = client
        self.settings = settings

    async def extract(self, session_id: str) -> ExtractionReport:
        messages, user_id, trace_id = await self._transcript(session_id)
        if not messages:
            return ExtractionReport(session_id, considered=0, written=0,
                                    discarded_low_confidence=0)

        transcript = render_transcript(messages)
        if not transcript.strip():
            return ExtractionReport(session_id, considered=0, written=0,
                                    discarded_low_confidence=0)

        facts = await self._call(transcript)
        keep = [f for f in facts if admissible(f)]
        written: list[Memory] = []
        for fact in keep:
            statement = fact["statement"].strip()
            memory = await self.store.write(
                user_id,
                "episodic",
                statement,
                embedding=self.embedder.embed([statement])[0],
                entities=list(fact.get("entities") or []),
                confidence=float(fact["confidence"]),
                created_by="extraction",
                source_session_id=session_id,
                # The provenance trail §5.5 requires: which messages this fact
                # came from, so a bad memory can be traced to what produced it.
                source_ids=list(fact.get("source_message_ids") or []),
                trace_id=trace_id,
            )
            written.append(memory)

        return ExtractionReport(
            session_id=session_id,
            considered=len(facts),
            written=len(written),
            discarded_low_confidence=len(facts) - len(keep),
            memories=written,
        )

    async def _transcript(self, session_id: str) -> tuple[list[dict], str, str | None]:
        rows = await self.pool.fetch(
            "SELECT m.id, m.role, m.content, m.trace_id, s.user_id"
            " FROM messages m JOIN sessions s ON s.id = m.session_id"
            " WHERE m.session_id = $1 AND m.role IN ('user','assistant')"
            " ORDER BY m.turn, m.role",
            session_id,
        )
        if not rows:
            return [], "", None
        messages = [
            {"id": r["id"], "role": r["role"], "content": json.loads(r["content"])}
            for r in rows
        ]
        return messages, rows[0]["user_id"], rows[-1]["trace_id"]

    async def _call(self, transcript: str) -> list[dict]:
        """One structured-output call. Never raises.

        Extraction runs in the background, after the user already has their
        answer. A failure here should cost the next session some context, not
        surface as an error on a request that already succeeded — so this
        swallows and logs rather than propagating.
        """
        as_of = datetime.now(UTC).date().isoformat()
        try:
            response = await self.client.messages.create(
                model=self.settings.agent_model,
                max_tokens=EXTRACTION_MAX_TOKENS,
                output_config={
                    "effort": self.settings.batch_effort,
                    "format": {"type": "json_schema", "schema": EXTRACTION_SCHEMA},
                },
                messages=[
                    {
                        "role": "user",
                        "content": EXTRACTION_PROMPT.format(
                            as_of=as_of, transcript=transcript
                        ),
                    }
                ],
            )
        except Exception:  # noqa: BLE001 — background work, never fails a turn
            log.warning("extraction call failed", exc_info=True)
            return []

        # §10: check stop_reason before reading content. A refusal returns 200
        # with an empty content list, and indexing it would raise inside what
        # is supposed to be a best-effort background task.
        if response.stop_reason == "refusal":
            log.warning("extraction refused: %s", response.stop_details)
            return []
        text = next((b.text for b in response.content if b.type == "text"), "")
        if not text:
            return []
        try:
            return list(json.loads(text).get("facts") or [])
        except (json.JSONDecodeError, AttributeError):
            log.warning("extraction returned unparseable output")
            return []


# §10: 16000 on non-streaming batch routes, sized so thinking plus the response
# fits. `max_tokens` caps thinking *and* output together, so a value tuned on a
# non-thinking model truncates mid-answer.
EXTRACTION_MAX_TOKENS = 16000


def should_extract(turn: int) -> bool:
    """§8.1's mid-session trigger.

    Turns are 0-indexed, so this fires after the 10th, 20th, … completed turn.
    A manager who never formally ends a session still accumulates memory, which
    matters because nothing in the API forces a session to end.
    """
    return turn > 0 and (turn + 1) % TURNS_PER_EXTRACTION == 0
