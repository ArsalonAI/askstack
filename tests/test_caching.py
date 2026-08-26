"""Prompt caching and tracing — TRD §9, §12.

§9 asks for one assertion: `cache_read_input_tokens > 0` on the second turn of
a session, because a caching claim nobody verifies is usually false. That one
needs the Anthropic API and is skipped without a key.

The rest run with no network, and they are the ones that catch real breakage.
A cached prefix is a byte match, so caching dies silently — a timestamp in the
system prompt or a tool list that reorders costs the entire cache with no
symptom other than the bill. Those are checkable offline and worth checking on
every run.
"""

import asyncpg
import pytest
from alembic import command

from app.config import Settings, settings
from app.facts.store import PostgresFactsStore
from app.orchestrator import SYSTEM_PROMPT, Orchestrator
from app.retrieval.hybrid import HybridRetriever
from app.tools.registry import CATALOG, ToolRegistry
from app.tools.selector import FullToolSelector, to_api_tools
from app.tracing import NullTrace, Tracer

# Opus 5's minimum cacheable prefix. A shorter prompt silently does not cache —
# no error, just cache_creation_input_tokens: 0.
MIN_CACHEABLE_TOKENS = 512
CHARS_PER_TOKEN = 4


class TestCacheInvalidants:
    """No network. These are the silent invalidators §9 audits for."""

    @pytest.mark.parametrize(
        "volatile", ["now()", "datetime", "uuid", "as_of", "session_id", "timestamp"]
    )
    def test_system_prompt_carries_nothing_per_request(self, volatile):
        """Anything interpolated into the system prompt sits at the front of
        the prefix and invalidates everything after it, every request."""
        assert volatile not in SYSTEM_PROMPT.lower()

    def test_system_prompt_is_a_plain_constant(self):
        """An f-string or `.format()` here would be the same failure arriving
        later, once someone adds a field."""
        assert "{" not in SYSTEM_PROMPT and "}" not in SYSTEM_PROMPT

    def test_the_cacheable_prefix_clears_the_minimum(self):
        """The prefix is `tools` *then* `system`, so both count toward the 512
        tokens. The system prompt alone is under it — measuring that in
        isolation would fail for a prefix that caches perfectly well."""
        import json

        prefix = json.dumps(to_api_tools(CATALOG)) + SYSTEM_PROMPT
        assert len(prefix) / CHARS_PER_TOKEN > MIN_CACHEABLE_TOKENS

    def test_tool_payload_is_byte_identical_regardless_of_input_order(self):
        """§10. `tools` renders *before* `system`, so a reordered tool list
        invalidates every breakpoint downstream of it."""
        forward = to_api_tools(CATALOG)
        reversed_ = to_api_tools(sorted(CATALOG, key=lambda t: t.name, reverse=True))
        assert forward == reversed_

    def test_the_breakpoint_sits_on_the_system_block(self):
        """§9's layout: the stable system prompt carries `cache_control`, and
        everything volatile goes after it. ADR 8 is why the memory block is
        not in there — it changes per session and would evict the prefix."""
        import inspect

        source = inspect.getsource(Orchestrator.run)
        assert '"cache_control": {"type": "ephemeral"}' in source
        breakpoint_at = source.index("cache_control")
        assert source.index("SYSTEM_PROMPT") < breakpoint_at


class TestTracingIsOptional:
    """§14.4 needs retrieval metrics reproducible with no network. A tracer
    that required a backend to no-op would make every eval depend on Langfuse."""

    def test_unconfigured_tracer_is_disabled_but_still_usable(self):
        tracer = Tracer(Settings(langfuse_public_key="", langfuse_secret_key=""))
        assert not tracer.enabled
        trace = tracer.trace("chat_request")
        assert isinstance(trace, NullTrace)
        with trace.span("tools.select") as span:
            span.update(metadata={"anything": 1})
        with trace.generation("llm.generate", model="x") as gen:
            gen.update(usage={"input": 1, "output": 2})
        trace.update(output={})
        tracer.flush()

    def test_a_span_closes_even_when_the_body_raises(self):
        """An unclosed span reads as an infinite-duration outlier in the
        latency dashboard, which is worse than no span at all."""
        with pytest.raises(ValueError, match="boom"), NullTrace().span("agent.loop"):
            raise ValueError("boom")


@pytest.mark.skipif(
    not settings.anthropic_api_key, reason="needs ANTHROPIC_API_KEY"
)
class TestLiveCaching:
    """§9's actual assertion. Two real turns; costs a few cents."""

    @pytest.fixture
    async def orchestrator(self, alembic_config, test_database):
        command.downgrade(alembic_config, "base")
        command.upgrade(alembic_config, "head")
        pool = await asyncpg.create_pool(test_database, min_size=1, max_size=4)
        from anthropic import AsyncAnthropic

        try:
            yield Orchestrator(
                pool,
                ToolRegistry(
                    PostgresFactsStore(pool), HybridRetriever(pool, None), top_k=5
                ),
                FullToolSelector(CATALOG),
                AsyncAnthropic(api_key=settings.anthropic_api_key),
                settings,
            )
        finally:
            await pool.close()
            command.downgrade(alembic_config, "base")

    async def _turn(self, orchestrator, session_id, message):
        usage, new_session = {}, session_id
        async for event, data in orchestrator.run("cachetest", session_id, message):
            if event == "session":
                new_session = data["session_id"]
            elif event == "done":
                usage = data["usage"]
            elif event == "error":
                pytest.fail(f"{data['code']}: {data['message']}")
        return new_session, usage

    async def test_second_turn_reads_the_cache(self, orchestrator):
        """The prefix is tools + system prompt, unchanged between turns, so the
        second request must hit it. A zero here means a silent invalidator has
        crept into prompt assembly."""
        message = "Reply with the single word: acknowledged. Do not call any tool."
        session_id, first = await self._turn(orchestrator, None, message)
        _, second = await self._turn(orchestrator, session_id, message)

        # Turn one may write or read depending on whether another run warmed
        # the same prefix in the last five minutes. Asserting it *wrote* would
        # make this pass or fail on test ordering rather than on caching.
        assert (
            first.get("cache_creation_input_tokens", 0)
            + first.get("cache_read_input_tokens", 0)
            > 0
        ), (
            "first turn neither wrote nor read a cache entry — the prefix is "
            f"probably under {MIN_CACHEABLE_TOKENS} tokens"
        )
        assert second.get("cache_read_input_tokens", 0) > 0, (
            "second turn read nothing from cache — something in tools[] or "
            "system[] changes between requests"
        )
