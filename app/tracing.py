"""Langfuse spans — TRD §12.

Every §13 latency budget maps to a span here, which is what makes those
numbers observable rather than aspirational. The span tree is also how a
memory row is traced back to the request that wrote it (ADR 5) from M3.

**Tracing is optional and silent when unconfigured.** No Langfuse keys means a
null tracer with the same interface, because §14.4 requires retrieval metrics
to be reproducible with no network at all — a CI run that needs an observability
backend to compute recall@5 is a CI run that breaks for reasons unrelated to
retrieval. A tracing failure must never take a request with it, either: the
whole module is best-effort by construction.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

from app.config import Settings

log = logging.getLogger(__name__)


class NullSpan:
    def update(self, **kwargs: Any) -> None: ...
    def end(self, **kwargs: Any) -> None: ...


class NullTrace:
    id: str | None = None

    @contextmanager
    def span(self, name: str, **kwargs: Any):
        yield NullSpan()

    @contextmanager
    def generation(self, name: str, **kwargs: Any):
        yield NullSpan()

    def update(self, **kwargs: Any) -> None: ...


class LangfuseTrace:
    """One `/chat` request. Spans are context managers so a raised exception
    still closes them — an unclosed span is worse than no span, because it
    shows up as an infinite-duration outlier in the latency dashboard."""

    def __init__(self, trace) -> None:
        self._trace = trace
        self.id = trace.id

    @contextmanager
    def _observation(self, factory, name: str, **kwargs: Any):
        try:
            observation = factory(name=name, **kwargs)
        except Exception:  # noqa: BLE001
            log.debug("tracing: could not open span %s", name, exc_info=True)
            yield NullSpan()
            return
        try:
            yield observation
        finally:
            try:
                observation.end()
            except Exception:  # noqa: BLE001
                log.debug("tracing: could not close span %s", name, exc_info=True)

    @contextmanager
    def span(self, name: str, **kwargs: Any):
        with self._observation(self._trace.span, name, **kwargs) as span:
            yield span

    @contextmanager
    def generation(self, name: str, **kwargs: Any):
        with self._observation(self._trace.generation, name, **kwargs) as gen:
            yield gen

    def update(self, **kwargs: Any) -> None:
        try:
            self._trace.update(**kwargs)
        except Exception:  # noqa: BLE001
            log.debug("tracing: could not update trace", exc_info=True)


class Tracer:
    def __init__(self, settings: Settings) -> None:
        self._client = None
        if not (settings.langfuse_public_key and settings.langfuse_secret_key):
            log.info("langfuse not configured; tracing disabled")
            return
        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
        except Exception:  # noqa: BLE001
            # An unreachable or misconfigured backend must not stop the service.
            log.warning("langfuse unavailable; tracing disabled", exc_info=True)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def trace(self, name: str, **kwargs: Any) -> LangfuseTrace | NullTrace:
        if self._client is None:
            return NullTrace()
        try:
            return LangfuseTrace(self._client.trace(name=name, **kwargs))
        except Exception:  # noqa: BLE001
            log.debug("tracing: could not open trace", exc_info=True)
            return NullTrace()

    def flush(self) -> None:
        """Langfuse batches in a background thread; a short-lived process exits
        before the queue drains without this."""
        if self._client is not None:
            try:
                self._client.flush()
            except Exception:  # noqa: BLE001
                log.debug("tracing: flush failed", exc_info=True)
