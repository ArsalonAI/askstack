"""GitHub REST client for corpus ingest — TRD §5.4.

Two things make this more than a thin httpx wrapper:

* **The cache.** Three calls per pull request dominate ingest wall time, so
  per-object responses are cached under `.cache/gh/` keyed by object ID and
  `updated_at`. A PR that has not been touched since the last run costs zero
  API calls on the next one.
* **The rate limiter.** A first ingest of FastAPI runs to roughly 4,500 calls
  against a 5,000/hour budget. That is close enough that a retry storm would
  push it over, so the client waits for the reset window rather than failing.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

BASE_URL = "https://api.github.com"
PER_PAGE = 100
MAX_RETRIES = 5
# Transport failures get their own, more patient budget. A 5xx is the server
# saying no and retrying fast is right; a DNS or connection failure usually
# means the local network is down for tens of seconds, and 1+2+4+8s of backoff
# is not enough to ride that out. 1,2,4,8,16,32,60,60 ≈ 3 minutes.
TRANSPORT_MAX_RETRIES = 8
TRANSPORT_MAX_DELAY = 60.0
# Generous: a `/pulls/{n}/files` call on a large translation PR is slow, and a
# timeout there costs a retry rather than just waiting a beat longer.
TIMEOUT_SECONDS = 60.0
# Leave headroom so a burst of concurrent requests can't overshoot zero.
RATE_LIMIT_FLOOR = 10


class GitHubError(RuntimeError):
    pass


@dataclass
class FetchStats:
    """What actually cost us something. Reported at the end of a run so a
    cache that has quietly stopped working is visible rather than just slow."""

    requests: int = 0
    cache_hits: int = 0
    rate_limit_waits: int = 0
    seconds_waiting: float = 0.0
    transport_retries: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "cache_hits": self.cache_hits,
            "rate_limit_waits": self.rate_limit_waits,
            "seconds_waiting": round(self.seconds_waiting, 1),
            "transport_retries": self.transport_retries,
        }


def _slug(value: str) -> str:
    """A filename-safe form of an arbitrary cache key component."""
    return "".join(c if c.isalnum() or c in "-._" else "-" for c in value)


@dataclass
class GitHubClient:
    token: str
    repo: str
    cache_dir: Path = Path(".cache/gh")
    use_cache: bool = True
    stats: FetchStats = field(default_factory=FetchStats)
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)

    async def __aenter__(self) -> GitHubClient:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "askstack-ingest",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers=headers,
            timeout=httpx.Timeout(TIMEOUT_SECONDS),
            # `tiangolo/fastapi` 301-redirects to `fastapi/fastapi`; without
            # this every call against a renamed repo returns an empty body.
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client is not None:
            await self._client.aclose()

    # -- caching ---------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        head, _, tail = key.partition("/")
        return self.cache_dir / _slug(head) / f"{_slug(tail)}.json"

    def _read_cache(self, key: str) -> Any | None:
        if not self.use_cache:
            return None
        path = self._cache_path(key)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            # A half-written file from an interrupted run. Refetch rather than
            # letting a truncated payload become a missing PR.
            log.warning("discarding corrupt cache entry %s", path)
            path.unlink(missing_ok=True)
            return None

    def _write_cache(self, key: str, payload: Any) -> None:
        if not self.use_cache:
            return
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(path)  # atomic, so a Ctrl-C can't leave a partial entry

    # -- transport -------------------------------------------------------

    async def _sleep_until_reset(self, response: httpx.Response) -> None:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            delay = float(retry_after)
        else:
            reset = response.headers.get("x-ratelimit-reset")
            if reset is None:
                delay = 60.0
            else:
                delay = max(0.0, float(reset) - datetime.now(UTC).timestamp()) + 1.0
        self.stats.rate_limit_waits += 1
        self.stats.seconds_waiting += delay
        log.warning("rate limited; sleeping %.0fs", delay)
        await asyncio.sleep(delay)

    def _note_budget(self, response: httpx.Response) -> float | None:
        """Returns seconds to wait if the remaining budget is nearly gone."""
        remaining = response.headers.get("x-ratelimit-remaining")
        reset = response.headers.get("x-ratelimit-reset")
        if remaining is None or reset is None:
            return None
        if int(remaining) > RATE_LIMIT_FLOOR:
            return None
        return max(0.0, float(reset) - datetime.now(UTC).timestamp()) + 1.0

    async def _request(self, path: str, params: Mapping[str, Any] | None) -> httpx.Response:
        assert self._client is not None, "use GitHubClient as an async context manager"
        attempt = 0
        transport_attempt = 0
        while attempt < MAX_RETRIES:
            try:
                response = await self._client.get(path, params=params)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                # A dropped connection across ~10,000 requests must not take
                # down an hour-long ingest. Status-code retries alone are not
                # enough: these never produce a response to inspect. Counted
                # separately so a network outage does not consume the budget
                # reserved for server errors.
                self.stats.transport_retries += 1
                transport_attempt += 1
                if transport_attempt >= TRANSPORT_MAX_RETRIES:
                    raise GitHubError(f"transport failure on {path}: {exc!r}") from exc
                delay = min(2.0 ** (transport_attempt - 1), TRANSPORT_MAX_DELAY)
                log.warning("%s on %s; retrying in %.0fs", type(exc).__name__, path, delay)
                await asyncio.sleep(delay)
                continue
            attempt += 1
            self.stats.requests += 1

            if response.status_code in (403, 429):
                # Secondary rate limit, or the primary budget exhausted.
                await self._sleep_until_reset(response)
                continue
            if response.status_code >= 500:
                delay = 2.0**attempt
                log.warning("%s on %s; retrying in %.0fs", response.status_code, path, delay)
                await asyncio.sleep(delay)
                continue
            if response.status_code == 404:
                raise GitHubError(f"404 for {path} — check CORPUS_REPO")
            if response.status_code >= 400:
                raise GitHubError(f"{response.status_code} for {path}: {response.text[:200]}")

            wait = self._note_budget(response)
            if wait is not None:
                self.stats.rate_limit_waits += 1
                self.stats.seconds_waiting += wait
                log.warning("rate budget nearly spent; sleeping %.0fs", wait)
                await asyncio.sleep(wait)
            return response

        raise GitHubError(f"giving up on {path} after {MAX_RETRIES} attempts")

    async def get(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        response = await self._request(path, params)
        return response.json()

    async def get_cached(
        self,
        cache_key: str,
        path: str,
        params: Mapping[str, Any] | None = None,
        *,
        paginated: bool = False,
    ) -> Any:
        cached = self._read_cache(cache_key)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached
        payload = (
            [item async for item in self.paginate(path, params)]
            if paginated
            else await self.get(path, params)
        )
        self._write_cache(cache_key, payload)
        return payload

    async def paginate(
        self, path: str, params: Mapping[str, Any] | None = None
    ) -> AsyncIterator[dict]:
        """Walk `Link: rel="next"` rather than incrementing a page counter —
        GitHub caps `page` at 100 on some endpoints and the cursor does not."""
        query = {"per_page": PER_PAGE, **(params or {})}
        url: str | None = path
        while url:
            response = await self._request(url, query)
            page = response.json()
            if not isinstance(page, list):
                raise GitHubError(f"expected a list from {url}, got {type(page).__name__}")
            for item in page:
                yield item
            url = response.links.get("next", {}).get("url")
            query = None  # the next link already carries the query string

    # -- repository endpoints (TRD §5.4) ---------------------------------

    async def resolve_ref(self, ref: str) -> tuple[str, datetime]:
        """`CORPUS_REF` -> (sha, committed_at). A moving branch is not a pinned
        revision (TRD §5.1); this is what pins it."""
        payload = await self.get(f"/repos/{self.repo}/commits/{ref}")
        sha = payload["sha"]
        committed = payload["commit"]["committer"]["date"]
        return sha, datetime.fromisoformat(committed)

    async def pull_requests(self, since: datetime | None) -> AsyncIterator[dict]:
        """Newest-updated first, stopping at `since`.

        Sorting by `updated` rather than `created` is deliberate: a PR merged
        inside the window necessarily has `updated_at >= merged_at`, so this
        cannot miss one that was opened before the floor and merged after it.
        Sorting by `created` would.
        """
        params = {"state": "all", "sort": "updated", "direction": "desc"}
        async for pr in self.paginate(f"/repos/{self.repo}/pulls", params):
            if since and datetime.fromisoformat(pr["updated_at"]) < since:
                return
            yield pr

    async def pull_request_files(self, number: int, updated_at: str) -> list[dict]:
        return await self.get_cached(
            f"pulls/{number}@{updated_at}.files",
            f"/repos/{self.repo}/pulls/{number}/files",
            paginated=True,
        )

    async def pull_request_reviews(self, number: int, updated_at: str) -> list[dict]:
        return await self.get_cached(
            f"pulls/{number}@{updated_at}.reviews",
            f"/repos/{self.repo}/pulls/{number}/reviews",
            paginated=True,
        )

    async def issues(self, since: datetime | None) -> AsyncIterator[dict]:
        """Issues only — this endpoint returns pull requests too, and letting
        one through collides the `issues` and `pull_requests` tables."""
        params: dict[str, Any] = {"state": "all", "sort": "updated", "direction": "desc"}
        if since:
            params["since"] = since.isoformat()
        async for item in self.paginate(f"/repos/{self.repo}/issues", params):
            if "pull_request" in item:
                continue
            yield item

    async def commits(self, since: datetime | None) -> AsyncIterator[dict]:
        params: dict[str, Any] = {}
        if since:
            params["since"] = since.isoformat()
        async for commit in self.paginate(f"/repos/{self.repo}/commits", params):
            yield commit

    async def commit_detail(self, sha: str) -> dict:
        """The list endpoint omits `files`; only the detail endpoint has them.
        Commits are immutable, so the cache key needs no version component."""
        return await self.get_cached(f"commits/{sha}", f"/repos/{self.repo}/commits/{sha}")

    async def issue_comments(self, number: int) -> list[dict]:
        """Comment threads for issue chunking (§5.2).

        Keyed by number alone: a closed issue at a pinned revision does not
        acquire new comments that matter to us, and re-fetching 3,500 threads
        on every run would double ingest time for nothing.
        """
        return await self.get_cached(
            f"issue-comments/{number}",
            f"/repos/{self.repo}/issues/{number}/comments",
            paginated=True,
        )

    async def releases(self) -> list[dict]:
        """Small, and re-fetched whole each run (TRD §5.4)."""
        return [r async for r in self.paginate(f"/repos/{self.repo}/releases")]
