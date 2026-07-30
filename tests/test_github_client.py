"""Transport behaviour of the ingest client — TRD §5.4.

Driven through httpx's MockTransport rather than the live API, so these run
offline and deterministically. What is worth testing here is the failure
handling: a full ingest is ~4,500 requests, and anything that turns one bad
response into a dead run costs an hour.
"""

import httpx
import pytest

from app.ingest.github import GitHubClient, GitHubError


def _client(handler, tmp_path, **kw) -> GitHubClient:
    gh = GitHubClient(token="t", repo="fastapi/fastapi", cache_dir=tmp_path, **kw)
    gh._client = httpx.AsyncClient(
        base_url="https://api.github.com", transport=httpx.MockTransport(handler)
    )
    return gh


def _ok(payload, **headers) -> httpx.Response:
    base = {"x-ratelimit-remaining": "4999", "x-ratelimit-reset": "9999999999"}
    return httpx.Response(200, json=payload, headers={**base, **headers})


async def test_transport_failure_is_retried(tmp_path, monkeypatch):
    """A dropped connection must not end the run. This is the bug that killed
    the first full ingest at 474 of ~1,100 pull requests."""
    monkeypatch.setattr("app.ingest.github.asyncio.sleep", _noop)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ReadTimeout("boom", request=request)
        return _ok({"sha": "abc"})

    gh = _client(handler, tmp_path)
    assert await gh.get("/x") == {"sha": "abc"}
    assert calls["n"] == 3
    assert gh.stats.transport_retries == 2


async def test_transport_failure_eventually_gives_up(tmp_path, monkeypatch):
    monkeypatch.setattr("app.ingest.github.asyncio.sleep", _noop)

    def handler(request):
        raise httpx.ConnectError("no route", request=request)

    gh = _client(handler, tmp_path)
    with pytest.raises(GitHubError, match="transport failure"):
        await gh.get("/x")


async def test_transport_retries_ride_out_a_multi_minute_outage(tmp_path, monkeypatch):
    """A DNS outage lasting tens of seconds killed a full ingest once. The
    transport budget is separate from the status-code one so a network blip
    cannot exhaust the retries reserved for server errors."""
    slept = []

    async def record(delay):
        slept.append(delay)

    monkeypatch.setattr("app.ingest.github.asyncio.sleep", record)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] <= 6:
            raise httpx.ConnectError("nodename nor servname provided", request=request)
        return _ok({"ok": True})

    gh = _client(handler, tmp_path)
    assert await gh.get("/x") == {"ok": True}
    assert slept == [1, 2, 4, 8, 16, 32]
    assert sum(slept) > 60, "must survive an outage longer than a minute"


async def test_secondary_rate_limit_waits_and_retries(tmp_path, monkeypatch):
    slept = []

    async def record(delay):
        slept.append(delay)

    monkeypatch.setattr("app.ingest.github.asyncio.sleep", record)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(403, json={}, headers={"retry-after": "7"})
        return _ok({"ok": True})

    gh = _client(handler, tmp_path)
    assert await gh.get("/x") == {"ok": True}
    assert slept == [7.0]
    assert gh.stats.rate_limit_waits == 1


async def test_exhausted_budget_pauses_before_the_next_call(tmp_path, monkeypatch):
    """Waiting beats failing: a first ingest runs close to the hourly ceiling."""
    slept = []

    async def record(delay):
        slept.append(delay)

    monkeypatch.setattr("app.ingest.github.asyncio.sleep", record)
    monkeypatch.setattr(
        "app.ingest.github.datetime", _FrozenDatetime(1_000_000.0)
    )

    def handler(request):
        return _ok({"ok": True}, **{"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1000060"})

    gh = _client(handler, tmp_path)
    await gh.get("/x")
    assert slept and 55 < slept[0] < 65


async def test_404_names_the_setting_to_check(tmp_path):
    gh = _client(lambda r: httpx.Response(404, json={}), tmp_path)
    with pytest.raises(GitHubError, match="CORPUS_REPO"):
        await gh.get("/x")


async def test_cache_hit_skips_the_network(tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return _ok([{"filename": "a.py"}])

    gh = _client(handler, tmp_path)
    first = await gh.pull_request_files(1, "2026-01-01T00:00:00Z")
    second = await gh.pull_request_files(1, "2026-01-01T00:00:00Z")
    assert first == second == [{"filename": "a.py"}]
    assert calls["n"] == 1
    assert gh.stats.cache_hits == 1


async def test_a_touched_pr_misses_the_cache(tmp_path):
    """The key carries `updated_at`, so an edited PR refetches rather than
    serving a stale file list."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return _ok([{"filename": f"{calls['n']}.py"}])

    gh = _client(handler, tmp_path)
    await gh.pull_request_files(1, "2026-01-01T00:00:00Z")
    await gh.pull_request_files(1, "2026-02-01T00:00:00Z")
    assert calls["n"] == 2


async def test_corrupt_cache_entry_is_refetched(tmp_path):
    """An interrupted run can leave a half-written file; a truncated payload
    must not become a silently missing pull request."""

    def handler(request):
        return _ok([{"filename": "a.py"}])

    gh = _client(handler, tmp_path)
    path = gh._cache_path("pulls/1@x.files")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('[{"filename": "a.p')
    assert await gh.get_cached("pulls/1@x.files", "/x", paginated=True) == [
        {"filename": "a.py"}
    ]


async def test_issues_endpoint_filters_out_pull_requests(tmp_path):
    """GitHub returns PRs from /issues; letting one through collides the
    `issues` and `pull_requests` tables."""

    def handler(request):
        return _ok(
            [
                {"number": 1},
                {"number": 2, "pull_request": {"url": "..."}},
                {"number": 3},
            ]
        )

    gh = _client(handler, tmp_path)
    assert [i["number"] async for i in gh.issues(None)] == [1, 3]


async def test_pull_requests_stop_at_the_window_floor(tmp_path):
    from datetime import UTC, datetime

    def handler(request):
        return _ok(
            [
                {"number": 3, "updated_at": "2026-01-03T00:00:00Z"},
                {"number": 2, "updated_at": "2026-01-02T00:00:00Z"},
                {"number": 1, "updated_at": "2024-01-01T00:00:00Z"},
            ]
        )

    gh = _client(handler, tmp_path)
    since = datetime(2025, 1, 1, tzinfo=UTC)
    assert [p["number"] async for p in gh.pull_requests(since)] == [3, 2]


async def _noop(_delay):
    return None


class _FrozenDatetime:
    """Stands in for the module's `datetime` so `now(UTC).timestamp()` is fixed."""

    def __init__(self, epoch: float) -> None:
        self._epoch = epoch

    def now(self, _tz=None):
        class _Now:
            def __init__(self, epoch):
                self._epoch = epoch

            def timestamp(self):
                return self._epoch

        return _Now(self._epoch)
